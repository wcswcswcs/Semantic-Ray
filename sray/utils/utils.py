# GeoNeRF is a generalizable NeRF model that renders novel views
# without requiring per-scene optimization. This software is the 
# implementation of the paper "GeoNeRF: Generalizing NeRF with 
# Geometry Priors" by Mohammad Mahdi Johari, Yann Lepoittevin,
# and Francois Fleuret.

# Copyright (c) 2022 ams International AG

# This file is part of GeoNeRF.
# GeoNeRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# GeoNeRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GeoNeRF. If not, see <http://www.gnu.org/licenses/>.

# This file incorporates work covered by the following copyright and  
# permission notice:

    # MIT License

    # Copyright (c) 2021 apchenstu

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import cv2
import re

from PIL import Image
from kornia.utils import create_meshgrid

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).to(x.device))


def load_ckpt(network, ckpt_file, key_prefix, strict=True):
    ckpt_dict = torch.load(ckpt_file)

    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]

    state_dict = {}
    for key, val in ckpt_dict.items():
        if key_prefix in key:
            state_dict[key[len(key_prefix) + 1 :]] = val
    network.load_state_dict(state_dict, strict)


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction="mean")
        self.loss_ray = nn.SmoothL1Loss(reduction="none")

    def forward(self, inputs, targets):
        loss = 0
        if isinstance(inputs, dict):
            for l in range(self.levels):
                depth_pred_l = inputs[f"level_{l}"]
                V = depth_pred_l.shape[1]

                depth_gt_l = targets[f"level_{l}"]
                depth_gt_l = depth_gt_l[:, :V]
                mask_l = depth_gt_l > 0

                loss = loss + self.loss(
                    depth_pred_l[mask_l], depth_gt_l[mask_l]
                ) * 2 ** (1 - l)
        else:
            mask = targets > 0
            loss = loss + (self.loss_ray(inputs, targets) * mask).sum() / len(mask)

        return loss


def self_supervision_loss(
    loss_fn,
    rays_pixs,
    rendered_depth,
    depth_map,
    rays_gt_rgb,
    unpre_imgs,
    rendered_rgb,
    intrinsics,
    c2ws,
    w2cs,
):
    loss = 0
    target_points = torch.stack(
        [rays_pixs[1], rays_pixs[0], torch.ones(rays_pixs[0].shape[0]).cuda()], dim=-1
    )
    target_points = rendered_depth.view(-1, 1) * (
        target_points @ torch.inverse(intrinsics[0, -1]).t()
    )
    target_points = target_points @ c2ws[0, -1][:3, :3].t() + c2ws[0, -1][:3, 3]

    rgb_mask = (rendered_rgb - rays_gt_rgb).abs().mean(dim=-1) < 0.02

    for v in range(len(w2cs[0]) - 1):
        points_v = target_points @ w2cs[0, v][:3, :3].t() + w2cs[0, v][:3, 3]
        points_v = points_v @ intrinsics[0, v].t()
        z_pred = points_v[:, -1].clone()
        points_v = points_v[:, :2] / points_v[:, -1:]

        points_unit = points_v.clone()
        H, W = depth_map["level_0"].shape[-2:]
        points_unit[:, 0] = points_unit[:, 0] / W
        points_unit[:, 1] = points_unit[:, 1] / H
        grid = 2 * points_unit - 1

        warped_rgbs = F.grid_sample(
            unpre_imgs[:, v],
            grid.view(1, -1, 1, 2),
            align_corners=True,
            mode="bilinear",
            padding_mode="zeros",
        ).squeeze()
        photo_mask = (warped_rgbs.t() - rays_gt_rgb).abs().mean(dim=-1) < 0.02

        pixel_coor = points_v.round().long()
        k = 5
        pixel_coor[:, 0] = pixel_coor[:, 0].clip(k // 2, W - (k // 2) - 1)
        pixel_coor[:, 1] = pixel_coor[:, 1].clip(2, H - (k // 2) - 1)
        lower_b = pixel_coor - (k // 2)
        higher_b = pixel_coor + (k // 2)

        ind_h = (
            lower_b[:, 1:] * torch.arange(k - 1, -1, -1).view(1, -1).cuda()
            + higher_b[:, 1:] * torch.arange(0, k).view(1, -1).cuda()
        ) // (k - 1)
        ind_w = (
            lower_b[:, 0:1] * torch.arange(k - 1, -1, -1).view(1, -1).cuda()
            + higher_b[:, 0:1] * torch.arange(0, k).view(1, -1).cuda()
        ) // (k - 1)

        patches_h = torch.gather(
            unpre_imgs[:, v].mean(dim=1).expand(ind_h.shape[0], -1, -1),
            1,
            ind_h.unsqueeze(-1).expand(-1, -1, W),
        )
        patches = torch.gather(patches_h, 2, ind_w.unsqueeze(1).expand(-1, k, -1))
        ent_mask = patches.view(-1, k * k).std(dim=-1) > 0.05

        for l in range(3):
            depth = F.grid_sample(
                depth_map[f"level_{l}"][:, v : v + 1],
                grid.view(1, -1, 1, 2),
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros",
            ).squeeze()
            in_mask = (grid > -1.0) * (grid < 1.0)
            in_mask = (in_mask[..., 0] * in_mask[..., 1]).float()
            loss = loss + loss_fn(
                depth, z_pred * in_mask * photo_mask * ent_mask * rgb_mask
            ) * 2 ** (1 - l)
    loss = loss / (len(w2cs[0]) - 1)

    return loss


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]


def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    err = depth_pred - depth_gt
    return np.abs(err) if type(depth_pred) is np.ndarray else err.abs()


def acc_threshold(depth_pred, depth_gt, mask, threshold):
    errors = abs_error(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return (
        acc_mask.astype("float") if type(depth_pred) is np.ndarray else acc_mask.float()
    )


# Ray helpers
def get_rays(
    H,
    W,
    intrinsics_target,
    c2w_target,
    chunk=-1,
    chunk_id=-1,
    train=True,
    train_batch_size=-1,
    mask=None,
    que_imgs_info=None,
):
    # if train:
    #     if mask is None:
    #         xs, ys = (
    #             torch.randint(0, W, (train_batch_size,)).float().cuda(),
    #             torch.randint(0, H, (train_batch_size,)).float().cuda(),
    #         )
    #     else:  # Sample 8 times more points to get mask points as much as possible
    #         xs, ys = (
    #             torch.randint(0, W, (8 * train_batch_size,)).float().cuda(),
    #             torch.randint(0, H, (8 * train_batch_size,)).float().cuda(),
    #         )
    #         masked_points = mask[ys.long(), xs.long()]
    #         xs_, ys_ = xs[~masked_points], ys[~masked_points]
    #         xs, ys = xs[masked_points], ys[masked_points]
    #         xs, ys = torch.cat([xs, xs_]), torch.cat([ys, ys_])
    #         xs, ys = xs[:train_batch_size], ys[:train_batch_size]
    # else:
    #     ys, xs = torch.meshgrid(
    #         torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)
    #     )  # pytorch's meshgrid has indexing='ij'
    #     ys, xs = ys.cuda().reshape(-1), xs.cuda().reshape(-1)
    #     if chunk > 0:
    #         ys, xs = (
    #             ys[chunk_id * chunk : (chunk_id + 1) * chunk],
    #             xs[chunk_id * chunk : (chunk_id + 1) * chunk],
    #         )
    xs = que_imgs_info['coords'][0,:,0]
    ys = que_imgs_info['coords'][0,:,1]
    dirs = torch.stack(
        [
            (xs - intrinsics_target[0, 2]) / intrinsics_target[0, 0],
            (ys - intrinsics_target[1, 2]) / intrinsics_target[1, 1],
            torch.ones_like(xs),
        ],
        -1,
    )  # use 1 instead of -1

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_dir = (
        dirs @ c2w_target[:3, :3].t()
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_orig = c2w_target[:3, -1].clone().reshape(1, 3).expand(rays_dir.shape[0], -1)

    rays_pixs = torch.stack((ys, xs))  # row col

    return rays_orig, rays_dir, rays_pixs


def conver_to_ndc(ray_pts, w2c_ref, intrinsics_ref, W_H, depth_values):
    nb_rays, nb_samples = ray_pts.shape[:2]
    ray_pts = ray_pts.reshape(-1, 3)

    R = w2c_ref[:3, :3]  # (3, 3)
    T = w2c_ref[:3, 3:]  # (3, 1)
    ray_pts = torch.matmul(ray_pts, R.t()) + T.reshape(1, 3)

    ray_pts_ndc = ray_pts @ intrinsics_ref.t()
    ray_pts_ndc[:, :2] = ray_pts_ndc[:, :2] / (
        ray_pts_ndc[:, -1:] * W_H.reshape(1, 2)
    )  # normalize x,y to 0~1

    grid = ray_pts_ndc[None, None, :, :2] * 2 - 1
    near = F.grid_sample(
        depth_values[:, :1],
        grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    ).squeeze()
    far = F.grid_sample(
        depth_values[:, -1:],
        grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    ).squeeze()
    ray_pts_ndc[:, 2] = (ray_pts_ndc[:, 2] - near) / (far - near)  # normalize z to 0~1

    ray_pts_ndc = ray_pts_ndc.view(nb_rays, nb_samples, 3)

    return ray_pts_ndc


def get_sample_points(
    nb_coarse,
    nb_fine,
    near,
    far,
    rays_o,
    rays_d,
    nb_views,
    w2cs,
    intrinsics,
    depth_values,
    W_H,
    with_noise=False,
):
    device = rays_o.device
    nb_rays = rays_o.shape[0]

    with torch.no_grad():
        t_vals = torch.linspace(0.0, 1.0, steps=nb_coarse).view(1, nb_coarse).to(device)
        pts_depth = near * (1.0 - t_vals) + far * (t_vals)
        pts_depth = pts_depth.expand([nb_rays, nb_coarse])
        ray_pts = rays_o.unsqueeze(1) + pts_depth.unsqueeze(-1) * rays_d.unsqueeze(1)

        ## Counting the number of source views for which the points are valid
        valid_points = torch.zeros([nb_rays, nb_coarse]).to(device)
        for idx in range(nb_views):
            w2c_ref, intrinsic_ref = w2cs[idx], intrinsics[idx]
            ray_pts_ndc = conver_to_ndc(
                ray_pts,
                w2c_ref,
                intrinsic_ref,
                W_H,
                depth_values=depth_values[f"level_0"][:, idx],
            )
            valid_points += (
                ((ray_pts_ndc >= 0) & (ray_pts_ndc <= 1)).sum(dim=-1) == 3
            ).float()

        ## Creating a distribution based on the counted values and sample more points
        if nb_fine > 0:
            point_distr = torch.distributions.categorical.Categorical(
                logits=valid_points
            )
            t_vals = (
                point_distr.sample([nb_fine]).t()
                - torch.rand([nb_rays, nb_fine]).cuda()
            ) / (nb_coarse - 1)
            pts_depth_fine = near * (1.0 - t_vals) + far * (t_vals)

            pts_depth = torch.cat([pts_depth, pts_depth_fine], dim=-1)
            pts_depth, _ = torch.sort(pts_depth)

        if with_noise:  ## Add noise to sample points during training
            # get intervals between samples
            mids = 0.5 * (pts_depth[..., 1:] + pts_depth[..., :-1])
            upper = torch.cat([mids, pts_depth[..., -1:]], -1)
            lower = torch.cat([pts_depth[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(pts_depth.shape, device=device)
            pts_depth = lower + (upper - lower) * t_rand

        ray_pts = rays_o.unsqueeze(1) + pts_depth.unsqueeze(-1) * rays_d.unsqueeze(1)

        ray_pts_ndc = {"level_0": [], "level_1": [], "level_2": []}
        for idx in range(nb_views):
            w2c_ref, intrinsic_ref = w2cs[idx], intrinsics[idx]
            for l in range(3):
                ray_pts_ndc[f"level_{l}"].append(
                    conver_to_ndc(
                        ray_pts,
                        w2c_ref,
                        intrinsic_ref,
                        W_H,
                        depth_values=depth_values[f"level_{l}"][:, idx],
                    )
                )
        for l in range(3):
            ray_pts_ndc[f"level_{l}"] = torch.stack(ray_pts_ndc[f"level_{l}"], dim=2)

        return pts_depth, ray_pts, ray_pts_ndc


def get_rays_pts(
    H,
    W,
    c2ws,
    w2cs,
    intrinsics,
    near_fars,
    depth_values,
    nb_coarse,
    nb_fine,
    nb_views,
    chunk=-1,
    chunk_idx=-1,
    train=False,
    train_batch_size=-1,
    target_img=None,
    target_depth=None,
    que_imgs_info=None
):
    # if train:
    #     if target_depth.sum() > 0:
    #         depth_mask = target_depth > 0
    #     else:
    #         depth_mask = None
    # else:
    #     depth_mask = None
    depth_mask = None
    rays_orig, rays_dir, rays_pixs = get_rays(
        H,
        W,
        que_imgs_info['Ks'][0],
        que_imgs_info['c2ws'][0],
        chunk=chunk,
        chunk_id=chunk_idx,
        train=train,
        train_batch_size=train_batch_size,
        mask=depth_mask,
        que_imgs_info=que_imgs_info
    )

    ## Extracting ground truth color and depth of target view
    if train:
        rays_pixs_int = rays_pixs.long()
        rays_gt_rgb = target_img[:,:, rays_pixs_int[0], rays_pixs_int[1]]
        # rays_gt_depth = target_depth[rays_pixs_int[0], rays_pixs_int[1]]
        rays_gt_depth = None
    else:
        rays_gt_rgb = None
        rays_gt_depth = None

    # travel along the rays
    near, far = que_imgs_info['near_fars'][0, 0], que_imgs_info['near_fars'][0, 1]  ## near/far of the target view
    W_H = torch.tensor([W - 1, H - 1]).cuda()
    pts_depth, ray_pts, ray_pts_ndc = get_sample_points(
        nb_coarse,
        nb_fine,
        near,
        far,
        rays_orig,
        rays_dir,
        nb_views,
        w2cs,
        intrinsics,
        depth_values,
        W_H,
        with_noise=train,
    )

    return (
        pts_depth,
        ray_pts,
        ray_pts_ndc,
        rays_dir,
        rays_gt_rgb,
        rays_gt_depth,
        rays_pixs,
    )


def normal_vect(vect, dim=-1):
    return vect / (torch.sqrt(torch.sum(vect**2, dim=dim, keepdim=True)) + 1e-7)


def interpolate_3D(feats, pts_ndc):
    H, W = pts_ndc.shape[-3:-1]
    grid = pts_ndc.view(-1, 1, H, W, 3) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
    features = (
        F.grid_sample(
            feats, grid, align_corners=True, mode="bilinear", padding_mode="border"
        )[:, :, 0]
        .permute(2, 3, 0, 1)
        .squeeze()
    )

    return features


def interpolate_2D(feats, imgs, pts_ndc):
    H, W = pts_ndc.shape[-3:-1]
    grid = pts_ndc[..., :2].view(-1, H, W, 2) * 2 - 1.0  # [1 H W 2] (x,y)
    features = (
        F.grid_sample(
            feats, grid, align_corners=True, mode="bilinear", padding_mode="border"
        )
        .permute(2, 3, 1, 0)
        .squeeze()
    )
    images = (
        F.grid_sample(
            imgs, grid, align_corners=True, mode="bilinear", padding_mode="border"
        )
        .permute(2, 3, 1, 0)
        .squeeze()
    )
    with torch.no_grad():
        in_mask = (grid > -1.0) * (grid < 1.0)
        in_mask = (in_mask[..., 0] * in_mask[..., 1]).float().permute(1, 2, 0)

    return features, images, in_mask


def read_pfm(filename):
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    if src_grid == None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad > 0:
            H_pad, W_pad = H + pad * 2, W + pad * 2
        else:
            H_pad, W_pad = H, W

        if depth_values.dim() != 4:
            depth_values = depth_values[..., None, None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(
            H_pad, W_pad, normalized_coordinates=False, device=device
        )  # (1, H, W, 2)
        if pad > 0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat(
            (ref_grid, torch.ones_like(ref_grid[:, :1])), 1
        )  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.reshape(B, 1, D * W_pad * H_pad)
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

        src_grid = (
            src_grid_d[:, :2] / src_grid_d[:, 2:]
        )  # divide by depth (B, 2, D*H*W)
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(
        src_feat,
        src_grid.view(B, D, W_pad * H_pad, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid


##### Functions for view selection
TINY_NUMBER = 1e-5  # float32 only has 7 decimal digits precision

def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(
        np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0)
    )
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    assert (
        R1.shape[-1] == 3
        and R2.shape[-1] == 3
        and R1.shape[-2] == 3
        and R2.shape[-2] == 3
    )
    return np.arccos(
        np.clip(
            (np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1)
            / 2.0,
            a_min=-1 + TINY_NUMBER,
            a_max=1 - TINY_NUMBER,
        )
    )


def get_nearest_pose_ids(
    tar_pose,
    ref_poses,
    num_select,
    tar_id=-1,
    angular_dist_method="dist",
    scene_center=(0, 0, 0),
):
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams - 1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == "matrix":
        dists = batched_angular_dist_rot_matrix(
            batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3]
        )
    elif angular_dist_method == "vector":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == "dist":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception("unknown angular distance calculation method!")

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]

    return selected_ids

def positional_encoding(t, num_encodings=10):
    """
    Encodes time t with a positional encoding.

    :param t: Time tensor (shape: [batch_size, 1])
    :param num_encodings: The number of positional encodings to generate
    :return: Positionally encoded time (shape: [batch_size, num_encodings])
    """
    # Generate a range of frequencies
    frequencies = 2.0 ** torch.linspace(0.0, num_encodings - 1, num_encodings).unsqueeze(0)
    # frequencies = frequencies.to(t.device)
    # frequencies = frequencies.unsqueeze(0)
    # assert False, f"frequencies.shape {frequencies.shape} t.shape {t.shape}"
    # Encode t with these frequencies
    pre_fix = t.shape[:-1]
    dim  = t.shape[-1]
    frequencies = frequencies.expand(np.prod(pre_fix)*dim,num_encodings)
    t = t.reshape((-1,1)).expand(np.prod(pre_fix)*dim,num_encodings)
    # encoded_t = t * frequencies
    try:
        encoded_t = t * frequencies.to(t.device)
    except:
        assert False, f"t.shape: {t.shape}, frequencies.shape: {frequencies.shape}"
    # encoded_t = t * frequencies.to(t.device)
    encoded_t = encoded_t.reshape(list(pre_fix)+[-1])

    # Apply sin and cos to generate positional encodings
    encoded_t = torch.cat([encoded_t.sin(), encoded_t.cos()], dim=-1)

    return encoded_t.reshape(list(pre_fix)+[-1])

def L2_norm(x,epsilon=1e-9):
    if len(x.shape) == 2:
        return (torch.sum(x**2, dim=-1) + epsilon).mean()+1e-9
    else:
        return (x**2 + epsilon).mean()+1e-9
    
def normalize_min_max(x):
    min_ = x.min(-1,keepdim=True)[0]
    max_ = x.max(-1,keepdim=True)[0]
    return (x - min_) / (max_ - min_ +1e-9)    

def normalized_depth_scale_and_shift(
    prediction, target, mask = None
):
    """
    More info here: https://arxiv.org/pdf/2206.00665.pdf supplementary section A2 Depth Consistency Loss
    This function computes scale/shift required to normalizes predicted depth map,
    to allow for using normalized depth maps as input from monocular depth estimation networks.
    These networks are trained such that they predict normalized depth maps.

    Solves for scale/shift using a least squares approach with a closed form solution:
    Based on:
    https://github.com/autonomousvision/monosdf/blob/d9619e948bf3d85c6adec1a643f679e2e8e84d4b/code/model/loss.py#L7
    Args:
        prediction: predicted depth map
        target: ground truth depth map
        mask: mask of valid pixels
    Returns:
        scale and shift for depth prediction
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    if mask is None:
        mask = torch.ones_like(prediction)
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale, shift

def compute_depth_loss_scale_and_shift(pred,tar):
    pred = pred.reshape((-1,1))
    tar = tar.reshape((-1,1))
    # tar = normalize_min_max(tar)
    scale, shift = normalized_depth_scale_and_shift(pred[None,None], tar[None,None])
    prediction_ssi = scale * pred + shift

    return L2_norm(prediction_ssi - tar)

def replace_NaN(tensor_with_nan):
    replacement_value = 0.0

    # 使用 torch.isnan 检测 NaN 值并使用 torch.where 进行替换
    tensor_no_nan = torch.where(torch.isnan(tensor_with_nan), replacement_value, tensor_with_nan)
    return tensor_no_nan

def select_cameras(aabb,target_c2w,target_w2c,target_k,cam_ids,c2ws, w2cs,Ks, N1,N2):
    target_cam = Camera('',target_w2c,target_k)

    remaining_cameras = np.asarray([Camera(idx,w2c,K) for idx,w2c,K in zip(cam_ids,w2cs,Ks)])
    resolution = (50, 50, 20)
    aabb_points = generate_sample_points(aabb, resolution)
    valid_points_inds = target_cam.check_points_inside_frustum(aabb_points)
    valid_points_num = np.sum(valid_points_inds)
    valid_points = aabb_points[valid_points_inds]

    target_cam_pts = np.array(target_c2w[:3,3])
    remaining_cam_pts = np.array([c2w[:3,3] for c2w in c2ws])
    dists = np.linalg.norm(target_cam_pts[None,None] - remaining_cam_pts[None], 2, 2)
    dists_idx = np.argsort(dists,1)[0]
    nearest_id = dists_idx[:N1]

    selected_cameras = remaining_cameras[nearest_id].tolist()
    remaining_cameras = np.delete(remaining_cameras,nearest_id, axis=0)

    rets2 = []
    remaining_cameras = np.random.permutation(remaining_cameras)
    for camera in remaining_cameras[:100]:
        intersection1 = camera.check_points_inside_frustum(aabb_points)
        intersection2 = np.logical_and(valid_points_inds,intersection1)
        # intersection3 = np.logical_and(select_ind,intersection1)
        if np.sum(intersection2) > 0.3*valid_points_num :
            intersection = np.logical_and(intersection1,~intersection2)
            rets2.append(np.sum(intersection))
            valid_points_inds = np.logical_or(valid_points_inds,intersection1)
            valid_points_num = np.sum(valid_points_inds)
        else:
            rets2.append(-1*np.inf)
    nearest_id2 = top_k_indices(rets2,N2)
    cameras = selected_cameras + remaining_cameras[nearest_id2].tolist()
    
    # return cameras,aabb_points[valid_points_inds]
    return [cam.cam_id for cam in cameras]

def top_k_indices(arr, k):
    # 使用 argsort 对数组进行排序并获取索引
    sorted_indices = np.argsort(arr)
    
    # 返回 top k 的索引
    return sorted_indices[-k:][::-1]

def generate_sample_points(aabb, resolution):
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    x_res, y_res, z_res = resolution

    # 使用linspace生成在每个轴上的均匀间隔的点
    x = np.linspace(xmin, xmax, x_res)
    y = np.linspace(ymin, ymax, y_res)
    z = np.linspace(zmin, zmax, z_res)

    # 使用meshgrid在三维空间中创建一个坐标网格
    X, Y, Z = np.meshgrid(x, y, z)

    # 将坐标网格重塑为点的列表
    sample_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return sample_points

class Camera:
    def __init__(self, cam_id,W2C, K, near_clip=None, far_clip=None):
        # 相机到世界坐标的变换
        self.cam_id = cam_id
        W2C = np.array(W2C)
        self.World2Camera = np.eye(4)
        self.World2Camera[:3,:4]  =W2C
        # 近裁剪面和远裁剪面
        self.near_clip = near_clip
        self.far_clip = far_clip
        # 从内参矩阵中提取相机参数
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
    def check_points_inside_frustum(self, points):
        # 将点从世界坐标转换到相机坐标
        homogeneous_points = np.column_stack((points, np.ones(points.shape[0])))
        camera_coords = self.World2Camera.dot(homogeneous_points.T).T[:, :3]
        # 计算点与相机的距离
        distances = np.linalg.norm(camera_coords, axis=1)
        # 远近裁剪面判断
        if self.near_clip is not None and self.far_clip is not None:
            inside_near_far = (distances > self.near_clip) & (distances < self.far_clip)
        else:
            inside_near_far = (distances > 0.1) & (distances < 8)
            # inside_near_far = np.ones_like(camera_coords[:, 0]).astype(bool)
        # 计算点在相机坐标中的x,y坐标
        projected_x = camera_coords[:, 0] * self.fx / camera_coords[:, 2] + self.cx
        projected_y = camera_coords[:, 1] * self.fy / camera_coords[:, 2] + self.cy
         
        # 判断是否在视锥内
        inside_frustum_x = np.abs(projected_x - self.cx) < self.cx
        inside_frustum_y = np.abs(projected_y - self.cy) < self.cy
        fron_cam = camera_coords[:,2] >= 1e-2
        inside_frustum = inside_frustum_x & inside_frustum_y & inside_near_far & fron_cam
        return inside_frustum