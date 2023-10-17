import torch
import torch.nn.functional as F


def compute_projection(intrinsics,extrinsics):
    # projection = []
    # for ind, extrinsic in enumerate(extrinsics):
    #     projection.append(intrinsics[ind] @ extrinsic[:3])
    return torch.bmm(intrinsics,extrinsics[:,:3])
    # return torch.stack(projection)

@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    # origin: point-cloud center.
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]), # 40 W width, x
        torch.arange(n_voxels[1]), # 40 D depth, y
        torch.arange(n_voxels[2]) # 16 H Heigh, z
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points

# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection, depth, voxel_size):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)

    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    ##### below is using depth to sample feature ########
    if depth is not None:
        depth = F.interpolate(depth.unsqueeze(1), size=(height, width), mode="bilinear").squeeze(1)
        for i in range(n_images):
            z_mask = z.clone() > 0
            z_mask[i, valid[i]] = (z[i, valid[i]] > depth[i, y[i, valid[i]], x[i, valid[i]]] - voxel_size[-1]) & \
                (z[i, valid[i]] < depth[i, y[i, valid[i]], x[i, valid[i]]] + voxel_size[-1])
            valid = valid & z_mask

    ######################################################
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)

    return volume, valid