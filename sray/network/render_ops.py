import torch
import torch.nn.functional as F
from sray.network.ops import interpolate_feats
import einops
import os
from torch.utils.cpp_extension import load

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply

def coords2rays(coords, poses, Ks):
    """
    :param coords:   [rfn,rn,2]
    :param poses:    [rfn,3,4]
    :param Ks:       [rfn,3,3]
    :return:
        ref_rays:
            centers:    [rfn,rn,3]
            directions: [rfn,rn,3]
    """
    rot = poses[:, :, :3].unsqueeze(1).permute(0, 1, 3, 2)  # rfn,1,3,3
    trans = -rot @ poses[:, :, 3:].unsqueeze(1)  # rfn,1,3,1

    rfn, rn, _ = coords.shape
    centers = trans.repeat(1, rn, 1, 1).squeeze(-1)  # rfn,rn,3
    coords = torch.cat([coords, torch.ones([rfn, rn, 1], dtype=torch.float32, device=coords.device)], 2)  # rfn,rn,3
    Ks_inv = torch.inverse(Ks).unsqueeze(1)
    cam_xyz = Ks_inv @ coords.unsqueeze(3)
    cam_xyz = rot @ cam_xyz + trans
    directions = cam_xyz.squeeze(3) - centers
    # directions = directions / torch.clamp(torch.norm(directions, dim=2, keepdim=True), min=1e-4)
    return centers, directions

def depth2points(que_imgs_info, que_depth):
    """
    :param que_imgs_info:
    :param que_depth:       qn,rn,dn
    :return:
    """
    cneters, directions = coords2rays(que_imgs_info['coords'],que_imgs_info['poses'],que_imgs_info['Ks']) # centers, directions qn,rn,3
    qn, rn, _ = cneters.shape
    que_pts = cneters.unsqueeze(2) + directions.unsqueeze(2) * que_depth.unsqueeze(3) # qn,rn,dn,3
    qn, rn, dn, _ = que_pts.shape
    que_dir = -directions / torch.norm(directions, dim=2, keepdim=True)  # qn,rn,3
    que_dir = que_dir.unsqueeze(2).repeat(1, 1, dn, 1)
    return que_pts, que_dir # qn,rn,dn,3

def depth2dists(depth):
    device = depth.device
    dists = depth[...,1:]-depth[...,:-1]
    return torch.cat([dists, torch.full([*depth.shape[:-1], 1], 1e6, dtype=torch.float32, device=device)], -1)

def depth2inv_dists(depth,depth_range):
    near, far = -1 / depth_range[:, 0], -1 / depth_range[:, 1]
    near, far = near[:, None, None], far[:, None, None]
    depth_inv = -1 / depth  # qn,rn,dn
    depth_inv = (depth_inv - near) / (far - near)
    dists = depth2dists(depth_inv)  # qn,rn,dn
    return dists

def interpolate_feature_map(ray_feats, coords, mask, h, w, border_type='border'):
    """
    :param ray_feats:       rfn,f,h,w
    :param coords:          rfn,pn,2
    :param mask:            rfn,pn
    :param h:
    :param w:
    :param border_type:
    :return:
    """
    fh, fw = ray_feats.shape[-2:]
    if fh == h and fw == w:
        cur_ray_feats = interpolate_feats(ray_feats, coords, h, w, border_type, True)  # rfn,pn,f
    else:
        cur_ray_feats = interpolate_feats(ray_feats, coords, h, w, border_type, False)  # rfn,pn,f
    cur_ray_feats = cur_ray_feats * mask.float().unsqueeze(-1) # rfn,pn,f
    return cur_ray_feats

def alpha_values2hit_prob(alpha_values):
    """
    :param alpha_values: qn,rn,dn
    :return: qn,rn,dn
    """
    no_hit_density = torch.cat([torch.ones((*alpha_values.shape[:-1], 1))
                               .to(alpha_values.device), 1. - alpha_values + 1e-10], -1)  # rn,k+1
    hit_prob = alpha_values * torch.cumprod(no_hit_density, -1)[..., :-1]  # [n,k]
    return hit_prob

def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]
    :param K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn]
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=torch.float32)],1)
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=torch.float32)
    last_row[:,:,3] = 1.0
    H = torch.cat([KRt,last_row],1) # rfn,4,4
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]
    pts_cam = pts_cam[:,:,:3,0]
    depth = pts_cam[:,:,2:]
    invalid_mask = torch.abs(depth)<1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:,:,:2]/depth
    return pts_2d, ~(invalid_mask[...,0]), depth

def project_points_directions(poses,points):
    """
    :param poses:       rfn,3,4
    :param points:      pn,3
    :return: rfn,pn,3
    """
    cam_pts = -poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:]  # rfn,3,1
    dir = points.unsqueeze(0) - cam_pts.permute(0, 2, 1)  # [1,pn,3] - [rfn,1,3] -> rfn,pn,3
    dir = -dir / torch.clamp_min(torch.norm(dir, dim=2, keepdim=True), min=1e-5)  # rfn,pn,3
    return dir

def project_points_ref_views(ref_imgs_info, que_points,nb_views=-1):
    """
    :param ref_imgs_info:
    :param que_points:      pn,3
    :return:
    """
    if nb_views ==-1:
        nb_views = ref_imgs_info['imgs'].shape[0]
    prj_pts, prj_valid_mask, prj_depth = project_points_coords(
        que_points, ref_imgs_info['poses'][:nb_views], ref_imgs_info['Ks'][:nb_views]) # rfn,pn,2
    h,w=ref_imgs_info['imgs'][:nb_views].shape[-2:]
    prj_img_invalid_mask = (prj_pts[..., 0] < -0.5) | (prj_pts[..., 0] >= w - 0.5) | \
                           (prj_pts[..., 1] < -0.5) | (prj_pts[..., 1] >= h - 0.5)
    valid_mask = prj_valid_mask & (~prj_img_invalid_mask)
    prj_dir = project_points_directions(ref_imgs_info['poses'][:nb_views], que_points) # rfn,pn,3
    return prj_dir, prj_pts, prj_depth, valid_mask

def project_points_dict(ref_imgs_info, que_pts):
    # project all points
    qn, rn, dn, _ = que_pts.shape
    prj_dir, prj_pts, prj_depth, prj_mask = project_points_ref_views(ref_imgs_info, que_pts.reshape([qn * rn * dn, 3]))
    rfn, _, h, w = ref_imgs_info['imgs'].shape
    prj_ray_feats = interpolate_feature_map(ref_imgs_info['ray_feats'], prj_pts, prj_mask, h, w)
    prj_rgb = interpolate_feature_map(ref_imgs_info['imgs'], prj_pts, prj_mask, h, w)
    
    if 'sem_feats' in ref_imgs_info:
        prj_sem_feats = interpolate_feature_map(ref_imgs_info['sem_feats'], prj_pts, prj_mask, h, w)
        prj_dict = {
            'dir': prj_dir,
            'pts': prj_pts,
            'depth': prj_depth,
            'mask': prj_mask.float(),
            'ray_feats': prj_ray_feats,
            'rgb': prj_rgb,
            'sem_feats': prj_sem_feats,
        }
    else:
        prj_dict = {
            'dir':prj_dir, 'pts':prj_pts, 'depth':prj_depth, 
            'mask': prj_mask.float(), 'ray_feats':prj_ray_feats, 'rgb':prj_rgb}
    ret = {}
    if 'seg_logits' in ref_imgs_info:
        
        seg_logits = interpolate_feature_map(ref_imgs_info['seg_logits'], prj_pts, prj_mask, h, w)
        ret['seg_logits'] = seg_logits
    if 'global_volume' in ref_imgs_info:
        n_ref = ref_imgs_info['imgs'].shape[0]
        aabb = ref_imgs_info['aabb']
        min_ = aabb[0]
        max_ = aabb[1]
        qn,rn,dn,_  = que_pts.shape
        ind = que_pts.reshape((-1,3)) #(qn rn dn) 3 qn:1
        ind = (ind - min_)/(max_ - min_)
        ind = ind*2-1.
        global_volume = ref_imgs_info['global_volume'].permute(0,1,4,3,2)
        global_volume_feats = F.grid_sample(global_volume,ind[None,None,None],mode='bilinear',align_corners=True)
        global_volume_feats = global_volume_feats.reshape((rn,1,dn,-1)).repeat(1,n_ref,1,1)
        ret['global_volume_feats'] = global_volume_feats # rn n_ref dn c
    if 'density_volume' in ref_imgs_info:
        n_ref = ref_imgs_info['imgs'].shape[0]
        aabb = ref_imgs_info['aabb']
        min_ = aabb[0]
        max_ = aabb[1]
        qn,rn,dn,_  = que_pts.shape
        ind = que_pts.reshape((-1,3)) #(qn rn dn) 3 qn:1
        ind = (ind - min_)/(max_ - min_)
        ind = ind*2-1.
        density_volume = ref_imgs_info['density_volume'].permute(0,1,4,3,2)
        density_coarse = F.grid_sample(density_volume,ind[None,None,None],mode='bilinear',align_corners=True)
        density_coarse = density_coarse.reshape((rn,1,dn,-1)).repeat(1,n_ref,1,1)
        ret['density_coarse'] = density_coarse # rn n_ref dn c
    prj_dict.update(ret)

    # post process
    for k, v in prj_dict.items():
        prj_dict[k]=v.reshape(rfn,qn,rn,dn,-1)
    return prj_dict

def project_points_dict_v2(ref_imgs_info, que_pts,nb_views):
    # project all points
    qn, rn, dn, _ = que_pts.shape
    prj_dir, prj_pts, prj_depth, prj_mask = project_points_ref_views(ref_imgs_info, que_pts.reshape([qn * rn * dn, 3]),nb_views)
    rfn, _, h, w = ref_imgs_info['imgs'][:nb_views].shape
    # prj_ray_feats = interpolate_feature_map(ref_imgs_info['ray_feats'], prj_pts, prj_mask, h, w)
    prj_rgb = interpolate_feature_map(ref_imgs_info['imgs'][:nb_views], prj_pts, prj_mask, h, w)
    
    if 'sem_feats' in ref_imgs_info:
        prj_sem_feats = interpolate_feature_map(ref_imgs_info['sem_feats'], prj_pts, prj_mask, h, w)
        prj_dict = {
            'dir': -1*prj_dir.reshape((rfn,rn, dn,-1)).permute(1,0,2,3),
            'pts': prj_pts.reshape((rfn,rn, dn,-1)).permute(1,0,2,3),
            'depth': prj_depth.reshape((rfn,rn, dn,-1)).permute(1,0,2,3),
            'mask': prj_mask.float().reshape((rfn,rn, dn,-1)).permute(1,0,2,3),
            # 'ray_feats': prj_ray_feats,
            'rgb': prj_rgb.reshape((rfn,rn, dn,-1)).permute(1,0,2,3),
            'sem_feats': prj_sem_feats,
        }
    else:
        prj_dict = {
            'dir':prj_dir.reshape((rfn,rn, dn,-1)).permute(1,0,2,3), 
            'pts':prj_pts.reshape((rfn,rn, dn,-1)).permute(1,0,2,3), 
            'depth':prj_depth.reshape((rfn,rn, dn,-1)).permute(1,0,2,3), 
            'mask': prj_mask.float().reshape((rfn,rn, dn,-1)).permute(1,0,2,3), 
            # 'ray_feats':prj_ray_feats, 
            'rgb':prj_rgb.reshape((rfn,rn, dn,-1)).permute(1,0,2,3)}
    ret = {}
    if 'seg_logits' in ref_imgs_info:
        
        seg_logits = interpolate_feature_map(ref_imgs_info['seg_logits'][:nb_views], prj_pts, prj_mask, h, w)
        ret['seg_logits'] = seg_logits.reshape((rfn,rn, dn,-1)).permute(1,0,2,3)
    for key in [ 'pred_sem_seg','labels']:
        if key in ref_imgs_info:
            prj_dict[key] = interpolate_feature_map(
                ref_imgs_info[key][:nb_views].float(), 
                prj_pts, prj_mask, h, w,).reshape((rfn,rn, dn,-1)).permute(1,0,2,3)

    if 'global_volume' in ref_imgs_info:
        n_ref = ref_imgs_info['imgs'].shape[0]
        aabb = ref_imgs_info['aabb']
        min_ = aabb[0]
        max_ = aabb[1]
        qn,rn,dn,_  = que_pts.shape
        ind = que_pts.reshape((-1,3)) #(qn rn dn) 3 qn:1
        ind = (ind - min_)/(max_ - min_)
        ind = ind*2-1.
        global_volume = ref_imgs_info['global_volume'].permute(0,1,4,3,2)
        global_volume_feats = F.grid_sample(global_volume,ind[None,None,None],mode='bilinear',align_corners=True)
        global_volume_feats = global_volume_feats.reshape((rn,1,dn,-1)).repeat(1,n_ref,1,1)
        ret['global_volume_feats'] = global_volume_feats # rn n_ref dn c
    if 'tpv_hw' in ref_imgs_info:
        n_ref = nb_views
        aabb = ref_imgs_info['aabb']
        min_ = aabb[0]
        max_ = aabb[1]
        qn,rn,dn,_  = que_pts.shape
        ind = que_pts.reshape((-1,3)) #(qn rn dn) 3 qn:1
        ind = (ind - min_)/(max_ - min_)
        ind = ind*2-1.


        tpv_hw = ref_imgs_info['tpv_hw']
        tpv_zh = ref_imgs_info['tpv_zh']
        tpv_wz = ref_imgs_info['tpv_wz']
        ind_wh,ind_hz,ind_zw = ind[...,[0,1]],ind[...,[1,2]],ind[...,[2,0]]
        tpv_hw_feat = F.grid_sample(tpv_hw,ind_wh[None,None],mode='bilinear',align_corners=True).permute(0,2,3,1)
        tpv_zh_feat = F.grid_sample(tpv_zh,ind_hz[None,None],mode='bilinear',align_corners=True).permute(0,2,3,1)
        tpv_wz_feat = F.grid_sample(tpv_wz,ind_zw[None,None],mode='bilinear',align_corners=True).permute(0,2,3,1)

        ret['tpv_hw_feat'] = tpv_hw_feat.reshape((rn, dn,1,-1)).repeat(1,1,n_ref,1).permute(0,2,1,3)
        ret['tpv_zh_feat'] = tpv_zh_feat.reshape((rn, dn,1,-1)).repeat(1,1,n_ref,1).permute(0,2,1,3)
        ret['tpv_wz_feat'] = tpv_wz_feat.reshape((rn, dn,1,-1)).repeat(1,1,n_ref,1).permute(0,2,1,3)
    if 'density_volume' in ref_imgs_info:
        n_ref = ref_imgs_info['imgs'].shape[0]
        aabb = ref_imgs_info['aabb']
        min_ = aabb[0]
        max_ = aabb[1]
        qn,rn,dn,_  = que_pts.shape
        ind = que_pts.reshape((-1,3)) #(qn rn dn) 3 qn:1
        ind = (ind - min_)/(max_ - min_)
        ind = ind*2-1.
        density_volume = ref_imgs_info['density_volume'].permute(0,1,4,3,2)
        density_coarse = F.grid_sample(density_volume,ind[None,None,None],mode='bilinear',align_corners=True)
        density_coarse = density_coarse.reshape((rn,1,dn,-1)).repeat(1,n_ref,1,1)
        ret['density_coarse'] = density_coarse # rn n_ref dn c
    prj_dict.update(ret)

    # post process
    # for k, v in prj_dict.items():
    #     prj_dict[k]=v.reshape(rfn,qn,rn,dn,-1)
    return prj_dict


def sample_depth(depth_range, coords, sample_num, random_sample):
    """
    :param depth_range: qn,2
    :param sample_num:
    :param random_sample:
    :return:
    """
    qn, rn, _ = coords.shape
    device = coords.device
    near, far = depth_range[:,0], depth_range[:,1] # qn,2
    dn = sample_num
    assert(dn>2)
    interval = (1 / far - 1 / near) / (dn - 1)  # qn
    val = torch.arange(1, dn - 1, dtype=torch.float32, device=near.device)[None, None, :]
    if random_sample:
        val = val + (torch.rand(qn, rn, dn-2, dtype=torch.float32, device=device) - 0.5) * 0.999
    else:
        val = val + torch.zeros(qn, rn, dn-2, dtype=torch.float32, device=device)
    ticks = interval[:, None, None] * val

    diff = (1 / far - 1 / near)
    ticks = torch.cat([torch.zeros(qn,rn,1,dtype=torch.float32,device=device),ticks,diff[:,None,None].repeat(1,rn,1)],-1)
    que_depth = 1 / (1 / near[:, None, None] + ticks)  # qn, dn,
    que_dists = torch.cat([que_depth[...,1:],torch.full([*que_depth.shape[:-1],1],1e6,dtype=torch.float32,device=device)],-1) - que_depth
    return que_depth, que_dists # qn, rn, dn


def near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.05):
    # rays: [N, 3], [N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [N, 1], far [N, 1]

    tmin = (aabb[1] - rays_o) / (rays_d + 1e-15) # [N, 3]
    tmax = (aabb[0]  - rays_o) / (rays_d + 1e-15)
    near = torch.where(tmin < tmax, tmin, tmax).amax(dim=-1, keepdim=True)
    far = torch.where(tmin > tmax, tmin, tmax).amin(dim=-1, keepdim=True)
    # if far < near, means no intersection, set both near and far to inf (1e9 here)
    mask = far < near
    near[mask] = 1e9
    far[mask] = 1e9
    # restrict near to a minimal value
    near = torch.clamp(near, min=min_near)

    return near, far

def sample_depth_from_density_volume(density_volume, que_imgs_info, depth_sample_num,random_sample=False):
    """
    :param depth_range: qn,2
    :param sample_num:
    :param random_sample:
    :return:
    """
    cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
    raymarch_kernel = load(name='raymarch_kernel',
                        extra_cuda_cflags=[],
                        sources=[f'{cuda_dir}/raymarcher.cpp',
                                    f'{cuda_dir}/raymarcher.cu'])
    (depth_range, coords, poses,Ks,aabb) = (
                                        que_imgs_info['depth_range'], 
                                        que_imgs_info['coords'], 
                                        que_imgs_info['poses'],
                                        que_imgs_info['Ks'],
                                        que_imgs_info['aabb'],
                                        )
    qn, rn, _ = coords.shape
    device = coords.device
    near, far = depth_range[:,0], depth_range[:,1] # qn,2
    replacement_value = 0.0
    density_volume = torch.where(torch.isnan(density_volume), replacement_value, density_volume)
    density_volume = trunc_exp(density_volume)[0]
    density_volume = 1 - torch.exp(-density_volume)
    # density_volume = torch.clamp(density_volume,min=0.3,max=1.)


    rays_o, rays_d = coords2rays(coords, poses, Ks)
    rays_d = rays_d/torch.norm(rays_d, dim=2, keepdim=True)
    rays_d = rays_d.reshape(-1,3)
    rays_o = rays_o.reshape(-1,3)
    near = near.reshape((1,)).repeat((rn,))
    far = far.reshape((1,)).repeat((rn,))
    # near,far= near_far_from_aabb(rays_o, rays_d, aabb)
    # step = (far-near)/100.
    step = torch.tensor([3e-3]).repeat((rn,)).to(rays_o.device)
    z = raymarch_kernel.raymarch_train(rays_o, rays_d, near.squeeze(-1), far.squeeze(-1),
                                                density_volume.gt(0.1), aabb[1]-aabb[0], aabb[0],
                                                step.squeeze(-1), depth_sample_num)

    
    que_depth = z.reshape(qn,rn,-1) 

    return que_depth




def sample_fine_depth(depth, hit_prob, depth_range, sample_num, random_sample, inv_mode=True):
    """
    :param depth:       qn,rn,dn
    :param hit_prob:    qn,rn,dn
    :param depth_range: qn,2
    :param sample_num:
    :param random_sample:
    :param inv_mode:
    :return: qn,rn,dn
    """
    if inv_mode:
        near, far = depth_range[0,0], depth_range[0,1]
        near, far = -1/near, -1/far
        depth_inv = -1 / depth  # qn,rn,dn
        depth_inv = (depth_inv - near) / (far - near)
        depth = depth_inv

    depth_center = (depth[...,1:] + depth[...,:-1])/2
    depth_center = torch.cat([depth[...,0:1],depth_center,depth[...,-1:]],-1) # rfn,pn,dn+1
    fdn = sample_num
    # Get pdf
    hit_prob = hit_prob + 1e-5  # prevent nans
    pdf = hit_prob / torch.sum(hit_prob, -1, keepdim=True) # rfn,pn,dn-1
    cdf = torch.cumsum(pdf, -1) # rfn,pn,dn-1
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # rfn,pn,dn

    # Take uniform samples
    if not random_sample:
        interval = 1 / fdn
        u = 0.5*interval+torch.arange(fdn)*interval
        # u = torch.linspace(0., 1., steps=fdn)
        u = u.expand(list(cdf.shape[:-1]) + [fdn]) # rfn,pn,fdn
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [fdn])

    # Invert CDF
    device = pdf.device
    u = u.to(device).contiguous() # rfn,pn,fdn
    inds = torch.searchsorted(cdf, u, right=True)                       # rfn,pn,fdn
    below = torch.max(torch.zeros_like(inds-1), inds-1)                 # rfn,pn,fdn
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # rfn,pn,fdn
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)   # rfn,pn,fdn,2

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)    # rfn,pn,fdn,2
    bins_g = torch.gather(depth_center.unsqueeze(-2).expand(matched_shape), -1, inds_g) # rfn,pn,fdn,2

    denom = (cdf_g[...,1]-cdf_g[...,0]) # rfn,pn,fdn
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    fine_depth = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    if inv_mode:
        near, far = depth_range[0,0], depth_range[0,1]
        near, far = -1/near, -1/far
        fine_depth = fine_depth * (far - near) + near
        fine_depth = -1/fine_depth
    return fine_depth
