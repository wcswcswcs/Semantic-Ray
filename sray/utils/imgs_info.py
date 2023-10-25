import numpy as np
import torch

from sray.utils.base_utils import color_map_forward, pad_img_end

def random_crop(ref_imgs_info, que_imgs_info, target_size):
    imgs = ref_imgs_info['imgs']
    n, _, h, w = imgs.shape
    out_h, out_w = target_size[0], target_size[1]
    if out_w >= w or out_h >= h:
        return ref_imgs_info

    center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
    center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)

    def crop(tensor):
        tensor = tensor[:, :, center_h - out_h // 2:center_h + out_h // 2,
                              center_w - out_w // 2:center_w + out_w // 2]
        return tensor

    def crop_imgs_info(imgs_info):
        imgs_info['imgs'] = crop(imgs_info['imgs'])
        if 'depth' in imgs_info: imgs_info['depth'] = crop(imgs_info['depth'])
        if 'true_depth' in imgs_info: imgs_info['true_depth'] = crop(imgs_info['true_depth'])
        if 'masks' in imgs_info: imgs_info['masks'] = crop(imgs_info['masks'])

        Ks = imgs_info['Ks'] # n, 3, 3
        h_init = center_h - out_h // 2
        w_init = center_w - out_w // 2
        Ks[:,0,2]-=w_init
        Ks[:,1,2]-=h_init
        imgs_info['Ks']=Ks
        return imgs_info

    return crop_imgs_info(ref_imgs_info), crop_imgs_info(que_imgs_info)

def random_flip(ref_imgs_info,que_imgs_info):
    def flip(tensor):
        tensor = np.flip(tensor.transpose([0, 2, 3, 1]), 2)  # n,h,w,3
        tensor = np.ascontiguousarray(tensor.transpose([0, 3, 1, 2]))
        return tensor

    def flip_imgs_info(imgs_info):
        imgs_info['imgs'] = flip(imgs_info['imgs'])
        if 'depth' in imgs_info: imgs_info['depth'] = flip(imgs_info['depth'])
        if 'true_depth' in imgs_info: imgs_info['true_depth'] = flip(imgs_info['true_depth'])
        if 'masks' in imgs_info: imgs_info['masks'] = flip(imgs_info['masks'])

        Ks = imgs_info['Ks']  # n, 3, 3
        Ks[:, 0, :] *= -1
        w = imgs_info['imgs'].shape[-1]
        Ks[:, 0, 2] += w - 1
        imgs_info['Ks'] = Ks
        return imgs_info

    ref_imgs_info = flip_imgs_info(ref_imgs_info)
    que_imgs_info = flip_imgs_info(que_imgs_info)
    return ref_imgs_info, que_imgs_info

def pad_imgs_info(ref_imgs_info,pad_interval):
    ref_imgs, ref_depths, ref_masks = ref_imgs_info['imgs'], ref_imgs_info['depth'], ref_imgs_info['masks']
    ref_depth_gt = ref_imgs_info['true_depth'] if 'true_depth' in ref_imgs_info else None
    rfn, _, h, w = ref_imgs.shape
    ph = (pad_interval - (h % pad_interval)) % pad_interval
    pw = (pad_interval - (w % pad_interval)) % pad_interval
    if ph != 0 or pw != 0:
        ref_imgs = np.pad(ref_imgs, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
        ref_depths = np.pad(ref_depths, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
        ref_masks = np.pad(ref_masks, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
        if ref_depth_gt is not None:
            ref_depth_gt = np.pad(ref_depth_gt, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
    ref_imgs_info['imgs'], ref_imgs_info['depth'], ref_imgs_info['masks'] = ref_imgs, ref_depths, ref_masks
    if ref_depth_gt is not None:
        ref_imgs_info['true_depth'] = ref_depth_gt
    return ref_imgs_info

def build_imgs_info(database, ref_ids, pad_interval=-1, is_aligned=True, align_depth_range=False, has_depth=True, replace_none_depth=False, add_label=True, num_geo_src_views=0):
    if not is_aligned:
        assert has_depth
        rfn = len(ref_ids)
        ref_imgs, ref_labels, ref_masks, ref_depths, shapes,ref_imgs_mmseg,ref_norm_imgs = [], [], [], [], [],[],[]
        for ref_id in ref_ids:
            img = database.get_image(ref_id)
            norm_img = database.get_norm_image(ref_id)
            img_mmseg = database.get_image_for_mmseg(ref_id)
            if add_label:
                label = database.get_label(ref_id)
                ref_labels.append(label)
            shapes.append([img.shape[0], img.shape[1]])
            ref_imgs.append(img)
            ref_imgs_mmseg.append(img_mmseg)
            ref_norm_imgs.append(norm_img)
            ref_masks.append(database.get_mask(ref_id))
            ref_depths.append(database.get_depth(ref_id))

        shapes = np.asarray(shapes)
        th, tw = np.max(shapes, 0)
        for rfi in range(rfn):
            ref_imgs[rfi] = pad_img_end(ref_imgs[rfi], th, tw, 'reflect')
            ref_imgs_mmseg[rfi] = pad_img_end(ref_imgs_mmseg[rfi], th, tw, 'reflect')
            ref_norm_imgs[rfi] = pad_img_end(ref_norm_imgs[rfi], th, tw, 'reflect')
            ref_labels[rfi] = pad_img_end(ref_labels[rfi], th, tw, 'reflect')
            ref_masks[rfi] = pad_img_end(ref_masks[rfi][:, :, None], th, tw, 'constant', 0)[..., 0]
            ref_depths[rfi] = pad_img_end(ref_depths[rfi][:, :, None], th, tw, 'constant', 0)[..., 0]
        ref_imgs = color_map_forward(np.stack(ref_imgs, 0)).transpose([0, 3, 1, 2])
        # ref_imgs_mmseg = color_map_forward(np.stack(ref_imgs_mmseg, 0)).transpose([0, 3, 1, 2])
        ref_labels = np.stack(ref_labels, 0).transpose([0, 3, 1, 2])
        ref_masks = np.stack(ref_masks, 0)[:, None, :, :]
        ref_depths = np.stack(ref_depths, 0)[:, None, :, :]
    else:
        ref_imgs = color_map_forward(np.asarray([database.get_image(ref_id) for ref_id in ref_ids])).transpose([0, 3, 1, 2])
        ref_imgs_norm = np.asarray([database.get_norm_image(ref_id) for ref_id in ref_ids]).transpose([0, 3, 1, 2])
        ref_imgs_mmseg = [database.get_image_for_mmseg(ref_id) for ref_id in ref_ids]
        ref_labels = np.asarray([database.get_label(ref_id) for ref_id in ref_ids])[:, None, :, :]
        ref_masks =  np.asarray([database.get_mask(ref_id) for ref_id in ref_ids], dtype=np.float32)[:, None, :, :]
        if has_depth:
            ref_depths = [database.get_depth(ref_id) for ref_id in ref_ids]
            if replace_none_depth:
                b, _, h, w = ref_imgs.shape
                for i, depth in enumerate(ref_depths):
                    if depth is None: ref_depths[i] = np.zeros([h, w], dtype=np.float32)
            ref_depths = np.asarray(ref_depths, dtype=np.float32)[:, None, :, :]
        else: ref_depths = None
    seg_logits = torch.stack([database.get_m2f_out(ref_id) for ref_id in ref_ids],0)
    # m2f = [database.get_m2f_out(ref_id) for ref_id in ref_ids]
    # seg_logits, pred_sem_seg, mlvl_feats = zip(*m2f)
    # seg_logits = torch.stack(seg_logits,0)
    # pred_sem_seg  = torch.stack(pred_sem_seg,0)
    # mlvl_feats = torch.stack(mlvl_feats,1)

    ref_poses = np.asarray([database.get_pose(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_Ks = np.asarray([database.get_K(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_depth_range = np.asarray([database.get_depth_range(ref_id) for ref_id in ref_ids], dtype=np.float32)
    if align_depth_range:
        ref_depth_range[:,0]=np.min(ref_depth_range[:,0])
        ref_depth_range[:,1]=np.max(ref_depth_range[:,1])

    

    ref_imgs_info = {
        'imgs_mmseg':ref_imgs_mmseg,
        'seg_logits':seg_logits,
        # 'pred_sem_seg':pred_sem_seg,
        # 'mlvl_feats':mlvl_feats,
        
        # 'affine_mats':, #w2c2i
        # 'affine_mats_inv':,
        # 'near_fars':,
        # 'closest_idxs':,
        # 'depths_aug':,

        'imgs': ref_imgs,
        'norm_imgs': ref_imgs_norm, 
        'poses': ref_poses, 
        'Ks': ref_Ks, 
        'depth_range': ref_depth_range, 
        'masks': ref_masks, 
        'labels': ref_labels}
    # if has_depth: ref_imgs_info['depth'] = ref_depths
    ref_imgs_info['depth'] = ref_depths
    if pad_interval!=-1:
        ref_imgs_info = pad_imgs_info(ref_imgs_info, pad_interval)
    ref_imgs_info['img_metas'] = build_img_metas(ref_imgs_info)
    if num_geo_src_views != 0: # query_cam
        ref_imgs_info = build_info_for_geo(ref_imgs_info,num_geo_src_views)
    return ref_imgs_info

def build_img_metas(ref_img_info,img_H = 280,img_W = 320):

    img_metas=[]
    d = {
        'img_shape' : [[img_H, img_W]],
    }
    img_metas = []
    for pose,k in zip(ref_img_info['poses'],ref_img_info['Ks']):
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :4] = pose[:3,:4]
        intrinsic = np.eye(4)
        intrinsic[:k.shape[0], :k.shape[1]] = k
        lidar2img = intrinsic  @ lidar2cam_rt
        ret = d.copy()
        ret['lidar2img'] = lidar2img
        img_metas.append(ret)
    return img_metas

def build_info_for_geo(ref_img_info,num_src_views):
    affine_mats, affine_mats_inv = [], []
    depths, depths_h, depths_aug = [], [], []
    intrinsics, w2cs, c2ws, near_fars = [], [], [], []
    near_far = [0.1, 10.0]
    h = 240
    w = 320
    # num_src_views = len(ref_img_info['poses'])
    for vid in range(num_src_views):

        
        w2c = ref_img_info['poses'][vid]
        w2c_ = np.eye(4)
        w2c_[:3,:4] = w2c
        w2cs.append(w2c_)
        c2w = np.linalg.inv(w2c_)
        c2ws.append(c2w)
        aff = []
        aff_inv = []
        intrinsic = ref_img_info['Ks'][vid]
        intrinsics.append(intrinsic)

        for l in range(3):
            proj_mat_l = np.eye(4)
            intrinsic_temp = intrinsic.copy()
            intrinsic_temp[:2] = intrinsic_temp[:2] / (2**l)
            proj_mat_l[:3, :4] = intrinsic_temp @ w2c[:3, :4]
            aff.append(proj_mat_l.copy())
            aff_inv.append(np.linalg.inv(proj_mat_l))
        aff = np.stack(aff, axis=-1)
        aff_inv = np.stack(aff_inv, axis=-1)
        affine_mats.append(aff)
        affine_mats_inv.append(aff_inv)

        near_fars.append(near_far)
        depths_h.append(np.zeros([h, w]))
        depths.append(np.zeros([h // 4, w // 4]))
        depths_aug.append(np.zeros([h // 4, w // 4]))
    intrinsics = np.stack(intrinsics)
    w2cs = np.stack(w2cs)
    c2ws = np.stack(c2ws)
    closest_idxs = []
    for pose in c2ws:
        ids = get_nearest_pose_ids(
                pose, ref_poses=c2ws, num_select=num_src_views, angular_dist_method="dist"
            )
        closest_idxs.append(ids)
    closest_idxs = np.stack(closest_idxs, axis=0)
    affine_mats = np.stack(affine_mats)
    affine_mats_inv = np.stack(affine_mats_inv)
    depths_h = np.stack(depths_h)
    depths_aug = np.stack(depths_aug)
    affine_mats = np.stack(affine_mats)
    affine_mats_inv = np.stack(affine_mats_inv)
    near_fars = np.stack(near_fars)

    
    ref_img_info["depths"] = depths
    ref_img_info["depths_h"] = depths_h
    ref_img_info["depths_aug"] = depths_aug
    ref_img_info["w2cs"] = w2cs.astype("float32")
    ref_img_info["c2ws"] = c2ws.astype("float32")
    ref_img_info["near_fars"] = near_fars
    ref_img_info["affine_mats"] = affine_mats
    ref_img_info["affine_mats_inv"] = affine_mats_inv
    ref_img_info["intrinsics"] = intrinsics.astype("float32")
    ref_img_info["closest_idxs"] = closest_idxs
    
    
    return ref_img_info

def build_render_imgs_info(que_pose,que_K,que_shape,que_depth_range):
    h, w = que_shape
    h, w = int(h), int(w)
    que_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h), indexing='xy'), -1)
    que_coords = que_coords.reshape([1, -1, 2]).astype(np.float32)
    return {'poses': que_pose.astype(np.float32)[None,:,:],  # 1,3,4
            'Ks': que_K.astype(np.float32)[None,:,:],  # 1,3,3
            'coords': que_coords,
            'depth_range': np.asarray(que_depth_range, np.float32)[None, :],
            'shape': (h,w)}

def imgs_info_to_torch(imgs_info):
    for k, v in imgs_info.items():
        if isinstance(v,np.ndarray):
            imgs_info[k] = torch.from_numpy(v)
    return imgs_info

def imgs_info_slice(imgs_info, indices):
    imgs_info_out={}
    for k, v in imgs_info.items():
        imgs_info_out[k] = v[indices]
    return imgs_info_out


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


    if angular_dist_method == "dist":
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