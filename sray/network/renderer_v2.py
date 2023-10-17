import torch
import torch.nn as nn
import numpy as np
from sray.network.aggregate_net import name2agg_net
from sray.network.dist_decoder import name2dist_decoder
from sray.network.init_net import name2init_net
from sray.network.ops import ResUNetLight,upconv
from sray.network.vis_encoder import name2vis_encoder
from sray.network.render_ops import *
from sray.network.dat.dat import DAT
from sray.network.mask2former.mask2former import *
from sray.network.nerfdet.utils import *
from sray.network.nerfdet.neck import FastIndoorImVoxelNeck
class BaseRenderer(nn.Module):
    base_cfg = {
        'vis_encoder_type': 'default',
        'vis_encoder_cfg': {},

        'dist_decoder_type': 'mixture_logistics',
        'dist_decoder_cfg': {},

        'agg_net_type': 'default',
        'agg_net_cfg': {},

        'use_hierarchical_sampling': False,
        'fine_agg_net_cfg': {},
        'fine_dist_decoder_cfg': {},
        'fine_depth_sample_num': 64,
        'fine_depth_use_all': False,

        'ray_batch_num': 2048,
        'depth_sample_num': 64,
        'alpha_value_ground_state': -15,
        'use_nr_color_for_dr': False,
        'use_self_hit_prob': False,
        'use_ray_mask': True,
        'ray_mask_view_num': 2,
        'ray_mask_point_num': 8,

        'render_depth': False,
        'render_label': False,
        'num_classes': 20,
        'use_ref_sem_loss': False,
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.base_cfg}
        self.cfg.update(cfg)
        self.vis_encoder = name2vis_encoder[self.cfg['vis_encoder_type']](
            self.cfg['vis_encoder_cfg'])
        self.dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](
            self.cfg['dist_decoder_cfg'])
        self.image_encoder = ResUNetLight(3, [1, 2, 6, 4], 32, inplanes=16)
        self.m2f = get_m2f(self.cfg['M2F'])
        self.use_dat = True if 'DAT' in cfg.keys() else False
        if self.use_dat:
            self.dat = DAT(**cfg['DAT'])
        self.sem_head_2d =nn.Sequential(
            upconv(32,64,3,4),
            nn.SiLU(),
            nn.Conv2d(64,cfg['num_classes']+1,stride=1,kernel_size=1)
        ) 
        self.coarse_density_decoder = nn.Sequential(
            nn.Conv3d(256*3,256,kernel_size=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256,1,kernel_size=1),
        )

        self.mean_mapping = nn.ModuleList([nn.Conv3d(256,128 , kernel_size=1) for _ in range(3)])
        self.cov_mapping = nn.ModuleList([nn.Conv3d(256,128 , kernel_size=1) for _ in range(3)])
        self.mapping = nn.ModuleList([nn.Conv3d(256,256 , kernel_size=1) for _ in range(3)])
        self.necks_3d = nn.ModuleList([
                            FastIndoorImVoxelNeck(
                                (256+3)*2,[1, 1, 1],256
                            ) for _ in range(3)])
        
        self.agg_net = name2agg_net[self.cfg['agg_net_type']](
            self.cfg['agg_net_cfg'])
        if self.cfg['use_hierarchical_sampling']:
            self.fine_dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](
                self.cfg['fine_dist_decoder_cfg'])
            self.fine_agg_net = name2agg_net[self.cfg['agg_net_type']](
                self.cfg['fine_agg_net_cfg'])


    def predict_proj_ray_prob(self, prj_dict, ref_imgs_info, que_dists, is_fine):
        rfn, qn, rn, dn, _ = prj_dict['mask'].shape
        # decode ray prob
        if is_fine:
            prj_mean, prj_var, prj_vis, prj_aw = self.fine_dist_decoder(
                prj_dict['ray_feats'])
        else:
            prj_mean, prj_var, prj_vis, prj_aw = self.dist_decoder(
                prj_dict['ray_feats'])

        alpha_values, visibility, hit_prob = self.dist_decoder.compute_prob(
            prj_dict['depth'].squeeze(
                -1), que_dists.unsqueeze(0), prj_mean, prj_var,
            prj_vis, prj_aw, True, ref_imgs_info['depth_range'])
        # post process
        prj_dict['alpha'] = alpha_values.reshape(rfn, qn, rn, dn, 1) * prj_dict['mask'] + \
            (1 - prj_dict['mask']) * self.cfg['alpha_value_ground_state']
        prj_dict['vis'] = visibility.reshape(
            rfn, qn, rn, dn, 1) * prj_dict['mask']
        prj_dict['hit_prob'] = hit_prob.reshape(
            rfn, qn, rn, dn, 1) * prj_dict['mask']
        return prj_dict

    def predict_alpha_values_dr(self, prj_dict):
        eps = 1e-5
        # predict alpha values for query ray
        prj_alpha, prj_vis = prj_dict['alpha'], prj_dict['vis']
        alpha = torch.sum(prj_vis * prj_alpha, 0) / \
            (torch.sum(prj_vis, 0) + eps)  # qn,rn,dn,1
        invalid_ray_mask = torch.sum(
            prj_dict['mask'].int().squeeze(-1), 0) == 0
        alpha = alpha * (1 - invalid_ray_mask.float().unsqueeze(-1)) + \
            invalid_ray_mask.float().unsqueeze(-1) * \
            self.cfg['alpha_value_ground_state']
        rfn, qn, rn, dn, _ = prj_alpha.shape
        return alpha.reshape(qn, rn, dn)

    def get_img_feats(self, ref_imgs_info, prj_dict):
        rfn, _, h, w = ref_imgs_info['imgs'].shape
        rfn, qn, rn, dn, _ = prj_dict['pts'].shape

        img_feats = ref_imgs_info['img_feats']
        prj_img_feats = interpolate_feature_map(img_feats, prj_dict['pts'].reshape(rfn, qn * rn * dn, 2),
                                                prj_dict['mask'].reshape(rfn, qn * rn * dn), h, w,)
        prj_dict['img_feats'] = prj_img_feats.reshape(rfn, qn, rn, dn, -1)
        img_feats_dat = ref_imgs_info.get('img_feats_dat',None)
        pts = prj_dict['pts'].reshape(rfn, qn * rn * dn, 2)
        mask = prj_dict['mask'].reshape(rfn, qn * rn * dn)
        if img_feats_dat is not None:
            prj_img_feats_dat = interpolate_feature_map(img_feats_dat, pts, mask, h, w,)
            prj_dict['img_feats_dat'] = prj_img_feats_dat.reshape(rfn, qn, rn, dn, -1)
        for key in ['seg_logits', 'pred_sem_seg']:
            if key in ref_imgs_info:
                prj_dict[key] = interpolate_feature_map(ref_imgs_info[key].float(), pts, mask, h, w,)
        for key in ['global_volume','mlvl_feats']:
            if key in ref_imgs_info:
                prj_dict[key] = ref_imgs_info[key]
        if 'mlvl_feats' in ref_imgs_info:
            m2f_feats = []
            for i in ref_imgs_info['mlvl_feats']:
                h,w = i.shape[-2:]
                m2f_feats.append(interpolate_feature_map(i, pts, mask, h, w,))
            prj_dict[key] = m2f_feats


        return prj_dict

    def predict_self_hit_prob_impl(self, que_ray_feats, que_depth, que_dists, depth_range, is_fine):
        if is_fine:
            ops = self.fine_dist_decoder
        else:
            ops = self.dist_decoder
        mean, var, vis, aw = ops(que_ray_feats)  # qn,rn,1
        if aw is not None:
            aw = aw.unsqueeze(2)
        if vis is not None:
            vis = vis.unsqueeze(2)
        if mean is not None:
            mean = mean.unsqueeze(2)
        if var is not None:
            var = var.unsqueeze(2)
        # qn, rn, dn
        _, _, hit_prob_que = ops.compute_prob(
            que_depth, que_dists, mean, var, vis, aw, False, depth_range)
        return hit_prob_que

    def predict_self_hit_prob(self, que_imgs_info, que_depth, que_dists, is_fine):
        _, _, h, w = que_imgs_info['imgs'].shape
        qn, rn, _ = que_imgs_info['coords'].shape
        mask = torch.ones([qn, rn], dtype=torch.float32,
                          device=que_imgs_info['coords'].device)
        que_ray_feats = interpolate_feature_map(
            que_imgs_info['ray_feats'], que_imgs_info['coords'], mask, h, w)  # qn,rn,f
        hit_prob_que = self.predict_self_hit_prob_impl(
            que_ray_feats, que_depth, que_dists, que_imgs_info['depth_range'], is_fine)
        return hit_prob_que

    def network_rendering(self, prj_dict, que_dir, is_fine):
        if is_fine:
            agg_net_out = self.fine_agg_net(prj_dict, que_dir)
        else:
            agg_net_out = self.agg_net(prj_dict, que_dir)

        render_label = self.cfg['render_label']
        if render_label:
            density, colors, label = agg_net_out
        else:
            density, colors = agg_net_out
            label = None

        alpha_values = 1.0 - torch.exp(-torch.relu(density))
        hit_prob = alpha_values2hit_prob(alpha_values)
        pixel_colors = torch.sum(hit_prob.unsqueeze(-1)*colors, 2)

        if render_label:
            pixel_label = torch.sum(hit_prob.unsqueeze(-1)*label, 2)
            return hit_prob, colors, pixel_colors, label, pixel_label
        else:
            return hit_prob, colors, pixel_colors

    def render_by_depth(self, que_depth, que_imgs_info, ref_imgs_info, is_train, is_fine):
        ref_imgs_info = ref_imgs_info.copy()
        que_imgs_info = que_imgs_info.copy()
        que_dists = depth2inv_dists(que_depth, que_imgs_info['depth_range'])
        que_pts, que_dir = depth2points(que_imgs_info, que_depth)

        
        prj_dict = project_points_dict(ref_imgs_info, que_pts)
        prj_dict['depth_mask'] = que_imgs_info['depth_mask']
        prj_dict = self.predict_proj_ray_prob(
            prj_dict, ref_imgs_info, que_dists, is_fine)
        prj_dict = self.get_img_feats(ref_imgs_info, prj_dict)
        prj_dict['sem_feats'] = ref_imgs_info['img_feats']
        if 'img_feats_dat' in ref_imgs_info.keys():
            prj_dict['sem_feats_dat'] = ref_imgs_info['img_feats_dat']

        nr_out = self.network_rendering(prj_dict, que_dir, is_fine )
        if self.cfg['render_label']:
            hit_prob_nr, colors_nr, pixel_colors_nr, label_nr, pixel_label_nr = nr_out
            outputs = {
                'pixel_colors_nr': pixel_colors_nr,
                'pixel_label_nr': pixel_label_nr,
                'hit_prob_nr': hit_prob_nr,
            }
        else:
            hit_prob_nr, colors_nr, pixel_colors_nr = nr_out
            outputs = {'pixel_colors_nr': pixel_colors_nr,
                       'hit_prob_nr': hit_prob_nr}

        # predict query hit prob
        # if is_train and self.cfg['use_self_hit_prob']:
        #     outputs['hit_prob_self'] = self.predict_self_hit_prob(
        #         que_imgs_info, que_depth, que_dists, is_fine)

        if 'imgs' in que_imgs_info:
            outputs['pixel_colors_gt'] = interpolate_feats(
                que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)

        if 'labels' in que_imgs_info:
            outputs['pixel_label_gt'] = interpolate_feats(
                que_imgs_info['labels'].float(),
                que_imgs_info['coords'],
                align_corners=True,
                inter_mode='nearest'
            )
        
        if self.cfg['use_ray_mask']:
            outputs['ray_mask'] = torch.sum(prj_dict['mask'].int(
            ), 0) > self.cfg['ray_mask_view_num']  # qn,rn,dn,1
            outputs['ray_mask'] = torch.sum(
                outputs['ray_mask'], 2) > self.cfg['ray_mask_point_num']  # qn,rn
            outputs['ray_mask'] = outputs['ray_mask'][..., 0]

        if self.cfg['render_depth']:
            # qn,rn,dn
            outputs['render_depth'] = torch.sum(
                hit_prob_nr * que_depth, -1)  # qn,rn
        return outputs

    def fine_render_impl(self, coarse_render_info, que_imgs_info, ref_imgs_info, is_train):
        fine_depth = sample_fine_depth(coarse_render_info['depth'], coarse_render_info['hit_prob'].detach(),
                                       que_imgs_info['depth_range'], self.cfg['fine_depth_sample_num'], is_train)

        # qn, rn, fdn+dn
        if self.cfg['fine_depth_use_all']:
            que_depth = torch.sort(
                torch.cat([coarse_render_info['depth'], fine_depth], -1), -1)[0]
        else:
            que_depth = torch.sort(fine_depth, -1)[0]
        outputs = self.render_by_depth(
            que_depth, que_imgs_info, ref_imgs_info, is_train, True)
        return outputs

    def render_impl(self, que_imgs_info, ref_imgs_info, is_train):
        # [qn,rn,dn]
        # que_depth, _ = sample_depth(
        #     que_imgs_info['depth_range'], que_imgs_info['coords'], self.cfg['depth_sample_num'], False)
        density_volume = self.coarse_density_decoder(ref_imgs_info['global_volume'])[0]
        que_depth = sample_depth_from_density_volume(density_volume, que_imgs_info, self.cfg['depth_sample_num'], False)
        depth_mask = (que_depth==0.)
        que_imgs_info['depth_mask'] =depth_mask
        outputs = self.render_by_depth(
            que_depth, que_imgs_info, ref_imgs_info, is_train, False)
        if self.cfg['use_hierarchical_sampling']:
            coarse_render_info = {'depth': que_depth,
                                  'hit_prob': outputs['hit_prob_nr']}
            fine_outputs = self.fine_render_impl(
                coarse_render_info, que_imgs_info, ref_imgs_info, is_train)
            for k, v in fine_outputs.items():
                outputs[k + "_fine"] = v
        return outputs

    def render(self, que_imgs_info, ref_imgs_info, is_train):
        ref_img_feats = self.image_encoder(ref_imgs_info['imgs'])
        seg_logits, pred_sem_seg, mlvl_feats = get_m2f_inference_outputs(self.m2f,ref_imgs_info['imgs_mmseg'])
        ref_imgs_info['seg_logits'] = seg_logits
        ref_imgs_info['pred_sem_seg'] = pred_sem_seg
        ref_imgs_info['mlvl_feats'] = mlvl_feats
        volumes, _ = self.build_ms_voxel(ref_imgs_info)
        global_volume = einops.einops.rearrange(volumes[0],'b c w h d -> 1 (b c) w h d')
        ref_imgs_info['global_volume'] = global_volume
        if 'aabb' in self.cfg.keys():
            ref_imgs_info['aabb'] = torch.tensor(self.cfg['aabb']).to(global_volume.device)
            que_imgs_info['aabb'] = torch.tensor(self.cfg['aabb']).to(global_volume.device)

        if self.use_dat:
            _,ref_img_feats_dat = self.dat(ref_img_feats)
        
        # ref_sem_pred = self.sem_head_2d(ref_img_feats)
        ref_imgs_info['img_feats'] = ref_img_feats
        if self.use_dat:
            ref_imgs_info['img_feats_dat'] = ref_img_feats_dat
        ref_imgs_info['ray_feats'] = self.vis_encoder(
            ref_imgs_info['ray_feats'], ref_img_feats)

        # if is_train and self.cfg['use_self_hit_prob']:
        #     que_img_feats = self.image_encoder(que_imgs_info['imgs'])
        #     que_imgs_info['ray_feats'] = self.vis_encoder(
        #         que_imgs_info['ray_feats'], que_img_feats)

        ray_batch_num = self.cfg["ray_batch_num"]
        coords = que_imgs_info['coords']
        ray_num = coords.shape[1]
        render_info_all = {}
        for ray_id in range(0, ray_num, ray_batch_num):
            que_imgs_info['coords'] = coords[:, ray_id:ray_id+ray_batch_num]
            render_info = self.render_impl(
                que_imgs_info, ref_imgs_info, is_train)
            output_keys = [k for k in render_info.keys(
            ) if is_train or (not k.startswith('hit_prob'))]
            for k in output_keys:
                v = render_info[k]
                if k not in render_info_all:
                    render_info_all[k] = []
                render_info_all[k].append(v)

        for k, v in render_info_all.items():
            render_info_all[k] = torch.cat(v, 1)

        # render_info['ref_sem_pred'] = ref_sem_pred
        if self.use_dat:
            render_info_all['ref_img_feats_dat'] = ref_img_feats_dat

        return render_info_all


class Renderer(BaseRenderer):
    default_cfg = {
        'init_net_type': 'depth',
        'init_net_cfg': {},
        'use_depth_loss': False,
        'depth_loss_coords_num': 8192,
        
        'use_semantic_encoder': False,
        'use_strong_encoder': False,
    }

    def build_ms_voxel(self, ref_info):
        mlvl_feats =  ref_info['mlvl_feats']
        volumes, valids = [], []
        for ind, feature in enumerate(mlvl_feats):
            # x = feature
            img_info = ref_info
            # use predicted pitch and roll for SUNRGBDTotal test
            intrinsics = ref_info['Ks']
            extrinsics = ref_info['poses']
            projection = compute_projection(intrinsics,extrinsics).to(feature.device)
            points = get_points(
                n_voxels=torch.tensor(self.cfg['n_voxels']),
                voxel_size=torch.tensor(self.cfg['voxel_size']),
                origin=torch.tensor(self.cfg['origin'])
            ).to(feature.device)
            
            
            volume, valid = backproject(
                feature,
                points,
                projection,
                depth = None,
                voxel_size = self.cfg['voxel_size'])
            density = None

            volume_sum = volume.sum(dim=0)
            # cov_valid = valid.clone().detach()
            valid = valid.sum(dim=0)
            # TODO: Maintain a mask and use a learnable token to fill in the unobserved place.
            volume_mean = volume_sum / (valid + 1e-8)
            volume_mean[:, valid[0]==0] = .0
            # volume_cov = (volume - volume_mean.unsqueeze(0)) ** 2 * cov_valid
            # volume_cov = torch.sum(volume_cov, dim=0) / (valid + 1e-8)
            volume_cov = torch.sum((volume - volume_mean.unsqueeze(0)) ** 2, dim=0) / (valid + 1e-8)
            volume_cov[:, valid[0]==0] = 1e6
            volume_cov = torch.exp(-volume_cov) # default setting
            # be careful here, the smaller the cov, the larger the weight.
            n_channels, n_x_voxels, n_y_voxels, n_z_voxels = volume_mean.shape
            mean_volume = self.mean_mapping[ind](volume_mean.unsqueeze(0))
            cov_volume = self.cov_mapping[ind](volume_cov.unsqueeze(0))
            volume = torch.cat([mean_volume,cov_volume],1)
            

            denorm_images = torch.asarray(np.array(ref_info['imgs_mmseg'])).float().permute(0,3,1,2).to(volume.device)
            # denorm_images = denorm_images.reshape([-1] + list(denorm_images.shape)[2:])
            rgb_projection =  compute_projection(intrinsics,extrinsics).to(feature.device)
            rgb_volume, _ = backproject(
                    denorm_images,
                    points,
                    rgb_projection,
                    None,
                    self.cfg['voxel_size'])
            n_v, C, n_x_voxels, n_y_voxels, n_z_voxels = volume.shape
            # volume = volume.view(n_v, C, -1).permute(0, 2, 1).contiguous()
            mapping_volume = self.mapping[ind](volume).repeat(rgb_volume.shape[0],1,1,1,1)

            mapping_volume = torch.cat([rgb_volume, mapping_volume], dim=1)
            mapping_volume_sum = mapping_volume.sum(dim=0)
            mapping_volume_mean = mapping_volume_sum /  (valid + 1e-8)
            # mapping_volume_cov = (
            #         mapping_volume - mapping_volume_mean.unsqueeze(0)
            #     ) ** 2 * cov_valid
            mapping_volume_cov = (
                    mapping_volume - mapping_volume_mean.unsqueeze(0)
                ) ** 2
            mapping_volume_cov = torch.sum(mapping_volume_cov, dim=0) / (valid + 1e-8)
            mapping_volume_cov[:, valid[0]==0] = 1e6
            mapping_volume_cov = torch.exp(-mapping_volume_cov) # default setting
            global_volume = torch.cat([mapping_volume_mean, mapping_volume_cov], dim=0)
            # global_volume = global_volume.view(-1, n_x_voxels*n_y_voxels*n_z_voxels
            #     ).permute(1, 0).contiguous()
            # points = points.view(3, -1).permute(1, 0).contiguous()

            # density = self.nerf_mlp.query_density(points, global_volume
            #     )
            # alpha = 1 - torch.exp(-density)
            # # density -> alpha # (1, n_x_voxels, n_y_voxels, n_z_voxels)
            # volume = alpha.view(
            #     1, n_x_voxels, n_y_voxels, n_z_voxels) * volume_mean
            global_volume[:, valid[0]==0] = .0
            volumes.append(self.necks_3d[ind](global_volume[None]))
            valids.append(valid)
        y =[]
        for i in zip(*volumes):
            y.append(torch.cat(i))
        valids = torch.stack(valids)
        return y, valids 

    def __init__(self, cfg):
        cfg = {**self.default_cfg, **cfg}
        super().__init__(cfg)
        self.init_net = name2init_net[self.cfg['init_net_type']](
            self.cfg['init_net_cfg'])

    def render_call(self, que_imgs_info, ref_imgs_info, is_train, src_imgs_info=None):
        ref_imgs_info['ray_feats'] = self.init_net(
            ref_imgs_info, src_imgs_info, is_train)
        return self.render(que_imgs_info, ref_imgs_info, is_train)

    def gen_depth_loss_coords(self, h, w, device):
        coords = torch.stack(
            torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'), -1).reshape(-1, 2).to(device)
        num = self.cfg['depth_loss_coords_num']
        idxs = torch.randperm(coords.shape[0])
        idxs = idxs[:num]
        coords = coords[idxs]
        return coords

    def predict_mean_for_depth_loss(self, ref_imgs_info):
        ray_feats = ref_imgs_info['ray_feats']  # rfn,f,h',w'
        ref_imgs = ref_imgs_info['imgs']  # rfn,3,h,w
        rfn, _, h, w = ref_imgs.shape
        coords = self.gen_depth_loss_coords(h, w, ref_imgs.device)  # pn,2
        coords = coords.unsqueeze(0).repeat(rfn, 1, 1)  # rfn,pn,2

        batch_num = self.cfg['depth_loss_coords_num']
        pn = coords.shape[1]
        coords_dist_mean, coords_dist_mean_2, coords_dist_mean_fine, coords_dist_mean_fine_2 = [], [], [], []
        for ci in range(0, pn, batch_num):
            coords_ = coords[:, ci:ci+batch_num]
            mask_ = torch.ones(
                coords_.shape[:2], dtype=torch.float32, device=ref_imgs.device)
            coords_ray_feats_ = interpolate_feature_map(
                ray_feats, coords_, mask_, h, w)  # rfn,pn,f
            coords_dist_mean_ = self.dist_decoder.predict_mean(
                coords_ray_feats_)  # rfn,pn
            coords_dist_mean_2.append(coords_dist_mean_[..., 1])
            coords_dist_mean_ = coords_dist_mean_[..., 0]

            coords_dist_mean.append(coords_dist_mean_)
            if self.cfg['use_hierarchical_sampling']:
                coords_dist_mean_fine_ = self.fine_dist_decoder.predict_mean(
                    coords_ray_feats_)
                coords_dist_mean_fine_2.append(coords_dist_mean_fine_[..., 1])
                # use 0 for depth supervision
                coords_dist_mean_fine_ = coords_dist_mean_fine_[..., 0]
                coords_dist_mean_fine.append(coords_dist_mean_fine_)

        coords_dist_mean = torch.cat(coords_dist_mean, 1)
        outputs = {'depth_mean': coords_dist_mean, 'depth_coords': coords}
        if len(coords_dist_mean_2) > 0:
            coords_dist_mean_2 = torch.cat(coords_dist_mean_2, 1)
            outputs['depth_mean_2'] = coords_dist_mean_2
        if self.cfg['use_hierarchical_sampling']:
            coords_dist_mean_fine = torch.cat(coords_dist_mean_fine, 1)
            outputs['depth_mean_fine'] = coords_dist_mean_fine
            if len(coords_dist_mean_fine_2) > 0:
                coords_dist_mean_fine_2 = torch.cat(coords_dist_mean_fine_2, 1)
                outputs['depth_mean_fine_2'] = coords_dist_mean_fine_2
        return outputs
        
    def forward(self, data):
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()
        is_train = 'eval' not in data

        src_imgs_info = data['src_imgs_info'].copy(
        ) if 'src_imgs_info' in data else None
        render_outputs = self.render_call(
            que_imgs_info, ref_imgs_info, is_train, src_imgs_info)
        if (self.cfg['use_depth_loss'] and 'true_depth' in ref_imgs_info) or (not is_train):
            render_outputs.update(
                self.predict_mean_for_depth_loss(ref_imgs_info))
        if self.use_dat:
            y = render_outputs['ref_img_feats_dat']
        else:
            y = ref_imgs_info['img_feats']
        render_outputs['ref_sem_pred'] = self.sem_head_2d(y)
        return render_outputs
