import torch
import torch.nn as nn
import numpy as np
from sray.network.aggregate_net_v2 import name2agg_net
from sray.network.dist_decoder import name2dist_decoder
from sray.network.init_net import name2init_net
from sray.network.ops import ResUNetLight,upconv
from sray.network.vis_encoder import name2vis_encoder
from sray.network.render_ops import *
# from sray.network.dat.dat import DAT
# from sray.network.mask2former.mask2former import *
from sray.network.nerfdet.utils import *
from sray.network.nerfdet.neck import FastIndoorImVoxelNeck
from sray.network.geo_reasoner import CasMVSNet
from mmengine.config import Config
from mmseg.models import build_segmentor
from sray.network.tpvformer10 import *
from sray.utils.utils import get_rays_pts
from sray.utils.rendering import *
from sray.utils.draw_utils import draw_aabb, draw_cam

def model_builder(model_config):
    model = build_segmentor(model_config)
    model.init_weights()
    return model
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
        'fine_depth_sample_num': 32,
        'fine_depth_use_all': False,

        'ray_batch_num': 2048,
        'depth_sample_num': 64,
        'alpha_value_ground_state': -15,
        'use_nr_color_for_dr': False,
        'use_self_hit_prob': False,
        'use_ray_mask': False,#True,
        'ray_mask_view_num': 2,
        'ray_mask_point_num': 8,

        'render_depth': False,
        'render_label': False,
        'num_classes': 20,
        'use_ref_sem_loss': False,

        # 'nb_views': 8 ,
        'tpv_h' : 160,
        'tpv_w' : 160,
        'tpv_z' : 64,
        
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.base_cfg}
        self.cfg.update(cfg)
        self.cfg['nb_views'] = self.cfg['train_dataset_cfg']['num_geo_src_views']
        
        self.geo_reasoner = CasMVSNet(use_depth=False).cuda()
        
        self.agg_net = name2agg_net[self.cfg['agg_net_type']](
            self.cfg['agg_net_cfg'])
        if self.cfg['use_hierarchical_sampling']:
            self.fine_dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](
                self.cfg['fine_dist_decoder_cfg'])
            self.fine_agg_net = name2agg_net[self.cfg['agg_net_type']](
                self.cfg['fine_agg_net_cfg'])
        tpv_cfg = Config.fromfile('sray/network/tpvformer10/tpv_config.py')
        tpv_cfg._num_cams_ = self.cfg['nb_views']
        self.tpvformer = model_builder(tpv_cfg.model).cuda()
        
        self.mean_mapping = nn.ModuleList([nn.Conv3d(21,128 , kernel_size=1) for _ in range(1)])
        self.cov_mapping = nn.ModuleList([nn.Conv3d(21,128 , kernel_size=1) for _ in range(1)])
        self.mapping = nn.ModuleList([nn.Conv3d(256,128 , kernel_size=1) for _ in range(1)])
        self.necks_3d = FastIndoorImVoxelNeck(
                                (128+3)*2,[1, 1, 1],128
                            )

 
    def render_impl(self, que_imgs_info, ref_imgs_info, is_train):
        # [qn,rn,dn]
        H,W = ref_imgs_info['imgs'][0].shape[-2:]
        (
            pts_depth,
            rays_pts,
            rays_pts_ndc,
            rays_dir,
            rays_gt_rgb,
            rays_gt_depth,
            rays_pixs,
        ) = get_rays_pts(
            H,
            W,
            ref_imgs_info["c2ws"],
            ref_imgs_info["w2cs"],
            ref_imgs_info["intrinsics"],
            ref_imgs_info["near_fars"],
            ref_imgs_info['depth_values'],
            self.cfg['depth_sample_num'],
            self.cfg['fine_depth_sample_num'],
            nb_views=self.cfg['nb_views'],
            train=is_train,
            train_batch_size=self.cfg['train_dataset_cfg']['train_ray_num'],
            target_img=que_imgs_info['imgs'],
            target_depth=None,
            que_imgs_info=que_imgs_info
        )

        pts_feat = interpolate_pts_feats(
            ref_imgs_info['imgs'][:self.cfg['nb_views']], 
            ref_imgs_info['feats_fpn'], 
            ref_imgs_info['feats_vol'], 
            rays_pts_ndc)
        occ_masks = get_occ_masks( ref_imgs_info['depth_map_norm'], rays_pts_ndc)
        
        rays_dir_unit = rays_dir / torch.norm(rays_dir, dim=-1, keepdim=True)
        angles = get_angle_wrt_src_cams(ref_imgs_info['c2ws'], rays_pts, rays_dir_unit)
        embedded_angles = get_embedder()(angles)
        rays_pts = rays_pts.unsqueeze(0)
        prj_dict = project_points_dict_v2(ref_imgs_info, rays_pts,self.cfg['nb_views'])
        prj_dict['rays_pts'] = rays_pts
        prj_dict['geonerf_feats'] = pts_feat.permute(0,2,1,3)
        prj_dict['occ_masks'] = occ_masks.permute(0,2,1,3)
        prj_dict['embedded_angles'] = embedded_angles.permute(0,2,1,3)
        prj_dict['viewdirs'] = rays_dir_unit
        agg_net_out = self.agg_net(prj_dict,rays_dir_unit)
        rgb = agg_net_out['colors']
        sigma = agg_net_out['density']
        semantic = agg_net_out['semantic']
        sem_out = agg_net_out['sem_out']
        rgb_sigma = torch.cat([rgb,sigma],-1)
        rendered_rgb, render_depth,weights = volume_rendering(rgb_sigma, pts_depth,semantic)
        semantic_map =  torch.sum(weights[..., None] * semantic, -2)
        sem_out = torch.sum(weights[..., None] * sem_out, -2)
        outputs = {
            'pixel_colors_nr': rendered_rgb[None],
            'render_depth':render_depth[None,...,None],
            }
        outputs['pixel_label_gt'] = interpolate_feats(
                    que_imgs_info['labels'].float(),
                    que_imgs_info['coords'],
                    align_corners=True,
                    inter_mode='nearest'
                )
        
        outputs['pixel_depth_gt'] = interpolate_feats(
            que_imgs_info['depth'], que_imgs_info['coords'], align_corners=True)
        
        outputs['pixel_colors_gt'] = interpolate_feats(
                        que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)
        
        outputs['pixel_label_nr'] = semantic_map[None]
        outputs['sem_out'] = sem_out[None]

        
        return outputs

    def render(self, que_imgs_info, ref_imgs_info, is_train):
        nb_views = self.cfg['nb_views']
        # nb_views = 8
        feats_vol, feats_fpn, depth_map, depth_values = self.geo_reasoner(
            imgs=ref_imgs_info["norm_imgs"][None, :nb_views],
            affine_mats=ref_imgs_info["affine_mats"][None, :nb_views],
            affine_mats_inv=ref_imgs_info["affine_mats_inv"][None, :nb_views],
            near_far=ref_imgs_info["near_fars"][None, :nb_views],
            closest_idxs=ref_imgs_info["closest_idxs"][None, :nb_views],
            gt_depths=ref_imgs_info["depths_aug"][None, :nb_views],
        )
        y, valids = self.build_ms_voxel(ref_imgs_info)
        ## Normalizing depth maps in NDC coordinate
        depth_map_norm = {}
        for l in range(3):
            depth_map_norm[f"level_{l}"] = (
                depth_map[f"level_{l}"].detach() - depth_values[f"level_{l}"][:, :, 0]
            ) / (
                depth_values[f"level_{l}"][:, :, -1]
                - depth_values[f"level_{l}"][:, :, 0]
            )
        ref_imgs_info['feats_vol'] = feats_vol
        ref_imgs_info['feats_fpn'] = feats_fpn 
        ref_imgs_info['depth_map'] = depth_map 
        ref_imgs_info['depth_values'] = depth_values
        ref_imgs_info['depth_map_norm'] = depth_map_norm
        if self.cfg['use_tpvformer']:
            self.tpvformer.tpv_head.encoder.pc_range = ref_imgs_info['pc_range']
            tpv_hw,tpv_zh,tpv_wz = self.tpvformer(
                    img_metas=ref_imgs_info['img_metas'],
                    img=ref_imgs_info['norm_imgs'][None],
                    # img_feat2 = ref_imgs_info['seg_logits']
                    )
            tpv_h = self.cfg['tpv_h']
            tpv_w = self.cfg['tpv_w']
            tpv_z = self.cfg['tpv_z']
            tpv_hw = tpv_hw.permute(0,2,1).reshape((1,-1,tpv_h,tpv_w))
            tpv_zh = tpv_zh.permute(0,2,1).reshape((1,-1,tpv_z,tpv_h))
            tpv_wz = tpv_wz.permute(0,2,1).reshape((1,-1,tpv_w,tpv_z))
            ref_imgs_info['tpv_hw'] = tpv_hw
            ref_imgs_info['tpv_zh'] = tpv_zh
            ref_imgs_info['tpv_wz'] = tpv_wz

        # if 'aabb' in self.cfg.keys():
        #     ref_imgs_info['aabb'] = torch.tensor(self.cfg['aabb']).to(que_imgs_info['Ks'].device)
        #     que_imgs_info['aabb'] = torch.tensor(self.cfg['aabb']).to(que_imgs_info['Ks'].device)

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
        render_info_all['depth_map'] = ref_imgs_info['depth_map']
        # render_info['ref_sem_pred'] = ref_sem_pred
        # if self.use_dat:
        #     render_info_all['ref_img_feats_dat'] = ref_img_feats_dat

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
        # mlvl_feats =  ref_info['mlvl_feats']
        mlvl_feats = [ref_info['seg_logits']]
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
            origin = torch.as_tensor(self.cfg['origin']).to(points.device)
            n_voxels = torch.as_tensor(self.cfg['n_voxels']).to(points.device)
            voxel_size = torch.as_tensor(self.cfg['voxel_size']).to(points.device)
            size_ = n_voxels*voxel_size
            points = points - (origin - n_voxels / 2. * voxel_size)[...,None,None,None]
            points = points / size_[...,None,None,None] * torch.as_tensor(ref_info['voxel_size'])[...,None,None,None]
            points = points + ref_info['aabb'][0][...,None,None,None] 
            points = points.float()
            # aabb = ref_info['aabb'].cpu().numpy().tolist()
            # p_ = points.permute(1,2,3,0).reshape((-1,3)).cpu().numpy()
            # fig = draw_aabb(aabb_min=aabb[0],aabb_max=aabb[1])
            # fig = draw_cam(fig,p_)
            # fig.write_html("global_volume.html")
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
            

            denorm_images = ref_info['imgs'].float().to(volume.device)
            # denorm_images = denorm_images.reshape([-1] + list(denorm_images.shape)[2:])
            rgb_projection =  compute_projection(intrinsics,extrinsics).to(feature.device)
            rgb_volume, _ = backproject(
                    denorm_images,
                    points,
                    rgb_projection,
                    None,
                    None)
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
            global_volume[:, valid[0]==0] = .0
            volumes.append(self.necks_3d(global_volume[None]))
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

    def forward(self, data):
        self.step = data['step']
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()
        is_train = 'eval' not in data

        src_imgs_info = data['src_imgs_info'].copy(
        ) if 'src_imgs_info' in data else None
        render_outputs = self.render_call(
            que_imgs_info, ref_imgs_info, is_train, src_imgs_info)

        return render_outputs
