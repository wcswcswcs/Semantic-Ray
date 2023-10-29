import torch
import torch.nn as nn
import torch.nn.functional as F
from sray.network.ops import interpolate_feats
from sray.utils.utils import SL1Loss,compute_depth_loss_scale_and_shift
import cv2

def calculate_weights(label, fine_label, threshold=1.):
    # 计算label与fine_label之间的差异
    diff = torch.abs(label - fine_label)/(label+1e-3)
    
    # 使用差异来计算权重，当差异小于阈值时，权重接近1，否则接近0
    w = 1 - (diff / threshold)
    
    # 将小于0的值截断为3e-2
    w[w < 0] = 3e-2
    
    return w

class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys = keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass


class RenderLoss(Loss):
    default_cfg = {
        'use_ray_mask': False,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
        'render_loss_scale': 1.0,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__([f'loss_rgb'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgb_gt = data_pr['pixel_colors_gt']  # 1,rn,3
        rgb_nr = data_pr['pixel_colors_nr']  # 1,rn,3
        # rgb_nr_c = data_pr['pixel_colors_c']

        def compute_loss(rgb_pr, rgb_gt):
            loss = torch.sum((rgb_pr-rgb_gt)**2, -1)        # b,n
            if self.cfg['use_ray_mask']:
                ray_mask = data_pr['ray_mask'].float()  # 1,rn
                loss = torch.sum(loss*ray_mask, 1) / \
                    (torch.sum(ray_mask, 1)+1e-3)
            else:
                loss = torch.mean(loss, 1)
            return loss * self.cfg['render_loss_scale']

        results = {'loss_rgb_nr': compute_loss(rgb_nr, rgb_gt)}
        # results['loss_rgb_nr_c'] = compute_loss(rgb_nr_c, rgb_gt)
        if self.cfg['use_dr_loss']:
            rgb_dr = data_pr['pixel_colors_dr']  # 1,rn,3
            results['loss_rgb_dr'] = compute_loss(rgb_dr, rgb_gt)
        if self.cfg['use_dr_fine_loss']:
            results['loss_rgb_dr_fine'] = compute_loss(
                data_pr['pixel_colors_dr_fine'], rgb_gt)
        if self.cfg['use_nr_fine_loss']:
            results['loss_rgb_nr_fine'] = compute_loss(
                data_pr['pixel_colors_nr_fine'], rgb_gt)
        return results


class DepthLoss(Loss):
    default_cfg = {
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
    }

    def __init__(self, cfg):
        super().__init__(['loss_depth'])
        self.cfg = {**self.default_cfg, **cfg}
        if self.cfg['depth_loss_type'] == 'smooth_l1':
            self.loss_op = nn.SmoothL1Loss(
                reduction='none', beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'true_depth' not in data_gt['ref_imgs_info']:
            return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        coords = data_pr['depth_coords']  # rfn,pn,2
        depth_pr = data_pr['depth_mean']  # rfn,pn
        depth_maps = data_gt['ref_imgs_info']['true_depth']  # rfn,1,h,w
        rfn, _, h, w = depth_maps.shape
        depth_gt = interpolate_feats(
            depth_maps, coords, h, w, padding_mode='border', align_corners=True)[..., 0]   # rfn,pn

        # transform to inverse depth coordinate
        depth_range = data_gt['ref_imgs_info']['depth_range']  # rfn,2
        near, far = -1/depth_range[:, 0:1], -1/depth_range[:, 1:2]  # rfn,1

        def process(depth):
            depth = torch.clamp(depth, min=1e-5)
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth
        depth_gt = process(depth_gt)

        # compute loss
        def compute_loss(depth_pr):
            if self.cfg['depth_loss_type'] == 'l2':
                loss = (depth_gt - depth_pr)**2
            elif self.cfg['depth_loss_type'] == 'smooth_l1':
                loss = self.loss_op(depth_gt, depth_pr)

            if data_gt['scene_name'].startswith('gso'):
                # rfn,1,h,w
                depth_maps_noise = data_gt['ref_imgs_info']['depth']
                depth_aug = interpolate_feats(
                    depth_maps_noise, coords, h, w, padding_mode='border', align_corners=True)[..., 0]  # rfn,pn
                depth_aug = process(depth_aug)
                mask = (torch.abs(depth_aug-depth_gt) <
                        self.cfg['depth_correct_thresh']).float()
                loss = torch.sum(loss * mask, 1) / (torch.sum(mask, 1) + 1e-4)
            else:
                loss = torch.mean(loss, 1)
            return loss

        outputs = {'loss_depth': compute_loss(depth_pr)}
        if 'depth_mean_fine' in data_pr:
            outputs['loss_depth_fine'] = compute_loss(
                data_pr['depth_mean_fine'])
        return outputs
class DepthLoss_v2(Loss):
    default_cfg = {
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
    }

    def __init__(self, cfg):
        super().__init__(['loss_depth'])
        self.cfg = {**self.default_cfg, **cfg}
        if self.cfg['depth_loss_type'] == 'smooth_l1':
            self.loss_op = nn.SmoothL1Loss(
                reduction='none', beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, **kwargs):

        depth_gt = data_pr['pixel_depth_gt']
        render_depth = data_pr['render_depth']

        # loss = torch.mean((render_depth-depth_gt)**2)
        loss = compute_depth_loss_scale_and_shift(render_depth,depth_gt).unsqueeze(0)   

        outputs = {'loss_depth': loss* self.cfg.get('depth_loss_scale',0.)}

        return outputs

class RefDepthLoss(Loss):
    default_cfg = {
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
    }

    def __init__(self, cfg):
        super().__init__(['loss_depth'])
        self.cfg = {**self.default_cfg, **cfg}
        if self.cfg['depth_loss_type'] == 'smooth_l1':
            self.loss_op = nn.SmoothL1Loss(
                reduction='none', beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        
        pred_depth = data_pr['depth_map']
        nb_view = pred_depth['level_1'].shape[1]
        gt_depth = data_gt['ref_imgs_info']['depth'][:nb_view]
        gt_depth_dict = {}
        depth_h = gt_depth
        for i in range(3):
            gt_depth_dict[f"level_{i}"] = F.interpolate(
                depth_h,
                scale_factor=1/ (2**i),
                mode='nearest'
            ).permute(1,0,2,3)
            
        loss = SL1Loss()(pred_depth,gt_depth_dict).unsqueeze(0)
        outputs = {'loss_ref_depth': loss* self.cfg.get('ref_depth_scale',0.)}
        return outputs

class SemanticLoss(Loss):
    def __init__(self, cfg):
        super().__init__(['loss_semantic'])
        self.cfg = cfg
        self.ignore_label = cfg['ignore_label']

    def __call__(self, data_pr, data_gt, step, **kwargs):
        num_classes = data_pr['pixel_label_nr'].shape[-1]
        def compute_loss(label_pr, label_gt):
            label_pr = label_pr.reshape(-1, num_classes)
            label_gt = label_gt.reshape(-1).long()
            valid_mask = (label_gt != self.ignore_label)
            label_pr = label_pr[valid_mask]
            label_gt = label_gt[valid_mask]
            return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)
        
        pixel_label_gt = data_pr['pixel_label_gt']
        pixel_label_nr = data_pr['pixel_label_nr']
        coarse_loss = compute_loss(pixel_label_nr, pixel_label_gt)
        
        if 'pixel_label_gt_fine' in data_pr:
            pixel_label_gt_fine = data_pr['pixel_label_gt_fine']
            pixel_label_nr_fine = data_pr['pixel_label_nr_fine']
            fine_loss = compute_loss(pixel_label_nr_fine, pixel_label_gt_fine)
        else:
            fine_loss = torch.zeros_like(coarse_loss)
        
        loss = (coarse_loss + fine_loss) * self.cfg['semantic_loss_scale']
        ret ={'loss_semantic': loss}
        if 'ref_sem_pred' in data_pr:
            ref_labels_pr = data_pr['ref_sem_pred'].permute(0, 2, 3, 1)
            ref_labels_gt = data_gt['ref_imgs_info']['labels'].permute(0, 2, 3, 1)
            ref_loss = compute_loss(ref_labels_pr, ref_labels_gt)
            loss_2d = ref_loss * self.cfg.get('semantic_loss_2d_scale',0.)
            ret.update({
                'loss_semantic_2d':loss_2d
            })
        return ret

class SemanticLoss_v2(Loss):
    def __init__(self, cfg):
        super().__init__(['loss_semantic'])
        self.cfg = cfg
        self.ignore_label = cfg['ignore_label']

    def __call__(self, data_pr, data_gt, step, **kwargs):
        num_classes = data_pr['pixel_label_nr'].shape[-1]
        def compute_loss(label_pr, label_gt,mask =None):
            label_pr = label_pr.reshape(-1, num_classes)
            label_gt = label_gt.reshape(-1).long()
            valid_mask = (label_gt != self.ignore_label)
            if mask is not None:
                if all(mask):
                    return torch.tensor(0.)
                valid_mask = valid_mask*(~mask)
            label_pr = label_pr[valid_mask]
            label_gt = label_gt[valid_mask]
            return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)
        
        pixel_label_gt = data_pr['pixel_label_gt']
        pixel_label_nr = data_pr['pixel_label_nr']
        sem_out = data_pr['sem_out']
        # d = (pixel_label_gt-sem_out).abs()
        # label_loss = torch.mean(d.gt(1.).float()).unsqueeze(0)* self.cfg.get('label_loss_scale',0.)
        # mask = (d>1.).detach()[0,:,0]
        # coarse_loss = compute_loss(pixel_label_nr, pixel_label_gt,mask)
        coarse_loss = compute_loss(pixel_label_nr, pixel_label_gt)
        
        if 'pixel_label_gt_fine' in data_pr:
            pixel_label_gt_fine = data_pr['pixel_label_gt_fine']
            pixel_label_nr_fine = data_pr['pixel_label_nr_fine']
            fine_loss = compute_loss(pixel_label_nr_fine, pixel_label_gt_fine)
        else:
            fine_loss = torch.zeros_like(coarse_loss)
        
        loss = (coarse_loss + fine_loss) * self.cfg['semantic_loss_scale']
        ret ={
            'loss_semantic': loss,
            # '_loss_label':label_loss
            }
        if 'ref_sem_pred' in data_pr:
            ref_labels_pr = data_pr['ref_sem_pred'].permute(0, 2, 3, 1)
            ref_labels_gt = data_gt['ref_imgs_info']['labels'].permute(0, 2, 3, 1)
            ref_loss = compute_loss(ref_labels_pr, ref_labels_gt)
            loss_2d = ref_loss * self.cfg.get('semantic_loss_2d_scale',0.)
            ret.update({
                'loss_semantic_2d':loss_2d
            })
        return ret

name2loss = {
    'render': RenderLoss,
    'depth': DepthLoss_v2,
    'ref_depth': RefDepthLoss,
    'semantic': SemanticLoss,
    'semantic_v2': SemanticLoss_v2,
}
