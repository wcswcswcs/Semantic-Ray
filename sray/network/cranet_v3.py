import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import einops
from einops import rearrange, reduce, repeat
from sray.network.cranet import *
from timm.models.layers import to_2tuple, trunc_normal_
from sray.network.ops import Conv2DMod, conv, has_nan

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAttention(nn.Module):

    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride, 
        offset_range_factor, use_pe, dwc_pe,
        no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Linear(
            self.nc, self.nc,
        )
        self.proj_q_md = Conv2DMod(self.nc,self.nc,1)

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Linear(
            self.nc, self.nc,
        )

        self.proj_drop = nn.Dropout(proj_drop, )
        self.attn_drop = nn.Dropout(attn_drop,)

        

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def forward(self, x, q_,mask_=None):

        # q_ : (rn ref_n)  dn c -> (ref_n rn)  c
        rn = int(q_.shape[0] / x.shape[0])
        q_cond = torch.sum(q_*mask_,1)/(torch.sum(mask_,1)+1e-6)
        # q_ = rearrange('')
        q_cond = self.proj_q(q_cond)


        _, C, H, W = x.size()
        B = q_cond.shape[0]
        dtype, device = x.dtype, x.device
        x = einops.repeat(x,"ref_n c h w -> (rn ref_n) c h w",rn = rn)
        q_t = self.proj_q_md(x,q_cond)
        q_off = einops.rearrange(q_t, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
                grid=pos[..., (1, 0)], # y, x -> x, y
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
                

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        q_ = q_.permute(0,2,1).reshape(B * self.n_heads, self.n_head_channels,-1)
        attn = torch.einsum('b c m, b c n -> b m n', q_, k) # B * h, HW, Ns
        
        attn = attn.mul(self.scale)

        
        if mask_ is not None:
            mask_ = einops.repeat(mask_,'b n c -> (b gn) n c',gn = self.n_groups)
            attn.masked_fill(mask_==0,-1e9)
            
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        out = einops.rearrange(out,'(b gn) c l -> b l (gn c)',gn = self.n_groups)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)



class DAttention_v2(nn.Module):

    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride, 
        offset_range_factor, use_pe, dwc_pe,
        no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Linear(
            self.nc, self.nc,
        )
        self.proj_q_md = Conv2DMod(self.nc,self.nc,1)

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Linear(
            self.nc, self.nc,
        )

        self.proj_drop = nn.Dropout(proj_drop, )
        self.attn_drop = nn.Dropout(attn_drop,)

        

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def forward(self, x, y, q_,mask_=None):

        # q_ : (rn ref_n)  dn c -> (ref_n rn)  c
        rn = int(q_.shape[0] / x.shape[0])
        q_cond = torch.sum(q_*mask_,1)/(torch.sum(mask_,1)+1e-6)
        # q_ = rearrange('')
        q_cond = self.proj_q(q_cond)


        _, C, H, W = x.size()
        B = q_cond.shape[0]
        dtype, device = x.dtype, x.device
        x = einops.repeat(x,"ref_n c h w -> (rn ref_n) c h w",rn = rn)
        q_t = self.proj_q_md(x,q_cond)
        q_off = einops.rearrange(q_t, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            pos = pos[..., (1, 0)] # y, x -> x, y
            pos = einops.rearrange(pos,'(rn refn gn) h w n -> (refn gn) rn (h w) n',gn = self.n_groups,refn = y.shape[0])
            x_sampled = F.grid_sample(
                input=y.reshape(y.shape[0] * self.n_groups, self.n_group_channels, y.shape[-2], y.shape[-1]), 
                grid=pos, 
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            x_sampled = einops.rearrange(
                                    x_sampled,
                                    'refn c (rn gn) (h w) -> (rn refn gn) c h w',
                                    h=offset.shape[-3],
                                    gn=self.n_groups,
                                    )
                

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        q_ = q_.permute(0,2,1).reshape(B * self.n_heads, self.n_head_channels,-1)
        attn = torch.einsum('b c m, b c n -> b m n', q_, k) # B * h, HW, Ns
        
        attn = attn.mul(self.scale)

        
        if mask_ is not None:
            mask_ = einops.repeat(mask_,'b n c -> (b gn) n c',gn = self.n_groups)
            attn.masked_fill(mask_==0,-1e9)
            
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        out = einops.rearrange(out,'(b gn) c l -> b l (gn c)',gn = self.n_groups)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class CRANet_Def(IBRNetWithNeuRay):
    def __init__(self,
                 neuray_in_dim=32,
                 in_feat_ch=32,
                 n_samples=64,
                 num_classes=20,
                 use_ptrans=False,
                 ptrans_first=False,
                 sem_only=False,
                 label_hidden=[],
                 color_cal_type='rgb_in',
                 **kwargs):
        super().__init__(
            neuray_in_dim=neuray_in_dim,
            in_feat_ch=in_feat_ch,
            n_samples=n_samples,
            **kwargs
        )
        self.color_cal_type = color_cal_type
        if len(label_hidden) > 0:
            self.semantic_fc = nn.Sequential()
            for i in range(len(label_hidden)):
                self.semantic_fc.add_module(
                    "fc{}".format(i),
                    nn.Linear(label_hidden[i-1] if i > 0 else 16,label_hidden[i])
                )
            # The output of the last layer is semantic logits
            self.semantic_fc.add_module(
                "fc{}".format(len(label_hidden)),
                nn.Linear(label_hidden[-1], num_classes + 1)  # +1 for invalid
            )
        else:
            self.semantic_fc = None
            
        self.use_ptrans = use_ptrans
        self.ptrans_first = ptrans_first
        self.sem_only = sem_only
        if use_ptrans:
            self.point_attention_new = MultiHeadAttention(4, 32, 4, 4)
            self.sem_rtrans_new = MultiHeadAttention(4, 32, 4, 4)
        self.sem_pos_encoding = self.posenc(32, n_samples=n_samples)
        self.gf2sgf = nn.Linear(16, 16)
        self.sem_out = nn.Linear(32, num_classes + 1)
        self.relu = nn.ReLU()

        #####
        self.ds_ref_img_f = nn.Sequential(
            conv(32,32,3,2),
            nn.ELU() ,
            conv(32,32,3,2),
            nn.ELU() ,
            conv(32,32,3,5)

        )
        self.pos_encoding_2 = self.posenc(d_hid=32, n_samples=self.n_samples)
        self.sem_w_fc = nn.Sequential(
          nn.Linear(32,16) ,
          nn.ELU() ,
          nn.Linear(16,1) ,
        )
        self.sem_fc = nn.Sequential(
          nn.Linear(64,32) ,
        )
        self.ray_attention_2 = MultiHeadAttention(8, 32, 8, 8)

        if self.color_cal_type != 'rgb_in':
            self.rgb_out = nn.Sequential(nn.Linear(32+1+4, 32),
                                        nn.ELU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, 16),
                                        nn.ELU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(16, 3),
                                        nn.Sigmoid())
        self.dattn = DAttention((15,20),None,4,8,4,0.1,0.1,1, 1.0 ,False,False,False,False,3,False)

        
        
    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, ref_sem_feats=None,ref_sem_feats_dat=None,rgb_feat_dat=None):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod,
                      dim=2, keepdim=True)[0]) * mask
            # means it will trust the one more with more consistent view point
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
        num_valid_obs = torch.sum(mask, dim=2)
        # neuray layer 0
        weight0 = torch.sigmoid(self.neuray_fc(
            neuray_feat)) * weight  # [rn,dn,rfn,f]
        # [n_rays, n_samples, 1, n_feat]
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)
        # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)
        # [n_rays, n_samples, 1, 2*n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)

        # [n_rays, n_samples, n_views, 3*n_feat]
        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1),
                      rgb_feat, neuray_feat], dim=-1)
        x = self.base_fc(x)
        sem_latent = x
        
        if self.use_ptrans and self.ptrans_first:
            # ptrans
            b, n, v, f = sem_latent.shape
            sem_latent = sem_latent.reshape(-1, num_views, f)
            sem_latent, _ = self.point_attention_new(
                sem_latent, sem_latent, sem_latent, mask=mask.reshape(-1, num_views, 1).float())
            sem_latent = sem_latent.reshape(b, n, v, f)
            # rtrans
            sem_latent = sem_latent.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_latent = sem_latent + self.sem_pos_encoding
            sem_latent, _ = self.sem_rtrans_new(
                sem_latent, sem_latent, sem_latent, mask=mask.permute(0, 2, 1, 3).reshape(-1, n, 1))
            # sem_latent, _ = self.sem_rtrans_new(
            #     sem_latent, sem_latent, sem_latent, mask=trans_mask)
            sem_latent = sem_latent.reshape(b, v, n, f).permute(0, 2, 1, 3)
        elif self.use_ptrans and not self.ptrans_first:
            # rtrans
            b, n, v, f = sem_latent.shape
            sem_latent = sem_latent.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_latent = sem_latent + self.sem_pos_encoding
            sem_latent, _ = self.sem_rtrans_new(
                sem_latent, sem_latent, sem_latent, mask=mask.permute(0, 2, 1, 3).reshape(-1, n, 1))
            # sem_latent, _ = self.sem_rtrans_new(
            #     sem_latent, sem_latent, sem_latent, mask=trans_mask)
            sem_latent = sem_latent.reshape(b, v, n, f).permute(0, 2, 1, 3)
            # ptrans
            sem_latent = sem_latent.reshape(-1, num_views, f)
            sem_latent, _ = self.point_attention_new(
                sem_latent, sem_latent, sem_latent, mask=mask.reshape(-1, num_views, 1).float())
            sem_latent = sem_latent.reshape(b, n, v, f)
        
        x_vis = self.vis_fc(sem_latent * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = sem_latent + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(
            2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        # set the sigma of invalid point to zero
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)

        x_ = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x_)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        # rgb computation
        if self.color_cal_type == 'rgb_in':
            rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        elif self.color_cal_type == 'feat_pred':
            x_ = self.rgb_out(x_) 
            rgb_out = torch.sum(x_*blending_weights_valid, dim=2)
        elif self.color_cal_type == 'hydrid':
            x_ = self.rgb_out(x_) + rgb_in
            rgb_out = torch.sum(x_*blending_weights_valid, dim=2)

        # semantic feature
        
        sem_global = self.gf2sgf(globalfeat)
        sigma_feat = self.out_geometry_fc[0](globalfeat)
        sigma_feat = self.out_geometry_fc[1](sigma_feat)
        sem_feat = torch.cat([
            sem_global.unsqueeze(2).expand(-1, -1, num_views, -1),
            sigma_feat.unsqueeze(2).expand(-1, -1, num_views, -1),
            sem_latent
        ], dim=-1)

        sem_feat = self.sem_fc(sem_feat)
        b, n, v, f = sem_feat.shape
        sem_feat = sem_feat.permute(0, 2, 1, 3).reshape(-1, n, f)
        ref_sem_feats = self.ds_ref_img_f(ref_sem_feats)

        sem_feat,_,_ = self.dattn(
            ref_sem_feats,
            sem_feat,
            einops.rearrange(mask,'rn dn refn c -> (rn refn) dn c')
        )

        sem_feat = einops.rearrange(sem_feat,'(rn refn) dn c -> rn dn refn c',refn = v)
        x = self.sem_w_fc(sem_feat)
        
        blending_weights_sem = F.softmax(x, dim=2)  # color blending
        sem_feat = torch.sum(sem_feat * blending_weights_sem, dim=2)
        sem_feat = sem_feat + self.pos_encoding_2
        sem_feat, _ = self.ray_attention_2(sem_feat, sem_feat, sem_feat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sem_feat = self.sem_out(sem_feat)

        
        out = torch.cat([rgb_out, sigma_out, sem_feat], dim=-1)
        return out

class CRANet_Def_IBR(IBRNetWithNeuRay):
    def __init__(self,
                 neuray_in_dim=32,
                 in_feat_ch=32,
                 n_samples=64,
                 num_classes=20,
                 use_ptrans=False,
                 ptrans_first=False,
                 sem_only=False,
                 label_hidden=[],
                 color_cal_type='rgb_in',
                 **kwargs):
        super().__init__(
            neuray_in_dim=neuray_in_dim,
            in_feat_ch=in_feat_ch,
            n_samples=n_samples,
            **kwargs
        )
        self.color_cal_type = color_cal_type
        if len(label_hidden) > 0:
            self.semantic_fc = nn.Sequential()
            for i in range(len(label_hidden)):
                self.semantic_fc.add_module(
                    "fc{}".format(i),
                    nn.Linear(label_hidden[i-1] if i > 0 else 16,label_hidden[i])
                )
            # The output of the last layer is semantic logits
            self.semantic_fc.add_module(
                "fc{}".format(len(label_hidden)),
                nn.Linear(label_hidden[-1], num_classes + 1)  # +1 for invalid
            )
        else:
            self.semantic_fc = None
            
        self.use_ptrans = use_ptrans
        self.ptrans_first = ptrans_first
        self.sem_only = sem_only
        if use_ptrans:
            # self.point_attention_new = MultiHeadAttention(4, 32, 4, 4)
            # self.sem_rtrans_new = MultiHeadAttention(4, 32, 4, 4)
            self.point_attention_2 = MultiHeadAttention(4, 32, 4, 4)
            self.sem_rtrans_2 = MultiHeadAttention(4, 32, 4, 4)
        self.sem_pos_encoding = self.posenc(32, n_samples=n_samples)
        self.gf2sgf = nn.Linear(16, 16)
        self.sem_out = nn.Linear(32, num_classes + 1)
        self.relu = nn.ReLU()
        self.sem_fuse = nn.Linear(64,32)
        #####
        self.ds_ref_img_f = nn.Sequential(
            conv(32,32,3,2),
            nn.ELU() ,
            conv(32,32,3,2),
            nn.ELU() ,
            conv(32,32,3,5)
        )
        self.pos_encoding_2 = self.posenc(d_hid=32, n_samples=self.n_samples)
        self.sem_w_fc = nn.Sequential(
          nn.Linear(32,16) ,
          nn.ELU() ,
          nn.Linear(16,1) ,
        )
        self.sem_fc = nn.Sequential(
          nn.Linear(64,32) ,
        )
        self.ray_attention_2 = MultiHeadAttention(8, 32, 8, 8)

        if self.color_cal_type != 'rgb_in':
            self.rgb_out = nn.Sequential(nn.Linear(32+1+4, 32),
                                        nn.ELU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, 16),
                                        nn.ELU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(16, 3),
                                        nn.Sigmoid())
        self.dattn = DAttention_v2((15,20),None,4,8,4,0.1,0.1,1, 1.0 ,False,False,False,False,3,False)

        
        
    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, ref_sem_feats=None,ref_sem_feats_dat=None,rgb_feat_dat=None):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod,
                      dim=2, keepdim=True)[0]) * mask
            # means it will trust the one more with more consistent view point
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        weight0 = torch.sigmoid(self.neuray_fc(
            neuray_feat)) * weight  # [rn,dn,rfn,f]
        # [n_rays, n_samples, 1, n_feat]
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)
        # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)
        # [n_rays, n_samples, 1, 2*n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)

        # [n_rays, n_samples, n_views, 3*n_feat]
        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1),
                      rgb_feat, neuray_feat], dim=-1)
        x = self.base_fc(x)
        sem_latent = x

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(
            2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        # set the sigma of invalid point to zero
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)

        # semantic feature
        
        # sem_global = self.gf2sgf(globalfeat)
        # sigma_feat = self.out_geometry_fc[0](globalfeat)
        # sigma_feat = self.out_geometry_fc[1](sigma_feat)
        # sem_feat = torch.cat([
        #     sem_global.unsqueeze(2).expand(-1, -1, num_views, -1),
        #     sigma_feat.unsqueeze(2).expand(-1, -1, num_views, -1),
        #     sem_latent
        # ], dim=-1)

        # sem_feat = self.sem_fc(sem_feat)
        sem_feat = sem_latent
        b, n, v, f = sem_feat.shape
        sem_feat = sem_feat.permute(0, 2, 1, 3).reshape(-1, n, f)
        ref_sem_feats_ds = self.ds_ref_img_f(ref_sem_feats)

        sem_feat_res,_,_ = self.dattn(
            ref_sem_feats_ds,
            ref_sem_feats,
            sem_feat,
            einops.rearrange(mask,'rn dn refn c -> (rn refn) dn c')
        )
        

        sem_feat_res = einops.rearrange(sem_feat_res,'(rn refn) dn c -> rn dn refn c',refn = v)
        sem_global = self.gf2sgf(globalfeat)
        sigma_feat = self.out_geometry_fc[0](globalfeat)
        sigma_feat = self.out_geometry_fc[1](sigma_feat)
        sem_feat = einops.rearrange(sem_feat,'(rn refn) dn c -> rn dn refn c',refn=sem_feat_res.shape[-2])  
        sem_feat = torch.cat([sem_feat,sem_feat_res],-1)
        sem_feat = self.sem_fuse(sem_feat)
            
        
        if self.use_ptrans and self.ptrans_first:
            # ptrans
            b, n, v, f = sem_feat.shape
            sem_feat = sem_feat.reshape(-1, num_views, f)
            sem_feat, _ = self.point_attention_2(
                sem_feat, sem_feat, sem_feat, mask=mask.reshape(-1, num_views, 1).float())
            sem_feat = sem_feat.reshape(b, n, v, f)
            # rtrans
            sem_feat = sem_feat.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_feat = sem_feat + self.sem_pos_encoding
            sem_feat, _ = self.sem_rtrans_2(
                sem_feat, sem_feat, sem_feat, mask=(mask.permute(0, 2, 1, 3).reshape(-1, n, 1))*trans_mask)
            # sem_feat, _ = self.sem_rtrans_2(
            #     sem_feat, sem_feat, sem_feat, mask=trans_mask)
            sem_feat = sem_feat.reshape(b, v, n, f).permute(0, 2, 1, 3)
        elif self.use_ptrans and not self.ptrans_first:
            # rtrans
            b, n, v, f = sem_feat.shape
            sem_feat = sem_feat.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_feat = sem_feat + self.sem_pos_encoding
            sem_feat, _ = self.sem_rtrans_2(
                sem_feat, sem_feat, sem_feat, mask=(mask.permute(0, 2, 1, 3).reshape(-1, n, 1))*trans_mask)
            # sem_feat, _ = self.sem_rtrans_2(
            #     sem_feat, sem_feat, sem_feat, mask=trans_mask)
            sem_feat = sem_feat.reshape(b, v, n, f).permute(0, 2, 1, 3)
            # ptrans
            sem_feat= sem_feat.reshape(-1, num_views, f)
            sem_feat, _ = self.point_attention_2(
                sem_feat, sem_feat, sem_feat, mask=mask.reshape(-1, num_views, 1).float())
            sem_feat = sem_feat.reshape(b, n, v, f)
        
        sem_feat = torch.cat([
            sem_global.unsqueeze(2).expand(-1, -1, num_views, -1),
            sigma_feat.unsqueeze(2).expand(-1, -1, num_views, -1),
            sem_feat
        ], dim=-1)

        sem_feat = self.sem_fc(sem_feat)
        x = self.sem_w_fc(sem_feat)
        
        blending_weights_sem = F.softmax(x, dim=2)  # color blending
        sem_feat = torch.sum(sem_feat * blending_weights_sem, dim=2)
        sem_feat = sem_feat + self.pos_encoding_2
        sem_feat, _ = self.ray_attention_2(sem_feat, sem_feat, sem_feat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sem_feat = self.sem_out(sem_feat)
        # sem_feat = sem_feat.masked_fill(num_valid_obs < 1, 0.)

        
        out = torch.cat([rgb_out, sigma_out, sem_feat], dim=-1)
        return out

class CRANet_Def_IBR_v2(IBRNetWithNeuRay):
    def __init__(self,
                 neuray_in_dim=32,
                 in_feat_ch=32,
                 n_samples=64,
                 num_classes=20,
                 use_ptrans=False,
                 ptrans_first=False,
                 sem_only=False,
                 label_hidden=[],
                 color_cal_type='rgb_in',
                 **kwargs):
        super().__init__(
            neuray_in_dim=neuray_in_dim,
            in_feat_ch=in_feat_ch,
            n_samples=n_samples,
            **kwargs
        )
        self.color_cal_type = color_cal_type
        if len(label_hidden) > 0:
            self.semantic_fc = nn.Sequential()
            for i in range(len(label_hidden)):
                self.semantic_fc.add_module(
                    "fc{}".format(i),
                    nn.Linear(label_hidden[i-1] if i > 0 else 16,label_hidden[i])
                )
            # The output of the last layer is semantic logits
            self.semantic_fc.add_module(
                "fc{}".format(len(label_hidden)),
                nn.Linear(label_hidden[-1], num_classes + 1)  # +1 for invalid
            )
        else:
            self.semantic_fc = None
            
        self.use_ptrans = use_ptrans
        self.ptrans_first = ptrans_first
        self.sem_only = sem_only
        if use_ptrans:
            # self.point_attention_new = MultiHeadAttention(4, 32, 4, 4)
            # self.sem_rtrans_new = MultiHeadAttention(4, 32, 4, 4)
            self.point_attention_2 = MultiHeadAttention(4, 32, 4, 4)
            self.sem_rtrans_2 = MultiHeadAttention(4, 32, 4, 4)
        self.sem_pos_encoding = self.posenc(32, n_samples=n_samples)
        self.gf2sgf = nn.Linear(16, 16)
        self.sem_out = nn.Linear(32, num_classes + 1)
        self.relu = nn.ReLU()
        self.sem_fuse = nn.Linear(64,32)
        #####
        self.ds_ref_img_f = nn.Sequential(
            conv(32,32,3,2),
            nn.ELU() ,
            conv(32,32,3,2),
            nn.ELU() ,
            conv(32,32,3,5)
        )
        self.pos_encoding_2 = self.posenc(d_hid=32, n_samples=self.n_samples)
        self.sem_w_fc = nn.Sequential(
          nn.Linear(32,16) ,
          nn.ELU() ,
          nn.Linear(16,1) ,
        )
        self.sem_fc = nn.Sequential(
          nn.Linear(64,32) ,
        )
        self.ray_attention_2 = MultiHeadAttention(8, 32, 8, 8)

        if self.color_cal_type != 'rgb_in':
            self.rgb_out = nn.Sequential(nn.Linear(32+1+4, 32),
                                        nn.ELU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, 16),
                                        nn.ELU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(16, 3),
                                        nn.Sigmoid())
        self.dattn = DAttention_v2((15,20),None,4,8,4,0.1,0.1,1, 1.0 ,False,False,False,False,3,False)

        
        
    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, ref_sem_feats=None,ref_sem_feats_dat=None,rgb_feat_dat=None):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod,
                      dim=2, keepdim=True)[0]) * mask
            # means it will trust the one more with more consistent view point
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        weight0 = torch.sigmoid(self.neuray_fc(
            neuray_feat)) * weight  # [rn,dn,rfn,f]
        # [n_rays, n_samples, 1, n_feat]
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)
        # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)
        # [n_rays, n_samples, 1, 2*n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)

        # [n_rays, n_samples, n_views, 3*n_feat]
        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1),
                      rgb_feat, neuray_feat], dim=-1)
        x = self.base_fc(x)
        sem_latent = x

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(
            2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        # set the sigma of invalid point to zero
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)

        # semantic feature
        
        # sem_global = self.gf2sgf(globalfeat)
        # sigma_feat = self.out_geometry_fc[0](globalfeat)
        # sigma_feat = self.out_geometry_fc[1](sigma_feat)
        # sem_feat = torch.cat([
        #     sem_global.unsqueeze(2).expand(-1, -1, num_views, -1),
        #     sigma_feat.unsqueeze(2).expand(-1, -1, num_views, -1),
        #     sem_latent
        # ], dim=-1)

        # sem_feat = self.sem_fc(sem_feat)
        sem_feat = rgb_feat_dat
        b, n, v, f = sem_feat.shape
        sem_feat = sem_feat.permute(0, 2, 1, 3).reshape(-1, n, f)
        ref_sem_feats_ds = self.ds_ref_img_f(ref_sem_feats_dat)

        sem_feat_res,_,_ = self.dattn(
            ref_sem_feats_ds,
            ref_sem_feats_dat,
            sem_feat,
            einops.rearrange(mask,'rn dn refn c -> (rn refn) dn c')
        )
        

        sem_feat_res = einops.rearrange(sem_feat_res,'(rn refn) dn c -> rn dn refn c',refn = v)
        sem_global = self.gf2sgf(globalfeat)
        sigma_feat = self.out_geometry_fc[0](globalfeat)
        sigma_feat = self.out_geometry_fc[1](sigma_feat)
        sem_feat = einops.rearrange(sem_feat,'(rn refn) dn c -> rn dn refn c',refn=sem_feat_res.shape[-2])  
        sem_feat = torch.cat([sem_feat,sem_feat_res],-1)
        sem_feat = self.sem_fuse(sem_feat)
            
        
        if self.use_ptrans and self.ptrans_first:
            # ptrans
            b, n, v, f = sem_feat.shape
            sem_feat = sem_feat.reshape(-1, num_views, f)
            sem_feat, _ = self.point_attention_2(
                sem_feat, sem_feat, sem_feat, mask=mask.reshape(-1, num_views, 1).float())
            sem_feat = sem_feat.reshape(b, n, v, f)
            # rtrans
            sem_feat = sem_feat.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_feat = sem_feat + self.sem_pos_encoding
            sem_feat, _ = self.sem_rtrans_2(
                sem_feat, sem_feat, sem_feat, mask=(mask.permute(0, 2, 1, 3).reshape(-1, n, 1))*trans_mask)
            # sem_feat, _ = self.sem_rtrans_2(
            #     sem_feat, sem_feat, sem_feat, mask=trans_mask)
            sem_feat = sem_feat.reshape(b, v, n, f).permute(0, 2, 1, 3)
        elif self.use_ptrans and not self.ptrans_first:
            # rtrans
            b, n, v, f = sem_feat.shape
            sem_feat = sem_feat.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_feat = sem_feat + self.sem_pos_encoding
            sem_feat, _ = self.sem_rtrans_2(
                sem_feat, sem_feat, sem_feat, mask=(mask.permute(0, 2, 1, 3).reshape(-1, n, 1))*trans_mask)
            # sem_feat, _ = self.sem_rtrans_2(
            #     sem_feat, sem_feat, sem_feat, mask=trans_mask)
            sem_feat = sem_feat.reshape(b, v, n, f).permute(0, 2, 1, 3)
            # ptrans
            sem_feat= sem_feat.reshape(-1, num_views, f)
            sem_feat, _ = self.point_attention_2(
                sem_feat, sem_feat, sem_feat, mask=mask.reshape(-1, num_views, 1).float())
            sem_feat = sem_feat.reshape(b, n, v, f)
        
        sem_feat = torch.cat([
            sem_global.unsqueeze(2).expand(-1, -1, num_views, -1),
            sigma_feat.unsqueeze(2).expand(-1, -1, num_views, -1),
            sem_feat
        ], dim=-1)

        sem_feat = self.sem_fc(sem_feat)
        x = self.sem_w_fc(sem_feat)
        
        # blending_weights_sem = F.softmax(x, dim=2)  # color blending
        sem_feat = torch.sum(sem_feat * blending_weights_valid, dim=2)
        sem_feat = sem_feat + self.pos_encoding_2
        sem_feat, _ = self.ray_attention_2(sem_feat, sem_feat, sem_feat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sem_feat = self.sem_out(sem_feat)
        # sem_feat = sem_feat.masked_fill(num_valid_obs < 1, 0.)

        
        out = torch.cat([rgb_out, sigma_out, sem_feat], dim=-1)
        return out


name2network = {
    'CRANet':CRANet,
    'CRANet_v1':CRANet_v1,
    'CRANet_v2':CRANet_v2,
    'CRANet_Def':CRANet_Def,
    'CRANet_Def_IBR':CRANet_Def_IBR,
    'CRANet_Def_IBR_v2':CRANet_Def_IBR_v2,
    # 'CRANet_Def_2':CRANet_Def_2,
    'IBRNet':IBRNet,
    'IBRNetWithNeuRay':IBRNetWithNeuRay,

}

