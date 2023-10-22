import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import einops
from einops import rearrange, reduce, repeat
from sray.network.cranet import *
from sray.network.cranet_v3 import *
from sray.network.cranet_v4 import *
from timm.models.layers import to_2tuple, trunc_normal_
from sray.network.ops import Conv2DMod, conv, has_nan


class CRANet_CV_IBR(IBRNetWithNeuRay):
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
        # self.ds_ref_img_f = nn.Sequential(
        #     conv(32,32,3,2),
        #     nn.ELU(inplace=True),
        #     conv(32,32,3,2),
        #     nn.ELU(inplace=True),
        #     conv(32,32,3,5)
        # )
        self.pos_encoding_2 = self.posenc(d_hid=32, n_samples=self.n_samples)
        self.sem_w_fc1 = nn.Sequential(
          nn.Linear(32,16) ,
          nn.ELU(inplace=True),
          nn.Linear(16,1) ,
        )
        self.sem_w_fc2 = nn.Sequential(
          nn.Linear(32,16) ,
          nn.ELU(inplace=True),
          nn.Linear(16,1) ,
        )
        self.sem_fc = nn.Sequential(
          nn.Linear(64,32) ,
        )
        self.ray_attention_2 = MultiHeadAttention(8, 32, 8, 8)

        if self.color_cal_type != 'rgb_in':
            self.rgb_out = nn.Sequential(nn.Linear(32+1+4, 32),
                                        nn.ELU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, 16),
                                        nn.ELU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(16, 3),
                                        nn.Sigmoid())
        self.dattn = DAttention_v2((15,20),None,4,8,4,0.1,0.1,1, 1.0 ,False,False,False,False,3,False)

        self.m2f_feat_prj = nn.Linear(256*3, 32)
        self.global_volume_prj = nn.Linear(256*3,32)
        
        self.global_volume_fc = nn.Linear(32, 16)
        
        
    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, prj_dict):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        rn,dn,rfn,_ = rgb_feat.shape
        global_volume_feats = prj_dict['global_volume_feats'].squeeze(1).permute(1,2,0,3) # rn,dn,rfn,_
        global_volume_feats = self.global_volume_prj(global_volume_feats)[:,:,0]
        seg_logits = prj_dict['seg_logits'].permute(1,0,2).reshape((rn,dn,rfn,-1))
        m2f_feats = []
        for f in prj_dict['mlvl_feats']:
            f = f.permute(1,0,2).reshape((rn,dn,rfn,-1))
            m2f_feats.append(f)
        m2f_feats = torch.cat(m2f_feats,-1)
        
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        # rgb_feat = rgb_feat + direction_feat
        rgb_feat =torch.cat([rgb_in,self.m2f_feat_prj(m2f_feats)],-1) + direction_feat
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
        # globalfeat = self.geometry_fc(globalfeat)
        # globalfeat = self.geometry_fc(globalfeat) + self.global_volume_fc(global_volume_feats)  # [n_rays, n_samples, 16]
        globalfeat = self.global_volume_fc(global_volume_feats)
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

        sem_feat = sem_latent
        # sem_feat = self.m2f_feat_prj(m2f_feats) #+ sem_latent #rgb_feat_dat
        # b, n, v, f = sem_feat.shape
        # sem_feat = sem_feat.permute(0, 2, 1, 3).reshape(-1, n, f)
        # ref_sem_feats_ds = self.ds_ref_img_f(ref_sem_feats_dat)

        # sem_feat_res,_,_ = self.dattn(
        #     ref_sem_feats_ds,
        #     ref_sem_feats_dat,
        #     sem_feat,
        #     einops.rearrange(mask,'rn dn refn c -> (rn refn) dn c')
        # )
        

        # sem_feat_res = einops.rearrange(sem_feat_res,'(rn refn) dn c -> rn dn refn c',refn = v)
        sem_global = self.gf2sgf(globalfeat)
        sigma_feat = self.out_geometry_fc[0](globalfeat)
        sigma_feat = self.out_geometry_fc[1](sigma_feat)
        # sem_feat = einops.rearrange(sem_feat,'(rn refn) dn c -> rn dn refn c',refn=sem_feat_res.shape[-2])  
        # sem_feat = torch.cat([sem_feat,sem_feat_res],-1)
        # sem_feat = self.sem_fuse(sem_feat)
            
        
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
        x = self.sem_w_fc1(sem_feat)
        
        blending_weights_sem1 = F.softmax(x, dim=2)  # color blending
        # blending_weights_sem2 = self.sem_w_fc2(sem_feat).softmax(2)
        sem_feat = torch.sum(sem_feat * blending_weights_sem1, dim=2)
        # seg_logits = torch.sum(seg_logits * blending_weights_sem2, dim=2)
        # seg_logits = torch.sum(seg_logits * blending_weights_sem1, dim=2)
        # sem_feat = sem_feat + self.pos_encoding_2
        # sem_feat, _ = self.ray_attention_2(sem_feat, sem_feat, sem_feat,
        #                                    mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        
        # sem_feat = seg_logits #+ self.sem_out(sem_feat)  
        sem_feat =  self.sem_out(sem_feat)  
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
    'CRANet_Def_IBR_v3':CRANet_Def_IBR_v3,
    'CRANet_CV_IBR':CRANet_CV_IBR,
    # 'CRANet_Def_2':CRANet_Def_2,
    'IBRNet':IBRNet,
    'IBRNetWithNeuRay':IBRNetWithNeuRay,

}

