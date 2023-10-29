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
from sray.utils.utils import positional_encoding
from sray.network.ops import has_nan


class CRANet_CV_IBR(nn.Module):
    def __init__(self,
                 n_samples=32 + 64,
                 num_classes=20,
                 use_ptrans=False,
                 ptrans_first=False,
                 sem_only=False,
                 label_hidden=[],
                 color_cal_type='rgb_in',
                 **kwargs):
        super().__init__()
        self.color_cal_type = color_cal_type
        if len(label_hidden) > 0:
            self.semantic_fc = nn.Sequential()
            for i in range(len(label_hidden)):
                self.semantic_fc.add_module(
                    "fc{}".format(i),
                    nn.Linear(label_hidden[i - 1] if i > 0 else 16, label_hidden[i])
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

        self.point_attention = MultiHeadAttention(4, 64, 4, 4)
        self.sem_rtrans = MultiHeadAttention(4, 64, 4, 4)

        self.pos_encoding = self.posenc(64, n_samples=n_samples)
        self.sem_out = nn.Linear(32, num_classes + 1)
        self.relu = nn.ReLU()
        self.sem_fuse = nn.Linear(64, 32)
        #####
        # self.ds_ref_img_f = nn.Sequential(
        #     conv(32,32,3,2),
        #     nn.ELU(inplace=True),
        #     conv(32,32,3,2),
        #     nn.ELU(inplace=True),
        #     conv(32,32,3,5)
        # )
        # self.pos_encoding_2 = self.posenc(d_hid=32, n_samples=self.n_samples)
        self.sem_w_fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 1),
        )
        # self.sem_w_fc2 = nn.Sequential(
        #   nn.Linear(32,16) ,
        #   nn.ELU(inplace=True),
        #   nn.Linear(16,1) ,
        # )
        self.sem_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(inplace=True)
        )

        if self.color_cal_type != 'rgb_in':
            self.rgb_out = nn.Sequential(nn.Linear(32 + 1 + 4, 32),
                                         nn.ELU(inplace=True),
                                         nn.Dropout(0.2),
                                         nn.Linear(32, 16),
                                         nn.ELU(inplace=True),
                                         nn.Dropout(0.2),
                                         nn.Linear(16, 3),
                                         nn.Sigmoid())

        activation_func = nn.ELU(inplace=True)
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, 16),
                                        activation_func)

        self.tpv_fc = nn.Linear(64 * 3, 32)
        self.dim = 32
        d_inner = self.dim
        n_head = 4
        d_k = self.dim // n_head
        d_v = self.dim // n_head
        self.attn_token_gen = nn.Linear(24 + 1 + 8, self.dim)
        num_layers = 4
        self.attn_layers = nn.ModuleList(
            [
                MultiHeadAttention(4, 32, 4, 4)  # n_head, d_model, d_k, d_v
                # EncoderLayer(self.dim, d_inner, n_head, d_k, d_v)
                for i in range(num_layers)
            ]
        )

        ## Processing the mean and variance of input features
        self.var_mean_fc1 = nn.Linear(16, self.dim)
        self.var_mean_fc2 = nn.Linear(self.dim, self.dim)

        self.seg_logits_prj = nn.Linear(21, 32)
        self.var_mean_fc3 = nn.Linear(64, self.dim)
        self.var_mean_fc4 = nn.Linear(self.dim, self.dim)

        self.var_mean_fuse = nn.Linear(2 * self.dim, self.dim)

        ## Setting mask of var_mean always enabled
        self.var_mean_mask = torch.tensor([1]).cuda()
        self.var_mean_mask.requires_grad = False

        ## For aggregating data along ray samples
        self.auto_enc = ConvAutoEncoder(self.dim, 64 + 32)

        self.sigma_fc1 = nn.Linear(self.dim, self.dim)
        self.sigma_fc2 = nn.Linear(self.dim, self.dim // 2)
        self.sigma_fc3 = nn.Linear(self.dim // 2, 1)

        self.rgb_fc1 = nn.Linear(self.dim + 9 + 16, self.dim)
        self.rgb_fc2 = nn.Linear(self.dim, self.dim // 2)
        self.rgb_fc3 = nn.Linear(self.dim // 2, 3)

        self.sem_fuse_1 = nn.Linear(32 * 2, 72)
        self.sem_fuse_2 = nn.Linear(72, 32)

    def forward(self, prj_dict):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        # viewdirs, feat, occ_masks
        embedded_angles = prj_dict['embedded_angles']
        occ_masks = prj_dict['occ_masks']
        viewdirs = prj_dict['viewdirs']
        geonerf_feats = prj_dict['geonerf_feats']
        seg_logits = prj_dict['seg_logits']
        rays_pts = prj_dict['rays_pts'][0]  # N S 3
        dir_diff = prj_dict['dir_diff']
        rays_pts_pos = positional_encoding(rays_pts, 12)
        N, V, S = geonerf_feats.shape[:3]
        viewdirs = viewdirs[:, None, None].repeat(1, S, V, 1)
        tpv_hw_feat = prj_dict['tpv_hw_feat'][:, 0]
        tpv_zh_feat = prj_dict['tpv_zh_feat'][:, 0]
        tpv_wz_feat = prj_dict['tpv_wz_feat'][:, 0]
        tpv_feat = torch.cat([tpv_hw_feat, tpv_zh_feat, tpv_wz_feat], -1)
        tpv_feat = self.tpv_fc(tpv_feat)

        geonerf_feats = geonerf_feats.permute(0, 2, 1, 3)
        embedded_angles = embedded_angles.permute(0, 2, 1, 3)
        dir_diff = dir_diff.permute(0, 2, 1, 3)
        occ_masks = occ_masks.permute(0, 2, 1, 3)
        seg_logits = seg_logits.permute(0, 2, 1, 3)
        m2f_feats = self.seg_logits_prj(seg_logits)
        m2f_feats = m2f_feats.view(-1, *m2f_feats.shape[2:])

        geonerf_feats = geonerf_feats.view(-1, *geonerf_feats.shape[2:])
        v_feat = geonerf_feats[..., :24]  # 3*8 cv
        s_feat = geonerf_feats[..., 24: 24 + 8]  # 8 fpn_lvl_0
        colors = geonerf_feats[..., 24 + 8: -1]
        vis_mask = geonerf_feats[..., -1:].detach()

        occ_masks = occ_masks.view(-1, *occ_masks.shape[2:])
        viewdirs = viewdirs.view(-1, *viewdirs.shape[2:])
        embedded_angles = embedded_angles.view(-1, *embedded_angles.shape[2:])
        dir_diff = dir_diff.reshape((-1, *dir_diff.shape[2:]))

        # s_feat: (N S) V 8 | var_mean (N S) 16
        ## Mean and variance of 2D features provide view-independent tokens
        var_mean = torch.var_mean(s_feat, dim=1, unbiased=False, keepdim=True)
        var_mean = torch.cat(var_mean, dim=-1)
        var_mean = F.elu(self.var_mean_fc1(var_mean))
        var_mean = F.elu(self.var_mean_fc2(var_mean))

        var_mean_m2f = torch.var_mean(m2f_feats, dim=1, unbiased=False, keepdim=True)
        var_mean_m2f = torch.cat(var_mean_m2f, dim=-1)
        var_mean_m2f = F.elu(self.var_mean_fc3(var_mean_m2f))
        var_mean_m2f = F.elu(self.var_mean_fc4(var_mean_m2f))

        ## Converting the input features to tokens (view-dependent) before self-attention
        tokens = F.elu(
            self.attn_token_gen(torch.cat([v_feat, vis_mask, s_feat], dim=-1))
        )
        var_mean = torch.cat([var_mean, var_mean_m2f], -1)
        var_mean = F.elu(self.var_mean_fuse(var_mean))

        tokens = torch.cat([tokens, var_mean], dim=1)

        ## Adding a new channel to mask for var_mean
        vis_mask = torch.cat(
            [vis_mask, self.var_mean_mask.view(1, 1, 1).expand(N * S, -1, -1)], dim=1
        )
        ## If a point is not visible by any source view, force its masks to enabled
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)

        ## Taking occ_masks into account, but remembering if there were any visibility before that
        mask_cloned = vis_mask.clone()
        vis_mask[:, :-1] *= occ_masks
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)
        masks = vis_mask * mask_cloned

        ## Performing self-attention
        for layer in self.attn_layers:
            tokens, _ = layer(tokens, tokens, tokens, masks)

        ## Predicting sigma with an Auto-Encoder and MLP
        sigma_tokens = tokens[:, -1:]
        sigma_tokens = sigma_tokens.view(N, S, self.dim).transpose(1, 2)
        sigma_tokens = self.auto_enc(sigma_tokens)
        sigma_tokens = sigma_tokens.transpose(1, 2).reshape(N * S, 1, self.dim)

        h = F.elu(self.sigma_fc1(sigma_tokens))
        h = F.elu(self.sigma_fc2(h))
        sigma = torch.relu(self.sigma_fc3(h[:, 0]))

        ## Concatenating positional encodings and predicting RGB weights
        direction_feat = self.ray_dir_fc(dir_diff)
        # 32+16+9
        rgb_tokens = torch.cat([tokens[:, :-1], embedded_angles, direction_feat], dim=-1)
        rgb_tokens = F.elu(self.rgb_fc1(rgb_tokens))
        rgb_tokens = F.elu(self.rgb_fc2(rgb_tokens))
        rgb_w = self.rgb_fc3(rgb_tokens)
        rgb_w = masked_softmax(rgb_w, masks[:, :-1], dim=1)

        rgb = (colors * rgb_w).sum(1)

        # outputs = torch.cat([rgb, sigma], -1)
        # outputs = outputs.reshape(N, S, -1)

        sigma_tokens = sigma_tokens.reshape((N, S, -1))
        sem_tokens = torch.cat([sigma_tokens, tpv_feat], -1)
        sem_tokens = F.elu(self.sem_fuse_1(sem_tokens)) * rays_pts_pos
        sem_tokens = self.sem_fuse_2(sem_tokens)
        sem_feat = torch.cat([
            sem_tokens[:, None].repeat(1, V, 1, 1),
            m2f_feats.reshape((N, S, V, -1)).permute(0, 2, 1, 3)
        ], -1)

        # intra
        N, V, S, C = sem_feat.shape
        sem_feat = sem_feat.reshape((-1, S, C))
        occ_masks = occ_masks.reshape((N, S, V, 1)).permute(0, 2, 1, 3)
        mask = prj_dict['mask'] * occ_masks
        num_valid_obs = mask.sum(1, keepdim=True) > 0.
        trans_mask = num_valid_obs.expand(-1, V, -1, -1).reshape(N * V, S, 1)
        sem_feat = sem_feat + self.pos_encoding
        sem_feat, _ = self.point_attention(
            sem_feat, sem_feat, sem_feat, mask=(mask.reshape(-1, S, 1)) * trans_mask)

        sem_feat = sem_feat.reshape((N, V, S, -1)).permute(0, 2, 1, 3)
        # inter
        sem_feat = sem_feat.reshape(-1, V, C)
        sem_feat, _ = self.sem_rtrans(
            sem_feat, sem_feat, sem_feat, mask=mask.permute(0, 2, 1, 3).reshape(-1, V, 1).float())

        sem_feat = sem_feat.reshape((N, S, V, -1)).permute(0, 2, 1, 3)
        sem_feat = self.sem_fc(sem_feat)
        x = self.sem_w_fc(sem_feat)

        blending_weights_sem = F.softmax(x, dim=2)  # color blending
        # blending_weights_sem2 = self.sem_w_fc2(sem_feat).softmax(2)
        seg_logits = seg_logits.permute(0, 2, 1, 3)
        sem_out = torch.sum(seg_logits * blending_weights_sem, dim=1)
        labels = prj_dict['labels']
        sem_out2 = torch.sum(labels * blending_weights_sem, dim=1)

        rgb = rgb.reshape((N, S, -1))
        sigma = sigma.reshape((N, S, -1))

        out = torch.cat([rgb, sigma, sem_out, sem_out2], dim=-1)
        if has_nan(out):
            print('FFFFFuck')
        return out

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i)
                                   for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).cuda().float().unsqueeze(0)
        return sinusoid_table


class CRANet_CV_IBR_v2(CRANet_CV_IBR):
    def __init__(self,
                 n_samples=32 + 64,
                 num_classes=20,
                 use_ptrans=False,
                 ptrans_first=False,
                 sem_only=False,
                 label_hidden=[],
                 color_cal_type='rgb_in',
                 **kwargs):
        super().__init__(
            n_samples=32 + 64,
            num_classes=20,
            use_ptrans=False,
            ptrans_first=False,
            sem_only=False,
            label_hidden=[],
            color_cal_type='rgb_in',
            **kwargs)
        self.rgb_out = nn.Sequential(nn.Linear(self.dim //2 , 32),
                                        nn.ELU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, 16),
                                        nn.ELU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(16, 3),
                                        nn.Sigmoid())

    def forward(self, prj_dict):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        # viewdirs, feat, occ_masks
        embedded_angles = prj_dict['embedded_angles']
        occ_masks = prj_dict['occ_masks']
        viewdirs = prj_dict['viewdirs']
        geonerf_feats = prj_dict['geonerf_feats']
        seg_logits = prj_dict['seg_logits']
        rays_pts = prj_dict['rays_pts'][0]  # N S 3
        dir_diff = prj_dict['dir_diff']
        rays_pts_pos = positional_encoding(rays_pts, 12)
        N, V, S = geonerf_feats.shape[:3]
        viewdirs = viewdirs[:, None, None].repeat(1, S, V, 1)

        geonerf_feats = geonerf_feats.permute(0, 2, 1, 3)
        embedded_angles = embedded_angles.permute(0, 2, 1, 3)
        dir_diff = dir_diff.permute(0, 2, 1, 3)
        occ_masks = occ_masks.permute(0, 2, 1, 3)
        seg_logits = seg_logits.permute(0, 2, 1, 3)
        m2f_feats = self.seg_logits_prj(seg_logits)
        m2f_feats = m2f_feats.view(-1, *m2f_feats.shape[2:])

        geonerf_feats = geonerf_feats.view(-1, *geonerf_feats.shape[2:])
        v_feat = geonerf_feats[..., :24]  # 3*8 cv
        s_feat = geonerf_feats[..., 24: 24 + 8]  # 8 fpn_lvl_0
        colors = geonerf_feats[..., 24 + 8: -1]
        vis_mask = geonerf_feats[..., -1:].detach()

        occ_masks = occ_masks.view(-1, *occ_masks.shape[2:])
        viewdirs = viewdirs.view(-1, *viewdirs.shape[2:])
        embedded_angles = embedded_angles.view(-1, *embedded_angles.shape[2:])
        dir_diff = dir_diff.reshape((-1, *dir_diff.shape[2:]))

        # s_feat: (N S) V 8 | var_mean (N S) 16
        ## Mean and variance of 2D features provide view-independent tokens
        var_mean = torch.var_mean(s_feat, dim=1, unbiased=False, keepdim=True)
        var_mean = torch.cat(var_mean, dim=-1)
        var_mean = F.elu(self.var_mean_fc1(var_mean))
        var_mean = F.elu(self.var_mean_fc2(var_mean))

        var_mean_m2f = torch.var_mean(m2f_feats, dim=1, unbiased=False, keepdim=True)
        var_mean_m2f = torch.cat(var_mean_m2f, dim=-1)
        var_mean_m2f = F.elu(self.var_mean_fc3(var_mean_m2f))
        var_mean_m2f = F.elu(self.var_mean_fc4(var_mean_m2f))

        ## Converting the input features to tokens (view-dependent) before self-attention
        tokens = F.elu(
            self.attn_token_gen(torch.cat([v_feat, vis_mask, s_feat], dim=-1))
        )
        var_mean = torch.cat([var_mean, var_mean_m2f], -1)
        var_mean = F.elu(self.var_mean_fuse(var_mean))

        tokens = torch.cat([tokens, var_mean], dim=1)

        ## Adding a new channel to mask for var_mean
        vis_mask = torch.cat(
            [vis_mask, self.var_mean_mask.view(1, 1, 1).expand(N * S, -1, -1)], dim=1
        )
        ## If a point is not visible by any source view, force its masks to enabled
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)

        ## Taking occ_masks into account, but remembering if there were any visibility before that
        mask_cloned = vis_mask.clone()
        vis_mask[:, :-1] *= occ_masks
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)
        masks = vis_mask * mask_cloned

        ## Performing self-attention
        for layer in self.attn_layers:
            tokens, _ = layer(tokens, tokens, tokens, masks)
            # tokens, _ = layer(tokens, tokens, tokens, None)

        ## Predicting sigma with an Auto-Encoder and MLP
        sigma_tokens = tokens[:, -1:]
        sigma_tokens = sigma_tokens.view(N, S, self.dim).transpose(1, 2)
        sigma_tokens = self.auto_enc(sigma_tokens)
        sigma_tokens = sigma_tokens.transpose(1, 2).reshape(N * S, 1, self.dim)

        h = F.elu(self.sigma_fc1(sigma_tokens))
        h = F.elu(self.sigma_fc2(h))
        sigma = torch.relu(self.sigma_fc3(h[:, 0]))

        ## Concatenating positional encodings and predicting RGB weights
        direction_feat = self.ray_dir_fc(dir_diff)
        # 32+16+9
        rgb_tokens = torch.cat([tokens[:, :-1], embedded_angles, direction_feat], dim=-1)
        rgb_tokens = F.elu(self.rgb_fc1(rgb_tokens))
        rgb_tokens = F.elu(self.rgb_fc2(rgb_tokens))
        rgb_w = self.rgb_fc3(rgb_tokens)
        rgb_w = masked_softmax(rgb_w, masks[:, :-1], dim=1)
        # rgb_w = rgb_w, masks[:, :-1], dim=1)
        # rgb_b = self.rgb_out(rgb_tokens)

        rgb = (colors * rgb_w).sum(1)
        # rgb = (rgb_b * rgb_w).sum(1)

        # outputs = torch.cat([rgb, sigma], -1)
        # outputs = outputs.reshape(N, S, -1)

        sigma_tokens = sigma_tokens.reshape((N, S, -1))
        # sem_tokens = torch.cat([sigma_tokens,tpv_feat],-1)
        # sem_tokens = F.elu(self.sem_fuse_1(sem_tokens))*rays_pts_pos
        # sem_tokens = self.sem_fuse_2(sem_tokens)
        sem_feat = torch.cat([
            sigma_tokens[:, None].repeat(1, V, 1, 1),
            m2f_feats.reshape((N, S, V, -1)).permute(0, 2, 1, 3)
        ], -1)

        # intra
        N, V, S, C = sem_feat.shape
        sem_feat = sem_feat.reshape((-1, S, C))
        occ_masks = occ_masks.reshape((N, S, V, 1)).permute(0, 2, 1, 3)
        mask = prj_dict['mask'] * occ_masks
        num_valid_obs = mask.sum(1, keepdim=True) > 0.
        trans_mask = num_valid_obs.expand(-1, V, -1, -1).reshape(N * V, S, 1)
        sem_feat = sem_feat + self.pos_encoding
        sem_feat, _ = self.point_attention(
            sem_feat, sem_feat, sem_feat, mask=(mask.reshape(-1, S, 1)) * trans_mask)

        sem_feat = sem_feat.reshape((N, V, S, -1)).permute(0, 2, 1, 3)
        # inter
        sem_feat = sem_feat.reshape(-1, V, C)
        sem_feat, _ = self.sem_rtrans(
            sem_feat, sem_feat, sem_feat, mask=mask.permute(0, 2, 1, 3).reshape(-1, V, 1).float())

        sem_feat = sem_feat.reshape((N, S, V, -1)).permute(0, 2, 1, 3)
        sem_feat = self.sem_fc(sem_feat)
        x = self.sem_w_fc(sem_feat)

        blending_weights_sem = F.softmax(x, dim=2)  # color blending
        # blending_weights_sem2 = self.sem_w_fc2(sem_feat).softmax(2)
        seg_logits = seg_logits.permute(0, 2, 1, 3)
        sem_out = torch.sum(seg_logits * blending_weights_sem, dim=1)
        labels = prj_dict['labels']
        sem_out2 = torch.sum(labels * blending_weights_sem, dim=1)

        rgb = rgb.reshape((N, S, -1))
        sigma = sigma.reshape((N, S, -1))

        out = torch.cat([rgb, sigma, sem_out, sem_out2], dim=-1)
        if has_nan(out):
            print('FFFFFuck')
        return out


def masked_softmax(x, mask, **kwargs):
    x_masked = x.masked_fill(mask == 0, -float("inf"))

    return torch.softmax(x_masked, **kwargs)


class ConvAutoEncoder(nn.Module):
    def __init__(self, num_ch, S):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_ch, num_ch * 2, 3, stride=1, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch * 4, 3, stride=1, padding=1),
            nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_ch * 4, num_ch * 4, 3, stride=1, padding=1),
            nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )

        # Decoder
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch * 4, 4, stride=2, padding=1),
            nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 8, num_ch * 2, 4, stride=2, padding=1),
            nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch, 4, stride=2, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        # Output
        self.conv_out = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch, 3, stride=1, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        conv1_out = x
        x = self.conv2(x)
        conv2_out = x
        x = self.conv3(x)

        x = self.t_conv1(x)
        x = self.t_conv2(torch.cat([x, conv2_out], dim=1))
        x = self.t_conv3(torch.cat([x, conv1_out], dim=1))

        x = self.conv_out(torch.cat([x, input], dim=1))

        return x


name2network = {
    'CRANet': CRANet,
    'CRANet_v1': CRANet_v1,
    'CRANet_v2': CRANet_v2,
    'CRANet_Def': CRANet_Def,
    'CRANet_Def_IBR': CRANet_Def_IBR,
    'CRANet_Def_IBR_v2': CRANet_Def_IBR_v2,
    'CRANet_Def_IBR_v3': CRANet_Def_IBR_v3,
    'CRANet_CV_IBR': CRANet_CV_IBR,
    'CRANet_CV_IBR_v2': CRANet_CV_IBR_v2,
    # 'CRANet_Def_2':CRANet_Def_2,
    'IBRNet': IBRNet,
    'IBRNetWithNeuRay': IBRNetWithNeuRay,

}
