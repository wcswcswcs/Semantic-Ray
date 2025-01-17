import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange, reduce, repeat

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q =  torch.nan_to_num(self.layer_norm(q))

        return q, attn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var


class IBRNet(nn.Module):
    def __init__(self, in_feat_ch=32, n_samples=64, **kwargs):
        super(IBRNet, self).__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i)
                                  for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to(
            "cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask):
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

        # compute mean and variance across different views for each point
        # [n_rays, n_samples, 1, n_feat]
        mean, var = fused_mean_variance(rgb_feat, weight)
        # [n_rays, n_samples, 1, 2*n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)

        # [n_rays, n_samples, n_views, 3*n_feat]
        x = torch.cat(
            [globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)
        x = self.base_fc(x)

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
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out


class IBRNetWithNeuRay(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        
        

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)

    def change_pos_encoding(self, n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i)
                                  for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).cuda().float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask,**_):
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
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out


class CRANet(IBRNetWithNeuRay):
    def __init__(self,
                 neuray_in_dim=32,
                 in_feat_ch=32,
                 n_samples=64,
                 num_classes=20,
                 use_ptrans=False,
                 ptrans_first=False,
                 sem_only=False,
                 label_hidden=[],
                 **kwargs):
        super().__init__(
            neuray_in_dim=neuray_in_dim,
            in_feat_ch=in_feat_ch,
            n_samples=n_samples,
            **kwargs
        )
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
        self.semantic_fc1_new = nn.Linear(16, 16)
        self.semantic_fc2_new = nn.Linear(64, num_classes + 1)

        self.relu = nn.ReLU()
        
    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, prj_dict):
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
        if self.use_ptrans and self.ptrans_first:
            # ptrans
            b, n, v, f = sem_latent.shape
            sem_latent = sem_latent.reshape(-1, num_views, f)
            sem_latent, _ = self.point_attention_new(
                sem_latent, sem_latent, sem_latent, mask=mask.reshape(-1, num_views, 1).float())
            sem_latent = sem_latent.reshape(b, n, v, f)
            # rtrans
            sem_latent = sem_latent.permute(0, 2, 1, 3).reshape(-1, n, f)
            # trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            # trans_mask = (trans_mask > 1).float()
            sem_latent = sem_latent + self.sem_pos_encoding
            sem_latent, _ = self.sem_rtrans_new(
                sem_latent, sem_latent, sem_latent, mask=trans_mask)
            sem_latent = sem_latent.reshape(b, v, n, f).permute(0, 2, 1, 3)
        elif self.use_ptrans and not self.ptrans_first:
            # rtrans
            b, n, v, f = sem_latent.shape
            sem_latent = sem_latent.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_latent = sem_latent + self.sem_pos_encoding
            sem_latent, _ = self.sem_rtrans_new(
                sem_latent, sem_latent, sem_latent, mask=trans_mask)
            sem_latent = sem_latent.reshape(b, v, n, f).permute(0, 2, 1, 3)
            # ptrans
            sem_latent = sem_latent.reshape(-1, num_views, f)
            sem_latent, _ = self.point_attention_new(
                sem_latent, sem_latent, sem_latent, mask=mask.reshape(-1, num_views, 1).float())
            sem_latent = sem_latent.reshape(b, n, v, f)
        
        sem_global = self.semantic_fc1_new(globalfeat)
        sigma_feat = self.out_geometry_fc[0](globalfeat)
        sigma_feat = self.out_geometry_fc[1](sigma_feat)
        sem_feat = torch.cat([
            sem_global.unsqueeze(2).expand(-1, -1, num_views, -1),
            sigma_feat.unsqueeze(2).expand(-1, -1, num_views, -1),
            sem_latent
        ], dim=-1)
        sem_feat = self.semantic_fc2_new(sem_feat)
        sem_feat = torch.sum(sem_feat * blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out, sem_feat], dim=-1)
        return out


class CRANet_v1(IBRNetWithNeuRay):
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
        self.semantic_fc1_new = nn.Linear(16, 16)
        self.semantic_fc2_new = nn.Linear(64, num_classes + 1)
        self.relu = nn.ReLU()

        if self.color_cal_type != 'rgb_in':
            self.rgb_out = nn.Sequential(nn.Linear(32+1+4, 32),
                                        nn.ELU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, 16),
                                        nn.ELU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(16, 3),
                                        nn.Sigmoid())

        
        
    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, prj_dict):
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
                sem_latent, sem_latent, sem_latent, mask=trans_mask)
            sem_latent = sem_latent.reshape(b, v, n, f).permute(0, 2, 1, 3)
        elif self.use_ptrans and not self.ptrans_first:
            # rtrans
            b, n, v, f = sem_latent.shape
            sem_latent = sem_latent.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_latent = sem_latent + self.sem_pos_encoding
            sem_latent, _ = self.sem_rtrans_new(
                sem_latent, sem_latent, sem_latent, mask=trans_mask)
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

        # semantic feature
        
        sem_global = self.semantic_fc1_new(globalfeat)
        sigma_feat = self.out_geometry_fc[0](globalfeat)
        sigma_feat = self.out_geometry_fc[1](sigma_feat)
        sem_feat = torch.cat([
            sem_global.unsqueeze(2).expand(-1, -1, num_views, -1),
            sigma_feat.unsqueeze(2).expand(-1, -1, num_views, -1),
            sem_latent
        ], dim=-1)
        sem_feat = self.semantic_fc2_new(sem_feat)

        sem_feat = torch.sum(sem_feat * blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out, sem_feat], dim=-1)
        return out

class CRANet_v2(IBRNetWithNeuRay):
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
        self.semantic_fc1_new = nn.Linear(16, 16)
        self.semantic_fc2_new = nn.Linear(64, num_classes + 1)
        self.relu = nn.ReLU()

        #####
        self.pos_encoding_2 = self.posenc(d_hid=64, n_samples=self.n_samples)
        self.ray_attention_2 = MultiHeadAttention(8, 64, 8, 8)

        if self.color_cal_type != 'rgb_in':
            self.rgb_out = nn.Sequential(nn.Linear(32+1+4, 32),
                                        nn.ELU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, 16),
                                        nn.ELU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(16, 3),
                                        nn.Sigmoid())

        
        
    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, prj_dict):
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
                sem_latent, sem_latent, sem_latent, mask=trans_mask)
            sem_latent = sem_latent.reshape(b, v, n, f).permute(0, 2, 1, 3)
        elif self.use_ptrans and not self.ptrans_first:
            # rtrans
            b, n, v, f = sem_latent.shape
            sem_latent = sem_latent.permute(0, 2, 1, 3).reshape(-1, n, f)
            trans_mask = num_valid_obs.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(b * v, n, 1)
            trans_mask = (trans_mask > 1).float()
            sem_latent = sem_latent + self.sem_pos_encoding
            sem_latent, _ = self.sem_rtrans_new(
                sem_latent, sem_latent, sem_latent, mask=trans_mask)
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
        
        sem_global = self.semantic_fc1_new(globalfeat)
        sigma_feat = self.out_geometry_fc[0](globalfeat)
        sigma_feat = self.out_geometry_fc[1](sigma_feat)
        sem_feat = torch.cat([
            sem_global.unsqueeze(2).expand(-1, -1, num_views, -1),
            sigma_feat.unsqueeze(2).expand(-1, -1, num_views, -1),
            sem_latent
        ], dim=-1)
        
        sem_feat = torch.sum(sem_feat * blending_weights_valid, dim=2)
        sem_feat = sem_feat + self.pos_encoding_2
        sem_feat, _ = self.ray_attention_2(sem_feat, sem_feat, sem_feat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sem_feat = self.semantic_fc2_new(sem_feat)

        
        out = torch.cat([rgb_out, sigma_out, sem_feat], dim=-1)
        return out


name2network = {
    'CRANet':CRANet,
    'CRANet_v1':CRANet_v1,
    'CRANet_v2':CRANet_v2,
    'IBRNet':IBRNet,
    'IBRNetWithNeuRay':IBRNetWithNeuRay,

}