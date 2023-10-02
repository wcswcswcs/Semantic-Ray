# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

from sray.network.dat.dat import DAT

def build_model(config):

    model_type = config.MODEL.TYPE
    if model_type == 'dat':
        model = DAT(**config.MODEL.DAT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

if __name__=='__main__':
    from sray.utils.base_utils import load_cfg
    import torch
    cfg = load_cfg('configs/cra/test_dat.yaml')['DAT']
    print(cfg)
    model = DAT(**cfg).cuda()
    data = torch.rand((8,32,60,80)).cuda()
    y,outs, = model(data)
    print(f"y's shape: {y.shape}")
    print(f"outs's shape: {outs.shape}")


