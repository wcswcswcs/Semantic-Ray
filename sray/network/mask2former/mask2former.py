from mmseg.apis import inference_model,init_model,model_forward
import torch
import einops
config_file ='/home/chengshun.wang/pjs/mmsegmentation/m2f/mask2former_r50_scannet_2d_240x320.py'
checkpoint_file = '/home/chengshun.wang/pjs/mmsegmentation/m2f/iter_30000.pth'

def get_m2f(cfg, device='cuda:0'):
    config_file = cfg['config_file'] 
    checkpoint_file = cfg['checkpoint_file']
    model = init_model(config_file, checkpoint_file, device=device)
    return model

def get_m2f_inference_outputs(model,imgs):
    intermediate_features = []
    def forward_hook(module, input, output):
        # 收集指定模块的输出张量
        intermediate_features.append(output[1])

    hook = model.decode_head.pixel_decoder.register_forward_hook(forward_hook) 
    # imgs = imgs.split(imgs.shape[0])
    # imgs = [i.cpu().numpy() for i in imgs]
    results = inference_model(model,imgs)
    hook.remove() 
    seg_logits = []
    pred_sem_seg = []
    for res in results:
        seg_logits.append(res.seg_logits.data)
        pred_sem_seg.append(res.pred_sem_seg.data)
    seg_logits = torch.stack(seg_logits,0)
    pred_sem_seg = torch.stack(pred_sem_seg,0)

    k = model.decode_head.query_embed.weight.requires_grad_(False) + \
        model.decode_head.query_feat.weight.requires_grad_(False)
    v = k
    s = [(15, 20), (30, 40),(60, 80)]
    mlvl_feats = []
    for ind,i in enumerate(intermediate_features[0]):
        c,h,w = i.shape[-3:]
        i = einops.einops.rearrange(i,'b c h w -> b (h w) c ',h=h,w=w)
        att = torch.einsum('b L c,n c -> b L n',i,k)/(c**0.5)
        att = att.softmax(-1)
        y = torch.einsum('b L n, n c -> b L c',att,v)
        y = einops.einops.rearrange(y,'b (h w) c -> b c h w ',h=h,w=w)
        y = torch.nn.functional.interpolate(y,size=s[ind],mode='bilinear')
        mlvl_feats.append(y)
    
    
    return seg_logits, pred_sem_seg, mlvl_feats