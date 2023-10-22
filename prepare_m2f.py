from sray.network.mask2former.mask2former import *
import torch
import numpy as np
import os
import glob
import cv2




cfg = {
    'config_file' : '/home/chengshun.wang/pjs/mmsegmentation/m2f_pt/mask2former_r50_scannet_2d_240x320_pretrain.py',
'checkpoint_file' : '/home/chengshun.wang/pjs/mmsegmentation/m2f_pt/best_mIoU_iter_85000.pth',
}
model = get_m2f(cfg)
def get_img_id_from_img_path(img_path):
    return img_path.split('/')[-1].split('.')[0]

def get_scene_fold_from_img_path(img_path):
    return os.path.join(*(img_path.split('/')[:-2]))


def get_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(downsample_gaussian_blur(
        img, 320/1296), (320, 240), interpolation=cv2.INTER_LINEAR)
    return img
def downsample_gaussian_blur(img, ratio):
    sigma = (1 / ratio) / 3
    # ksize=np.ceil(2*sigma)
    ksize = int(np.ceil(((sigma - 0.8) / 0.3 + 1) * 2 + 1))
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT101)
    return img

# for i in img_files:
#     scene_dir = get_scene_fold_from_img_path(i)
#     img_id = get_img_id_from_img_path(i)
#     os.makedirs(os.path.join(scene_dir,'m2f'),exist_ok=True)
#     seg_logits, pred_sem_seg, mlvl_feats = get_m2f_inference_outputs(model,[get_img(i)])
#     seg_logits = seg_logits[0].cpu()
#     pred_sem_seg  =pred_sem_seg[0].cpu()
#     mlvl_feats = torch.stack(mlvl_feats,1)[0].cpu()
#     # print(seg_logits.shape)
#     # print(pred_sem_seg.shape)
#     # print(mlvl_feats.shape)
#     ret=dict(seg_logits=seg_logits, pred_sem_seg = pred_sem_seg, mlvl_feats = mlvl_feats)
#     torch.save(ret,os.path.join(scene_dir,'m2f',f'm2f_{img_id}.pt'))


def f(img_files):
    for i in img_files:
        scene_dir = get_scene_fold_from_img_path(i)
        img_id = get_img_id_from_img_path(i)
        os.makedirs(os.path.join(scene_dir,'m2f'),exist_ok=True)
        seg_logits, pred_sem_seg, mlvl_feats = get_m2f_inference_outputs(model,[get_img(i)])
        seg_logits = seg_logits[0].cpu()
        pred_sem_seg  =pred_sem_seg[0].cpu()
        mlvl_feats = torch.stack(mlvl_feats,1)[0].cpu()
        # print(seg_logits.shape)
        # print(pred_sem_seg.shape)
        # print(mlvl_feats.shape)
        ret=dict(seg_logits=seg_logits, pred_sem_seg = pred_sem_seg, mlvl_feats = mlvl_feats)
        torch.save(ret,os.path.join(scene_dir,'m2f',f'm2f_{img_id}.pt'))




if __name__ == '__main__':
    
    import multiprocessing 
    multiprocessing.set_start_method('spawn')
    # from multiprocessing import Pool
    # from multiprocessing import get_context
    scannet_dir = 'data/scannet'
    img_files = glob.glob(os.path.join(scannet_dir,"*","color","*.jpg"))
    num_processes = 5
    chunk_size = len(img_files) // num_processes
    processes = []
    # with multiprocessing.get_start_method() as start_method:
    data_chunks = [img_files[i:i + chunk_size] for i in range(0, len(img_files), chunk_size)]
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else len(img_files)
        part = img_files[start:end]
        p = multiprocessing.Process(target=f, args=([part]))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    # with multiprocessing.get_start_method() as start_method:
    #     with multiprocessing.get_context(start_method).Pool(processes=5) as pool:
    #         pool.map(f, data_chunks)
    
    # with multiprocessing.get_start_method() as start_method:
    #     pool = multiprocessing.get_start_method("spawn")
    #     manager = multiprocessing.Manager()
    #     result = manager.list([None] * num_processes)

    #     jobs = []
        

    #     for i in range(num_processes):
    #         start = i * chunk_size
    #         end = (i + 1) * chunk_size if i < num_processes - 1 else len(img_files)
    #         part = img_files[start:end]
    #         job = pool.Process(target=f, args=(part))
    #         jobs.append(job)
    #         job.start()

    #     for job in jobs:
    #         job.join()

    # with get_context("spawn").Pool() as pool:
    #     slice_len = len(img_files)//10
    #     data=[]
    #     for i in range(5):
    #         data.append(img_files[i*slice_len: (i+1)*slice_len]) 
    #     pool.map(f,data)



