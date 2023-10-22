#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sray.utils.imgs_info import build_imgs_info, imgs_info_to_torch
from sray.dataset.database import ScannetDatabase
from sray.utils.imgs_info import build_imgs_info
import numpy as np
from mmengine.config import Config
from mmseg.models import build_segmentor
from sray.network.tpvformer10 import *
from sray.network.tpvformer10.tpv_head import TPVFormerHead,CustomPositionalEncoding
from mmcv.cnn.bricks.registry import POSITIONAL_ENCODING


# In[2]:


dataset_name = 'scannet/scene0188_00/black_320'
dataset = ScannetDatabase(dataset_name)


# In[3]:


ref_img_info = build_imgs_info(dataset,[10,20,30,40,50,60,70,80])


# In[4]:


list(ref_img_info.keys())


# In[5]:


ref_img_info['poses'].shape


# In[6]:


def build_img_metas(ref_img_info,img_H = 280,img_W = 320):

    img_metas=[]
    d = {
        'img_shape' : [[img_H, img_W]],
    }
    img_metas = []
    for pose,k in zip(ref_img_info['poses'],ref_img_info['Ks']):
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :4] = pose[:3,:4]
        intrinsic = np.eye(4)
        intrinsic[:k.shape[0], :k.shape[1]] = k
        lidar2img = intrinsic  @ lidar2cam_rt
        ret = d.copy()
        ret['lidar2img'] = lidar2img
        img_metas.append(ret)
    return img_metas


# In[7]:


img_metas = build_img_metas(ref_img_info)
ref_img_info['img_metas']  =img_metas


# In[8]:


ref_img_info = imgs_info_to_torch(ref_img_info)


# In[9]:


def model_builder(model_config):
    model = build_segmentor(model_config)
    model.init_weights()
    return model


# In[10]:


cfg = Config.fromfile('sray/network/tpvformer10/tpv_config.py')


# In[11]:


cfg.model


# In[12]:


my_model = model_builder(cfg.model).cuda()


# In[17]:


res = my_model(img_metas=ref_img_info['img_metas'],img=ref_img_info['imgs'][None])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




