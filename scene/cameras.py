#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image #懒加载机制

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image,gt_alpha_mask, invdepthmap,
                 image_name, uid,
                 image_path=None, #懒加载机制
                 white_background=None, #懒加载机制
                 cache_images=False, #图像缓存机制
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.resolution = resolution #懒加载机制
        self.image_path = image_path #懒加载机制
        self._lazy_image = image #懒加载机制：兼容旧调用，如果外部仍传入PIL图像则可直接使用
        self._gt_alpha_mask = gt_alpha_mask #懒加载机制
        self._train_test_exp = train_test_exp #懒加载机制
        self._is_test_dataset = is_test_dataset #懒加载机制
        self._is_test_view = is_test_view #懒加载机制
        self._white_background = white_background #懒加载机制
        self.cache_images = cache_images #图像缓存机制
        self._cached_image_tensors = None #图像缓存机制

        self.image_width = resolution[0] #懒加载机制：宽高直接来自目标分辨率，不触发图像加载
        self.image_height = resolution[1] #懒加载机制：宽高直接来自目标分辨率，不触发图像加载

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones((1, resolution[1], resolution[0]), device=self.data_device) #懒加载机制：深度mask按分辨率创建，不触发RGB加载
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def _load_image_tensors(self): #懒加载机制：按需临时加载当前图像，不在Camera对象中长期缓存GPU tensor
        if self.cache_images and self._cached_image_tensors is not None: #图像缓存机制
            return self._cached_image_tensors #图像缓存机制

        opened_image = None #懒加载机制
        if self._lazy_image is not None: #懒加载机制
            image = self._lazy_image #懒加载机制
        else: #懒加载机制
            opened_image = Image.open(self.image_path) #懒加载机制：第一次真正需要图像tensor时才打开RGB图像
            image = opened_image #懒加载机制

        if self._white_background is not None: #懒加载机制：Blender/Transforms数据集保持原来的RGBA背景合成逻辑
            im_data = np.array(image.convert("RGBA")) #懒加载机制
            bg = np.array([1,1,1]) if self._white_background else np.array([0, 0, 0]) #懒加载机制
            norm_data = im_data / 255.0 #懒加载机制
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4]) #懒加载机制
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB") #懒加载机制

        resized_image_rgb = PILtoTorch(image, self.resolution) #懒加载机制
        if opened_image is not None: #懒加载机制
            opened_image.close() #懒加载机制：像素已经转成tensor后关闭文件句柄
        gt_image = resized_image_rgb[:3, ...] #懒加载机制

        if resized_image_rgb.shape[0] == 4: #懒加载机制
            alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device) #懒加载机制
        else: #懒加载机制
            alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device)) #懒加载机制

        if self._train_test_exp and self._is_test_view: #懒加载机制
            if self._is_test_dataset: #懒加载机制
                alpha_mask[..., :alpha_mask.shape[-1] // 2] = 0 #懒加载机制
            else: #懒加载机制
                alpha_mask[..., alpha_mask.shape[-1] // 2:] = 0 #懒加载机制

        original_image = gt_image.clamp(0.0, 1.0).to(self.data_device) #懒加载机制：返回本次使用的GPU tensor，不保存到Camera中
        valid_mask = self._gt_alpha_mask[0:1,:,:].to(self.data_device) \
            if self._gt_alpha_mask is not None \
                else torch.ones_like(alpha_mask).to(self.data_device) #懒加载机制
        self._lazy_image = None #懒加载机制：释放PIL引用，不保留GPU图像缓存
        image_tensors = (original_image, alpha_mask, valid_mask) #图像缓存机制
        if self.cache_images: #图像缓存机制
            self._cached_image_tensors = image_tensors #图像缓存机制
        return image_tensors #懒加载机制

    def load_image_tensors(self): #懒加载机制：一次性返回同一轮训练需要的图像和mask，避免重复读图
        return self._load_image_tensors() #懒加载机制

    @property #懒加载机制
    def original_image(self): #懒加载机制
        original_image, _, _ = self._load_image_tensors() #懒加载机制
        return original_image #懒加载机制

    @property #懒加载机制
    def alpha_mask(self): #懒加载机制
        _, alpha_mask, _ = self._load_image_tensors() #懒加载机制
        return alpha_mask #懒加载机制

    @property #懒加载机制
    def valid_mask(self): #懒加载机制
        _, _, valid_mask = self._load_image_tensors() #懒加载机制
        return valid_mask #懒加载机制


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
