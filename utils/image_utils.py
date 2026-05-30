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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def masked_psnr(img1, img2, mask=None):
    if mask is None:
        return psnr(img1, img2)

    mask = mask.to(device=img1.device, dtype=img1.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1:
        mask = mask.expand_as(img1)

    squared_error = ((img1 - img2) ** 2) * mask
    mse = squared_error.reshape(img1.shape[0], -1).sum(1, keepdim=True) / mask.reshape(img1.shape[0], -1).sum(1, keepdim=True).clamp_min(1.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1e-12)))
