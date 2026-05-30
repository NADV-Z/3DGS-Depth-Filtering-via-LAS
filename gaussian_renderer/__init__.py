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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.depth_utils import load_processed_depth_info #深度图懒加载机制


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False,
            depth_dir: str = None, depth_tolerance: float = 1.0, depth_cache_size: int = 256, suppress_grad_warning: bool = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    depth_min_gpu = torch.empty(0, device="cuda", dtype=torch.float32) #CUDA动态深度筛选机制：Python只读取当前视角深度图，不再做高斯投影筛选
    depth_max_gpu = torch.empty(0, device="cuda", dtype=torch.float32) #CUDA动态深度筛选机制
    depth_filter_enabled = False #实验验证机制
    depth_status = "disabled" if depth_dir is None else "missing" #深度图懒加载机制
    valid_depth_ratio = 0.0 #深度图懒加载机制
    depth_source = "none" #深度图直接读取机制
    if depth_dir is not None: #CUDA动态深度筛选机制
        processed_depth = load_processed_depth_info(depth_dir, viewpoint_camera.image_name, cache_size=depth_cache_size) #深度图懒加载机制 #深度图直接读取机制
        if processed_depth is not None: #CUDA动态深度筛选机制 #深度图懒加载机制
            depth_min_gpu = processed_depth["min_depth"].to(device="cuda", non_blocking=True).contiguous() #深度图懒加载机制
            depth_max_gpu = processed_depth["max_depth"].to(device="cuda", non_blocking=True).contiguous() #深度图懒加载机制
            depth_filter_enabled = True #实验验证机制
            depth_status = "enabled" #深度图懒加载机制
            valid_depth_ratio = processed_depth["valid_depth_ratio"] #深度图懒加载机制
            depth_source = processed_depth.get("depth_source", "depth_file") #深度图直接读取机制

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        depth_min=depth_min_gpu, #CUDA动态深度筛选机制
        depth_max=depth_max_gpu, #CUDA动态深度筛选机制
        depth_tolerance=depth_tolerance, #CUDA动态深度筛选机制
        depth_missing_allow=True, #CUDA动态深度筛选机制：无深度像素仅保留视锥体裁剪
        debug=pipe.debug,
       # antialiasing=pipe.antialiasing
    )



    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    original_means3D = means3D #动态深度筛选机制：保留全量高斯引用，用于空图像梯度连接和索引映射
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # 默认情况下，所有高斯交给CUDA rasterizer，由CUDA内部完成视锥体裁剪和深度筛选
    visibility_mask = torch.ones(means3D.shape[0], dtype=torch.bool, device="cuda") #CUDA动态深度筛选机制：保持原始输入长度，CUDA preprocess把不合格高斯radii置0
    
    if not visibility_mask.any():
        H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
        empty_image = bg_color.view(3, 1, 1).expand(3, H, W).clone()
        gradient_connections = []
        
        # 连接到位置参数
        pos_connection = pc.get_xyz.mean(dim=0).sum() * 1e-12
        gradient_connections.append(pos_connection)
        
        # 连接到特征参数  
        feat_connection = pc.get_features.mean() * 1e-12
        gradient_connections.append(feat_connection)
        
        # 连接到不透明度参数
        opacity_connection = pc.get_opacity.mean() * 1e-12
        gradient_connections.append(opacity_connection)
        
        # 连接到缩放参数
        scale_connection = pc.get_scaling.mean() * 1e-12
        gradient_connections.append(scale_connection)
        
        # 连接到旋转参数
        rot_connection = pc.get_rotation.mean() * 1e-12
        gradient_connections.append(rot_connection)
        
        # 将所有连接合并
        total_connection = sum(gradient_connections)
        
        # 将梯度连接添加到图像中（数值上几乎为0，但保持梯度）
        empty_image = empty_image + total_connection.view(1, 1, 1) * torch.ones_like(empty_image) * 0.0

        return {
            "render": empty_image, 
            "viewspace_points": screenspace_points,
            "visibility_filter": torch.zeros(pc.get_xyz.shape[0], dtype=torch.bool, device="cuda"), #动态深度筛选机制
            "radii": torch.zeros(pc.get_xyz.shape[0], device="cuda"), #动态深度筛选机制
            "selected_count": torch.zeros((), dtype=torch.int64, device="cuda"), #实验验证机制
            "total_gaussians": torch.tensor(pc.get_xyz.shape[0], device="cuda"), #实验验证机制
            "depth_filter_enabled": depth_filter_enabled, #实验验证机制
            "depth_status": depth_status, #深度图懒加载机制
            "valid_depth_ratio": valid_depth_ratio, #深度图懒加载机制
            "depth_source": depth_source, #深度图直接读取机制
            "depth": torch.zeros(H, W, device="cuda")
        }

    # CUDA动态深度筛选机制：所有高斯保持原始长度传入rasterizer，CUDA preprocess阶段将未通过深度筛选的高斯radii/tiles_touched保持为0
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rasterizer_output= rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
            
    else:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        if len(rasterizer_output) == 2:
            rendered_image, radii = rasterizer_output
        elif len(rasterizer_output) == 3:
             rendered_image, radii, depth_image = rasterizer_output
        else:
            raise ValueError(f"意外的rasterizer输出数量: {len(rasterizer_output)}")

    if not rendered_image.requires_grad:
        if not suppress_grad_warning:
            print("警告：渲染输出缺少梯度，添加连接")
        excluded_connection = original_means3D[~visibility_mask].mean() * 1e-12 if (~visibility_mask).any() else torch.tensor(0.0, device="cuda") #动态深度筛选机制
        rendered_image = rendered_image + excluded_connection.view(1, 1, 1) * torch.zeros_like(rendered_image)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    full_radii = radii #CUDA动态深度筛选机制

    full_visibility_filter = radii > 0 #CUDA动态深度筛选机制：视锥体裁剪和深度筛选都由CUDA通过radii反映
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : full_visibility_filter, #动态深度筛选机制
        "radii": full_radii,
        "selected_count": full_visibility_filter.sum(), #实验验证机制
        "total_gaussians": torch.tensor(full_visibility_filter.numel(), device="cuda"), #实验验证机制
        "depth_filter_enabled": depth_filter_enabled, #实验验证机制
        "depth_status": depth_status, #深度图懒加载机制
        "valid_depth_ratio": valid_depth_ratio, #深度图懒加载机制
        "depth_source": depth_source, #深度图直接读取机制
        }
    
    return out
