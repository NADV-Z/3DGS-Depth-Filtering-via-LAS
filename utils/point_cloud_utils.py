# file: utils/point_cloud_utils.py

import numpy as np
import torch
import laspy
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import os
import hashlib #LAS坐标对齐机制
import xml.etree.ElementTree as ET #LAS坐标对齐机制

def _load_srs_origin(metadata_path: str): #LAS坐标对齐机制
    if metadata_path is None: #LAS坐标对齐机制
        raise ValueError("使用 LAS 初始化时必须提供 metadata_path，以便和 XML2Colmap/depth_map_generate 使用同一个 SRSOrigin。") #LAS坐标对齐机制

    tree = ET.parse(metadata_path) #LAS坐标对齐机制
    root = tree.getroot() #LAS坐标对齐机制
    origin_node = root.find(".//SRSOrigin") #LAS坐标对齐机制
    if origin_node is None or not origin_node.text: #LAS坐标对齐机制
        raise ValueError(f"metadata.xml 中没有找到 SRSOrigin: {metadata_path}") #LAS坐标对齐机制

    values = [float(v.strip()) for v in origin_node.text.split(",")] #LAS坐标对齐机制
    if len(values) != 3: #LAS坐标对齐机制
        raise ValueError(f"SRSOrigin 必须包含3个逗号分隔的数值，当前为: {origin_node.text}") #LAS坐标对齐机制
    return np.asarray(values, dtype=np.float32) #LAS坐标对齐机制

def read_las_for_gaussian_initialization(las_path: str, voxel_size: float = 0.05, metadata_path: str = None, las_scale_multiplier: float = 2.0): #LAS坐标对齐机制 #实验验证机制
    """
    专门为初始化高斯球体而读取LAS文件。
    此版本集成了缓存机制，避免重复进行耗时的预处理。
    
    返回:
        - xyz (np.array): Nx3 的点坐标 (已按 metadata.xml 的 SRSOrigin 对齐并下采样)
        - colors (np.array): Nx3 的点颜色, 范围在 0-1 之间
        - normals (np.array): Nx3 的法线
        - scales (np.array): Nx3 的初始缩放大小
    """
    base_name, _ = os.path.splitext(os.path.basename(las_path))
    cache_dir = os.path.dirname(las_path)
    srs_origin = _load_srs_origin(metadata_path) #LAS坐标对齐机制
    origin_key = hashlib.md5(srs_origin.tobytes()).hexdigest()[:8] #LAS坐标对齐机制
    cache_filename = f"{base_name}_downsampled_vs{voxel_size}_origin{origin_key}_scale{las_scale_multiplier:.3f}.npz" #LAS坐标对齐机制 #实验验证机制
    cache_path = os.path.join(cache_dir, cache_filename)
    print(f"[LAS坐标对齐机制] metadata_path: {metadata_path}") #LAS坐标对齐机制
    print(f"[LAS坐标对齐机制] SRSOrigin: {srs_origin}") #LAS坐标对齐机制
    print(f"[实验验证机制] LAS scale multiplier: {las_scale_multiplier}") #实验验证机制

    if os.path.exists(cache_path):
        print(f"✅ 找到预处理缓存文件，正在直接加载: {cache_path}")
        try:
            # 从 .npz 文件加载所有数组
            data = np.load(cache_path)
            xyz = data['xyz']
            colors = data['colors']
            normals = data['normals']
            scales = data['scales']
            print(f"✅ 成功从缓存加载 {len(xyz)} 个点。")
            return xyz, colors, normals, scales
        except Exception as e:
            print(f"⚠️ 加载缓存文件失败: {e}。将重新进行预处理。")

    print(f"🔎 未找到缓存文件，开始执行完整的预处理流程...")
    
    print(f"正在读取 LAS 文件: {las_path}")
    with laspy.open(las_path) as f:
        las = f.read()

    original_xyz_raw = np.vstack((las.x, las.y, las.z)).transpose()
    print(f"原始点云数量: {len(original_xyz_raw)}")

    print("[LAS坐标对齐机制] 使用 LAS_raw - SRSOrigin 对齐到 XML2Colmap 局部坐标系")
    original_xyz = original_xyz_raw.astype(np.float32) - srs_origin[None, :] #LAS坐标对齐机制
    print(f"[LAS坐标对齐机制] LAS raw min/max: {original_xyz_raw.min(axis=0)} / {original_xyz_raw.max(axis=0)}") #LAS坐标对齐机制
    print(f"[LAS坐标对齐机制] LAS local min/max: {original_xyz.min(axis=0)} / {original_xyz.max(axis=0)}") #LAS坐标对齐机制
    
    has_normals = hasattr(las, 'nx') and hasattr(las, 'ny') and hasattr(las, 'nz')
    original_normals = None
    if has_normals:
        original_normals_raw = np.vstack((las.nx, las.ny, las.nz)).transpose()
        original_normals = original_normals_raw #LAS坐标对齐机制：不再换轴，保持和 LAS_raw - SRSOrigin 的坐标轴一致
    
    has_colors = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
    original_colors = None
    if has_colors:
        original_colors = np.vstack((las.red / 65535.0, las.green / 65535.0, las.blue / 65535.0)).transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_xyz)
    if has_colors:
        pcd.colors = o3d.utility.Vector3dVector(original_colors)
    if has_normals:
        pcd.normals = o3d.utility.Vector3dVector(original_normals)

    print(f"正在使用 {voxel_size} 米的体素大小进行下采样...")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz = np.asarray(downsampled_pcd.points)
    print(f"下采样后点云数量: {len(xyz)}")

    if has_colors and len(downsampled_pcd.colors) == len(xyz):
        colors_normalized = np.asarray(downsampled_pcd.colors)
    else:
        print("正在为下采样点重新匹配颜色...")
        nn_color_matcher = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(original_xyz)
        _, indices = nn_color_matcher.kneighbors(xyz)
        colors_normalized = original_colors[indices.flatten()] if has_colors else np.full_like(xyz, 0.5)

    if has_normals and len(downsampled_pcd.normals) == len(xyz):
        normals = np.asarray(downsampled_pcd.normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(norms == 0, 1.0, norms)
    else:
        print("正在为下采样点重新匹配法线...")
        if has_normals:
            nn_normal_matcher = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(original_xyz)
            _, indices = nn_normal_matcher.kneighbors(xyz)
            normals = original_normals[indices.flatten()]
        else:
            normals = np.zeros_like(xyz)

    print("正在为下采样后的点云估算【自适应】初始缩放...")
    num_neighbors = 3
    nn = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='kd_tree').fit(xyz)
    distances, _ = nn.kneighbors(xyz)
    avg_dist_per_point = distances[:, 1:].mean(axis=1)
    scales = np.tile(avg_dist_per_point, (3, 1)).T * las_scale_multiplier #实验验证机制
    print(f"  - 缩放值范围: min={scales.min():.4f}, max={scales.max():.4f}, mean={scales.mean():.4f}")

    print(f"💾 预处理完成，正在将结果保存到缓存文件: {cache_path}")
    try:
        np.savez_compressed(
            cache_path,
            xyz=xyz,
            colors=colors_normalized,
            normals=normals,
            scales=scales
        )
        print("💾 缓存保存成功。")
    except Exception as e:
        print(f"⚠️ 保存缓存文件失败: {e}")

    return xyz, colors_normalized, normals, scales
