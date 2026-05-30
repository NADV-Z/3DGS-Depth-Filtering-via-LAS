#  utils/depth_utils.py

from collections import OrderedDict #深度图懒加载机制
import numpy as np
import os
import torch #深度图懒加载机制

_PROCESSED_DEPTH_CACHE = OrderedDict() #深度图懒加载机制

def load_depth_info(depth_dir, image_name):
    """
    为给定的图像加载预先计算好的最小和最大深度图。
    """
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    min_path = os.path.join(depth_dir, f"{base_name}_min.npy")
    max_path = os.path.join(depth_dir, f"{base_name}_max.npy")

    if not os.path.exists(min_path) or not os.path.exists(max_path):
        # 如果找不到深度图，可以返回None，或者返回一个“全通”的掩码
        return None, None 

    try:
        min_depth = np.load(min_path)
        max_depth = np.load(max_path)
        return min_depth, max_depth
    except Exception as e:
        print(f"Warning: Error loading depth maps for {image_name}: {e}")
        return None, None


def load_processed_depth_info(depth_dir, image_name, cache_size=256): #深度图懒加载机制 #深度图直接读取机制
    """
    深度图直接读取机制：
    按需读取当前--depth_dir路径中的min/max深度图，并缓存CPU tensor。
    """
    base_name = os.path.splitext(os.path.basename(image_name))[0] #深度图懒加载机制
    cache_key = (os.path.abspath(depth_dir), base_name) #深度图直接读取机制
    if cache_size > 0 and cache_key in _PROCESSED_DEPTH_CACHE: #深度图懒加载机制
        _PROCESSED_DEPTH_CACHE.move_to_end(cache_key) #深度图懒加载机制
        return _PROCESSED_DEPTH_CACHE[cache_key] #深度图懒加载机制

    depth_source = "depth_file" #深度图直接读取机制
    min_depth_np, max_depth_np = load_depth_info(depth_dir, image_name) #深度图直接读取机制
    if min_depth_np is None or max_depth_np is None: #深度图懒加载机制
        print(f"[深度图懒加载机制] missing depth map for image={base_name}, depth_dir={depth_dir}") #深度图懒加载机制
        return None #深度图懒加载机制

    try: #深度图懒加载机制
        min_depth = torch.from_numpy(np.asarray(min_depth_np)).float().cpu().contiguous() #深度图直接读取机制
        max_depth = torch.from_numpy(np.asarray(max_depth_np)).float().cpu().contiguous() #深度图直接读取机制
        valid_depth_ratio = (max_depth > 0).float().mean().item() #深度图直接读取机制

        processed = { #深度图懒加载机制
            "min_depth": min_depth, #深度图直接读取机制
            "max_depth": max_depth, #深度图直接读取机制
            "valid_depth_ratio": valid_depth_ratio, #深度图懒加载机制
            "depth_source": depth_source, #深度图直接读取机制
        } #深度图懒加载机制
    except Exception as e: #深度图懒加载机制
        print(f"[深度图懒加载机制] error processing depth map for image={base_name}: {e}") #深度图懒加载机制
        return None #深度图懒加载机制

    if cache_size > 0: #深度图懒加载机制
        _PROCESSED_DEPTH_CACHE[cache_key] = processed #深度图懒加载机制
        _PROCESSED_DEPTH_CACHE.move_to_end(cache_key) #深度图懒加载机制
        while len(_PROCESSED_DEPTH_CACHE) > cache_size: #深度图懒加载机制
            _PROCESSED_DEPTH_CACHE.popitem(last=False) #深度图懒加载机制

    return processed #深度图懒加载机制
