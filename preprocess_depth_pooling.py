import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def iter_depth_pairs(depth_dir): #深度图pooling机制
    for name in sorted(os.listdir(depth_dir)): #深度图pooling机制
        if not name.endswith("_min.npy") or "_pooled_k" in name: #深度图pooling机制
            continue #深度图pooling机制
        base = name[:-len("_min.npy")] #深度图pooling机制
        min_path = os.path.join(depth_dir, f"{base}_min.npy") #深度图pooling机制
        max_path = os.path.join(depth_dir, f"{base}_max.npy") #深度图pooling机制
        yield base, min_path, max_path #深度图pooling机制


def pool_depth_arrays_cuda(min_depth_np, max_depth_np, kernel_size=3, device=None): #深度图pooling机制
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu") #深度图pooling机制
    with torch.no_grad(): #深度图pooling机制
        min_depth = torch.from_numpy(np.asarray(min_depth_np)).float().to(device) #深度图pooling机制
        max_depth = torch.from_numpy(np.asarray(max_depth_np)).float().to(device) #深度图pooling机制
        valid_depth_map = max_depth > 0 #深度图pooling机制
        padding = kernel_size // 2 #深度图pooling机制
        large_depth = torch.full_like(min_depth, 1e10) #深度图pooling机制
        min_depth_for_pool = torch.where(valid_depth_map, min_depth, large_depth) #深度图pooling机制
        pooled_min = -F.max_pool2d((-min_depth_for_pool)[None, None], kernel_size=kernel_size, stride=1, padding=padding)[0, 0] #深度图pooling机制
        pooled_max = F.max_pool2d(max_depth[None, None], kernel_size=kernel_size, stride=1, padding=padding)[0, 0] #深度图pooling机制
        pooled_valid = F.max_pool2d(valid_depth_map.float()[None, None], kernel_size=kernel_size, stride=1, padding=padding)[0, 0] > 0 #深度图pooling机制
        pooled_min = torch.where(pooled_valid, pooled_min, torch.zeros_like(pooled_min)) #深度图pooling机制
    return pooled_min.cpu().numpy().astype(np.float32, copy=False), pooled_max.cpu().numpy().astype(np.float32, copy=False) #深度图pooling机制


def default_output_dir(depth_dir, kernel_size): #深度图pooling机制
    parent = os.path.dirname(os.path.abspath(depth_dir)) #深度图pooling机制
    return os.path.join(parent, f"depth_maps_pooling{kernel_size}x{kernel_size}") #深度图pooling机制


def process_depth_dir(depth_dir, output_dir=None, kernel_size=3, overwrite=False, limit=0): #深度图pooling机制
    success_count = 0 #深度图pooling机制
    skipped_count = 0 #深度图pooling机制
    missing_count = 0 #深度图pooling机制
    failed_count = 0 #深度图pooling机制

    output_dir = output_dir or default_output_dir(depth_dir, kernel_size) #深度图pooling机制
    os.makedirs(output_dir, exist_ok=True) #深度图pooling机制
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #深度图pooling机制
    depth_pairs = list(iter_depth_pairs(depth_dir)) #深度图pooling机制
    if limit and limit > 0: #深度图pooling机制
        depth_pairs = depth_pairs[:limit] #深度图pooling机制
    print(f"[深度图pooling机制] depth_dir: {depth_dir}") #深度图pooling机制
    print(f"[深度图pooling机制] output_dir: {output_dir}") #深度图pooling机制
    print(f"[深度图pooling机制] device: {device}") #深度图pooling机制
    print(f"[深度图pooling机制] kernel_size: {kernel_size}") #深度图pooling机制
    print(f"[深度图pooling机制] found min/max candidates: {len(depth_pairs)}") #深度图pooling机制

    for base, min_path, max_path in tqdm(depth_pairs, desc="depth pooling", unit="map"): #深度图pooling机制
        # out_min = os.path.join(depth_dir, f"{base}_pooled_k{kernel_size}_min.npy") #深度图pooling机制删除：输出文件名保持普通_min/_max
        # out_max = os.path.join(depth_dir, f"{base}_pooled_k{kernel_size}_max.npy") #深度图pooling机制删除
        out_min = os.path.join(output_dir, f"{base}_min.npy") #深度图pooling机制
        out_max = os.path.join(output_dir, f"{base}_max.npy") #深度图pooling机制

        if not os.path.exists(max_path): #深度图pooling机制
            missing_count += 1 #深度图pooling机制
            print(f"[深度图pooling机制] missing max pair: {max_path}") #深度图pooling机制
            continue #深度图pooling机制

        if not overwrite and os.path.exists(out_min) and os.path.exists(out_max): #深度图pooling机制
            skipped_count += 1 #深度图pooling机制
            continue #深度图pooling机制

        try: #深度图pooling机制
            tqdm.write(f"[深度图pooling机制] processing: {base}") #深度图pooling机制
            min_depth = np.load(min_path) #深度图pooling机制
            max_depth = np.load(max_path) #深度图pooling机制
            # pooled_min, pooled_max, _, _ = pool_depth_arrays(min_depth, max_depth, depth_kernel_size=kernel_size) #深度图pooling机制删除：离线pooling改为GPU执行
            pooled_min, pooled_max = pool_depth_arrays_cuda(min_depth, max_depth, kernel_size=kernel_size, device=device) #深度图pooling机制
            np.save(out_min, pooled_min) #深度图pooling机制
            np.save(out_max, pooled_max) #深度图pooling机制
            success_count += 1 #深度图pooling机制
        except Exception as exc: #深度图pooling机制
            failed_count += 1 #深度图pooling机制
            print(f"[深度图pooling机制] failed {base}: {exc}") #深度图pooling机制

    print("[深度图pooling机制] done") #深度图pooling机制
    print(f"  success: {success_count}") #深度图pooling机制
    print(f"  skipped: {skipped_count}") #深度图pooling机制
    print(f"  missing max: {missing_count}") #深度图pooling机制
    print(f"  failed: {failed_count}") #深度图pooling机制
    print(f"  output_dir: {output_dir}") #深度图pooling机制


def main(): #深度图pooling机制
    parser = argparse.ArgumentParser(description="Offline pooling for min/max depth maps") #深度图pooling机制
    parser.add_argument("--depth_dir", required=True, help="Directory containing *_min.npy and *_max.npy depth maps") #深度图pooling机制
    parser.add_argument("--output_dir", default=None, help="Output directory; default is sibling depth_maps_poolingKxK") #深度图pooling机制
    parser.add_argument("--kernel_size", type=int, default=3, help="Odd pooling kernel size, default 3") #深度图pooling机制
    parser.add_argument("--limit", type=int, default=0, help="Process only first N depth maps for testing; 0 means all") #深度图pooling机制
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing pooled files") #深度图pooling机制
    args = parser.parse_args() #深度图pooling机制

    if args.kernel_size <= 0 or args.kernel_size % 2 == 0: #深度图pooling机制
        raise ValueError(f"kernel_size must be a positive odd integer, got {args.kernel_size}") #深度图pooling机制
    process_depth_dir(args.depth_dir, output_dir=args.output_dir, kernel_size=args.kernel_size, overwrite=args.overwrite, limit=args.limit) #深度图pooling机制


if __name__ == "__main__": #深度图pooling机制
    main() #深度图pooling机制
