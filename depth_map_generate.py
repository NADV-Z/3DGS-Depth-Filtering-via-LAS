import argparse
import gc
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import laspy
import numpy as np
import torch
from tqdm import tqdm


class DepthMapGenerator:
    def __init__(
        self,
        las_file,
        xml_file,
        meta_file,
        image_dir,
        output_dir,
        device="cuda",
        z_sign=1.0,
        diagnostic_views=5,
    ):
        """
        Generate per-view min/max depth maps from a LAS point cloud and CubicBA XML cameras.
        The LAS points and XML camera centers are shifted by the same SRSOrigin from metadata.xml.
        """
        self.las_file = las_file
        self.xml_file = xml_file
        self.meta_file = meta_file
        self.image_dir = image_dir
        self.output_root = output_dir
        self.output_dir = os.path.join(output_dir, "depth_maps")
        self.z_sign = float(z_sign)
        self.diagnostic_views = max(0, int(diagnostic_views))
        self.srs_origin = self.load_srs_origin()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[device] {self.device}")
        print(f"[metadata] SRSOrigin = {self.srs_origin.numpy()}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[output] created: {self.output_dir}")
        else:
            print(f"[output] {self.output_dir}")

        self.check_disk_space()

        print(f"[las] loading: {las_file}")
        self.points = self.load_las()
        print(f"[las] loaded local points: {self.points.shape[0]}")

        print(f"[xml] parsing cameras: {xml_file}")
        self.cameras, self.cam_center_mean = self.parse_xml()
        print(f"[xml] parsed cameras: {len(self.cameras)}")

        self.print_alignment_diagnostics()

        print("[gpu] moving point cloud to device")
        self.points = self.points.to(self.device)

    def load_srs_origin(self):
        """Read SRSOrigin from metadata.xml."""
        if not self.meta_file:
            raise ValueError("--meta is required so depth generation uses the same origin as XML2Colmap.")

        tree = ET.parse(self.meta_file)
        root = tree.getroot()
        origin_node = root.find(".//SRSOrigin")
        if origin_node is None or not origin_node.text:
            raise ValueError(f"Cannot find SRSOrigin in metadata file: {self.meta_file}")

        values = [float(v.strip()) for v in origin_node.text.split(",")]
        if len(values) != 3:
            raise ValueError(f"SRSOrigin must contain 3 comma-separated values, got: {origin_node.text}")

        return torch.tensor(values, dtype=torch.float32)

    def check_disk_space(self, min_gb=1.0):
        try:
            total, used, free = shutil.disk_usage(self.output_root)
            free_gb = free / (1024**3)
            if free_gb < min_gb:
                print(f"\n[warning] low disk space: {free_gb:.2f} GB")
                input("Press Enter to continue, or Ctrl+C to stop...")
        except Exception as e:
            print(f"[warning] cannot check disk space: {e}")

    def load_las(self):
        """Load LAS points and shift them into the same local coordinate system as XML2Colmap."""
        try:
            las = laspy.read(self.las_file)
            points_raw = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        except Exception as e:
            print(f"[error] failed to read LAS file: {e}")
            sys.exit(1)

        points_torch = torch.from_numpy(points_raw)
        self.las_raw_min = points_torch.min(dim=0)[0]
        self.las_raw_max = points_torch.max(dim=0)[0]
        self.las_raw_mean = points_torch.mean(dim=0)

        # Use the same SRSOrigin as XML2Colmap. Do not use LAS mean shift here.
        points_torch = points_torch - self.srs_origin

        self.las_local_min = points_torch.min(dim=0)[0]
        self.las_local_max = points_torch.max(dim=0)[0]
        self.las_local_mean = points_torch.mean(dim=0)
        return points_torch

    def parse_xml(self):
        """Parse BlocksExchange XML cameras and shift camera centers by the same SRSOrigin."""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()

        cameras = []
        cam_centers = []
        raw_centers = []

        if root.tag != "BlocksExchange":
            raise ValueError(f"Unsupported XML format: {root.tag}. Expected BlocksExchange.")

        for pg in root.findall(".//Photogroup"):
            width = int(pg.find(".//ImageDimensions/Width").text)
            height = int(pg.find(".//ImageDimensions/Height").text)
            f = float(pg.find(".//FocalLength").text)
            cx = float(pg.find(".//PrincipalPoint/x").text)
            cy = float(pg.find(".//PrincipalPoint/y").text)

            for photo in pg.findall(".//Photo"):
                image_path_node = photo.find("ImagePath")
                if image_path_node is None:
                    continue

                R_raw = [
                    [float(photo.find(f".//Rotation/M_{r}{c}").text) for c in range(3)]
                    for r in range(3)
                ]
                R = torch.tensor(R_raw, dtype=torch.float32)

                C_raw = torch.tensor(
                    [
                        float(photo.find(".//Center/x").text),
                        float(photo.find(".//Center/y").text),
                        float(photo.find(".//Center/z").text),
                    ],
                    dtype=torch.float32,
                )
                C = C_raw - self.srs_origin

                cam_info = {
                    "image_name": os.path.basename(image_path_node.text),
                    "width": width,
                    "height": height,
                    "fx": f,
                    "fy": f,
                    "cx": cx,
                    "cy": cy,
                    "R": R,
                    "C": C,
                    "T_vec": -(R @ C),
                }
                raw_centers.append(C_raw.numpy())
                cam_centers.append(C.numpy())
                cameras.append(cam_info)

        if not cameras:
            raise ValueError("No cameras found in XML.")

        raw_centers_np = np.asarray(raw_centers, dtype=np.float32)
        cam_centers_np = np.asarray(cam_centers, dtype=np.float32)
        self.cam_raw_min = torch.from_numpy(raw_centers_np.min(axis=0))
        self.cam_raw_max = torch.from_numpy(raw_centers_np.max(axis=0))
        self.cam_raw_mean = torch.from_numpy(raw_centers_np.mean(axis=0))
        self.cam_local_min = torch.from_numpy(cam_centers_np.min(axis=0))
        self.cam_local_max = torch.from_numpy(cam_centers_np.max(axis=0))
        cam_center_mean = torch.from_numpy(cam_centers_np.mean(axis=0))
        return cameras, cam_center_mean

    def print_alignment_diagnostics(self):
        """Print enough information to verify metadata.xml and coordinate alignment."""
        print("-" * 60)
        print("[alignment check]")
        print(f"  SRSOrigin:             {self.srs_origin.numpy()}")
        print(f"  LAS raw min/max:       {self.las_raw_min.numpy()} / {self.las_raw_max.numpy()}")
        print(f"  LAS raw mean:          {self.las_raw_mean.numpy()}")
        print(f"  LAS local min/max:     {self.las_local_min.numpy()} / {self.las_local_max.numpy()}")
        print(f"  LAS local mean:        {self.las_local_mean.numpy()}")
        print(f"  XML camera raw min/max:{self.cam_raw_min.numpy()} / {self.cam_raw_max.numpy()}")
        print(f"  XML camera raw mean:   {self.cam_raw_mean.numpy()}")
        print(f"  XML camera local min/max:{self.cam_local_min.numpy()} / {self.cam_local_max.numpy()}")
        print(f"  XML camera local mean: {self.cam_center_mean.numpy()}")

        pc_center = (self.las_local_min + self.las_local_max) * 0.5
        cam_center = self.cam_center_mean
        center_dist = torch.norm(pc_center - cam_center)
        print(f"  local bbox center distance: {center_dist.item():.4f}")

        if torch.max(torch.abs(self.las_local_mean)) > 100000:
            print("  [warning] LAS local coordinates are still very large. Check SRSOrigin.")
        if torch.max(torch.abs(self.cam_center_mean)) > 100000:
            print("  [warning] Camera local coordinates are still very large. Check SRSOrigin.")
        print("-" * 60)

    def project_points_to_image(self, points, camera, collect_stats=False):
        """Project local 3D points into one XML camera."""
        R = camera["R"].to(self.device)
        T = camera["T_vec"].to(self.device)
        fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
        width, height = camera["width"], camera["height"]

        points_cam = torch.addmm(T, points, R.t())
        points_cam[:, 2] *= self.z_sign

        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        valid_z_mask = z > 0.1

        stats = None
        if collect_stats:
            z_cpu = z.detach().float()
            stats = {
                "total_points": int(points.shape[0]),
                "z_min": float(z_cpu.min().cpu()),
                "z_max": float(z_cpu.max().cpu()),
                "z_mean": float(z_cpu.mean().cpu()),
                "positive_z": int(valid_z_mask.sum().item()),
            }

        if valid_z_mask.sum() < 100:
            if collect_stats:
                stats["in_image"] = 0
                stats["valid_depth_min"] = None
                stats["valid_depth_max"] = None
            return None, None, None, stats

        x = x[valid_z_mask]
        y = y[valid_z_mask]
        z = z[valid_z_mask]

        inv_z = 1.0 / z
        u = fx * (x * inv_z) + cx
        v = fy * (y * inv_z) + cy

        valid_pixel_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if valid_pixel_mask.sum() == 0:
            if collect_stats:
                stats["in_image"] = 0
                stats["valid_depth_min"] = None
                stats["valid_depth_max"] = None
            return None, None, None, stats

        u = u[valid_pixel_mask]
        v = v[valid_pixel_mask]
        z = z[valid_pixel_mask]

        if collect_stats:
            stats["in_image"] = int(valid_pixel_mask.sum().item())
            stats["valid_depth_min"] = float(z.min().detach().cpu())
            stats["valid_depth_max"] = float(z.max().detach().cpu())
            stats["valid_depth_mean"] = float(z.mean().detach().cpu())

        return u, v, z, stats

    def create_depth_maps_gpu(self, camera, collect_stats=False):
        width, height = camera["width"], camera["height"]
        u, v, z, stats = self.project_points_to_image(self.points, camera, collect_stats=collect_stats)

        if u is None:
            return None, None, stats

        u = u.long()
        v = v.long()
        idx = v * width + u

        min_depth = torch.full((height * width,), float("inf"), device=self.device)
        max_depth = torch.zeros((height * width,), device=self.device)

        min_depth = torch.scatter_reduce(min_depth, 0, idx, z, reduce="amin", include_self=True)
        max_depth = torch.scatter_reduce(max_depth, 0, idx, z, reduce="amax", include_self=True)

        min_depth = min_depth.view(height, width)
        max_depth = max_depth.view(height, width)
        min_depth[min_depth == float("inf")] = 0

        if collect_stats and stats is not None:
            valid = min_depth > 0
            stats["valid_pixel_ratio"] = float(valid.float().mean().detach().cpu())

        return min_depth, max_depth, stats

    def save_depth_maps(self, min_depth, max_depth, base_output_path, colormap=None, save_jpg=True):
        try:
            min_cpu = min_depth.cpu().numpy()
            max_cpu = max_depth.cpu().numpy()

            np.save(f"{base_output_path}_min.npy", min_cpu)
            np.save(f"{base_output_path}_max.npy", max_cpu)

            if save_jpg:
                valid_mask = min_cpu > 0
                if valid_mask.any():
                    dmin, dmax = min_cpu[valid_mask].min(), min_cpu[valid_mask].max()
                    normalized = (min_cpu - dmin) / (dmax - dmin + 1e-8)
                    normalized[~valid_mask] = 0
                    depth_uint8 = (normalized * 255).astype(np.uint8)

                    if colormap is not None:
                        colored = cv2.applyColorMap(depth_uint8, colormap)
                        colored[min_cpu == 0] = [0, 0, 0]
                        cv2.imwrite(f"{base_output_path}_min_depth.jpg", colored)
                    else:
                        cv2.imwrite(f"{base_output_path}_min_depth.jpg", depth_uint8)

        except OSError as e:
            print(f"\n[error] cannot save depth map {base_output_path}: {e}")
            raise e

    def generate_all_depth_maps(self, colormap=cv2.COLORMAP_JET, preview_interval=10):
        print("\n[start] generating depth maps")
        print(f"[output] {self.output_dir}")
        print(f"[projection] z_sign = {self.z_sign}")

        success_count = 0
        fail_count = 0
        disk_check_interval = 500

        for idx, cam in enumerate(tqdm(self.cameras, desc="depth maps")):
            if idx % disk_check_interval == 0 and idx > 0:
                self.check_disk_space(min_gb=0.5)

            collect_stats = idx < self.diagnostic_views
            try:
                min_d, max_d, stats = self.create_depth_maps_gpu(cam, collect_stats=collect_stats)
            except Exception as e:
                print(f"\n[warning] projection failed: {cam['image_name']} - {e}")
                fail_count += 1
                continue

            if collect_stats:
                print(f"\n[view check] {cam['image_name']}")
                print(f"  stats: {stats}")

            if min_d is None:
                fail_count += 1
                continue

            base_name = os.path.splitext(cam["image_name"])[0]
            base_output = os.path.join(self.output_dir, base_name)
            save_jpg = idx % preview_interval == 0

            try:
                self.save_depth_maps(min_d, max_d, base_output, colormap, save_jpg)
                success_count += 1
            except OSError:
                print("[error] stopped due to disk write failure")
                break

            del min_d, max_d
            if idx % 100 == 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        print(f"\n[done] success: {success_count}, no projection/failed: {fail_count}")
        print(f"[done] files saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate LAS depth maps using CubicBA XML and metadata SRSOrigin.")
    parser.add_argument("--las", required=True, help="E:\3dgs\data\HLSRedHouse\input\20250418181906578.house_cut.las")
    parser.add_argument("--xml", required=True, help="E:\3dgs\data\HLSRedHouse\input\AT\CubicBA.xml")
    parser.add_argument("--meta", required=True, help="E:\3dgs\data\HLSRedHouse\input\AT\metadata.xml")
    parser.add_argument("--images", required=True, help="E:\3dgs\data\HLSRedHouse\input\images")
    parser.add_argument("--output", required=True, help="E:\3dgs\data\HLSRedHouse\input\depth_maps1")
    parser.add_argument("--colormap", default="jet", choices=["jet", "viridis", "hot", "gray"])
    parser.add_argument("--preview_interval", type=int, default=10, help="Generate one preview JPG every N images")
    parser.add_argument("--z_sign", type=float, default=1.0, choices=[1.0, -1.0], help="Set -1 only if diagnostics show all depths are behind cameras")
    parser.add_argument("--diagnostic_views", type=int, default=5, help="Print projection stats for the first N cameras")

    args = parser.parse_args()

    cmap_dict = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "hot": cv2.COLORMAP_HOT,
        "gray": None,
    }

    gen = DepthMapGenerator(
        args.las,
        args.xml,
        args.meta,
        args.images,
        args.output,
        z_sign=args.z_sign,
        diagnostic_views=args.diagnostic_views,
    )
    gen.generate_all_depth_maps(cmap_dict[args.colormap], args.preview_interval)


if __name__ == "__main__":
    main()
