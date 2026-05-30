import argparse
import csv
import os
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy

import laspy
import numpy as np
import torch
from tqdm import tqdm


def indent(elem, level=0): #LAS覆盖筛选机制
    i = "\n" + level * "    " #LAS覆盖筛选机制
    if len(elem): #LAS覆盖筛选机制
        if not elem.text or not elem.text.strip(): #LAS覆盖筛选机制
            elem.text = i + "    " #LAS覆盖筛选机制
        for child in elem: #LAS覆盖筛选机制
            indent(child, level + 1) #LAS覆盖筛选机制
        if not child.tail or not child.tail.strip(): #LAS覆盖筛选机制
            child.tail = i #LAS覆盖筛选机制
    if level and (not elem.tail or not elem.tail.strip()): #LAS覆盖筛选机制
        elem.tail = i #LAS覆盖筛选机制


def load_srs_origin(meta_path): #LAS覆盖筛选机制
    tree = ET.parse(meta_path) #LAS覆盖筛选机制
    root = tree.getroot() #LAS覆盖筛选机制
    origin_node = root.find(".//SRSOrigin") #LAS覆盖筛选机制
    if origin_node is None or not origin_node.text: #LAS覆盖筛选机制
        raise ValueError(f"Cannot find SRSOrigin in metadata file: {meta_path}") #LAS覆盖筛选机制
    values = [float(v.strip()) for v in origin_node.text.split(",")] #LAS覆盖筛选机制
    if len(values) != 3: #LAS覆盖筛选机制
        raise ValueError(f"SRSOrigin must contain 3 comma-separated values, got: {origin_node.text}") #LAS覆盖筛选机制
    return torch.tensor(values, dtype=torch.float32) #LAS覆盖筛选机制


def load_las_points(las_path, srs_origin): #LAS覆盖筛选机制
    print(f"[LAS覆盖筛选机制] loading LAS: {las_path}") #LAS覆盖筛选机制
    las = laspy.read(las_path) #LAS覆盖筛选机制
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32) #LAS覆盖筛选机制
    points_torch = torch.from_numpy(points) - srs_origin #LAS覆盖筛选机制
    print(f"[LAS覆盖筛选机制] loaded local points: {points_torch.shape[0]}") #LAS覆盖筛选机制
    return points_torch #LAS覆盖筛选机制


def parse_xml_cameras(xml_path, srs_origin): #LAS覆盖筛选机制
    tree = ET.parse(xml_path) #LAS覆盖筛选机制
    root = tree.getroot() #LAS覆盖筛选机制
    cameras = [] #LAS覆盖筛选机制

    for photogroup in root.findall(".//Photogroup"): #LAS覆盖筛选机制
        width = int(photogroup.find(".//ImageDimensions/Width").text) #LAS覆盖筛选机制
        height = int(photogroup.find(".//ImageDimensions/Height").text) #LAS覆盖筛选机制
        f = float(photogroup.find(".//FocalLength").text) #LAS覆盖筛选机制
        cx = float(photogroup.find(".//PrincipalPoint/x").text) #LAS覆盖筛选机制
        cy = float(photogroup.find(".//PrincipalPoint/y").text) #LAS覆盖筛选机制

        for photo in photogroup.findall("Photo"): #LAS覆盖筛选机制
            image_path_node = photo.find("ImagePath") #LAS覆盖筛选机制
            if image_path_node is None or not image_path_node.text: #LAS覆盖筛选机制
                continue #LAS覆盖筛选机制

            r_raw = [[float(photo.find(f".//Rotation/M_{r}{c}").text) for c in range(3)] for r in range(3)] #LAS覆盖筛选机制
            r_mat = torch.tensor(r_raw, dtype=torch.float32) #LAS覆盖筛选机制
            c_raw = torch.tensor([ #LAS覆盖筛选机制
                float(photo.find(".//Center/x").text), #LAS覆盖筛选机制
                float(photo.find(".//Center/y").text), #LAS覆盖筛选机制
                float(photo.find(".//Center/z").text), #LAS覆盖筛选机制
            ], dtype=torch.float32) #LAS覆盖筛选机制
            c_local = c_raw - srs_origin #LAS覆盖筛选机制
            t_vec = -(r_mat @ c_local) #LAS覆盖筛选机制

            image_name = os.path.basename(image_path_node.text) #LAS覆盖筛选机制
            cameras.append({ #LAS覆盖筛选机制
                "image_name": image_name, #LAS覆盖筛选机制
                "width": width, #LAS覆盖筛选机制
                "height": height, #LAS覆盖筛选机制
                "fx": f, #LAS覆盖筛选机制
                "fy": f, #LAS覆盖筛选机制
                "cx": cx, #LAS覆盖筛选机制
                "cy": cy, #LAS覆盖筛选机制
                "R": r_mat, #LAS覆盖筛选机制
                "T_vec": t_vec, #LAS覆盖筛选机制
            }) #LAS覆盖筛选机制

    print(f"[LAS覆盖筛选机制] parsed cameras: {len(cameras)}") #LAS覆盖筛选机制
    return tree, root, cameras #LAS覆盖筛选机制


def compute_coverage(points, camera, device, z_sign=1.0): #LAS覆盖筛选机制
    r_mat = camera["R"].to(device) #LAS覆盖筛选机制
    t_vec = camera["T_vec"].to(device) #LAS覆盖筛选机制
    width, height = camera["width"], camera["height"] #LAS覆盖筛选机制
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"] #LAS覆盖筛选机制

    points_cam = torch.addmm(t_vec, points, r_mat.t()) #LAS覆盖筛选机制
    if z_sign < 0: #LAS覆盖筛选机制
        points_cam = -points_cam #LAS覆盖筛选机制

    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2] #LAS覆盖筛选机制
    positive_z = int((z > 0.1).sum().item()) #LAS覆盖筛选机制
    if positive_z < 100: #LAS覆盖筛选机制
        return { #LAS覆盖筛选机制
            "positive_z": positive_z, #LAS覆盖筛选机制
            "in_image": 0, #LAS覆盖筛选机制
            "valid_pixel_ratio": 0.0, #LAS覆盖筛选机制
            "valid_depth_min": "", #LAS覆盖筛选机制
            "valid_depth_max": "", #LAS覆盖筛选机制
            "valid_depth_mean": "", #LAS覆盖筛选机制
            "reason": "too_few_positive_z", #LAS覆盖筛选机制
        } #LAS覆盖筛选机制

    mask_z = z > 0.1 #LAS覆盖筛选机制
    x, y, z = x[mask_z], y[mask_z], z[mask_z] #LAS覆盖筛选机制
    inv_z = 1.0 / z #LAS覆盖筛选机制
    u = fx * (x * inv_z) + cx #LAS覆盖筛选机制
    v = fy * (y * inv_z) + cy #LAS覆盖筛选机制
    pixel_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) #LAS覆盖筛选机制
    in_image = int(pixel_mask.sum().item()) #LAS覆盖筛选机制
    if in_image == 0: #LAS覆盖筛选机制
        return { #LAS覆盖筛选机制
            "positive_z": positive_z, #LAS覆盖筛选机制
            "in_image": 0, #LAS覆盖筛选机制
            "valid_pixel_ratio": 0.0, #LAS覆盖筛选机制
            "valid_depth_min": "", #LAS覆盖筛选机制
            "valid_depth_max": "", #LAS覆盖筛选机制
            "valid_depth_mean": "", #LAS覆盖筛选机制
            "reason": "no_pixel_projection", #LAS覆盖筛选机制
        } #LAS覆盖筛选机制

    u = u[pixel_mask].long() #LAS覆盖筛选机制
    v = v[pixel_mask].long() #LAS覆盖筛选机制
    z = z[pixel_mask] #LAS覆盖筛选机制
    pixel_idx = torch.unique(v * width + u) #LAS覆盖筛选机制
    valid_pixel_ratio = float(pixel_idx.numel() / float(width * height)) #LAS覆盖筛选机制

    return { #LAS覆盖筛选机制
        "positive_z": positive_z, #LAS覆盖筛选机制
        "in_image": in_image, #LAS覆盖筛选机制
        "valid_pixel_ratio": valid_pixel_ratio, #LAS覆盖筛选机制
        "valid_depth_min": float(z.min().detach().cpu()), #LAS覆盖筛选机制
        "valid_depth_max": float(z.max().detach().cpu()), #LAS覆盖筛选机制
        "valid_depth_mean": float(z.mean().detach().cpu()), #LAS覆盖筛选机制
        "reason": "covered", #LAS覆盖筛选机制
    } #LAS覆盖筛选机制


def build_output_xml(input_xml, output_xml, keep_names, output_images, keep_original_image_path=False): #LAS覆盖筛选机制
    tree = ET.parse(input_xml) #LAS覆盖筛选机制
    root = tree.getroot() #LAS覆盖筛选机制
    total_photos = 0 #LAS覆盖筛选机制
    kept_photos = 0 #LAS覆盖筛选机制

    for photogroup in root.findall(".//Photogroup"): #LAS覆盖筛选机制
        for photo in list(photogroup.findall("Photo")): #LAS覆盖筛选机制
            total_photos += 1 #LAS覆盖筛选机制
            image_path_node = photo.find("ImagePath") #LAS覆盖筛选机制
            image_name = os.path.basename(image_path_node.text) if image_path_node is not None and image_path_node.text else "" #LAS覆盖筛选机制
            if image_name.lower() not in keep_names: #LAS覆盖筛选机制
                photogroup.remove(photo) #LAS覆盖筛选机制
                continue #LAS覆盖筛选机制
            kept_photos += 1 #LAS覆盖筛选机制
            if not keep_original_image_path and image_path_node is not None: #LAS覆盖筛选机制
                image_path_node.text = os.path.join(output_images, image_name) #LAS覆盖筛选机制

    indent(root) #LAS覆盖筛选机制
    os.makedirs(os.path.dirname(output_xml), exist_ok=True) #LAS覆盖筛选机制
    tree.write(output_xml, encoding="utf-8", xml_declaration=True) #LAS覆盖筛选机制
    return total_photos, kept_photos #LAS覆盖筛选机制


def copy_kept_images(images_dir, output_images, image_name, overwrite=False): #LAS覆盖筛选机制
    src = os.path.join(images_dir, image_name) #LAS覆盖筛选机制
    dst = os.path.join(output_images, image_name) #LAS覆盖筛选机制
    if not os.path.exists(src): #LAS覆盖筛选机制
        return "missing_source_image" #LAS覆盖筛选机制
    os.makedirs(output_images, exist_ok=True) #LAS覆盖筛选机制
    if os.path.exists(dst) and not overwrite: #LAS覆盖筛选机制
        return "already_exists" #LAS覆盖筛选机制
    shutil.copy2(src, dst) #LAS覆盖筛选机制
    return "copied" #LAS覆盖筛选机制


def main(): #LAS覆盖筛选机制
    parser = argparse.ArgumentParser(description="Filter CubicBA images by coverage of a cropped LAS file") #LAS覆盖筛选机制
    parser.add_argument("--las", required=True) #LAS覆盖筛选机制
    parser.add_argument("--xml", required=True) #LAS覆盖筛选机制
    parser.add_argument("--meta", required=True) #LAS覆盖筛选机制
    parser.add_argument("--images", required=True) #LAS覆盖筛选机制
    parser.add_argument("--output_xml", required=True) #LAS覆盖筛选机制
    parser.add_argument("--report_csv", required=True) #LAS覆盖筛选机制
    parser.add_argument("--output_images", default=None) #LAS覆盖筛选机制
    parser.add_argument("--min_in_image", type=int, default=1000) #LAS覆盖筛选机制
    parser.add_argument("--min_valid_ratio", type=float, default=0.001) #LAS覆盖筛选机制
    parser.add_argument("--z_sign", type=float, default=1.0, choices=[1.0, -1.0]) #LAS覆盖筛选机制
    parser.add_argument("--limit", type=int, default=0) #LAS覆盖筛选机制
    parser.add_argument("--overwrite_images", action="store_true") #LAS覆盖筛选机制
    parser.add_argument("--keep_original_image_path", action="store_true") #LAS覆盖筛选机制
    args = parser.parse_args() #LAS覆盖筛选机制

    output_images = args.output_images or os.path.join(os.path.dirname(os.path.abspath(args.images)), "image_inlas") #LAS覆盖筛选机制
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #LAS覆盖筛选机制
    print(f"[LAS覆盖筛选机制] device: {device}") #LAS覆盖筛选机制
    print(f"[LAS覆盖筛选机制] output_images: {output_images}") #LAS覆盖筛选机制

    srs_origin = load_srs_origin(args.meta) #LAS覆盖筛选机制
    points = load_las_points(args.las, srs_origin).to(device) #LAS覆盖筛选机制
    _, _, cameras = parse_xml_cameras(args.xml, srs_origin) #LAS覆盖筛选机制
    if args.limit > 0: #LAS覆盖筛选机制
        cameras = cameras[:args.limit] #LAS覆盖筛选机制

    rows = [] #LAS覆盖筛选机制
    keep_names = set() #LAS覆盖筛选机制
    copied_count = 0 #LAS覆盖筛选机制
    missing_source_count = 0 #LAS覆盖筛选机制
    skipped_not_keep_count = 0 #LAS覆盖筛选机制

    for camera in tqdm(cameras, desc="LAS coverage"): #LAS覆盖筛选机制
        try: #LAS覆盖筛选机制
            stats = compute_coverage(points, camera, device, z_sign=args.z_sign) #LAS覆盖筛选机制
        except Exception as exc: #LAS覆盖筛选机制
            stats = { #LAS覆盖筛选机制
                "positive_z": 0, #LAS覆盖筛选机制
                "in_image": 0, #LAS覆盖筛选机制
                "valid_pixel_ratio": 0.0, #LAS覆盖筛选机制
                "valid_depth_min": "", #LAS覆盖筛选机制
                "valid_depth_max": "", #LAS覆盖筛选机制
                "valid_depth_mean": "", #LAS覆盖筛选机制
                "reason": f"exception:{exc}", #LAS覆盖筛选机制
            } #LAS覆盖筛选机制

        keep = stats["in_image"] >= args.min_in_image and stats["valid_pixel_ratio"] >= args.min_valid_ratio #LAS覆盖筛选机制
        copy_status = "skipped_not_keep" #LAS覆盖筛选机制
        if keep: #LAS覆盖筛选机制
            keep_names.add(camera["image_name"].lower()) #LAS覆盖筛选机制
            copy_status = copy_kept_images(args.images, output_images, camera["image_name"], overwrite=args.overwrite_images) #LAS覆盖筛选机制
            if copy_status in ("copied", "already_exists"): #LAS覆盖筛选机制
                copied_count += 1 #LAS覆盖筛选机制
            elif copy_status == "missing_source_image": #LAS覆盖筛选机制
                missing_source_count += 1 #LAS覆盖筛选机制
        else: #LAS覆盖筛选机制
            skipped_not_keep_count += 1 #LAS覆盖筛选机制

        rows.append({ #LAS覆盖筛选机制
            "image_name": camera["image_name"], #LAS覆盖筛选机制
            "width": camera["width"], #LAS覆盖筛选机制
            "height": camera["height"], #LAS覆盖筛选机制
            "positive_z": stats["positive_z"], #LAS覆盖筛选机制
            "in_image": stats["in_image"], #LAS覆盖筛选机制
            "valid_pixel_ratio": stats["valid_pixel_ratio"], #LAS覆盖筛选机制
            "valid_depth_min": stats["valid_depth_min"], #LAS覆盖筛选机制
            "valid_depth_max": stats["valid_depth_max"], #LAS覆盖筛选机制
            "valid_depth_mean": stats["valid_depth_mean"], #LAS覆盖筛选机制
            "keep": int(keep), #LAS覆盖筛选机制
            "reason": stats["reason"] if keep else f"below_threshold:{stats['reason']}", #LAS覆盖筛选机制
            "copy_status": copy_status, #LAS覆盖筛选机制
        }) #LAS覆盖筛选机制

    os.makedirs(os.path.dirname(args.report_csv), exist_ok=True) #LAS覆盖筛选机制
    with open(args.report_csv, "w", newline="", encoding="utf-8") as f: #LAS覆盖筛选机制
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_name"]) #LAS覆盖筛选机制
        writer.writeheader() #LAS覆盖筛选机制
        writer.writerows(rows) #LAS覆盖筛选机制

    total_photos, kept_photos = build_output_xml(args.xml, args.output_xml, keep_names, output_images, keep_original_image_path=args.keep_original_image_path) #LAS覆盖筛选机制

    print("[LAS覆盖筛选机制] done") #LAS覆盖筛选机制
    print(f"  XML original photos: {total_photos}") #LAS覆盖筛选机制
    print(f"  XML kept photos: {kept_photos}") #LAS覆盖筛选机制
    print(f"  report keep rows: {sum(int(r['keep']) for r in rows)}") #LAS覆盖筛选机制
    print(f"  copied/already_exists: {copied_count}") #LAS覆盖筛选机制
    print(f"  missing source images: {missing_source_count}") #LAS覆盖筛选机制
    print(f"  skipped not keep: {skipped_not_keep_count}") #LAS覆盖筛选机制
    print(f"  output XML: {args.output_xml}") #LAS覆盖筛选机制
    print(f"  report CSV: {args.report_csv}") #LAS覆盖筛选机制
    print(f"  output images: {output_images}") #LAS覆盖筛选机制


if __name__ == "__main__": #LAS覆盖筛选机制
    main() #LAS覆盖筛选机制
