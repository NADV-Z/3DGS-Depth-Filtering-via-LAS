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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImagePair(renders_dir, gt_dir, fname):
    with Image.open(renders_dir / fname) as render:
        render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    with Image.open(gt_dir / fname) as gt:
        gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
    return render_tensor, gt_tensor

def evaluate(model_paths, split):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            eval_dir = Path(scene_dir) / split

            for method in os.listdir(eval_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = eval_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                image_names = sorted(os.listdir(renders_dir))

                ssims = []
                psnrs = []
                lpipss = []

                for fname in tqdm(image_names, desc="Metric evaluation progress"):
                    render_image, gt_image = readImagePair(renders_dir, gt_dir, fname)
                    ssims.append(ssim(render_image, gt_image).detach().cpu())
                    psnrs.append(psnr(render_image, gt_image).detach().cpu())
                    lpipss.append(lpips(render_image, gt_image, net_type='vgg').detach().cpu())
                    del render_image, gt_image
                    torch.cuda.empty_cache()

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            results_name = "results.json" if split == "test" else f"results_{split}.json"
            per_view_name = "per_view.json" if split == "test" else f"per_view_{split}.json"
            with open(os.path.join(scene_dir, results_name), 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(os.path.join(scene_dir, per_view_name), 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("Reason:", repr(e))

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--split', choices=["train", "test"], default="test",
                        help="Rendered split to evaluate under each model path")
    args = parser.parse_args()
    evaluate(args.model_paths, args.split)
