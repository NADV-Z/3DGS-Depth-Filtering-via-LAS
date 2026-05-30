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

import os
import torch
from random import randint
from utils.loss_utils import masked_l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import masked_psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# 导入你自己的工具函数
from utils.point_cloud_utils import read_las_for_gaussian_initialization

# 自适应深度筛选固定策略：保留实验功能，但不再暴露无收益调度参数。
ADAPTIVE_DEPTH_METRIC = "psnr"
ADAPTIVE_DEPTH_UPDATE_INTERVAL = 100
ADAPTIVE_DEPTH_EMA = 0.3

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from,args):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed.")
    if args.las_file:
        las_file = args.las_file
        print(f"Using LAS file: {las_file}")
        print(f"Using LAS metadata file: {args.metadata_path}")
        print(f"Using LAS scale multiplier: {args.las_scale_multiplier}")

    if args.depth_dir:
        depth_dir = args.depth_dir
        print(f"Using depth maps from: {depth_dir}")
        print(f"CUDA depth filter enabled: {not args.disable_cuda_depth_filter}")
        print(f"Depth tolerance: {args.depth_tolerance}")
        print(f"Depth CPU cache size: {args.depth_cache_size}")
        print(f"Adaptive depth filter: {'on' if args.adaptive_depth_filter else 'off'}")
        if args.adaptive_depth_filter:
            print(f"Adaptive depth ratio: {args.adaptive_depth_ratio}")
            print(f"Adaptive depth metric: {ADAPTIVE_DEPTH_METRIC}")
            print(f"Adaptive depth update interval: {ADAPTIVE_DEPTH_UPDATE_INTERVAL}")
            print(f"Adaptive depth EMA: {ADAPTIVE_DEPTH_EMA}")
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    # LAS input replaces the original COLMAP point cloud initialization when provided.
    if args.las_file:
        print(f"\n[自定义流程] 从 LAS 文件初始化高斯: {args.las_file}")
        
        xyz, colors, normals, scales = read_las_for_gaussian_initialization(args.las_file, metadata_path=args.metadata_path, las_scale_multiplier=args.las_scale_multiplier)
        
        gaussians.create_from_las_data(xyz, colors, normals, scales)
        
        scene = Scene(dataset, gaussians, gaussians_preloaded=True)
        if args.depth_dir and not args.disable_cuda_depth_filter and not args.adaptive_depth_filter:
            print(f"\n[动态深度筛选] 训练时将在每个视角 forward 前临时筛选高斯。")
        elif args.depth_dir and not args.disable_cuda_depth_filter and args.adaptive_depth_filter:
            print(f"\n[自适应深度筛选机制] 默认快速训练，只对最近PSNR较低的困难图片临时启用CUDA深度筛选。")
        elif args.depth_dir and args.disable_cuda_depth_filter:
            print(f"\n[实验验证机制] 已提供深度图，但本次运行关闭CUDA深度筛选，用于无筛选对照实验。")
        else:
            print(f"\n[跳过深度筛选] 未提供深度图目录")
    else:
        # 如果没有提供 LAS 文件，就完全按照原始代码的流程进行
        print(f"\n[原始流程] 从 COLMAP 初始化高斯 (默认).")
        scene = Scene(dataset, gaussians, gaussians_preloaded=False)

    train_cameras = scene.getTrainCameras()
    eval_cameras_all = scene.getTestCameras()

    # Cache resized image tensors only for small datasets; larger scenes stay lazy-loaded.
    camera_count_for_cache = len(train_cameras) + len(eval_cameras_all)
    if args.cache_images is None:
        cache_images = camera_count_for_cache <= args.cache_images_threshold
        cache_mode = "auto"
    else:
        cache_images = args.cache_images
        cache_mode = "manual"
    for cam in train_cameras + eval_cameras_all:
        cam.cache_images = cache_images
    threshold_relation = "<=" if camera_count_for_cache <= args.cache_images_threshold else ">"
    print(f"[图像缓存机制] cache_images={'on' if cache_images else 'off'} ({cache_mode}, cameras={camera_count_for_cache} {threshold_relation} {args.cache_images_threshold})")

    gaussians.training_setup(opt, train_cameras)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    render_start = torch.cuda.Event(enable_timing = True)
    render_end = torch.cuda.Event(enable_timing = True)
    eval_render_start = torch.cuda.Event(enable_timing = True)
    eval_render_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    if len(eval_cameras_all) == 0:
        eval_cameras_all = train_cameras

    # Fixed validation always renders without depth filtering so different training modes
    # are compared by model quality rather than evaluation-time filtering.
    eval_cameras = eval_cameras_all[:min(args.eval_count, len(eval_cameras_all))] if args.eval_interval > 0 else []
    if args.eval_interval > 0:
        print(f"[固定验证机制] eval_count={len(eval_cameras)}, eval_interval={args.eval_interval}, eval_start_iter={args.eval_start_iter}, eval_depth_filter=off")
    viewpoint_stack = train_cameras.copy()

    # Adaptive depth filtering tracks recent per-image PSNR and enables CUDA depth
    # filtering only for the lowest-scoring training views.
    adaptive_depth_stats = {}
    adaptive_hard_images = set()
    adaptive_warmup_iters = len(train_cameras)
    adaptive_last_update_iter = 0
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_render_ms_for_log = 0.0
    ema_selected_ratio_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        image_key = viewpoint_cam.image_name
        depth_decision = "disabled"
        active_depth_dir = None
        if args.depth_dir and not args.disable_cuda_depth_filter:
            if args.adaptive_depth_filter:
                if iteration <= adaptive_warmup_iters:
                    depth_decision = "warmup"
                elif image_key in adaptive_hard_images:
                    active_depth_dir = args.depth_dir
                    depth_decision = f"hard_{ADAPTIVE_DEPTH_METRIC}"
                else:
                    depth_decision = "normal"
            else:
                active_depth_dir = args.depth_dir
                depth_decision = "global"
        render_start.record()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, depth_dir=active_depth_dir, depth_tolerance=args.depth_tolerance, depth_cache_size=args.depth_cache_size)
        render_end.record()

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image, _, valid_mask = viewpoint_cam.load_image_tensors()
        Ll1 = masked_l1_loss(image, gt_image, valid_mask)
        ssim_gt_image = gt_image * valid_mask + image.detach() * (1.0 - valid_mask)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, ssim_gt_image))
      
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            current_psnr = masked_psnr(image, gt_image, valid_mask).mean().item()
            valid_mask_ratio = valid_mask.float().mean().item()
            ema_psnr_for_log = 0.4 * current_psnr + 0.6 * ema_psnr_for_log
            render_ms = render_start.elapsed_time(render_end)
            selected_count = int(render_pkg["selected_count"].item())
            total_gaussians = int(render_pkg["total_gaussians"].item())
            selected_ratio = selected_count / max(total_gaussians, 1)
            depth_status = render_pkg.get("depth_status", "unknown")
            valid_depth_ratio = render_pkg.get("valid_depth_ratio", 0.0)
            depth_source = render_pkg.get("depth_source", "unknown")
            image_stats = adaptive_depth_stats.setdefault(image_key, {
                "count": 0,
                "ema_psnr": None,
                "ema_loss": None,
                "last_psnr": None,
                "last_loss": None,
                "last_depth_used": False,
            })
            image_stats["count"] += 1
            image_stats["last_psnr"] = current_psnr
            image_stats["last_loss"] = loss.item()
            image_stats["last_depth_used"] = bool(render_pkg["depth_filter_enabled"])
            if image_stats["ema_psnr"] is None:
                image_stats["ema_psnr"] = current_psnr
                image_stats["ema_loss"] = loss.item()
            else:
                image_stats["ema_psnr"] = ADAPTIVE_DEPTH_EMA * current_psnr + (1.0 - ADAPTIVE_DEPTH_EMA) * image_stats["ema_psnr"]
                image_stats["ema_loss"] = ADAPTIVE_DEPTH_EMA * loss.item() + (1.0 - ADAPTIVE_DEPTH_EMA) * image_stats["ema_loss"]
            if args.adaptive_depth_filter and args.depth_dir and not args.disable_cuda_depth_filter and iteration >= adaptive_warmup_iters:
                if iteration == adaptive_warmup_iters or iteration - adaptive_last_update_iter >= ADAPTIVE_DEPTH_UPDATE_INTERVAL:
                    scored_stats = [(key, stat) for key, stat in adaptive_depth_stats.items() if stat["count"] > 0 and stat["ema_psnr"] is not None]
                    if scored_stats:
                        hard_count = max(1, int(len(scored_stats) * args.adaptive_depth_ratio))
                        scored_stats.sort(key=lambda item: item[1]["ema_psnr"])
                        adaptive_hard_images = set(key for key, _ in scored_stats[:hard_count])
                        adaptive_last_update_iter = iteration
            ema_render_ms_for_log = 0.4 * render_ms + 0.6 * ema_render_ms_for_log
            ema_selected_ratio_for_log = 0.4 * selected_ratio + 0.6 * ema_selected_ratio_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "PSNR": f"{ema_psnr_for_log:.{2}f}", "RenderMs": f"{ema_render_ms_for_log:.2f}", "Selected": f"{ema_selected_ratio_for_log:.3f}"})
                progress_bar.update(10)
            if args.experiment_log_interval > 0 and iteration % args.experiment_log_interval == 0:
                render_mode = "cuda_depth_filtered" if render_pkg["depth_filter_enabled"] else "unfiltered"
                print(f"\n[实验验证机制][ITER {iteration}] mode={render_mode}, adaptive_depth={'on' if args.adaptive_depth_filter else 'off'}, depth_decision={depth_decision}, hard_images={len(adaptive_hard_images)}, depth_status={depth_status}, depth_source={depth_source}, image={viewpoint_cam.image_name}, render_ms={render_ms:.3f}, selected_count={selected_count}, total_gaussians={total_gaussians}, selected_ratio={selected_ratio:.6f}, valid_depth_ratio={valid_depth_ratio:.6f}, valid_mask_ratio={valid_mask_ratio:.6f}, psnr={current_psnr:.3f}, image_psnr_ema={image_stats['ema_psnr']:.3f}, loss={loss.item():.7f}, image_loss_ema={image_stats['ema_loss']:.7f}")
            if iteration == opt.iterations:
                progress_bar.close()

            if args.eval_interval > 0 and iteration >= args.eval_start_iter and (iteration - args.eval_start_iter) % args.eval_interval == 0 and len(eval_cameras) > 0:
                eval_psnr_sum = 0.0
                eval_loss_sum = 0.0
                eval_render_ms_sum = 0.0
                eval_mask_ratio_sum = 0.0
                for eval_cam in eval_cameras:
                    eval_render_start.record()
                    eval_pkg = render(eval_cam, gaussians, pipe, background, depth_dir=None, depth_tolerance=args.depth_tolerance, depth_cache_size=args.depth_cache_size, suppress_grad_warning=True)
                    eval_render_end.record()
                    torch.cuda.synchronize()
                    eval_image = eval_pkg["render"]
                    eval_gt_image, _, eval_valid_mask = eval_cam.load_image_tensors()
                    eval_l1 = masked_l1_loss(eval_image, eval_gt_image, eval_valid_mask)
                    eval_ssim_gt_image = eval_gt_image * eval_valid_mask + eval_image.detach() * (1.0 - eval_valid_mask)
                    eval_loss = (1.0 - opt.lambda_dssim) * eval_l1 + opt.lambda_dssim * (1.0 - ssim(eval_image, eval_ssim_gt_image))
                    eval_psnr_sum += masked_psnr(eval_image, eval_gt_image, eval_valid_mask).mean().item()
                    eval_loss_sum += eval_loss.item()
                    eval_render_ms_sum += eval_render_start.elapsed_time(eval_render_end)
                    eval_mask_ratio_sum += eval_valid_mask.float().mean().item()
                eval_count_actual = len(eval_cameras)
                print(f"\n[固定验证机制][ITER {iteration}] eval_count={eval_count_actual}, eval_depth_filter=off, eval_psnr_mean={eval_psnr_sum / eval_count_actual:.3f}, eval_loss_mean={eval_loss_sum / eval_count_actual:.7f}, eval_render_ms_mean={eval_render_ms_sum / eval_count_actual:.3f}, eval_valid_mask_ratio_mean={eval_mask_ratio_sum / eval_count_actual:.6f}")
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # Densification
            if iteration < opt.densify_until_iter:
                visibility_filter = render_pkg["visibility_filter"]
                full_radii = render_pkg["radii"]
    
                if visibility_filter.any() and viewspace_point_tensor.grad is not None:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                     gaussians.max_radii2D[visibility_filter],
                     full_radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if use_sparse_adam:
                    gaussians.optimizer.step(gaussians.get_xyz.grad > 0)
                else:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000,10_000,20_000,30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--las_file", type=str, default=None,
                    help="Path to input LAS file")
    parser.add_argument("--metadata_path", type=str, default=None,
                    help="Path to metadata.xml containing the SRSOrigin used by XML2Colmap")
    parser.add_argument("--depth_dir", type=str, default=None,
                    help="Directory containing depth maps")
    parser.add_argument("--disable_cuda_depth_filter", action="store_true", default=False,
                    help="Disable CUDA depth filtering even when --depth_dir is provided, for unfiltered baseline experiments")
    parser.add_argument("--depth_tolerance", type=float, default=1.0,
                    help="Depth tolerance used by CUDA depth filtering")
    parser.add_argument("--depth_cache_size", type=int, default=256,
                    help="Number of processed depth maps to keep in CPU LRU cache; 0 disables depth caching")
    parser.add_argument("--las_scale_multiplier", type=float, default=2.0,
                    help="Multiplier for LAS initialized Gaussian scale experiments")
    parser.add_argument("--experiment_log_interval", type=int, default=100,
                    help="Print render timing and selected ratio every N iterations; set 0 to disable")
    parser.add_argument("--adaptive_depth_filter", action="store_true", default=False,
                    help="Enable adaptive CUDA depth filtering only for hard images")
    parser.add_argument("--adaptive_depth_ratio", type=float, default=0.2,
                    help="Ratio of recently hardest images that use CUDA depth filtering")
    parser.add_argument("--eval_interval", type=int, default=0,
                    help="Run fixed no-depth-filter evaluation every N iterations; 0 disables fixed evaluation")
    parser.add_argument("--eval_count", type=int, default=100,
                    help="Number of test cameras used by fixed evaluation")
    parser.add_argument("--eval_start_iter", type=int, default=500,
                    help="First iteration for fixed no-depth-filter evaluation")
    parser.add_argument("--cache_images", action="store_true", default=None,
                    help="Force caching resized image tensors in Camera objects")
    parser.add_argument("--no_cache_images", action="store_false", dest="cache_images",
                    help="Force lazy image loading even for small datasets")
    parser.add_argument("--cache_images_threshold", type=int, default=100,
                    help="Auto-enable --cache_images when train+test camera count is at or below this threshold")
    parser.set_defaults(cache_images=None)

    args = parser.parse_args(sys.argv[1:])
    if args.adaptive_depth_ratio <= 0 or args.adaptive_depth_ratio > 1:
        raise ValueError("--adaptive_depth_ratio must be in (0, 1]")
    if args.eval_interval < 0:
        raise ValueError("--eval_interval must be >= 0")
    if args.eval_count <= 0:
        raise ValueError("--eval_count must be > 0")
    if args.eval_start_iter < 0:
        raise ValueError("--eval_start_iter must be >= 0")
    if args.cache_images_threshold < 0:
        raise ValueError("--cache_images_threshold must be >= 0")
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)

    print("\nTraining complete.")
