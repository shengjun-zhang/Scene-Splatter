import gzip
import os
import pickle
import hydra
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
import imageio
from tqdm import tqdm
from omegaconf import DictConfig
import datetime

from flash3d.models.model import GaussianPredictor, to_device
from flash3d.evaluation.evaluator import Evaluator
from flash3d.datasets3d.data import process_projs, data_to_c2w

from torchvision import transforms
from PIL import Image

import sys
from gaussianSplatting.scene import GaussianModel
from gaussianSplatting.utils.loss_utils import l1_loss, ssim
from argparse import ArgumentParser
from gaussianSplatting.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussianSplatting.gaussian_renderer import render

from diffusers.utils import logging, export_to_video
from viewcrafter.configs.infer_config import get_parser
from viewcrafterpipe import ViewCrafterPipeline, save_video

logger = logging.get_logger(__name__) 

def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

def prepare_data(
    cfg: DictConfig,
    num_render_frames: int = 16,
    image_path: str = None,
    camera_path: str = None,
):
    # load image
    image = Image.open(image_path).convert('RGB')
    h, w = cfg.dataset.height, cfg.dataset.width
    image = image.resize((w, h))
    
    # load camera poses
    with gzip.open(camera_path, 'rb') as f:
        camera_poses = pickle.load(f)

    # build inputs
    inputs = {}
    color_aug_fn = (lambda x: x)
    pad_border_fn = transforms.Pad((cfg.dataset.pad_border_aug, cfg.dataset.pad_border_aug))
    to_tensor_fn = transforms.ToTensor()
    for frame_idx in range(num_render_frames):
        K = process_projs(camera_poses["intrinsics"][frame_idx])
        c2w = data_to_c2w(camera_poses["poses"][frame_idx])
        inputs_color = to_tensor_fn(image)
        inputs_color_aug = to_tensor_fn(color_aug_fn(pad_border_fn(image)))
        
        # K_scale_target
        K_scale_target = K.copy()
        K_scale_target[0, :] *= w
        K_scale_target[1, :] *= h
        # K_scale_source
        K_scale_source = K.copy()
        # scale focal length by size of original image, scale principal point for the padded image
        K_scale_source[0, 0] *=  w
        K_scale_source[1, 1] *=  h
        K_scale_source[0, 2] *= (w + cfg.dataset.pad_border_aug * 2)
        K_scale_source[1, 2] *= (h + cfg.dataset.pad_border_aug * 2)
        inv_K_source = np.linalg.pinv(K_scale_source)
        
        inputs_K_scale_target = torch.from_numpy(K_scale_target)
        inputs_K_scale_source = torch.from_numpy(K_scale_source)
        inputs_inv_K_source = torch.from_numpy(inv_K_source)
        inputs_T_c2w = torch.from_numpy(c2w)
        
        inputs[("frame_id", 0)] = "inference"
        inputs[("K_tgt", frame_idx)] = inputs_K_scale_target.unsqueeze(0)
        inputs[("K_src", frame_idx)] = inputs_K_scale_source.unsqueeze(0)
        inputs[("inv_K_src", frame_idx)] = inputs_inv_K_source.unsqueeze(0)
        inputs[("color", frame_idx, 0)] = inputs_color.unsqueeze(0)              
        inputs[("color_aug", frame_idx, 0)] = inputs_color_aug.unsqueeze(0)
        inputs[("T_c2w", frame_idx)] = inputs_T_c2w.unsqueeze(0)
        inputs[("T_w2c", frame_idx)] = torch.linalg.inv(inputs_T_c2w).unsqueeze(0)
        inputs[("depth_sparse", frame_idx)] = None    
    
    return inputs

def image2gaussian(
    cfg: DictConfig, 
    device: Optional[Union[str, torch.device]],
    num_render_frames: int = 16,
    image_path: str = None,
    camera_path: str = None,
    ckpt_dir: str = None,
):
    cfg.dataset.scale_pose_by_depth = False
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1  
    model = GaussianPredictor(cfg)
    model.to(device)
    model.load_model(ckpt_dir, ckpt_ids=0)
    evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    evaluator.to(device)

    inputs = prepare_data(
            cfg,
            num_render_frames=num_render_frames,
            image_path=image_path,
            camera_path=camera_path
        )

    model_model = get_model_instance(model)
    model_model.set_eval()
    with torch.no_grad():
        to_device(inputs, device)
        inputs["target_frame_ids"] = list(range(1, num_render_frames))
        outputs = model(inputs)
        
    # prepare gaussians
    gaussians = GaussianModel(3)
    scaling = gaussians.scaling_inverse_activation(outputs['gauss_scaling'])
    opacity = gaussians.inverse_opacity_activation(outputs['gauss_opacity'])  
    xyz = outputs['gauss_means'][:, :3, :].permute(0, 2, 1) 
    dtype = xyz.dtype
    xyz = nn.Parameter(torch.tensor(xyz.reshape(-1, 3), dtype=dtype, device=device).requires_grad_(True))
    features = torch.zeros((xyz.shape[0],((gaussians.max_sh_degree + 1) ** 2), 3), dtype=dtype)
    features[:, 0:1, :] = outputs['gauss_features_dc'].reshape(-1,1,3)
    features[:, 1:4, :] = outputs['gauss_features_rest'].reshape(-1,3,3)
    features_dc = nn.Parameter(torch.tensor(features[:,0:1,:], dtype=dtype, device=device).contiguous().requires_grad_(True))
    features_rest = nn.Parameter(torch.tensor(features[:,1:,:], dtype=dtype, device=device).contiguous().requires_grad_(True))
    opacity = nn.Parameter(torch.tensor(opacity.reshape(-1,1), dtype=dtype, device=device).requires_grad_(True))
    scaling = nn.Parameter(torch.tensor(scaling.reshape(-1,3), dtype=dtype, device=device).requires_grad_(True))
    rotation = nn.Parameter(torch.tensor(outputs['gauss_rotation'].reshape(-1,4), dtype=dtype, device=device).requires_grad_(True))
    gaussians.init_gaussian_model(xyz, features_dc, features_rest, scaling,rotation, opacity, 5)
    
    # prepare rendering images
    image_list = []
    for frame_id in range(num_render_frames):
        if frame_id == 0:
            image_list.append(inputs["color", 0, 0]) # gt image
        else:
            image_list.append(outputs[("color_gauss", frame_id, 0)])
    
    # prepare camera list corresponding to the image list
    camera_list = outputs['camera_list']
    
    return gaussians, image_list, camera_list

def optimize_gaussian(
    gaussians: GaussianModel = None,
    images: list = None,
    cameras: list = None,
    render_cameras: List = None,
    iterations: int = 5000,
):
    assert len(images) == len(cameras)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    opt.iterations = iterations
    opt.position_lr_max_steps = iterations
    
    cameras_extent = 10 
    gaussians.training_setup(opt)    
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress") 
    
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        viewpoint_index = iteration % len(images)
        render_pkg = render(cameras, viewpoint_index, gaussians, pipe)
        render_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        images = images.type(render_image.dtype)

        Ll1 = l1_loss(render_image, images[viewpoint_index])
        Lssim = 1.0 - ssim(render_image, images[viewpoint_index])
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
        loss.backward()
        
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"point_number":f"{gaussians.get_xyz.shape[0]}","Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                    
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # Optimizer step  
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)   
     
    # render all target images               
    with torch.no_grad():
        render_images = []
        confidence_maps = []
        for render_idx in range(len(render_cameras)):
            render_pkg = render(render_cameras, render_idx, gaussians, pipe, confidence=False)
            render_image = render_pkg["render"]
            render_images.append(render_image.unsqueeze(0))
            
            confidence_pkg = render(render_cameras, render_idx, gaussians, pipe, confidence=True)
            confidence_map = confidence_pkg["render"]
            confidence_maps.append(confidence_map.unsqueeze(0))
        video_confidence_map = torch.cat(confidence_maps, dim=0) * 0.3 # [F, C, H, W]

    return gaussians, render_images, video_confidence_map.permute(0, 2, 3, 1)

def get_intrinsic(camera, h, w):
    tanfovx = camera["tanfovx"]
    tanfovy = camera["tanfovy"]
    fx = w / (2 * tanfovx)
    fy = h / (2 * tanfovy)
    cx = w / 2
    cy = h / 2
    K = torch.tensor([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return K

def projection(pts3d, w2c, K, h, w):
    N, _ = pts3d.shape
    ones = torch.ones((N, 1), dtype=pts3d.dtype, device=pts3d.device)
    homo_pts3d = torch.cat([pts3d, ones], dim=-1)
    pts_c = (w2c @ homo_pts3d.T).T
    K = K.to(pts3d.device)
    pts_i = (K @ pts_c.T).T

    pts_i[:, 0] /= pts_i[:, 2]
    pts_i[:, 1] /= pts_i[:, 2]

    mask_w = (pts_i[:, 0] > 0) & (pts_i[:, 0] < w)
    mask_h = (pts_i[:, 1] > 0) & (pts_i[:, 1] < h)

    return pts_i[:, :2], mask_w & mask_h

def video_mask(
    image: torch.Tensor = None,
    cameras: list = None,
    gaussians: GaussianModel = None,
):
    # TODO: compute confidence map for video
    # video [B, 3, F, H, W]
    # latents [B, F', C, H', W']
    print(image.shape)
    _, C, H, W = image.shape
    device = image.device
    pts3d = gaussians.get_xyz.clone().detach()
    pts3d = pts3d.reshape(-1, 3).to(device)
    
    mask_list = []
    for camera in cameras:
        # mask for w.o. valid projection
        mask_wo_proj = torch.zeros((H, W), dtype=image.dtype, device=device) * 4
        K = get_intrinsic(camera, H, W).to(device)
        w2c = camera["viewmatrix"]
        cur_pts_i, cur_mask = projection(pts3d, w2c, K, H, W)
        cur_proj_xs = cur_pts_i[:, 0].long()
        cur_proj_ys = cur_pts_i[:, 1].long()
        valid_x = cur_proj_xs[cur_mask]
        valid_y = cur_proj_ys[cur_mask]
        mask_wo_proj[valid_y, valid_x] = 1 # covered -> 0; uncovered -> 1
            
        mask_list.append(mask_wo_proj.unsqueeze(0))
        
    guide_mask = torch.cat(mask_list, dim=0).unsqueeze(1) # [F, 1, H, W]
    
    return guide_mask

def generate_video_from_frames(tensors: List[torch.Tensor], output_video_path: str = None, fps: int = 8) -> str:
    
    video_frames = []
    for tensor in tensors:
        frame = ((tensor.squeeze()).cpu()).permute(1, 2, 0).numpy()  # shape: (H, W, C)
        frame = (frame * 255).astype(np.uint8)
        video_frames.append(frame)
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    image_path = cfg.config.image_path
    camera_path = cfg.config.camera_path
    
    # checkpoint dir
    abs_path = os.path.abspath('..')
    ckpt_dir = os.path.join(abs_path, "flash3d/model_re10k_v2.pth")
    viewcrafter_ckpt = os.path.join(abs_path, "viewcrafter/checkpoints/model.ckpt")
    viewcrafter_config = os.path.join(abs_path, "viewcrafter/configs/inference_pvd_1024.yaml")
    if not os.path.isabs(image_path):
        image_path = os.path.join(abs_path, image_path)
    if not os.path.isabs(camera_path):
        camera_path = os.path.join(abs_path, camera_path)
    
    # exp dir
    exp_dir = os.path.join(abs_path, "results")                          
    sub_dir = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')                                                     
    save_dir = os.path.join(exp_dir, sub_dir)                               
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    device = torch.device("cuda:0")
    with gzip.open(camera_path, 'rb') as f:
        camera_poses = pickle.load(f)
    num_render_frames = len(camera_poses["poses"])

    # flash3d: generate gaussians from a single image
    cfg.dataset.height = 576
    cfg.dataset.width = 1024
    gaussians, image_list, camera_list = image2gaussian(cfg=cfg, device=device, num_render_frames=num_render_frames, image_path=image_path, camera_path=camera_path, ckpt_dir=ckpt_dir)  
    # camera sampling
    camera_sample_interval = cfg.config.camera_sample_interval
    image_list = image_list[::camera_sample_interval]
    camera_list = camera_list[::camera_sample_interval]
    # save flash3d video
    pil_images = []
    for i, input_image in enumerate(image_list):
        pil_image = transforms.ToPILImage()(input_image.squeeze())
        pil_images.append(pil_image)
    export_to_video(pil_images, os.path.join(save_dir, "flash3d_video.mp4"), fps=8)    

    # viewcrafter: enhance video
    parser = get_parser() 
    opts = parser.parse_args()
    opts.ckpt_path = viewcrafter_ckpt
    opts.config = viewcrafter_config

    video_iterations = len(camera_list) // (cfg.config.per_video_length - cfg.config.num_overlap_frames)
    per_video_length = cfg.config.per_video_length
    num_overlap_frames = cfg.config.num_overlap_frames       
    opts.video_length = per_video_length

    viewcrafter = ViewCrafterPipeline(opts)
    
    confidence_maps = None
    input_video = torch.cat(image_list[:per_video_length], dim=0).permute(0, 2, 3, 1)
    all_video = []
    cond_img = image_list[0]

    os.makedirs(os.path.join(save_dir,"confidence_maps"),exist_ok=True)

    for i in range(video_iterations):
        input_video = torch.clamp(input_video, min=0.0, max=1.0)
        save_video(input_video,  os.path.join(save_dir, f"input_video_{i}.mp4"))
        
        if i == 0:
            output_video = viewcrafter.run_diffusion(input_video, confidence_map=None, cond_img=cond_img)
        else:
            output_video_known = viewcrafter.run_diffusion(input_video, confidence_maps, cond_img)
            output_video_unknown = viewcrafter.run_diffusion(input_video, None, cond_img)
            output_video = (1 - confidence_maps) * output_video_unknown + confidence_maps * output_video_known
            output_video_known = (output_video_known + 1.) / 2.
            output_video_known = torch.clamp(output_video_known, min=0.0, max=1.0)
            save_video(output_video_known, os.path.join(save_dir, f"output_video_known_{i}.mp4"))
            output_video_unknown = (output_video_unknown + 1.) / 2.
            output_video_unknown = torch.clamp(output_video_unknown, min=0.0, max=1.0)
            save_video(output_video_unknown, os.path.join(save_dir, f"output_video_unknown_{i}.mp4"))

        output_video = (output_video + 1.) / 2.
        output_video = torch.clamp(output_video, min=0.0, max=1.0)
        save_video(output_video, os.path.join(save_dir, f"output_video_{i}.mp4"))

        if i == 0:
            all_video = output_video
        else:
            all_video = torch.cat((all_video[:(len(all_video)-num_overlap_frames)], output_video), dim=0)
        all_camera = camera_list[:len(all_video)]
        
        if i == video_iterations - 1:
            render_cameras = all_camera
        else:
            render_cameras = camera_list[(len(all_video) - num_overlap_frames): (len(all_video) + per_video_length - num_overlap_frames)]            
        
        gaussians, render_images, confidence_maps = optimize_gaussian(gaussians, all_video.permute(0, 3, 1, 2), all_camera, render_cameras, iterations=5000)        
                
        pil_images = []
        for render_image in render_images:
            pil_image = transforms.ToPILImage()(render_image.squeeze())
            pil_images.append(pil_image)
        export_to_video(pil_images, os.path.join(save_dir, f"render_video_{i}.mp4"), fps=8)
        
    return gaussians
    
if __name__ == "__main__":
    main()
   
   