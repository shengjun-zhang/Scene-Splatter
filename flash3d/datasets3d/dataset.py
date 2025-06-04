import os
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision
from PIL import Image
from typing import Optional
from pathlib import Path
from datasets3d.tardataset import TarDataset

from datasets3d.data import process_projs, data_to_c2w, pil_loader, get_sparse_depth
from misc.depth import estimate_depth_scale_ransac
from misc.localstorage import copy_to_local_storage, extract_tar, get_local_dir




class flash_Dataset(data.Dataset):
    def __init__(
            self,
            cfg,
            num_render_frames,
            pose_data,image_path,
            split: Optional[str]=None,
        ) -> None:
        super().__init__()
        self.cfg = cfg  
        self.num_render_frames=num_render_frames
        self.pose_data=pose_data                      # pose_data :dict.keys('intrinsics','poses') 
        self.image_path=image_path
        self.depth_path = None
        self.split = split
        self.image_size = (self.cfg.dataset.height, self.cfg.dataset.width)
        self.color_aug = self.cfg.dataset.color_aug
        if self.cfg.dataset.pad_border_aug != 0:
            self.pad_border_fn = T.Pad((self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug))
        self.num_scales = len(cfg.model.scales)
        self.novel_frames = list(cfg.model.gauss_novel_frames)
        self.frame_count = len(self.novel_frames) + 1
        self.max_fov = cfg.dataset.max_fov
        self.loader = pil_loader
        self.to_tensor = T.ToTensor()
        self.name = "inference"
        self.length=1
        

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.dilation = cfg.dataset.dilation
        self.max_dilation = cfg.dataset.max_dilation
        self._left_offset = 0
                 


    def _load_image(self, image_path):
        image = Image.open(image_path)
        height=self.image_size[0]
        width=self.image_size[1]
        resized_image = image.resize((width, height)) #先宽后高
        return resized_image.convert('RGB')
    
    def __len__(self) -> int:
        return self.length  

    def get_frame_data(self, frame_idx, pose_data, color_aug_fn,image_path):
        # load the image
        img = self._load_image(image_path=image_path)
        # load the intrinsics matrix
        K = process_projs(pose_data["intrinsics"][frame_idx])
        # load the extrinsic matrixself.num_scales
        c2w = data_to_c2w(pose_data["poses"][frame_idx])
        inputs_color = self.to_tensor(img)
        inputs_color_aug = self.to_tensor(color_aug_fn(self.pad_border_fn(img)))
        K_scale_target = K.copy()
        K_scale_target[0, :] *= self.image_size[1]
        K_scale_target[1, :] *= self.image_size[0]
        # scale K_inv for unprojection according to how much padding was added
        K_scale_source = K.copy()
        # scale focal length by size of original image, scale principal point for the padded image
        K_scale_source[0, 0] *=  self.image_size[1]
        K_scale_source[1, 1] *=  self.image_size[0]
        K_scale_source[0, 2] *= (self.image_size[1] + self.cfg.dataset.pad_border_aug * 2)
        K_scale_source[1, 2] *= (self.image_size[0] + self.cfg.dataset.pad_border_aug * 2)
        inv_K_source = np.linalg.pinv(K_scale_source)

        inputs_K_scale_target = torch.from_numpy(K_scale_target)
        inputs_K_scale_source = torch.from_numpy(K_scale_source)
        inputs_inv_K_source = torch.from_numpy(inv_K_source)

        # original world-to-camera matrix in row-major order and transfer to column-major order
        inputs_T_c2w = torch.from_numpy(c2w)

        return inputs_K_scale_target, inputs_K_scale_source, inputs_inv_K_source, inputs_color, inputs_color_aug, inputs_T_c2w
    
    def __getitem__(self, index):
                                                                                                                   
        src_and_tgt_frame_idxs =list(range(self.num_render_frames))      
        inputs = {}    

        color_aug = (lambda x: x)                     
        for frame_idx in src_and_tgt_frame_idxs:
            inputs_K_tgt, inputs_K_src, inputs_inv_K_src, inputs_color, inputs_color_aug, \
            inputs_T_c2w = self.get_frame_data(image_path=self.image_path, 
                                                frame_idx=frame_idx, 
                                                pose_data=self.pose_data,
                                                color_aug_fn=color_aug)
        
            inputs[("frame_id", 0)] = f"{self.name}"
            inputs[("K_tgt", frame_idx)] = inputs_K_tgt
            inputs[("K_src", frame_idx)] = inputs_K_src
            inputs[("inv_K_src", frame_idx)] = inputs_inv_K_src
            inputs[("color", frame_idx, 0)] = inputs_color              #输入图像
            inputs[("color_aug", frame_idx, 0)] = inputs_color_aug
            inputs[("T_c2w", frame_idx)] = inputs_T_c2w
            inputs[("T_w2c", frame_idx)] = torch.linalg.inv(inputs_T_c2w)
            inputs[("depth_sparse", frame_idx)] = None

        return inputs
