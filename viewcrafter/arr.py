import sys
sys.path.append('./extern/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2  
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from pathlib import Path
from torchvision.utils import save_image

from viewcrafter import ViewCrafter
import os
from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime

import os
from PIL import Image
import torch
import torchvision.transforms as transforms

# 设置图像文件夹路径
image_folder = '/home/zsj/ljz/flash3d/render_images/pred'  # 替换为你的图像文件夹路径

# 创建一个空的张量，用于存储图像
images_tensor = torch.empty((25, 576, 1024, 3), dtype=torch.float32)  # 注意: RGB 通道在最后

# 定义一个转换过程
transform = transforms.Compose([
    transforms.Resize((576, 1024)),  # 确保图像尺寸为 576x1024
    transforms.ToTensor(),            # 转换为张量，并归一化到 [0, 1]
])

# 读取图像文件

image_files = sorted(os.listdir(image_folder),key=lambda x:int(x.split('.')[0]))[35:60] # 读取前25张图像
for i, image_file in enumerate(image_files):
    
    image_path = os.path.join(image_folder, image_file)  # 构建图像路径
    image = Image.open(image_path).convert('RGB')  # 打开图像并确保是 RGB 格式
    tensor_image = transform(image)  # 转换为张量，格式为 [C, H, W]
    
    # 将张量从 [C, H, W] 转换为 [H, W, C]
    images_tensor[i] = tensor_image.permute(1, 2, 0)  # 变换为 [H, W, C]

# 现在 images_tensor 是一个形状为 [25, 576, 1024, 3] 的张量
print(images_tensor.shape)  # 输出: torch.Size([25, 576, 1024, 3])



parser = get_parser() # infer config.py
opts = parser.parse_args()
if opts.exp_name == None:
    prefix = datetime.now().strftime("%Y%m%d_%H%M")
    opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
os.makedirs(opts.save_dir,exist_ok=True)
pvd = ViewCrafter(opts)

print('ok')

#pvd.opts.prompt='Rotating view of a scene'

diffusion_results = pvd.run_diffusion(images_tensor)

save_video((diffusion_results + 1.0) / 2.0, os.path.join('diffusion', 'diffusion5.mp4'))