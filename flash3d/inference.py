# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# # 在其他 import 之前设置可见的 GPU



import os

import json
import hydra
import torch
import numpy as np
import pickle
import gzip

from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF

from models.model import GaussianPredictor, to_device
from evaluation.evaluator import Evaluator
from misc.visualise_3d import save_ply
from datasets3d.dataset import flash_Dataset
from datasets3d.util import create_datasets

def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

def inference(model, dataloader, num_render_frames, device=None, save_vis=True):
    print("start")
    model_model = get_model_instance(model)
    model_model.set_eval()
    num = num_render_frames
    target_frame_ids=list(range(1,num))              
    eval_frames = ["src"]
    for i in range(1,num):
        eval_frames.append(f"tgt{i}")                                       #用于创建渲染的图像名字，分为src,tgt1,tgt2,tgt3...

    dataloader_iter = iter(dataloader)   
    inputs = next(dataloader_iter)
        
    if save_vis:
        out_dir = Path("render_images")                                     #注意在我们设置的工作目录下，应该是在命令行可以改那个工作目录，即结果输出的目录
        out_dir.mkdir(parents=True,exist_ok=True)                           #设置parents=True可以创建多层目录
        print(f"saving images to: {out_dir.resolve()}")
        out_out_dir = out_dir
        out_out_dir.mkdir(exist_ok=True)
        out_pred_dir = out_out_dir / f"pred"
        out_pred_dir.mkdir(exist_ok=True)
        out_dir_ply = out_out_dir / "ply"
        out_dir_ply.mkdir(exist_ok=True)
    
    with torch.no_grad():
        if device is not None:
            to_device(inputs, device)
        inputs["target_frame_ids"] = target_frame_ids  
        outputs = model(inputs)
   
    for f_id in range(num):
        pred = outputs[('color_gauss', f_id, 0)]
    
        #  应该在这个时候把pred给存下来，已经是归一化好的tensor了   
               
        if save_vis:
            save_ply(outputs, out_dir_ply / f"{f_id}.ply", gaussians_per_pixel=model.cfg.model.gaussians_per_pixel)
            pred = pred[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            plt.imsave(str(out_pred_dir / f"{f_id:03}.png"), pred)
        
        

   

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    cfg.data_loader.batch_size=1
    cfg.data_loader.num_workers=1                                                         #这里别动！ 后续加到配置文件中
    model = GaussianPredictor(cfg)                                                        #这里有个download不知道是啥
    device = torch.device("cuda:0")
    model.to(device)
    ckpt_dir='exp/re10k_v2/checkpoints/model_re10k_v2.pth'                                 #存放模型文件的地方
    model.load_model(ckpt_dir, ckpt_ids=0)
    evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    evaluator.to(device)
    
    num_render_frames=16                                                                      #渲染帧数
    image_path='/home/zsj/ljz/flash3d/data/RealEstate10K/test/000c3ab189999a83/45979267.jpg'  #输入图像地址
    file_path = "/home/zsj/ljz/flash3d/data/RealEstate10K/test.pickle.gz"                     #相机内外参文件地址，可预先用convert_camera_para.py文件处理txt相机参数文件转为.gz文件
    save_vis = cfg.eval.save_vis                                                              #用来选择是否绘制flash3d渲染结果

    with gzip.open(file_path, "rb") as f:
        pose_data = pickle.load(f)                                        #读取相机内外参
  
          
    dataset= flash_Dataset(cfg, num_render_frames, pose_data, image_path)
    dataloader = create_datasets(cfg,dataset)
    inference(model, dataloader, num_render_frames,device=device, save_vis=save_vis)
    

if __name__ == "__main__":
    main()
