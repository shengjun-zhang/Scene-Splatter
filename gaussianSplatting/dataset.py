from scene.cameras import Camera
import torch
import numpy as np
import math
from scene.dataset_readers import CameraInfo

def process():
     lines_list=[]
     filename = './000c3ab189999a83.txt'
     with open(filename, 'r') as file:
          for line in file:
               elements = line.strip().split()  # 根据空格分割
               lines_list.append(elements)  # 将分割后的列表添加到主列表中,每行是一个列表，所有行被存在了一个大列表中。

     del lines_list[0]
     return lines_list


def fovx(width,focalx):
     return 2*math.atan(width/(2*focalx))

def fovy(height,focaly):
     return 2*math.atan(height/(2*focaly))

def get_camera_info():
     height=256
     width=384
     image_list_len=20
     focalx=width*0.482334223
     focaly=height*0.857483078
     FoVx=fovx(width,focalx)
     FoVy=fovy(height,focaly)
     file_path = 'enhanced_image.pth'                       #读取图像
     data = torch.load(file_path)
     lines_list=process()
     cam_infos = []
     for i in range(image_list_len):
          R=np.array([[lines_list[i][7],lines_list[i][11],lines_list[i][15]],
                      [lines_list[i][8],lines_list[i][12],lines_list[i][16]],
                      [lines_list[i][9],lines_list[i][13],lines_list[i][17]]])
          T=np.array([lines_list[i][10],lines_list[i][14],lines_list[i][18]])
          image=data[i].reshape(3,256,384)
          cam_info=CameraInfo(uid=i,R=R,T=T,FovY=FoVy,FovX=FoVx,image=image,image_path='',image_name=f"{i}",width=width,height=height)
          cam_infos.append(cam_info)
     return cam_infos
     
     
# def get_camera():
#      height=256
#      width=384
#      image_list_len=20
#      focalx=width*0.482334223
#      focaly=height*0.857483078
#      FoVx=fovx(width,focalx)
#      FoVy=fovy(height,focaly)
#      ##cx,cy在图形学的渲染过程中默认0.5处理了
     
#      file_path = 'camera.pth'
#      data = torch.load(file_path)
     
#      cam_list = []
#      for i in range(image_list):


     # lines_list=process()
     
     # device=torch.device("cuda")
     
     # cam_list=[]
     # for i in range(image_list_len):
     #      R=np.array([[lines_list[i][7],lines_list[i][11],lines_list[i][15]],
     #                  [lines_list[i][8],lines_list[i][12],lines_list[i][16]],
     #                  [lines_list[i][9],lines_list[i][13],lines_list[i][17]]]) #这里不确定是不是写成转置的形式
     #      T=np.array([lines_list[i][10],lines_list[i][14],lines_list[i][18]])
     #      image=data[i].reshape(3,256,384)
     #      cam=Camera(colmap_id=i,R=R,T=T,FoVx=FoVx,FoVy=FoVy,image=image,gt_alpha_mask=None,image_name=f"{i}",uid=i,data_device=device)
     #      cam_list.append(cam)
     # return cam_list
     










