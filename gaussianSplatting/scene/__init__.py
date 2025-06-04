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
import random
import json
from gaussianSplatting.utils.system_utils import searchForMaxIteration
from gaussianSplatting.scene.dataset_readers import sceneLoadTypeCallbacks
from gaussianSplatting.scene.gaussian_model import GaussianModel
from gaussianSplatting.arguments import ModelParams
from gaussianSplatting.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:                        #这里的scene其实就相当于传统深度学习任务的dataset，用来加载数据和相关相机模型

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:                                            #给render.py的scene来用，用来加载训练output中最大轮数的点云文件
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))  #这里记得改回来，这里是测试取了1
                
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))


        

    
                                                                   

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]