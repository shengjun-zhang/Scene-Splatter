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

#==================================================================


#ModelParams配置参数和Camera()类。以及train文件显示的cuda和device，全部已经调成了


#=================================================================

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

#调试的时候才需要用，先sh.train.sh,再在左侧选择相应调试的launch.json文件即可（有添加过内容的）
#==================================================================

import os

# 在其他 import 之前设置可见的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3" #调整GPU
import torchvision
from PIL import Image
from scene.dataset_readers import getNerfppNorm
from torch import nn
from dataset import get_camera_info
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_val=1.0):
    # img1 和 img2 是两个形状为 [3, h, w] 的张量
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations,  debug_from, recon_number, supervise_image_path, max_ind, iter):
    
    
    render_camera=torch.load("camera.pth")               #flash3d输出的渲染函数需要的相机相关参数，让flash3d一次性的输出所有相机参数
    first_iter = 0     
    gaussians = GaussianModel(3)                       #定义了一个高斯模型,原来这里是dataset.sh_degree，也即传入最大阶数，这里我们手动赋值
    scene = Scene(dataset, gaussians)                  #定义了一个场景类，用来加载或保存模型

    # train_camera_info=get_camera_info()    
    # scene_info=getNerfppNorm(train_camera_info)
    #cameras_extent=scene_info['radius']                    #这边有个优化场景的范围估计，可能后续得补一个计算半径的函数，这里直接用了默认值
    cameras_extent=3                                        #这个场景范围由于相机太近算出来太小，相当于里面不含点云，全别剪枝掉了

    if(iter==1):
        file_path="gaussian_point.pth"                      #flash3d输出的高斯点的相关参数信息，并完成下述模型初始化
        data=torch.load(file_path)
        device=torch.device("cuda")
        data['scaling']= gaussians.scaling_inverse_activation(data['scaling'])
        data['opacity']=gaussians.inverse_opacity_activation(data['opacity'])  #先做反激活
        xyz = nn.Parameter(torch.tensor(data['xyz'].reshape(-1,3), dtype=torch.float, device=device).requires_grad_(True))
        features = torch.zeros((xyz.shape[0],((gaussians.max_sh_degree + 1) ** 2),3)).float()
        features[:,0:1,:]=data['features_dc'].reshape(-1,1,3)
        features[:,1:4,:]=data['features_rest'].reshape(-1,3,3)
        features_dc = nn.Parameter(torch.tensor(features[:,0:1,:], dtype=torch.float, device=device).contiguous().requires_grad_(True))
        features_rest = nn.Parameter(torch.tensor(features[:,1:,:], dtype=torch.float, device=device).contiguous().requires_grad_(True))
        opacity = nn.Parameter(torch.tensor(data['opacity'].reshape(-1,1), dtype=torch.float, device=device).requires_grad_(True))
        scaling = nn.Parameter(torch.tensor(data['scaling'].reshape(-1,3), dtype=torch.float, device=device).requires_grad_(True))
        rotation = nn.Parameter(torch.tensor(data['rotation'].reshape(-1,4), dtype=torch.float, device=device).requires_grad_(True))
        gaussians.init_gaussian_model(xyz,features_dc,features_rest,scaling,rotation,opacity,5) 
        gaussians.training_setup(opt)                    
   
    else:
        model_params = torch.load(scene.model_path + f"/{iter}_chkpnt" + "5000" + ".pth")  #后续的重建直接调用此前的高斯初始化
        gaussians.restore(model_params, opt)


    


    iter_start = torch.cuda.Event(enable_timing = True)  #创建两个cuda事件，用来测量迭代时间
    iter_end = torch.cuda.Event(enable_timing = True)
    

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") #创建一个tqdm进度条，用来显示训练进度，如果不是像dataloader本身直接参与迭代，而只是作为计算epoch(iterration)完成情况的进度条(三维重建一次iteration只对一个相机进行计算），则在循环外生成tqdm对象
    first_iter += 1
    #接下来开始迭代训练
    for iteration in range(first_iter, opt.iterations + 1):   #从第一步开始到结束，opt.iterations里存了循环总次数，也即训练次数    
        if network_gui.conn == None:                          #检查gui是否连接
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe,scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        

        iter_start.record()                                  #用于测量迭代时间

        gaussians.update_learning_rate(iteration)            #更新学习率，各参数的学习率可以不一样，分别更新

     
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()                        #每一千次迭代，增加球谐函数的阶数

        
        viewpoint_stack=list(range(recon_number))                                #重建相机索引列表
        viewpoint_index = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))  #随机采样的相机索引
        
       
    

        # Render                                                                  
        if (iteration - 1) == debug_from:
            pipe.debug = True

    
       
        render_pkg = render(render_camera,viewpoint_index, gaussians, pipe) 
                                        
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        
        # os.makedirs('tmp', exist_ok=True)
        # if iteration % 100 == 0:
        #     torchvision.utils.save_image(image, os.path.join('tmp', f'demo_{iteration}.png'))
        
        img_path = supervise_image_path
        gt_image = Image.open(os.path.join(img_path, f"{viewpoint_index}.png"))
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        gt_image = transform(gt_image).to(image.device)                             #用来获得用来监督的gt


        # if iteration % 2000 == 0:
        #     os.makedirs('tmp', exist_ok=True)
        #     torchvision.utils.save_image(image, os.path.join('tmp', f'demo_{iteration}.png'))
        #     psnr = calculate_psnr(gt_image, image)
        #     print(f'Iter {iteration}, PSNR: {psnr}')

        if iteration == 5000:                                                                            #5000轮就阶数训练，渲染结果，保存点云
            for viewpoint_index in range(0,max_ind):                                               #选择min_ind和max_ind的进行渲染
                new_render_pkg = render(render_camera,viewpoint_index, gaussians, pipe) 
                new_image = new_render_pkg['render']
                os.makedirs(f"render_new/new_{iter}", exist_ok=True)                                    #储存每次渲染的结果        
                torchvision.utils.save_image(new_image, os.path.join(f"render_new/new_{iter}", f"{viewpoint_index}.png"))
        
            torch.save((gaussians.capture(), iteration), scene.model_path + f"/{iter}_chkpnt" + str(iteration) + ".pth")
            print("\n One iterative training complete.")
            return
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()                           
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"point_number":f"{gaussians.get_xyz.shape[0]}","Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()                

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)               #保存场景             #这个应该是保存三维信息，保存在点云文件中

            # Densification                         #在一定的迭代次数内进行密集化处理
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step                      #执行优化器的步骤，更新参数，再清空梯度
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # if (iteration in checkpoint_iterations):                      #达到检查点保存次数，保存检查点
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save(gaussians.capture(), scene.model_path + "/chkpnt" + str(iter) + ".pth")    
            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_001])           #[7_000, 30_000]
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_001])           #[7_000, 30_000]
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5_001])  #[7_000, 30_000]
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])              #args是读取到了所有group的参数，后面分组成立各自的解析对象，每个对象存了同一类相关参数，也即ex
    args.save_iterations.append(args.iterations)  
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)                            #初始化随机种子

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset=lp.extract(args)
    opt=op.extract(args)
    pipe=pp.extract(args)
    

    iter=1
    recon_number=24
    max_ind=recon_number+8
    supervise_image_path='recon_image/iter1'

    training(dataset, opt, pipe, args.save_iterations, args.checkpoint_iterations, args.debug_from, recon_number, supervise_image_path, max_ind, iter )





    # iter=1
    # recon_number=20  #初始监督图片数量
    # supervise_image_path="supervise_image"
    

    # while(1):                                         #后面这里给iter加一个最大迭代轮数限制条件
    #     min_ind=recon_number-5
    #     max_ind=recon_number+5                        #每次上下加5帧进行渲染+增强
    #     training(dataset, opt, pipe, args.save_iterations, args.checkpoint_iterations, args.debug_from, recon_number, supervise_image_path, max_ind, iter ) 
        
    #     #上述会生成 f"render_new/new_{iter}"
    #     #从 f"render_new/new_{iter}" 读image，送入diffusion,resize成256 384（flash3d),转成图片，存到f"diffusion_image/enhance{iter}"中
    #     #注意diffusion后png前的索引始终要保持着，最后每次做完都拍个序
    #     #用f"diffusion_image/enhance{iter}"中的图片替换掉对应索引的"supervise_image"，并补充上新的索引的图片

    #     recon_number+=5                                          #每迭代一轮总数+5，多5张新的
    #     iter+=1
       
    #    #所以其实supervise_image文件夹下的图片是不断变化的，他就是用来监督的



            
            

