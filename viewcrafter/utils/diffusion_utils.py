import importlib
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict
import os
from viewcrafter.lvdm.models.samplers.ddim import DDIMSampler
from viewcrafter.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from einops import rearrange, repeat
import torch.nn.functional as F

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

def video_confidence_map(
    video_confidence_map: torch.Tensor = None,
): 
    video_confidence_map = torch.prod(video_confidence_map, keepdim=True, dim=1) ** (1./3.) # [F, 1, H, W]
    f_down = nn.Upsample(size=(video_confidence_map.shape[2]//8, video_confidence_map.shape[3]//8), mode='bilinear')
    video_confidence_map = f_down(video_confidence_map)
    video_confidence_map = video_confidence_map.permute(1, 0, 2, 3) # [1, F, H, W]
    video_confidence_map = video_confidence_map.unsqueeze(0) # [1, 1, F, H, W]
    return video_confidence_map

def latent_confidence_map(
    img_latents: torch.Tensor = None,
    cond_img: torch.Tensor = None,
    ref_img_num: int = 2,
):
    '''
    use the cond_img and img_latents[:, :, :ref_img_num] as reference latents to
    compute the similarity of other latents
    img_latents: [B, C, F, H, W]
    cond_img: [B, C, 1, H, W]
    '''
    B, C, F, H, W = img_latents.shape
    ref = torch.cat((cond_img, img_latents[:, :, :ref_img_num]), dim=2).reshape(-1, C, H, W)
    ori = img_latents[:, :, ref_img_num:].reshape(-1, C, H, W)
    
    # downsample
    f_down = nn.Upsample(scale_factor=0.2, mode='bilinear')
    ref = f_down(ref) # (B*F1, C, h, w)
    ori = f_down(ori) # (B*F2, C, h, w)
    
    # cosine similarity
    _, _, h, w = ref.shape
    ref = ref.permute(0, 2, 3, 1).reshape(B, -1, C, 1) # (B, h*w*F1, C, 1)
    ori = ori.reshape(B, -1, C, h, w).permute(0, 2, 1, 3, 4) # (B, C, F2, h, w)
    ori = ori.reshape(B, 1, C, -1) # (B, 1, C, h*w*F2)
    cos = nn.CosineSimilarity(dim=2)
    sim_map = cos(ref, ori) #(B, h*w*F1, h*w*F2)
    sim_map = torch.max(sim_map, dim=1)[0] # (B, h*w*F2)
    sim_map = sim_map.reshape(B, -1, h, w) # (B, F2, h, w)
    
    #upsample
    f_up = nn.Upsample(size=(H, W), mode='bilinear')
    sim_map = f_up(sim_map) # (B, F2, H, W)
    
    cof_map = torch.ones(size=(B, F, H, W), dtype=img_latents.dtype, device=img_latents.device)
    cof_map[:, ref_img_num:] = sim_map
    cof_map = cof_map.unsqueeze(1) # (B, 1, F, H, W)
    
    # MinMaxScaler
    cof_map = (cof_map - torch.min(cof_map)) / (1 - torch.min(cof_map) + 1e-6)
    return cof_map 
    

def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, timestep_spacing='uniform', guidance_rescale=0.0, condition_index=None, confidence_map=None, cond_img=None, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size
    assert condition_index is not None, "Error: condition index is None!"


  
    if cond_img == None:
        img = videos[:,:,0] #bchw
        cond_img=img
    else:
        img = cond_img
    
    #img = (img.unsqueeze(0)).permute(0, 3, 1, 2)  


    
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)
    
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        # if loop or interp:
        #     img_cat_cond = torch.zeros_like(z)
        #     img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
        #     img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        # else:
        img_cat_cond = z
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    if confidence_map == None:
        cond_mask = None
        z0 = None
    else:
        img_cond = get_latent_z(model, cond_img.unsqueeze(2))
        cond_mask = latent_confidence_map(z, img_cond, 2) * 1   

        z0 = z

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)