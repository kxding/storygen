# %load_ext autoreload
# %autoreload 2
import gradio as gr
import numpy as np
import torch
import requests
import random
import os
import sys
import pickle
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from utils.gradio_utils import is_torch2_available
if is_torch2_available():
    from utils.gradio_utils import \
        AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils  import AttnProcessor

import diffusers
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from utils.gradio_utils import cal_attn_mask_xl
import copy
import os
from diffusers.utils import load_image
from utils.utils import get_comic
from utils.style_template import styles

from storygen_video_utils import SpatialAttnProcessor2_0, setup_seed

if "__name__" == "__main__":
        
    __import__('ipdb').set_trace()
    ## Global
    STYLE_NAMES = list(styles.keys())
    DEFAULT_STYLE_NAME = "(No style)"

    global attn_count, total_count, id_length, total_length,cur_step, cur_model_type
    global write
    global  sa32, sa64
    global height,width
    attn_count = 0
    total_count = 0
    cur_step = 0
    id_length = 4
    total_length = 5
    cur_model_type = ""
    device="cuda"
    global attn_procs,unet
    attn_procs = {}
    ###
    write = False
    ### strength of consistent self-attention: the larger, the stronger
    sa32 = 0.5
    sa64 = 0.5
    ### Res. of the Generated Comics. Please Note: SDXL models may do worse in a low-resolution! 
    height = 512
    width = 512
    ###
    global pipe
    global sd_model_path
    sd_model_path = "../models/sd_xl"
    ### LOAD Stable Diffusion Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16, use_safetensors=False)
    pipe = pipe.to(device)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)

    from animatediff.models.unet import UNet3DConditionModel
    from omegaconf import OmegaConf
    config = OmegaConf.load("./config/inference.yaml")
    from diffusers import AutoencoderKL, EulerDiscreteScheduler
    pipe.scheduler = EulerDiscreteScheduler(timestep_spacing='leading', steps_offset=1,	**config.noise_scheduler_kwargs)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler.set_timesteps(50)

    unet = UNet3DConditionModel.from_pretrained_2d(sd_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))



    # ### Insert PairedAttention
    # for name in unet.attn_processors.keys():
    #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #     if name.startswith("mid_block"):
    #         hidden_size = unet.config.block_out_channels[-1]
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")])
    #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet.config.block_out_channels[block_id]
    #     if cross_attention_dim is None and (name.startswith("up_blocks") ) :
    #         attn_procs[name] =  SpatialAttnProcessor2_0(id_length = id_length)
    #         total_count +=1
    #     else:
    #         attn_procs[name] = AttnProcessor()
    # print("successsfully load consistent self-attention")
    # print(f"number of the processor : {total_count}")

    # unet.set_attn_processor(copy.deepcopy(attn_procs))
    global mask1024,mask4096
    mask1024, mask4096 = cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width,device=device,dtype= torch.float16)


    motion_module_path = "/userhome/37/ahhfdkx/AnimateDiff_sdxl/models/Motion_Module/mm_sdxl_v10_beta.ckpt"
    motion_module_ckpt = torch.load(motion_module_path, map_location="cpu")
    motion_module_state_dict = {}
    m_k = None
    for k, v in motion_module_ckpt.items():
        if 'motion_module' in k and k in pipe.unet.state_dict().keys():
            motion_module_state_dict[k] = v
            m_k = k
            print("insert motion module")
        elif 'motion_module' in k and k not in pipe.unet.state_dict().keys():
            print(k)
    pipe.unet.load_state_dict(motion_module_state_dict, strict=False)   
    del motion_module_ckpt
    del motion_module_state_dict
    print(f'Loading motion module from {motion_module_path}...')
    # print(unet)

    guidance_scale = 5.0
    seed = 2047
    sa32 = 0.5
    sa64 = 0.5
    id_length = 4
    num_steps = 50

    general_prompt = "a man with a black suit"
    negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
    prompt_array = ["wake up in the bed",
                    "have breakfast",
                    "is on the road, go to the company",
                    "work in the company",
                    "running in the playground",
                    "reading book in the home"
                    ]


    from animatediff.pipelines.pipeline_animation import AnimationPipeline
    pipe = AnimationPipeline(
        unet=unet, vae=pipe.vae, tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, scheduler=pipe.scheduler,
        text_encoder_2=pipe.text_encoder_2, tokenizer_2=pipe.tokenizer_2,
    ).to(device)

    def apply_style_positive(style_name: str, positive: str):
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive) 
    def apply_style(style_name: str, positives: list, negative: str = ""):
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative
    ### Set the generated Style
    style_name = "Comic book"
    setup_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    print(id_length)
    prompts = [general_prompt+","+prompt for prompt in prompt_array]
    id_prompts = prompts[:id_length]
    real_prompts = prompts[id_length:]
    torch.cuda.empty_cache()
    write = False
    cur_step = 0
    attn_count = 0
    id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
    print("id_prompts", id_prompts)
    print(id_prompts[0])
    print(negative_prompt)

    pipe.unet = pipe.unet.half()
    pipe.text_encoder = pipe.text_encoder.half()
    pipe.text_encoder_2 = pipe.text_encoder_2.half()
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    import os
    os.environ["DEBUG_MODE"] = "false"
    from animatediff.utils.util import save_videos_grid
    write = True
    id_images = []
    __import__('ipdb').set_trace()
    for id_prompt in id_prompts:
        print("cur_step", cur_step)
        sample = pipe(
                    id_prompt,
                    num_inference_steps = num_steps, 
                    guidance_scale=guidance_scale, 
                    height = height, # height is global once initialized all the sames ！！！ [todo]
                    width = width,
                    negative_prompt = negative_prompt,
                    single_model_length = 16,
                    generator = generator
                ).videos
        id_images.append(sample)
        print("sample ", sample)
        save_videos_grid(sample, f"./output/sample_{cur_step}.mp4")