from torchvision.transforms import ToTensor, ToPILImage
from einops import rearrange, repeat
import gc
import folder_paths
import torch
import os
import math
import numpy as np

from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from magicanimate.models.unet_controlnet import UNet3DConditionModel
from magicanimate.models.controlnet import ControlNetModel
from magicanimate.models.appearance_encoder import AppearanceEncoderModel
from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.pipelines.pipeline_animation import AnimationPipeline
from magicanimate.utils.util import save_videos_grid
from magicanimate.utils.dist_tools import distributed_init
from accelerate.utils import set_seed
from collections import OrderedDict

class MagicAnimateModelLoader:
    def __init__(self):
        self.models = {}
        
    @classmethod
    def INPUT_TYPES(s):
        magic_animate_checkpoints = folder_paths.get_filename_list("magic_animate")

        devices = []
        if True: #torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")

        return {
            "required": {
                "controlnet" : (magic_animate_checkpoints ,{
                    "default" : magic_animate_checkpoints[0]
                }),
                "appearance_encoder" : (magic_animate_checkpoints ,{
                    "default" : magic_animate_checkpoints[0]
                }),
                "motion_module" : (magic_animate_checkpoints ,{
                    "default" : magic_animate_checkpoints[0]
                }),
                "device" : (devices,),
            },
        }

    RETURN_TYPES = ("MAGIC_ANIMATE_MODEL",)

    FUNCTION = "load_model"

    CATEGORY = "ComfyUI Magic Animate"

    def load_model(self, controlnet, appearance_encoder, motion_module, device):
        if self.models:
            # delete old models
            for key in self.models.keys():
                # clear memory
                del self.models[key]
            self.models = {}
            gc.collect()

        current_dir = os.path.dirname(os.path.realpath(__file__))
        config  = OmegaConf.load(os.path.join(current_dir, "configs", "prompts", "animation.yaml"))
        inference_config = OmegaConf.load(os.path.join(current_dir, "configs", "inference", "inference.yaml"))
        magic_animate_models_dir = folder_paths.get_folder_paths("magic_animate")[0]
        
        config.pretrained_model_path = os.path.join(magic_animate_models_dir, "stable-diffusion-v1-5")
        config.pretrained_vae_path = os.path.join(magic_animate_models_dir, "sd-vae-ft-mse")
        
        config.pretrained_appearance_encoder_path = os.path.join(magic_animate_models_dir, os.path.dirname(appearance_encoder))
        config.pretrained_controlnet_path = os.path.join(magic_animate_models_dir, os.path.dirname(controlnet))
        motion_module = os.path.join(magic_animate_models_dir, motion_module)
        config.motion_module = motion_module

        ### >>> create animation pipeline >>> ###
        tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
        if config.pretrained_unet_path:
            unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_unet_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        else:
            unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        
        appearance_encoder = AppearanceEncoderModel.from_pretrained(config.pretrained_appearance_encoder_path).to(device)
        reference_control_writer = ReferenceAttentionControl(appearance_encoder, do_classifier_free_guidance=True, mode='write', fusion_blocks=config.fusion_blocks)
        
        reference_control_reader = ReferenceAttentionControl(unet, do_classifier_free_guidance=True, mode='read', fusion_blocks=config.fusion_blocks)
        if config.pretrained_vae_path is not None:
            vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
        else:
            vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")

        ### Load controlnet
        controlnet   = ControlNetModel.from_pretrained(config.pretrained_controlnet_path)

        # unet.enable_xformers_memory_efficient_attention()
        # appearance_encoder.enable_xformers_memory_efficient_attention()
        # controlnet.enable_xformers_memory_efficient_attention()

        vae.to(torch.float16)
        unet.to(torch.float16)
        text_encoder.to(torch.float16)
        appearance_encoder.to(torch.float16)
        controlnet.to(torch.float16)

        pipeline = AnimationPipeline(
            vae=vae, 
            text_encoder=text_encoder, 
            tokenizer=tokenizer, 
            unet=unet, 
            controlnet=controlnet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(
                    inference_config.noise_scheduler_kwargs
                )
            ),
        )

        # 1. unet ckpt
        # 1.1 motion module
        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        # if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
        motion_module_state_dict = motion_module_state_dict['state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
        try:
            # extra steps for self-trained models
            state_dict = OrderedDict()
            for key in motion_module_state_dict.keys():
                if key.startswith("module."):
                    _key = key.split("module.")[-1]
                    state_dict[_key] = motion_module_state_dict[key]
                else:
                    state_dict[key] = motion_module_state_dict[key]
            motion_module_state_dict = state_dict
            del state_dict
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
        except:
            _tmp_ = OrderedDict()
            for key in motion_module_state_dict.keys():
                if "motion_modules" in key:
                    if key.startswith("unet."):
                        _key = key.split('unet.')[-1]
                        _tmp_[_key] = motion_module_state_dict[key]
                    else:
                        _tmp_[key] = motion_module_state_dict[key]
            missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
            assert len(unexpected) == 0
            del _tmp_
        del motion_module_state_dict

        pipeline.to(device)

        self.models['vae'] = vae
        self.models['text_encoder'] = text_encoder
        self.models['appearance_encoder'] = appearance_encoder
        self.models['tokenizer'] = tokenizer
        self.models['unet'] = unet
        self.models['controlnet'] = controlnet
        self.models['pipeline'] = pipeline
        self.models['config'] = config
        self.models['reference_control_writer'] = reference_control_writer
        self.models['reference_control_reader'] = reference_control_reader

        return (self.models,)

class MagicAnimate:
    def __init__(self):
        self.generator = torch.Generator(device=torch.device("cuda:0"))
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "magic_animate_model": ("MAGIC_ANIMATE_MODEL",),
                "image" : ("IMAGE",),
                "pose_video" : ("IMAGE",),
                "seed" : ("INT", {
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "inference_steps" : ("INT", {
                    "default" : 25,
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",) #() #("IMAGE",)
    # OUTPUT_NODE = True

    FUNCTION = "generate"

    CATEGORY = "ComfyUI Magic Animate"

    def resize_image_frame(self, image_tensor, size):
        # permute to C x H x W
        image_tensor = rearrange(image_tensor, 'h w c -> c h w')
        # print(image.shape)
        image_tensor = ToPILImage()(image_tensor)
        image_tensor = image_tensor.resize((size, size))
        image_tensor = ToTensor()(image_tensor)
        # permute back to H x W x C
        image_tensor = rearrange(image_tensor, 'c h w -> h w c')
        return image_tensor


    def generate(self, magic_animate_model, image, pose_video, seed, inference_steps):
        num_actual_inference_steps = inference_steps

        pipeline = magic_animate_model['pipeline']
        config = magic_animate_model['config']
        size = config.size
        appearance_encoder = magic_animate_model['appearance_encoder']
        reference_control_writer = magic_animate_model['reference_control_writer']
        reference_control_reader = magic_animate_model['reference_control_reader']

        assert image.shape[0] == 1, "Only one image input is supported"
        image = image[0]
        H, W, C = image.shape

        if H != size or W != size:
            # resize image to be (size, size)
            image = self.resize_image_frame(image, size)
            # print(image.shape)
            H, W, C = image.shape
        
        prompt = ""
        n_prompt = ""
        control = pose_video.detach().cpu().numpy() # (num_frames, H, W, C)
        # print("control shape:", control.shape)

        if control.shape[1] != size or control.shape[2] != size:
            # resize each frame in control to be (size, size)
            control = torch.stack([self.resize_image_frame(frame, size) for frame in control], dim=0)

        init_latents = None

        original_length = control.shape[0]
        if control.shape[0] % config.L > 0:
            control = np.pad(control, ((0, config.L-control.shape[0] % config.L), (0, 0), (0, 0), (0, 0)), mode='edge')
        
        self.generator.manual_seed(seed)

        dist_kwargs = {"rank":0, "world_size":1, "dist":False}

        sample = pipeline(
            prompt,
            negative_prompt         = n_prompt,
            num_inference_steps     = config.steps,
            guidance_scale          = config.guidance_scale,
            width                   = W,
            height                  = H,
            video_length            = len(control),
            controlnet_condition    = control,
            init_latents            = init_latents,
            generator               = self.generator,
            num_actual_inference_steps = num_actual_inference_steps,
            appearance_encoder       = appearance_encoder, 
            reference_control_writer = reference_control_writer,
            reference_control_reader = reference_control_reader,
            source_image             = image.detach().cpu().numpy(),
            **dist_kwargs,
        ).videos

        sample = sample[0, :, :original_length] # shape: (C, num_frames, H, W)

        # permute to (num_frames, H, W, C)
        sample = rearrange(sample, 'c f h w -> f h w c').detach().cpu()

        return (sample,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "MagicAnimateModelLoader" : MagicAnimateModelLoader,
    "MagicAnimate" : MagicAnimate,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicAnimateModelLoader" : "Load Magic Animate Model",
    "MagicAnimate" : "Magic Animate",
}