import os
import folder_paths
import sys
import urllib.request

magic_animate_models_dir = os.path.join(folder_paths.models_dir, "MagicAnimate")
os.makedirs(magic_animate_models_dir, exist_ok=True)

appearance_encoder_path = os.path.join(magic_animate_models_dir, "appearance_encoder")
densepose_controlnet_path = os.path.join(magic_animate_models_dir, "densepose_controlnet")
temporal_attention_path = os.path.join(magic_animate_models_dir, "temporal_attention")

def download_file(url, save_path):
    with urllib.request.urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read())

if not os.path.exists(appearance_encoder_path):
    print("Downloading Magic Animate's appearance encoder...")
    os.makedirs(appearance_encoder_path, exist_ok=True)
    download_file('https://huggingface.co/zcxu-eric/MagicAnimate/resolve/main/appearance_encoder/diffusion_pytorch_model.safetensors?download=true', os.path.join(appearance_encoder_path, 'diffusion_pytorch_model.safetensors'))
    download_file('https://huggingface.co/zcxu-eric/MagicAnimate/resolve/main/appearance_encoder/config.json?download=true', os.path.join(appearance_encoder_path, 'config.json'))

if not os.path.exists(densepose_controlnet_path):
    print("Downloading Magic Animate's densepose controlnet...")
    os.makedirs(densepose_controlnet_path, exist_ok=True)
    download_file('https://huggingface.co/zcxu-eric/MagicAnimate/resolve/main/densepose_controlnet/diffusion_pytorch_model.safetensors?download=true', os.path.join(densepose_controlnet_path, 'diffusion_pytorch_model.safetensors'))
    download_file('https://huggingface.co/zcxu-eric/MagicAnimate/resolve/main/densepose_controlnet/config.json?download=true', os.path.join(densepose_controlnet_path, 'config.json'))

if not os.path.exists(temporal_attention_path):
    print("Downloading Magic Animate's temporal attention...")
    os.makedirs(temporal_attention_path, exist_ok=True)
    download_file('https://huggingface.co/zcxu-eric/MagicAnimate/resolve/main/temporal_attention/temporal_attention.ckpt?download=true', os.path.join(temporal_attention_path, 'temporal_attention.ckpt'))

sd_15_path = os.path.join(magic_animate_models_dir, "stable-diffusion-v1-5")
if not os.path.exists(sd_15_path):
    print("Downloading stable-diffusion-v1-5 checkpoints from Huggingface...")
    os.makedirs(sd_15_path, exist_ok=True)
    os.makedirs(os.path.join(sd_15_path, "text_encoder"), exist_ok=True)
    os.makedirs(os.path.join(sd_15_path, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(sd_15_path, "unet"), exist_ok=True)

    download_file('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/config.json?download=true', os.path.join(sd_15_path, 'text_encoder', 'config.json'))
    download_file('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/model.safetensors?download=true', os.path.join(sd_15_path, 'text_encoder', 'model.safetensors'))
    
    download_file('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt?download=true', os.path.join(sd_15_path, 'tokenizer', 'merges.txt'))
    download_file('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/special_tokens_map.json?download=true', os.path.join(sd_15_path, 'tokenizer', 'special_tokens_map.json'))
    download_file('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/tokenizer_config.json?download=true', os.path.join(sd_15_path, 'tokenizer', 'tokenizer_config.json'))
    download_file('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json?download=true', os.path.join(sd_15_path, 'tokenizer', 'vocab.json'))

    download_file('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json?download=true', os.path.join(sd_15_path, 'unet', 'config.json'))
    download_file('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin?download=true', os.path.join(sd_15_path, 'unet', 'diffusion_pytorch_model.bin'))

sd_vae_ft_mse_path = os.path.join(magic_animate_models_dir, "sd-vae-ft-mse")
if not os.path.exists(sd_vae_ft_mse_path):
    print("Downloading sd-vae-ft-mse model from Huggingface...")
    os.makedirs(sd_vae_ft_mse_path, exist_ok=True)
    download_file('https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json?download=true', os.path.join(sd_vae_ft_mse_path, 'config.json'))
    download_file('https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors?download=true', os.path.join(sd_vae_ft_mse_path, 'diffusion_pytorch_model.safetensors'))

control_v11p_sd15_openpose_path = os.path.join(magic_animate_models_dir, "control_v11p_sd15_openpose")
if not os.path.exists(control_v11p_sd15_openpose_path):
    print("Downloading control_v11p_sd15_openpose model from Huggingface...")
    os.makedirs(control_v11p_sd15_openpose_path, exist_ok=True)
    download_file('https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors?download=true', os.path.join(control_v11p_sd15_openpose_path, 'diffusion_pytorch_model.safetensors'))
    download_file('https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/config.json?download=true', os.path.join(control_v11p_sd15_openpose_path, 'config.json'))
