import os
import folder_paths
import sys

magic_animate_models_dir = os.path.join(folder_paths.models_dir, "MagicAnimate")
os.makedirs(magic_animate_models_dir, exist_ok=True)

if not os.path.exists(os.path.join(magic_animate_models_dir, "appearance_encoder")) or not os.path.exists(os.path.join(magic_animate_models_dir, "densepose_controlnet")) or not os.path.exists(os.path.join(magic_animate_models_dir, "temporal_attention")):
    print("Downloading Magic Animate models...")
    assert os.system("git lfs install") == 0, "ERROR: Git LFS is not installed. Please install it and restart ComfyUI."
    assert os.system(f"cd {folder_paths.models_dir} && git clone https://huggingface.co/zcxu-eric/MagicAnimate") == 0, "ERROR: Failed to download Magic Animate models. Please check your internet connection and restart ComfyUI."

if not os.path.exists(os.path.join(magic_animate_models_dir, "stable-diffusion-v1-5")):
    print("Downloading stable-diffusion-v1-5 model from Huggingface...")
    assert os.system(f"cd {magic_animate_models_dir} && git clone https://huggingface.co/runwayml/stable-diffusion-v1-5") == 0, "ERROR: Failed to download stable-diffusion-v1-5 model from Huggingface. Please check your internet connection and restart ComfyUI."
    
if not os.path.exists(os.path.join(magic_animate_models_dir, "sd-vae-ft-mse")):
    print("Downloading sd-vae-ft-mse model from Huggingface...")
    assert os.system(f"cd {magic_animate_models_dir} && git clone https://huggingface.co/stabilityai/sd-vae-ft-mse") == 0, "ERROR: Failed to download sd-vae-ft-mse model from Huggingface. Please check your internet connection and restart ComfyUI."

control_v11p_sd15_openpose_path = os.path.join(magic_animate_models_dir, "control_v11p_sd15_openpose")
if not os.path.exists(control_v11p_sd15_openpose_path):
    print("Downloading control_v11p_sd15_openpose model from Huggingface...")
    os.makedirs(control_v11p_sd15_openpose_path, exist_ok=True)
    assert os.system(f"cd {control_v11p_sd15_openpose_path} && wget --content-disposition -q --show-progress 'https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors?download=true'") == 0, "ERROR: Failed to download control_v11p_sd15_openpose model from Huggingface. Please check your internet connection and restart ComfyUI."
    assert os.system(f"cd {control_v11p_sd15_openpose_path} && wget --content-disposition -q --show-progress 'https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/config.json?download=true'") == 0, "ERROR: Failed to download control_v11p_sd15_openpose model from Huggingface. Please check your internet connection and restart ComfyUI."
    