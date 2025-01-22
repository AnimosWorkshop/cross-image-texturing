from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
from diffusers.training_utils import set_seed
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from src.CIA.appearance_transfer_model import AppearanceTransferModel
from src.cit_configs import RunConfig
from src.pipeline import StableSyncMVDPipeline
from src.cit_utils import invert_images, load_size


views = [f"/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/face_view_{i}.jpg" for i in range(10)]
input_image_prompt = "Portrait photo of Kratos, god of war."

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float32)	
controlnet = controlnet.to("cuda:1")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
		"runwayml/stable-diffusion-v1-5",
        controlnet=controlnet, 
        torch_dtype=torch.float32,
	).to("cuda:1")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = lambda images, clip_input: (images, [False]*len(images))
cfg = RunConfig(input_image_prompt)
set_seed(cfg.seed)

def main():
    for i, view in enumerate(views):
        print(f"Processing view {i}")
        image = load_size(view, size=512)
        image = torch.tensor(image).permute(2, 0, 1).to(torch.float16)
        inverted_latents, _ = invert_images(pipe.to("cuda:0"), image.to("cuda:0"), cfg)
        torch.save(inverted_latents, f"/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/inverted/inverted_{i}.pt")
    
if __name__ == "__main__":
    main() 