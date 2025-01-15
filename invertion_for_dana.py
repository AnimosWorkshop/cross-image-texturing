from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, UniPCMultistepScheduler
from diffusers.training_utils import set_seed
import torch
from src.CIA.utils.latent_utils import invert_images

path_of_photo_for_invertion = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/show_views_at15Jan2025-152223.jpg"


controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float16)	
pipe = StableDiffusionControlNetPipeline.from_pretrained(
		"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
	)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
# invert_images(path_of_photo_for_invertion)