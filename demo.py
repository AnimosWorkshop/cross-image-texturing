from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
from PIL import Image
import numpy as np
import torch


cond = Image.open("/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/face_cond_4.jpg")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler == DDIMScheduler.from_config(pipe.scheduler.config)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

latent = torch.load("/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/inverted/noisy/inverted_4.pt").to(torch.float16).unsqueeze(0)

res = pipe("Portrait photo of Kratos, god of war.", cond, num_inference_steps=20, latents=latent).images[0]

res.save('./holy_shit_that_works.png')


