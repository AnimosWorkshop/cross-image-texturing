from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
from PIL import Image
import numpy as np
import torch

def demo_wrapper(prompt, latent_path):
    latent = torch.load(latent_path).to(torch.float16)
    demo(prompt, latent)

def demo(prompt, latent):
    """
    'latent' is a tensor of shape [1, 4, 64, 64] or [4, 64, 64]
    """
    cond = Image.open("/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/face_cond_4.jpg")

    if len(latent.shape) == 3:
        latent = latent.unsqueeze(0)

    latent = latent.to(torch.float16)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler == DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()    

    res = pipe(prompt, cond, num_inference_steps=100, latents=latent).images[0]

    res.save('./holy_shit_that_works.png')
    
    
if __name__ == "__main__":
    latent_path = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/dana/26Jan2025-093242_reverse/face_view_4/noisy_latents_991.pt"
    prompt = "Portrait photo of Kratos, god of war."
    demo_wrapper(latent_path=latent_path,
         prompt=prompt)

