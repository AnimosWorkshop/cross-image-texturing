from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler
from diffusers.training_utils import set_seed
import numpy as np
import torch
from PIL import Image

from src.CIA.appearance_transfer_model import AppearanceTransferModel
from src.cit_configs import RunConfig
from src.pipeline import StableSyncMVDPipeline
from src.CIA.utils.latent_utils import invert_images

def load_size(image_path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512) -> Image.Image:
    if isinstance(image_path, (str)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))  
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    h, w, c = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = np.array(Image.fromarray(image).resize((size, size)))
    return image



path_of_photo_for_invertion = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/show_views_at15Jan2025-152223.jpg"


controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float16)	
pipe = StableDiffusionControlNetPipeline.from_pretrained(
		"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
	)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

syncmvd = StableSyncMVDPipeline(**pipe.components)

model_cfg = RunConfig("Portrait photo of Kratos, god of war.")
set_seed(model_cfg.seed)
model = AppearanceTransferModel(model_cfg, pipe=syncmvd)
# invert_images(path_of_photo_for_invertion)

# image = Image.open(path_of_photo_for_invertion).convert("RGB")

# image_array = np.array(image).astype(np.float32)
# image_array = image_array / 127.5 - 1.0
# # photo_tensor.shape == [1, 3, 1536, 1536]

image = load_size(path_of_photo_for_invertion)
image_tensor = torch.from_numpy(image).permute(2, 0, 1).to("cuda:0") 


inverted = invert_images(model.pipe, image_tensor, struct_image=None, cfg=model.config)