from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
from diffusers.training_utils import set_seed
import numpy as np
import torch
from PIL import Image
from pathlib import Path
# from src.cit_utils import show_latent, show_latents, lidor_dir
from src.CIA.appearance_transfer_model import AppearanceTransferModel
from src.cit_configs import RunConfig
from src.pipeline import StableSyncMVDPipeline
from src.CIA.utils.latent_utils import invert_images

lidor_dir = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor"

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


# def save_step_callback(step, timestep, latents):
#     # Decode latents to image
#     print("hi")
#     images = pipe.decode_latents(latents)
#     for i, img in enumerate(images):
#         pil_image = Image.fromarray((img * 255).astype("uint8"))
#         pil_image.save(f"step_{step:03d}_image_{i}.png")

def main():
    path_of_photo_for_invertion = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/face_view_4.jpg"
    path_of_condition_photo = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/face_cond_4.jpg"
    input_image_prompt = "Portrait photo of Kratos, god of war."




    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float32)	
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet, 
            torch_dtype=torch.float32,
            # callback=save_step_callback,
            # callback_steps=1  # Call the callback at every step
        )
    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler.prediction_type = "sample"


    # syncmvd = StableSyncMVDPipeline(**pipe.components)

    model_cfg = RunConfig(input_image_prompt)
    set_seed(69)
    # model = AppearanceTransferModel(model_cfg, pipe=syncmvd)
    # model.config.latents_path = Path(model.config.output_path) / "latents"
    # model.config.latents_path.mkdir(parents=True, exist_ok=True)

    image = load_size(path_of_photo_for_invertion)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).to("cuda:0") 
    condition_photo = load_size(path_of_condition_photo)


    inverted_latents, _, _, _ = invert_images(pipe.to("cuda:0"), image_tensor, struct_image=None, cfg=model_cfg)
    # inverted_latents = torch.load(lidor_dir + "/inverted_latents.pt")

    pipe.safety_checker = lambda images, clip_input: (images, [False]*len(images))
    res = pipe(input_image_prompt, image = [condition_photo], latents=inverted_latents[-1][None].to(torch.float16), num_inference_steps=model_cfg.num_timesteps, guidance_scale=model_cfg.swap_guidance_scale)
    image = res.images[0]  # Assuming res.images[0] is in the format expected
    image.save("reconstructed.png")

    print("yeah")
    
if __name__ == "__main__":
    main()