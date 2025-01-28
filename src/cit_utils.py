from datetime import datetime
import numpy as np
from diffusers.utils import numpy_to_pil, randn_tensor
import torch
from src.SyncMVD.src.utils import decode_latents, get_rgb_texture
from src.CIA.appearance_transfer_model import AppearanceTransferModel
from PIL import Image
from src.cit_configs import RunConfig
from src.CIA.utils.ddpm_inversion import invert


##################################
######### CIT_utils.py ###########
##################################

# A fast decoding method based on linear projection of latents to rgb
@torch.no_grad()
def latent_preview(x):
	# adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
	v1_4_latent_rgb_factors = torch.tensor([
		#   R        G        B
		[0.298, 0.207, 0.208],  # L1
		[0.187, 0.286, 0.173],  # L2
		[-0.158, 0.189, 0.264],  # L3
		[-0.184, -0.271, -0.473],  # L4
	], dtype=x.dtype, device=x.device)
	image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	return image


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


def invert_images(sd_model: AppearanceTransferModel, app_image: Image.Image, cfg: RunConfig):
	if app_image is None:
		input_app =None
	elif torch.is_tensor(app_image):
		input_app = app_image.to(torch.float16) / 127.5 - 1.0
	else:
		input_app = torch.from_numpy(np.array(app_image)).to(torch.float16) / 127.5 - 1.0
	
	if input_app is None:
		zs_app, latents_app = None, None
	else:
		assert input_app.shape[1] == input_app.shape[2] == 512
		zs_app, latents_app = invert(x0=input_app.unsqueeze(0),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5)
        
	return latents_app, zs_app


'''
	Customized Step Function
	step on texture
	texture
'''
@torch.no_grad()
def step_tex(
		scheduler,
		uvp,
		model_output: torch.FloatTensor,
		timestep: int,
		prev_t: int,
		sample: torch.FloatTensor,
		texture: None,
		generator=None,
		return_dict: bool = True,
		guidance_scale = 1,
		main_views = [],
		hires_original_views = True,
		exp=None,
		cos_weighted=True,
		eta=None,
		is_app=False
):
	step_results = scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=eta,
        generator=generator,
        return_dict=return_dict,
    )
	pred_prev_sample = step_results['prev_sample']
	pred_original_sample = step_results['pred_original_sample']

	# Step texture
	if not is_app:
		prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
		alpha_prod_t = scheduler.alphas_cumprod[timestep]
		alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
		beta_prod_t = 1 - alpha_prod_t
		variance = scheduler._get_variance(timestep, prev_timestep)
		std_dev_t = eta * variance ** (0.5)

		if texture is None:
			sample_views = [view for view in sample]
			sample_views, texture, _ = uvp.bake_texture(views=sample_views, main_views=main_views, exp=exp)
			sample_views = torch.stack(sample_views, axis=0)[:,:-1,...]

		original_views = [view for view in pred_original_sample]
		original_views, original_tex, visibility_weights = uvp.bake_texture(views=original_views, main_views=main_views, exp=exp)
		uvp.set_texture_map(original_tex)
		original_views = uvp.render_textured_views()
		original_views = torch.stack(original_views, axis=0)[:,:-1,...]

		pred_tex_epsilon = (texture - alpha_prod_t ** (0.5) * original_tex) / beta_prod_t ** (0.5)
		pred_tex_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_tex_epsilon
		prev_tex = alpha_prod_t_prev ** (0.5) * original_tex + pred_tex_direction

	if not is_app:
		uvp.set_texture_map(prev_tex)
		prev_views = uvp.render_textured_views()
	pred_prev_sample = torch.clone(sample)
	if not is_app:
		for i, view in enumerate(prev_views):
			pred_prev_sample[i] = view[:-1]
		masks = [view[-1:] for view in prev_views]

	if not is_app:
		return {"prev_sample": pred_prev_sample, "pred_original_sample":pred_original_sample, "prev_tex": prev_tex}
	else:
		return {"prev_sample": pred_prev_sample, "pred_original_sample":pred_original_sample}

 
###############################
##### copy to DBG console #####
###############################

from datetime import datetime
import numpy as np
from diffusers.utils import numpy_to_pil
from src.SyncMVD.src.utils import decode_latents

lidor_dir = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor"

def image_to_tensor(image):
    return (torch.from_numpy(np.array(image)) / 255.0).permute(2, 0, 1)

def tensor_to_image(tensor):
    return Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def show_views(views, dest_dir=lidor_dir): # Working!
	result_images = []
	for view in views:
		rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
		print(rgb_image, rgb_image.shape)
		result_images.append(rgb_image)
	concatenated_image = np.concatenate(result_images, axis=1)
	numpy_to_pil(concatenated_image)[0].save(f"{dest_dir}/show_views_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
 
def save_all_views(views, dest_dir=lidor_dir):
	for i, view in enumerate(views):
		rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
		numpy_to_pil(rgb_image)[0].save(f"{dest_dir}/face_view_{i}.jpg")
 
def show_mesh(uvp, dest_dir=lidor_dir):
	views = uvp.render_textured_views()
	show_views(views, dest_dir)
 
def show_latents(latents, dest_dir=lidor_dir):
	"""
	Latents can be a tensor of shape (N, L) or (L,), or path.
	"""
	if isinstance(latents, str):
		latents = torch.load(latents)
 
	if (len(latents.shape) == 3):
		latents = latents.unsqueeze(0).unsqueeze(0)
	elif (len(latents.shape) == 4):
		latents = latents.unsqueeze(0)

	# latents = latents.to(torch.float16).to("cuda:0")
	# decoded_latents = latent_preview(latents)
	# concatenated_image = np.concatenate(decoded_latents, axis=1)
	# numpy_to_pil(concatenated_image)[0].save(f"{dest_dir}/show_latent_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
 
	# if len(latents.shape) != 5:
	# 	return
	views = []
	for view in latents:
		view = view.to(torch.float16).to("cuda:0")
		decoded_latents = latent_preview(view)
		concatenated_image = np.concatenate(decoded_latents, axis=1)
		views.append(concatenated_image)
	concatenated_image = np.concatenate(views, axis=0)
	numpy_to_pil(concatenated_image)[0].save(f"{dest_dir}/show_latent_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
 
		