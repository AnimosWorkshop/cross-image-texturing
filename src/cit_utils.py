from datetime import datetime
import numpy as np
from diffusers.utils import numpy_to_pil
import torch
from src.SyncMVD.src.utils import decode_latents, get_rgb_texture, latent_preview
from src.CIA.appearance_transfer_model import AppearanceTransferModel
from PIL import Image
from src.cit_configs import RunConfig
from src.CIA.utils.ddpm_inversion import invert

##################################
######### CIT_utils.py ###########
##################################

# def show_latents(latent_images, dest_dir): # TODO not yet working
# 	decoded_results = []
# 	# for latent_images in latents:
# 	images = latent_preview(latent_images.cpu())
# 	images = np.concatenate([img for img in images], axis=1)
# 	decoded_results.append(images)
# 	result_image = np.concatenate(decoded_results, axis=0)
# 	numpy_to_pil(result_image)[0].save(f"{dest_dir}/show_latents_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")

# def show_texture(uvp: 'UVProjection', dest_dir: str):
#     result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, pred_original_sample)
#     numpy_to_pil(result_tex_rgb_output)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")
	
	# texture_uv = uvp.mesh.textures
	# result_image = texture_uv.maps_list()[0].permute(1, 2, 0).cpu().numpy()
	# numpy_to_pil(result_image)[0].save(f"{dest_dir}/show_texture_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
 

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
 
###############################
##### copy to DBG console #####
###############################

from datetime import datetime
import numpy as np
from diffusers.utils import numpy_to_pil

lidor_dir = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor"

def show_views(views, dest_dir): # Working!
	result_images = []
	for view in views:
		rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
		print(rgb_image, rgb_image.shape)
		result_images.append(rgb_image)
	concatenated_image = np.concatenate(result_images, axis=1)
	numpy_to_pil(concatenated_image)[0].save(f"{dest_dir}/show_views_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
 
def show_mesh(uvp, dest_dir):
	views = uvp.render_textured_views()
	show_views(views, dest_dir)