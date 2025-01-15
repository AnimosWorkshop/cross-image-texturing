from datetime import datetime
import numpy as np
from diffusers.utils import numpy_to_pil
import torch
from SyncMVD.src.utils import decode_latents, get_rgb_texture, latent_preview

############################################
##### copied from SyncMVD/src/utils.py #####
############################################

# def get_rgb_texture(vae, uvp_rgb, latents):
# 	result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
# 	resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
# 	result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
# 	textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
# 	result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
# 	return result_tex_rgb, result_tex_rgb_output
	
# @torch.no_grad()
# def latent_preview(x):
# 	v1_4_latent_rgb_factors = torch.tensor([
# 		#   R        G        B
# 		[0.298, 0.207, 0.208],  # L1
# 		[0.187, 0.286, 0.173],  # L2
# 		[-0.158, 0.189, 0.264],  # L3
# 		[-0.184, -0.271, -0.473],  # L4
# 	], dtype=x.dtype, device=x.device)
# 	image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
# 	image = (image / 2 + 0.5).clamp(0, 1)
# 	image = image.float()
# 	image = image.cpu()
# 	image = image.numpy()
# 	return image
	
 
 

##################################
######### CIT_utils.py ###########
##################################

def show_views(views, dest_dir): # Working!
	view = views[0]
	rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
	print(rgb_image, rgb_image.shape)
	result_image = rgb_image
	numpy_to_pil(result_image)[0].save(f"{dest_dir}/show_views_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
	
	
# def show_latents(latent_images, dest_dir): # TODO not yet working
# 	decoded_results = []
# 	# for latent_images in latents:
# 	images = latent_preview(latent_images.cpu())
# 	images = np.concatenate([img for img in images], axis=1)
# 	decoded_results.append(images)
# 	result_image = np.concatenate(decoded_results, axis=0)
# 	numpy_to_pil(result_image)[0].save(f"{dest_dir}/show_latents_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")

def show_mesh(uvp, dest_dir):
	views = uvp.render_textured_views()
	show_views(views, dest_dir)

# def show_texture(uvp: 'UVProjection', dest_dir: str):
#     result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, pred_original_sample)
#     numpy_to_pil(result_tex_rgb_output)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")
	
	# texture_uv = uvp.mesh.textures
	# result_image = texture_uv.maps_list()[0].permute(1, 2, 0).cpu().numpy()
	# numpy_to_pil(result_image)[0].save(f"{dest_dir}/show_texture_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
 
 
 
 
###############################
##### copy to DBG console #####
###############################

from datetime import datetime
import numpy as np
from diffusers.utils import numpy_to_pil

lidor_dir = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor"

def show_views(views, dest_dir): # Working!
	view = views[0]
	rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
	print(rgb_image, rgb_image.shape)
	result_image = rgb_image
	numpy_to_pil(result_image)[0].save(f"{dest_dir}/show_views_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
 
def show_mesh(uvp, dest_dir):
	views = uvp.render_textured_views()
	show_views(views, dest_dir)