from datetime import datetime
import numpy as np
from diffusers.utils import numpy_to_pil
import torch
from PIL import Image

lidor_dir = "/home/ML_courses/03683533_2024/lidor_yael_snir/new_semester/cross-image-texturing/results/lidor"


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

def concat_images_vertically(images):
    # Get total height and maximum width
	total_height = sum(img.height for img in images)
	max_width = max(img.width for img in images)

	# Create a new blank image with the right size
	concatenated_img = Image.new("RGB", (max_width, total_height))

	# Paste images on top of each other
	y_offset = 0
	for img in images:
		concatenated_img.paste(img, (0, y_offset))
		y_offset += img.height

	return concatenated_img

def concat_images_horizontally(images):
    if type(images) == torch.Tensor:
        images = [tensor_to_image(img) for img in images]
    # Get total width and maximum height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the right size
    concatenated_img = Image.new("RGB", (total_width, max_height))

    # Paste images side by side
    x_offset = 0
    for img in images:
        concatenated_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return concatenated_img

def image_to_tensor(image):
    return (torch.from_numpy(np.array(image)) / 255.0).permute(2, 0, 1)

def tensor_to_image(tensor):
    return Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def show_views(views, dest_dir=lidor_dir, save=True): # Deprecated, can't remember what it does
	result_images = []
	for view in views:
		rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
		print(rgb_image, rgb_image.shape)
		result_images.append(rgb_image)
	concatenated_image = np.concatenate(result_images, axis=1)
	res_image = numpy_to_pil(concatenated_image)[0]
	if save:
		res_image.save(f"{dest_dir}/show_views_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
	return res_image
 
def save_all_views(views, dest_dir=lidor_dir):
	for i, view in enumerate(views):
		rgb_image = view[:3].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)
		numpy_to_pil(rgb_image)[0].save(f"{dest_dir}/face_view_{i}.jpg")
 
def show_mesh(uvp, dest_dir=lidor_dir, save=True, texture=None): #TODO what if the mesh is rendered in latent space?
	"""uvp can be a path to a saved model or a UVP object."""

	print(uvp)
	print(texture)

	if type(uvp) == str:
		from uvp_utils import build_uvp
		# if not texture:
		# 	texture = Image.new("RGB", (1024, 1024), "white")
		uvp = build_uvp(uvp, texture)
	device = uvp.device
  
	uvp.to("cuda:0")
	views = uvp.render_textured_views()
	uvp.to(device)

	return show_views(views, dest_dir, save)
 
def show_latents(latents, dest_dir=lidor_dir, save=True):
	"""
	Latents can be a tensor of shape (N, L) or (L,), or path.
	"""
	if isinstance(latents, str):
		latents = torch.load(latents)
 
	if (len(latents.shape) == 3):
		latents = latents.unsqueeze(0).unsqueeze(0)
	elif (len(latents.shape) == 4):
		latents = latents.unsqueeze(0)

	views = []
	for view in latents:
		view = view.to(torch.float16).to("cuda:0")
		decoded_latents = latent_preview(view)
		concatenated_image = np.concatenate(decoded_latents, axis=1)
		views.append(concatenated_image)
	concatenated_image = np.concatenate(views, axis=0)
	res_image = numpy_to_pil(concatenated_image)[0]
	if save:
		res_image.save(f"{dest_dir}/show_latent_at{datetime.now().strftime('%d%b%Y-%H%M%S')}.jpg")
	return res_image