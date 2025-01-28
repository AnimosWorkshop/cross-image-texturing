from demo import demo
from src.cit_utils import image_to_tensor, tensor_to_image, show_views
from src.project import UVProjection as UVP
from src.pipeline import get_conditioning_images # TODO move the function to here.
import torch 
from PIL import Image
from DANA_inversion_save_latents import Preprocess


def set_cameras(uvp_app, camera_centers, camera_azims=[-180, -135, -90, -45, 0, 45, 90, 135], top_cameras=True):
	camera_poses = []
	# attention_mask=[]
	centers = camera_centers

	cam_count = len(camera_azims)
	front_view_diff = 360
	back_view_diff = 360
	front_view_idx = 0
	back_view_idx = 0
	for i, azim in enumerate(camera_azims):
		if azim < 0:
			azim += 360
		camera_poses.append((0, azim))
		# attention_mask.append([(cam_count+i-1)%cam_count, i, (i+1)%cam_count])
		if abs(azim) < front_view_diff:
			front_view_idx = i
			front_view_diff = abs(azim)
		if abs(azim - 180) < back_view_diff:
			back_view_idx = i
			back_view_diff = abs(azim - 180)

	# Add two additional cameras for painting the top surfaces
	if top_cameras:
		camera_poses.append((30, 0))
		camera_poses.append((30, 180))

		# attention_mask.append([front_view_idx, cam_count])
		# attention_mask.append([back_view_idx, cam_count+1])
	# return camera_poses
	uvp_app.set_cameras_and_render_settings(camera_poses, centers=camera_centers, camera_distance=4.0)


def prepare_uvp(tex_app_path, mesh_path_app, texture_rgb_size_app = 1024, device = "cuda:0"):
	uvp_app = UVP(texture_size=texture_rgb_size_app, render_size=512, sampling_mode="nearest", channels=3, device=device)
	uvp_app.load_mesh(mesh_path_app, scale_factor=True, autouv=True)

	texture_image = Image.open(tex_app_path)
	texture_tensor = image_to_tensor(texture_image)
	uvp_app.set_texture_map(texture_tensor)
 
	return uvp_app
	
def invert_lidor(num_steps, image_tensor, prompt, device="cuda:0"):
    """
    @return: a 4D tensor of shape (num_steps,x,y,z), where the last index is the noise.
    """
    model = Preprocess(device=device, sd_version="1.5")
    image_pil = tensor_to_image(image_tensor)
    image = model.prepare_image(image_pil)
    inverted_latents = model.invert_latents(num_steps=num_steps, image=image)
    return inverted_latents

def reshape_latents(latents):
    """
    I want only the 100 noisy latents (all timesteps but 1)
	I want the order to be reversed: noise is at index 0 and clear is at 99
	I want to access the latents by [timestep, batch, channel, x, y]
    """
    return latents.permute(1, 0, 2, 3, 4)[:-1]#.flip(dims=[-1])
 
def get_inverted_views(tex_app_path, mesh_path_app, num_steps, prompt_for_inversion, save_path, camera_azims=[-180, -135, -90, -45, 0, 45, 90, 135], camera_centers=None) -> list:
	uvp_app = prepare_uvp(tex_app_path, mesh_path_app)
	set_cameras(uvp_app, camera_centers, camera_azims=camera_azims)
 
	app_views = uvp_app.render_textured_views() # List of 10 tensors, each is 1536x1536, but with 4 channels (last channel is mask)
	app_views = [view[:-1] for view in app_views]

	# show_views(app_views)

	fully_inverted_latents = []
	midproccesses = []
 
	for i in range(len(uvp_app.cameras)):
		image_tensor = app_views[i]
		# First index here is to grab the correct variable, second is to get the first(?) timestep
		inverted_x, mid = invert_lidor(num_steps=num_steps, image_tensor=image_tensor, prompt=prompt_for_inversion)
		# demo(prompt=prompt_for_inversion, latent=inverted_x) # TODO remove this line
		fully_inverted_latents.append(inverted_x)
		midproccesses.append(mid)
	fully_inverted_latents = torch.stack(fully_inverted_latents)
	midproccesses = torch.stack(midproccesses)
	midproccesses = reshape_latents(midproccesses)
	torch.save(fully_inverted_latents, save_path)
	torch.save(midproccesses, save_path.replace(".pt", "_midproccess.pt"))
	
# def get_cond(uvp_app):
#     conditioning_images_app, masks_app = get_conditioning_images(uvp_app, height, cond_type=)

if __name__ == "__main__":
	get_inverted_views(
		tex_app_path="/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/data/textured.png",
		mesh_path_app="/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/data/textured.obj",
		num_steps=100,
		prompt_for_inversion="Portrait photo of Kratos, god of war.",
		save_path="/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/cit_inveeeert.pt",
	)