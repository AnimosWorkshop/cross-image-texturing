from src.cit_utils import image_to_tensor, show_latents, tensor_to_image, show_views
from src.project import UVProjection as UVP
from PIL import Image
from src.pipeline import get_conditioning_images



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

def reshape_latents(latents):
    """
    I want only the 100 noisy latents (all timesteps but 1)
	I want the noise to be at index 0 and clear is at the last index
	I want to access the latents by [timestep, batch, channel, x, y]
    """
    # show_latents(latents.permute(1, 0, 2, 3, 4)[-1]) #This is the clear image
    return latents.permute(1, 0, 2, 3, 4)#[:-1]
 
def reshape_inverted_latents(latents):
	"""
	I want only the 100 noisy latents (all timesteps but 1)
	I want the noise to be at index 0 and clear is at the last index
	I want to access the latents by [timestep, batch, channel, x, y]
	"""
	return latents.permute(1, 0, 2, 3, 4)[1:].flip(dims=[-1])

def get_views_and_depth(tex_app_path, mesh_path_app, camera_azims=[-180, -135, -90, -45, 0, 45, 90, 135], camera_centers=None, height=512,cond_type="depth"):
	"""
 	Returns a list of PIL Images, each is a view of the mesh with the texture applied, and a tensot, which is a collection of the depth maps of the mesh"""
	uvp_app = prepare_uvp(tex_app_path, mesh_path_app)
	set_cameras(uvp_app, camera_centers, camera_azims=camera_azims)
 
	app_views = uvp_app.render_textured_views() # List of 10 tensors, each is 1536x1536, but with 4 channels (last channel is mask)
	app_views = [tensor_to_image(view[:-1]) for view in app_views]
 
	conditional_images, _ = get_conditioning_images(uvp_app, height, cond_type=cond_type)
	
	return app_views, conditional_images
