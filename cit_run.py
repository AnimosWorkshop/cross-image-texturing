import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, UniPCMultistepScheduler
from diffusers.training_utils import set_seed
from src.pipeline import StableSyncMVDPipeline
from src.cit_configs import *
from shutil import copy
from src.CIA.appearance_transfer_model import AppearanceTransferModel


# This part is copied from SyncMVD/run_experiment.py and prepares the pipeline.
# Need to make sure that the pipe receives the two meshes and appearance texture instead of only one mesh.

# opt is the options object that is created by parsing the command line arguments.
opt = parse_config()

def get_paths(opt):
	"""
	Generate and return the paths for mesh, mesh appearance, and output based on the given options.
	Args:
		opt (Namespace): A namespace object containing the options.
	Returns:
		tuple: A tuple containing three elements:
			- mesh_path (str): The absolute or relative path to the mesh file.
			- mesh_path_app (str): The absolute or relative path to the mesh appearance file.
			- output_root (str): The absolute path to the output directory.
	"""
    
	if opt.mesh_config_relative:
		mesh_path = join(dirname(opt.config), opt.mesh)
	else:
		mesh_path = abspath(opt.mesh)

	# app = appearance
	if opt.mesh_config_relative:
		mesh_path_app = join(dirname(opt.config), opt.mesh_app)
	else:
		mesh_path_app = abspath(opt.mesh_app)
	
	if opt.output:
		output_root = abspath(opt.output)
	else:
		output_root = dirname(opt.config)
  
	return mesh_path, mesh_path_app, output_root

def get_output_dir(opt, output_root, mesh_path):
	"""
	Generate and return the output directory path based on the given options.
	Args:
		opt (Namespace): A namespace object containing the options.
		output_root (str): The absolute path to the output root directory.
		mesh_path (str): The absolute or relative path to the mesh file.
	Returns:
		str: The absolute path to the output directory.
	"""
	output_name_components = []
	if opt.prefix and opt.prefix != "":
		output_name_components.append(opt.prefix)
	if opt.use_mesh_name:
		mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
		output_name_components.append(mesh_name)

	if opt.timeformat and opt.timeformat != "":
		output_name_components.append(datetime.now().strftime(opt.timeformat))
	output_name = "_".join(output_name_components)
	output_dir = join(output_root, output_name)
  
	return output_dir

def create_output_dir(output_dir:str):
	"""
	Create the output directory if it does not exist.
	Args:
		output_dir (str): The absolute path to the output directory.
	"""
	if not isdir(output_dir):
		os.mkdir(output_dir)
	else:
		print(f"Results exist in the output directory, use time string to avoid name collision.")
		exit(0)

def get_log_configurations(opt, output_dir):
	"""
	Generate and return the logging configurations based on the given options.
	Args:
		opt (Namespace): A namespace object containing the options.
	Returns:
		dict: A dictionary containing the logging configurations.
	"""
	logging_config = {
		"output_dir":output_dir, 
		# "output_name":None, 
		# "intermediate":False, 
		"log_interval":opt.log_interval,
		"view_fast_preview": opt.view_fast_preview,
		"tex_fast_preview": opt.tex_fast_preview,
	}
  
	return logging_config

def create_controlnet(opt):
	"""
	Create and return the controlnet based on the given options.
	Args:
		opt (Namespace): A namespace object containing the options.
	Returns:
		ControlNetModel: The controlnet model.
	"""
	if opt.cond_type == "normal":
		controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", variant="fp16", torch_dtype=torch.float16)
	elif opt.cond_type == "depth":
		controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float16)	
  
	return controlnet  

def create_pipe(controlnet):
	"""
	Create and return the pipeline based on the given options and controlnet.
	Args:
		opt (Namespace): A namespace object containing the options.
		controlnet (ControlNetModel): The controlnet model.
	Returns:
		StableDiffusionControlNetPipeline: The pipeline.
	"""
	pipe = StableDiffusionControlNetPipeline.from_pretrained(
		"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
	)
	pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

	return pipe

def create_model(opt, pipe):
	"""
	Create and return the appearance transfer model based on the given options, pipeline, and logging configurations.
	Args:
		opt (Namespace): A namespace object containing the options.
		pipe (StableDiffusionControlNetPipeline): The pipeline.
	Returns:
		AppearanceTransferModel: The appearance transfer model.
	"""
	syncmvd = StableSyncMVDPipeline(**pipe.components)
	model_cfg = RunConfig(opt.prompt)
	set_seed(model_cfg.seed)
	model = AppearanceTransferModel(model_cfg, pipe=syncmvd)

	return model

def run_the_pipeline(opt, model, mesh_path, mesh_path_app, logging_config):
	"""
	Executes the pipeline for cross-image texturing using the provided model and options.
	Args:
		opt (Namespace): A namespace containing various options and configurations for the pipeline.
		model (Model): The model to be used for the pipeline.
		mesh_path (str): Path to the primary mesh file.
		mesh_path_app (str): Path to the secondary mesh file for appearance.
		logging_config (dict): Configuration for logging.
	Returns:
		tuple: A tuple containing:
			- result_tex_rgb (Tensor): The resulting textured RGB image.
			- textured_views (list): A list of textured views.
			- v (Any): Additional output from the model's pipeline.
	"""

	result_tex_rgb, textured_views, v = model.pipe(
	prompt=opt.prompt,
	height=opt.latent_view_size*8,
	width=opt.latent_view_size*8,
	num_inference_steps=opt.steps,
	guidance_scale=opt.guidance_scale,
	negative_prompt=opt.negative_prompt,
	
	generator=torch.manual_seed(opt.seed),
	max_batch_size=48,
	controlnet_guess_mode=opt.guess_mode,
	controlnet_conditioning_scale = opt.conditioning_scale,
	controlnet_conditioning_end_scale= opt.conditioning_scale_end,
	control_guidance_start= opt.control_guidance_start,
	control_guidance_end = opt.control_guidance_end,
	guidance_rescale = opt.guidance_rescale,
	use_directional_prompt=True,

	mesh_path=mesh_path,
	mesh_transform={"scale":opt.mesh_scale},
	mesh_autouv=not opt.keep_mesh_uv,
	
    mesh_path_app=mesh_path_app,
	mesh_transform_app={"scale":opt.mesh_scale},
	mesh_autouv_app=not opt.keep_mesh_uv,

	camera_azims=opt.camera_azims,
	top_cameras=not opt.no_top_cameras,
	texture_size=opt.latent_tex_size,
	render_rgb_size=opt.rgb_view_size,
	texture_rgb_size=opt.rgb_tex_size,
	multiview_diffusion_end=opt.mvd_end,
	exp_start=opt.mvd_exp_start,
	exp_end=opt.mvd_exp_end,
	ref_attention_end=opt.ref_attention_end,
	shuffle_background_change=opt.shuffle_bg_change,
	shuffle_background_end=opt.shuffle_bg_end,

	logging_config=logging_config,
	cond_type=opt.cond_type,
	)

	return result_tex_rgb, textured_views, v


def main():
	mesh_path, mesh_path_app, output_root = get_paths(opt)
	output_dir = get_output_dir(opt, output_root, mesh_path)
	create_output_dir(output_dir)

	print(f"Saving to {output_dir}")

	copy(opt.config, join(output_dir, "config.yaml"))

	logging_config = get_log_configurations(opt, output_dir)
	controlnet = create_controlnet(opt)
	pipe = create_pipe(controlnet)
	model = create_model(opt, pipe, logging_config)
 
	result_tex_rgb, textured_views, v = run_the_pipeline(opt, model, mesh_path, mesh_path_app, logging_config)
	
	display(v)


if __name__ == "__main__":
	main()