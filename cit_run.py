from src.cit_configs import *
opt = parse_config()

from src.cit_utils import concat_images_horizontally, show_views
import torch
from src.cit_utils import tensor_to_image
import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from typing import List
from IPython.display import display
from datetime import datetime
from shutil import copy
from PIL import Image

# This part is copied from SyncMVD/run_experiment.py and prepares the pipeline.
# Need to make sure that the pipe receives the two meshes and appearance texture instead of only one mesh.


def make_dirs_for_app_proccessing(output_dir):
	inversion_dir = join(output_dir, "inversion")
	cond_dir = join(output_dir, "cond")
	data_dir = join(output_dir, "data") # 'data' is the terminology of invertsion_with_controlnet for the views.

	if not isdir(inversion_dir):
		os.mkdir(inversion_dir)
	if not isdir(cond_dir):
		os.mkdir(cond_dir)
	if not isdir(data_dir):
		os.mkdir(data_dir)
  
	return inversion_dir, cond_dir, data_dir

def save_list_of_tensors(base_name:str, tensor_list, output_dir:str):
	for i, tensor in enumerate(tensor_list):
		torch.save(tensor, join(output_dir, f"{base_name}_{i}.pt"))

def save_list_of_images(base_name:str, image_list:List[Image.Image], output_dir:str):
	for i, image in enumerate(image_list):
		image.save(join(output_dir, f"{base_name}_{i}.png"))

def prepare_appearance(prompt, steps, tex_app, mesh_app, output_dir, seed, bg_path=None, invert_with_controlnet=True, cond_type="depth", camera_azims=None):
	from cit_invert import get_views_and_depth
	from argparse import Namespace
	if invert_with_controlnet:
		from inversion_with_controlnet import run
	else:
		from inversion_save_latents_wo_cn import run
 
	inversion_dir, cond_dir, data_dir = make_dirs_for_app_proccessing(output_dir)
	cond_app_path = join(cond_dir, "cond_app.pt")
 
	print(f"Saving appearance views to {data_dir}")
	print(f"Saving appearance conditioning images to {cond_dir}")
	print(f"Saving inverted appearance latents to {inversion_dir}")
 
	app_views, app_conds = get_views_and_depth(tex_app_path=tex_app, mesh_path_app=mesh_app, camera_azims=camera_azims, cond_type=cond_type, bg_path=bg_path)
	save_list_of_images("view", app_views, data_dir)
	
	depth_conditional_images = [tensor_to_image(image) for image in app_conds]
	save_list_of_images("cond", depth_conditional_images, cond_dir)


	# Create opt object directly
	opt = Namespace(
		data_path=data_dir,
		control_image_path=cond_dir,
		save_dir=inversion_dir,
		sd_version='1.5',
		seed=seed,
		steps=steps,
		save_steps=1000,  # or another value if needed
		inversion_prompt=prompt,
		lora_weights_path=None,
		square_size=False,
		extract_reverse=False,
	)
	recon_latents_path = run(opt) # Should look like "inversion/latents.pt"
 
	concat_images_horizontally(app_views).save(join(data_dir, "all_views.png"))
	torch.save(app_conds, cond_app_path)
 
	return recon_latents_path, cond_app_path



if opt.mesh_config_relative:
	mesh_path = join(dirname(opt.config), opt.mesh)
	mesh_path_app = join(dirname(opt.config), opt.mesh_app)
	tex_app = join(dirname(opt.config), opt.tex_app)
	latents_save_path = join(dirname(opt.config), opt.latents_save_path)
	cond_app_path = join(dirname(opt.config), opt.cond_app_path)
	bg_path = join(dirname(opt.config), opt.bg_path)
else:
	mesh_path = abspath(opt.mesh)
	mesh_path_app = abspath(opt.mesh_app)
	tex_app = abspath(opt.tex_app)
	latents_save_path = abspath(opt.latents_save_path)
	cond_app_path = abspath(opt.cond_app_path)
	bg_path = abspath(opt.bg_path)

# app = appearance

if opt.output:
	output_root = abspath(opt.output)
else:
	output_root = dirname(opt.config)

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

if not isdir(output_dir):
	os.mkdir(output_dir)
else:
	print(f"Results exist in the output directory, use time string to avoid name collision.")
	exit(0)
 
print(f"Saving to {output_dir}")
if not opt.latents_load:
	print("Calculating new inverted latents and appearance conditioning images.")
	latents_save_path, cond_app_path = prepare_appearance(opt.prompt, opt.steps, tex_app, mesh_path_app, output_dir, opt.seed, bg_path, opt.invert_with_controlnet, opt.cond_type, opt.camera_azims)
	print("Done calculating new inverted latents and appearance conditioning images.")
else:
    print("Using provided latents and appearance conditioning images.")


# Slow imports here
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, UniPCMultistepScheduler
from diffusers.training_utils import set_seed
from src.pipeline import StableSyncMVDPipeline
from src.CIA.appearance_transfer_model import AppearanceTransferModel


copy(opt.config, join(output_dir, "config.yaml"))

logging_config = {
	"output_dir":output_dir, 
	# "output_name":None, 
	# "intermediate":False, 
	"log_interval":opt.log_interval,
	"view_fast_preview": opt.view_fast_preview,
	"tex_fast_preview": opt.tex_fast_preview,
	}

if opt.cond_type == "normal":
	# controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", torch_dtype=torch.float16)
	raise NotImplementedError("Normal controlnet is not supported yet.")
elif opt.cond_type == "depth":
	controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)			

pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

syncmvd = StableSyncMVDPipeline(**pipe.components)


model_cfg = RunConfig(opt.prompt)
set_seed(model_cfg.seed)
model = AppearanceTransferModel(model_cfg, pipe=syncmvd)





# Run the SyncMVD pipeline
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
	tex_app_path=tex_app,	
 
    mesh_path_app=mesh_path_app,
	mesh_transform_app={"scale":opt.mesh_scale},
	mesh_autouv_app=not opt.keep_mesh_uv,
	
	latents_load=True,
	latents_save_path=latents_save_path,
	cond_app_path=cond_app_path,


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

	app_transfer_model=model,
	)

display(v)