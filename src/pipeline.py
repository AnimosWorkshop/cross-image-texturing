import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from IPython.display import display
import numpy as np
import math
import random
import torch
import copy
from torch import functional as F
from torch import nn
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import (
	BaseOutput, 
	randn_tensor, 
	numpy_to_pil,
	pt_to_pil,
	# make_image_grid,
	is_accelerate_available,
	is_accelerate_version,
	is_compiled_module,
	logging,
	randn_tensor,
	replace_example_docstring
	)

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.training_utils import set_seed

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from src.project import UVProjection as UVP


from src.SyncMVD.src.syncmvd.attention import SamplewiseAttnProcessor2_0, replace_attention_processors
from src.SyncMVD.src.syncmvd.prompt import *
from src.SyncMVD.src.utils import *

from src.CIA.appearance_transfer_model import AppearanceTransferModel
from src.cit_configs import Range, RunConfig
from src.cit_utils import invert_images, show_latents, step_tex, show_views, save_all_views

from datetime import datetime


if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")

# Background colors
color_constants = {"black": [-1, -1, -1], "white": [1, 1, 1], "maroon": [0, -1, -1],
			"red": [1, -1, -1], "olive": [0, 0, -1], "yellow": [1, 1, -1],
			"green": [-1, 0, -1], "lime": [-1 ,1, -1], "teal": [-1, 0, 0],
			"aqua": [-1, 1, 1], "navy": [-1, -1, 0], "blue": [-1, -1, 1],
			"purple": [0, -1 , 0], "fuchsia": [1, -1, 1]}
color_names = list(color_constants.keys())


# Used to generate depth or normal conditioning images
@torch.no_grad()
def get_conditioning_images(uvp, output_size, render_size=512, blur_filter=5, cond_type="normal"):
	verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(image_size=render_size)
	masks = normals[...,3][:,None,...]
	masks = Resize((output_size//8,)*2, antialias=True)(masks)
	normals_transforms = Compose([
		Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True), 
		GaussianBlur(blur_filter, blur_filter//3+1)]
	)

	if cond_type == "normal":
		view_normals = uvp.decode_view_normal(normals).permute(0,3,1,2) *2 - 1
		conditional_images = normals_transforms(view_normals)
	# Some problem here, depth controlnet don't work when depth is normalized
	# But it do generate using the unnormalized form as below
	elif cond_type == "depth":
		view_depths = uvp.decode_normalized_depth(depths).permute(0,3,1,2)
		conditional_images = normals_transforms(view_depths)
	
	return conditional_images, masks


# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t):
	composited_images = []
	for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
		if t > 0:
			alphas_cumprod = scheduler.alphas_cumprod[t]
			noise = torch.normal(0, 1, background.shape, device=background.device)
			background = (1-alphas_cumprod) * noise + alphas_cumprod * background
		composited = foreground * mask + background * (1-mask)
		composited_images.append(composited)
	composited_tensor = torch.stack(composited_images)
	return composited_tensor


# Split into micro-batches to use less memory in each unet prediction
# But need more investigation on reducing memory usage
# Assume it has no positive effect and use a large "max_batch_size" to skip splitting
def split_groups(attention_mask, max_batch_size, ref_view=[]):
	group_sets = []
	group = set()
	ref_group = set()
	idx = 0
	while idx < len(attention_mask):
		new_group = group | set([idx])
		new_ref_group = (ref_group | set(attention_mask[idx] + ref_view)) - new_group 
		if len(new_group) + len(new_ref_group) <= max_batch_size:
			group = new_group
			ref_group = new_ref_group
			idx += 1
		else:
			assert len(group) != 0, "Cannot fit into a group"
			group_sets.append((group, ref_group))
			group = set()
			ref_group = set()
	if len(group)>0:
		group_sets.append((group, ref_group))

	group_metas = []
	for group, ref_group in group_sets:
		in_mask = sorted(list(group | ref_group))
		out_mask = []
		group_attention_masks = []
		for idx in in_mask:
			if idx in group:
				out_mask.append(in_mask.index(idx))
			group_attention_masks.append([in_mask.index(idxx) for idxx in attention_mask[idx] if idxx in in_mask])
		ref_attention_mask = [in_mask.index(idx) for idx in ref_view]
		group_metas.append([in_mask, out_mask, group_attention_masks, ref_attention_mask])

	return group_metas

'''

	MultiView-Diffusion Stable-Diffusion Pipeline
	Modified from a Diffusers StableDiffusionControlNetPipeline
	Just mimic the pipeline structure but did not follow any API convention

'''

class StableSyncMVDPipeline(StableDiffusionControlNetPipeline):
	def __init__(
		self, 
		vae: AutoencoderKL,
		text_encoder: CLIPTextModel,
		tokenizer: CLIPTokenizer,
		unet: UNet2DConditionModel,
		controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
		scheduler: KarrasDiffusionSchedulers,
		safety_checker: StableDiffusionSafetyChecker,
		feature_extractor: CLIPImageProcessor,
		requires_safety_checker: bool = False,
	):
		super().__init__(
			vae, text_encoder, tokenizer, unet, 
			controlnet, scheduler, safety_checker, 
			feature_extractor, requires_safety_checker
		)

		# CIT - Changed to DDIM to correspond with CIA
		self.scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
		self.scheduler.prediction_type = "sample"
		self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
		self.enable_model_cpu_offload()
		self.enable_vae_slicing()
		self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

	
	def initialize_pipeline(
			self,
			mesh_path=None,
			mesh_transform=None,
			mesh_autouv=None,
			mesh_path_app=None,
			mesh_transform_app=None,
			mesh_autouv_app=None,
			tex_app_path=None,
			camera_azims=None,
			camera_centers=None,
			top_cameras=True,
			ref_views=[],
			latent_size=None,
			latents_load=False,
			render_rgb_size=None,
			texture_size=None,
			texture_rgb_size=None,
			texture_rgb_size_app=None,

			max_batch_size=24,
			logging_config=None,
		):
		# Make output dir
		output_dir = logging_config["output_dir"]

		self.result_dir = f"{output_dir}/results"
		self.intermediate_dir = f"{output_dir}/intermediate"

		dirs = [output_dir, self.result_dir, self.intermediate_dir]
		for dir_ in dirs:
			if not os.path.isdir(dir_):
				os.mkdir(dir_)


		# 1. Initialize camera positions
		# Define the cameras for rendering
		self.camera_poses = []
		self.attention_mask=[]
		self.centers = camera_centers

		cam_count = len(camera_azims)
		front_view_diff = 360
		back_view_diff = 360
		front_view_idx = 0
		back_view_idx = 0
		for i, azim in enumerate(camera_azims):
			if azim < 0:
				azim += 360
			self.camera_poses.append((0, azim))
			self.attention_mask.append([(cam_count+i-1)%cam_count, i, (i+1)%cam_count])
			if abs(azim) < front_view_diff:
				front_view_idx = i
				front_view_diff = abs(azim)
			if abs(azim - 180) < back_view_diff:
				back_view_idx = i
				back_view_diff = abs(azim - 180)

		# Add two additional cameras for painting the top surfaces
		if top_cameras:
			self.camera_poses.append((30, 0))
			self.camera_poses.append((30, 180))

			self.attention_mask.append([front_view_idx, cam_count])
			self.attention_mask.append([back_view_idx, cam_count+1])

		# TODO: check that self.attention_mask doesn't screw us up

		# Reference view for attention (all views attend the the views in this list)
		# A forward view will be used if not specified
		if len(ref_views) == 0:
			ref_views = [front_view_idx]

		# Calculate in-group attention mask
		# self.group_metas = split_groups(self.attention_mask, max_batch_size, ref_views)
		

		# 2. Set up the UV mappings
		# Set up pytorch3D for projection between screen space and UV space
		# uvp is for latent and uvp_rgb for rgb color
		self.uvp = UVP(texture_size=texture_size, render_size=latent_size, sampling_mode="nearest", channels=4, device=self._execution_device)
		if mesh_path.lower().endswith(".obj"):
			self.uvp.load_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
		elif mesh_path.lower().endswith(".glb"):
			# mesh_autouv=False
			self.uvp.load_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
		else:
			assert False, "The mesh file format is not supported. Use .obj or .glb."

		self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0)

		self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size, sampling_mode="nearest", channels=3, device=self._execution_device)
		self.uvp_rgb.mesh = self.uvp.mesh.clone()
		self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0)
		_,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
		self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)

		self.uvp.to("cpu")
		self.uvp_rgb.to("cpu")

		if not latents_load:
			# CIT - Now also configuring for appearance mesh
			self.uvp_app = UVP(texture_size=texture_rgb_size_app, render_size=512, sampling_mode="nearest", channels=3, device=self._execution_device)
			if mesh_path_app.lower().endswith(".obj"):
				self.uvp_app.load_mesh(mesh_path_app, scale_factor=mesh_transform_app["scale"] or 1, autouv=mesh_autouv_app)
			elif mesh_path_app.lower().endswith(".glb"):
				mesh_autouv_app = False
				self.uvp_app.load_mesh(mesh_path_app, scale_factor=mesh_transform_app["scale"] or 1, autouv=mesh_autouv_app)
			else:
				assert False, "The mesh file format is not supported. Use .obj or .glb."

			
			texture_image = Image.open(tex_app_path)
			texture_tensor = (torch.from_numpy(np.array(texture_image)) / 255.0).permute(2, 0, 1)
			# texture_tensor = (torch.from_numpy(np.array(texture_image)).to(torch.float16) / 255.0).permute(2, 0, 1)
			self.uvp_app.set_texture_map(texture_tensor)
			
			self.uvp_app.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0)

			self.uvp_app.to("cpu")

		# Save some VRAM
		del _, cos_maps

		color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(-1,3,1,1).to(dtype=self.text_encoder.dtype, device=self._execution_device)
		color_images = torch.ones(
			(1,1,latent_size*8, latent_size*8), 
			device=self._execution_device, 
			dtype=self.text_encoder.dtype
		) * color_images
		color_images = ((0.5*color_images)+0.5)
		color_latents = encode_latents(self.vae, color_images)

		self.color_latents = {color[0]:color[1] for color in zip(color_names, [latent for latent in color_latents])}
		self.vae = self.vae.to("cpu")

		print("Done Initialization")




	'''
		Modified from a StableDiffusion ControlNet pipeline
		Multi ControlNet not supported yet
	'''
	@torch.no_grad()
	def __call__(
		self,
		prompt: str = None,
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: int = 50,
		guidance_scale: float = 7.5,
		negative_prompt: str = None,
		
		num_images_per_prompt: Optional[int] = 1,
		eta: float = 0.0,
		generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
		return_dict: bool = False,
		callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
		callback_steps: int = 1,
		max_batch_size=6,
		
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		controlnet_guess_mode: bool = False,
		controlnet_conditioning_scale: Union[float, List[float]] = 0.7,
		controlnet_conditioning_end_scale: Union[float, List[float]] = 0.9,
		control_guidance_start: Union[float, List[float]] = 0.0,
		control_guidance_end: Union[float, List[float]] = 0.99,
		guidance_rescale: float = 0.0,

		mesh_path: str = None,
		mesh_transform: dict = None,
		mesh_autouv = False,

		mesh_path_app: str = None,
		mesh_transform_app: dict = None,
		mesh_autouv_app = False,
		tex_app_path=None,

		latents_load: bool = False,
		latents_save_path: str = None,
		cond_app_path: str=None,

		camera_azims=None,
		camera_centers=None,
		top_cameras=True,
		texture_size = 1536,
		render_rgb_size=1024,
		texture_rgb_size = 1024,
		texture_rgb_size_app = 1024,
		multiview_diffusion_end=0.8,
		exp_start=0.0,
		exp_end=6.0,
		shuffle_background_change=0.4,
		shuffle_background_end=0.99, #0.4

		use_directional_prompt=True,
		
		ref_attention_end=0.2,

		logging_config=None,
		cond_type="depth",

		app_transfer_model=None,
	):
		
		if latents_load:
			if (not os.path.isfile(latents_save_path)) or (not os.path.isfile(cond_app_path)):
				latents_load = False
		

		# Setup pipeline settings
		self.initialize_pipeline(
				mesh_path=mesh_path,
				mesh_transform=mesh_transform,
				mesh_autouv=mesh_autouv,
				mesh_path_app=mesh_path_app,
				mesh_transform_app=mesh_transform_app,
				mesh_autouv_app=mesh_autouv_app,
				tex_app_path=tex_app_path,
				camera_azims=camera_azims,
				camera_centers=camera_centers,
				top_cameras=top_cameras,
				ref_views=[],
				latent_size=height//8,
				latents_load=latents_load,
				render_rgb_size=render_rgb_size,
				texture_size=texture_size,
				texture_rgb_size=texture_rgb_size,
				texture_rgb_size_app=texture_rgb_size_app,

				max_batch_size=max_batch_size,

				logging_config=logging_config
			)

		# CIT - add kwarg
		if cross_attention_kwargs is None:
			cross_attention_kwargs = {'perform_swap': True} 
		elif type(cross_attention_kwargs) == dict:
			cross_attention_kwargs['perform_swap'] = True
		else:
			raise(TypeError())
		
		num_timesteps = self.scheduler.config.num_train_timesteps
		initial_controlnet_conditioning_scale = controlnet_conditioning_scale
		log_interval = logging_config.get("log_interval", 10)
		view_fast_preview = logging_config.get("view_fast_preview", True)
		tex_fast_preview = logging_config.get("tex_fast_preview", True)

		controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

		# align format for control guidance
		if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
			control_guidance_start = len(control_guidance_end) * [control_guidance_start]
		elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
			control_guidance_end = len(control_guidance_start) * [control_guidance_end]
		elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
			# mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
			mult = 1
			control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
				control_guidance_end
			]


		# 0. Default height and width to unet
		height = height or self.unet.config.sample_size * self.vae_scale_factor
		width = width or self.unet.config.sample_size * self.vae_scale_factor


		# 1. Check inputs. Raise error if not correct
		self.check_inputs(
			prompt,
			torch.zeros((1,3,height,width), device=self._execution_device),
			callback_steps,
			negative_prompt,
			None,
			None,
			controlnet_conditioning_scale,
			control_guidance_start,
			control_guidance_end,
		)


		# 2. Define call parameters
		if prompt is not None and isinstance(prompt, list):
			assert len(prompt) == 1 and len(negative_prompt) == 1, "Only implemented for 1 (negative) prompt"  
		assert num_images_per_prompt == 1, "Only implemented for 1 image per-prompt"
		batch_size = len(self.uvp.cameras)


		device = self._execution_device
		# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
		# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
		# corresponds to doing no classifier free guidance.
		do_classifier_free_guidance = guidance_scale > 1.0

		# if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
		# 	controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

		global_pool_conditions = (
			controlnet.config.global_pool_conditions
			if isinstance(controlnet, ControlNetModel)
			else controlnet.nets[0].config.global_pool_conditions
		)
		guess_mode = controlnet_guess_mode or global_pool_conditions


		# 3. Encode input prompt
		prompt, negative_prompt = prepare_directional_prompt(prompt, negative_prompt)

		text_encoder_lora_scale = (
			cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
		)
		prompt_embeds = self._encode_prompt(
			prompt,
			device,
			num_images_per_prompt,
			do_classifier_free_guidance,
			negative_prompt,
			prompt_embeds=None,
			negative_prompt_embeds=None,
			lora_scale=text_encoder_lora_scale,
		)

		negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
		prompt_embed_dict = dict(zip(direction_names, [emb for emb in prompt_embeds]))
		negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in negative_prompt_embeds]))

		# (4. Prepare image) This pipeline use internal conditional images from Pytorch3D
		self.uvp.to(self._execution_device)
		conditioning_images, masks = get_conditioning_images(self.uvp, height, cond_type=cond_type)
		conditioning_images = conditioning_images.type(prompt_embeds.dtype)
		cond = (conditioning_images/2+0.5).permute(0,2,3,1).cpu().numpy()
		numpy_to_pil(cond[0])[0].save(f"{self.intermediate_dir}/first_cond.jpg")
		cond = np.concatenate([img for img in cond], axis=1)
		numpy_to_pil(cond)[0].save(f"{self.intermediate_dir}/cond.jpg")

		if not latents_load:
			self.uvp_app.to(self._execution_device)
			conditioning_images_app, masks_app = get_conditioning_images(self.uvp_app, height, cond_type=cond_type)
			conditioning_images_app = conditioning_images_app.type(prompt_embeds.dtype)
			torch.save(conditioning_images_app, cond_app_path)
		else:
			conditioning_images_app = torch.load(cond_app_path)

		# 5. Prepare timesteps
		self.scheduler.set_timesteps(num_inference_steps, device=device)
		timesteps = self.scheduler.timesteps

		# 6. Prepare latent variables
		num_channels_latents = self.unet.config.in_channels
		latents = self.prepare_latents( # [10,4,96,96]
			batch_size,
			num_channels_latents,
			height,
			width,
			prompt_embeds.dtype,
			device,
			generator,
			None,
		)

		latent_tex = self.uvp.set_noise_texture()
		noise_views = self.uvp.render_textured_views() # list of 10 tensors, each got 5 channels, the last one is the mask (aka transparency aka alpha), and the other 4 are latent channels
		foregrounds = [view[:-1] for view in noise_views] # here are the latent channels
		masks = [view[-1:] for view in noise_views] # here is the 5th channel, the mask
		composited_tensor = composite_rendered_view(self.scheduler, latents, foregrounds, masks, timesteps[0]+1)
		latents = composited_tensor.type(latents.dtype)
		self.uvp.to("cpu")

		# CIT
		if not latents_load:
			noise_backgrounds = torch.normal(0, 1, (len(self.uvp_app.cameras), 3, 512, 512), device=self._execution_device)
			app_views = self.uvp_app.render_textured_views() # List of 10 tensors, each is 1536x1536, but with 4 channels (last channel is mask)
   
			# CIT DBG - TODO remove this
			show_views(app_views, self.intermediate_dir)
			save_all_views(app_views, self.intermediate_dir)
   
			foregrounds_app = [view[:-1] for view in app_views]
			masks_app = [view[-1:] for view in app_views]
			composited_tensor_app = composite_rendered_view(self.scheduler, noise_backgrounds, foregrounds_app, masks_app, timesteps[0]+1) # shape is [10, 3, 1536, 1536]
			latents_app = []
			for i in range(len(self.camera_poses)):
				image_tensor = composited_tensor_app[i]
				# First index here is to grab the correct variable, second is to get the first(?) timestep
				latents_app.append(invert_images(app_transfer_model.pipe, app_image=image_tensor, cfg=app_transfer_model.config)[0][0])
			latents_app = torch.stack(latents_app)
			torch.save(latents_app, latents_save_path)
			# Cleanup
			del(noise_backgrounds)
			del(app_views)
			del(foregrounds_app)
			del(masks_spp)
			del(composited_tensor_app)
			del(self.uvp_app)
		else:
			latents_app = torch.load(latents_save_path)
   
		########################################################################################
		### Right now, latents_app is the noisiest latent: the shape is [10, 4, 64, 64] ########
		### The following change is that latents_app is all the latents [10, 100, 4, 64, 64] ###
		########################################################################################
   
		# 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

		# 7.1 Create tensor stating which controlnets to keep
		controlnet_keep = []

		for i in range(len(timesteps)):
			keeps = [
				1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
				for s, e in zip(control_guidance_start, control_guidance_end)
			]
			controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

		# 8. Denoising loop
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
		intermediate_results = []
		intermediate_results_app = []
		background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
		dbres_sizes_list = []
		mbres_size_list = []
		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):
				print(f"{datetime.now()}: iteration {i}, timestep {t}")

				# mix prompt embeds according to azim angle
				positive_prompt_embeds = [azim_prompt(prompt_embed_dict, pose) for pose in self.camera_poses]
				positive_prompt_embeds = torch.stack(positive_prompt_embeds, axis=0)

				negative_prompt_embeds = [azim_neg_prompt(negative_prompt_embed_dict, pose) for pose in self.camera_poses]
				negative_prompt_embeds = torch.stack(negative_prompt_embeds, axis=0)


				# expand the latents if we are doing classifier free guidance
				latent_model_input = self.scheduler.scale_model_input(latents, t).to(torch.float16)
				latent_model_input_app = self.scheduler.scale_model_input(latents_app[i], t)

				'''
					Use groups to manage prompt and results
					Make sure negative and positive prompt does not perform attention together
				'''
				prompt_embeds_groups = {"positive": positive_prompt_embeds}
				result_groups = {}
				result_groups_app = {}
				if do_classifier_free_guidance:
					prompt_embeds_groups["negative"] = negative_prompt_embeds

				for prompt_tag, prompt_embeds in prompt_embeds_groups.items(): # Lidor asks: how many times does this loop run?
					if prompt_tag == "positive" or not guess_mode:
						# controlnet(s) inference
						control_model_input = latent_model_input
						control_model_input_app = latent_model_input_app
						controlnet_prompt_embeds = prompt_embeds

						if isinstance(controlnet_keep[i], list):
							cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
						else:
							controlnet_cond_scale = controlnet_conditioning_scale
							if isinstance(controlnet_cond_scale, list):
								controlnet_cond_scale = controlnet_cond_scale[0]
							cond_scale = controlnet_cond_scale * controlnet_keep[i]

						# Split into micro-batches according to group meta info
						# Ignore this feature for now
						down_block_res_samples_list = []
						mid_block_res_sample_list = []

						# CIT - modify batches
						model_input_batches = [torch.stack((control_model_input[i], control_model_input[i], control_model_input_app[i])).to(torch.float16) for i in range(latents.shape[0])]
						prompt_embeds_batches = [torch.stack((embed, embed, embed)).to(torch.float16) for embed in controlnet_prompt_embeds]
						conditioning_images_batches = [torch.stack((conditioning_images[i], conditioning_images[i], conditioning_images_app[i])) for i in range(conditioning_images.shape[0])]

						for model_input_batch ,prompt_embeds_batch, conditioning_images_batch \
							in zip (model_input_batches, prompt_embeds_batches, conditioning_images_batches):
							down_block_res_samples, mid_block_res_sample = self.controlnet(
								model_input_batch,
								t,
								encoder_hidden_states=prompt_embeds_batch,
								controlnet_cond=conditioning_images_batch,
								conditioning_scale=cond_scale,
								guess_mode=guess_mode,
								return_dict=False,
							)
							down_block_res_samples_list.append(down_block_res_samples)
							mid_block_res_sample_list.append(mid_block_res_sample)

						''' For the ith element of down_block_res_samples, concat the ith element of all mini-batch result '''
						model_input_batches = prompt_embeds_batches = conditioning_images_batches = None

						if guess_mode:
							for dbres in down_block_res_samples_list:
								dbres_sizes = []
								for res in dbres:
									dbres_sizes.append(res.shape)
								dbres_sizes_list.append(dbres_sizes)

							for mbres in mid_block_res_sample_list:
								mbres_size_list.append(mbres.shape)

					else:
						# Infered ControlNet only for the conditional batch.
						# To apply the output of ControlNet to both the unconditional and conditional batches,
						# add 0 to the unconditional batch to keep it unchanged.
						# We copy the tensor shapes from a conditional batch
						down_block_res_samples_list = []
						mid_block_res_sample_list = []
						for dbres_sizes in dbres_sizes_list:
							down_block_res_samples_list.append([torch.zeros(shape, device=self._execution_device, dtype=latents.dtype) for shape in dbres_sizes])
						for mbres in mbres_size_list:
							mid_block_res_sample_list.append(torch.zeros(mbres, device=self._execution_device, dtype=latents.dtype))
						dbres_sizes_list = []
						mbres_size_list = []


					'''
						predict the noise residual, split into mini-batches
						Downblock res samples has n samples, we split each sample into m batches
						and re group them into m lists of n mini batch samples.
					
					'''
					noise_pred_list = []
					noise_pred_app_list = []
					# CIT - need to modify the batches so that they contain appearance latents
					model_input_batches = [torch.stack((latent_model_input[i], latent_model_input[i], latent_model_input_app[i])).to(torch.float16) for i in range(latents.shape[0])]
					prompt_embeds_batches = [torch.stack((embed, embed, embed)) for embed in prompt_embeds]

					for model_input_batch, prompt_embeds_batch, down_block_res_samples_batch, mid_block_res_sample_batch \
						in zip(model_input_batches, prompt_embeds_batches, down_block_res_samples_list, mid_block_res_sample_list):
						noise_pred = self.unet(
							model_input_batch,
							t,
							encoder_hidden_states=prompt_embeds_batch,
							cross_attention_kwargs=cross_attention_kwargs,
							down_block_additional_residuals=down_block_res_samples_batch,
							mid_block_additional_residual=mid_block_res_sample_batch,
							return_dict=False,
						)[0]
						noise_pred_list.append(noise_pred[0])
						noise_pred_app_list.append(noise_pred[2])

					# TODO: Make sure that the noise gets to the right place.

					noise_pred = torch.stack(noise_pred_list).to(torch.float16)
					noise_pred_app = torch.stack(noise_pred_app_list).to(torch.float16)
					down_block_res_samples_list = None
					mid_block_res_sample_list = None
					noise_pred_list = None
					model_input_batches = prompt_embeds_batches = down_block_res_samples_batches = mid_block_res_sample_batches = None

					result_groups[prompt_tag] = noise_pred
					result_groups_app[prompt_tag] = noise_pred_app

				positive_noise_pred = result_groups["positive"]
				positive_noise_pred_app = result_groups_app["positive"]

				# perform guidance
				if do_classifier_free_guidance:
					noise_pred = result_groups["negative"] + guidance_scale * (positive_noise_pred - result_groups["negative"])
					noise_pred_app = result_groups_app["negative"] + guidance_scale * (positive_noise_pred_app - result_groups_app["negative"])

				# CIT - This seems unreachable and references things that do not exist, so I comment it here
				# if do_classifier_free_guidance and guidance_rescale > 0.0:
					# Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
					# noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

				self.uvp.to(self._execution_device)
				# compute the previous noisy sample x_t -> x_t-1
				# Multi-View step or individual step
				current_exp = ((exp_end-exp_start) * i / num_inference_steps) + exp_start
				if t > (1-multiview_diffusion_end)*num_timesteps:
					prev_t = (t + (t - timesteps[i + 1])) if i == 0 else timesteps[i - 1]
					scheduler_copy = copy.deepcopy(self.scheduler)
					step_results = step_tex(
						scheduler=self.scheduler, 
						uvp=self.uvp, 
						model_output=noise_pred, 
						timestep=t,
						prev_t=prev_t,
						sample=latents, 
						texture=latent_tex,
						return_dict=True, 
						main_views=[], 
						exp=current_exp,
						**extra_step_kwargs
					)
					step_results_app = step_tex(
						scheduler=scheduler_copy, 
						uvp=None, 
						model_output=noise_pred_app, 
						timestep=t,
						prev_t=prev_t,
						sample=latents_app[i], 
						texture=latent_tex,
						return_dict=True, 
						main_views=[], 
						exp=current_exp,
						is_app=True,
						**extra_step_kwargs
					)

					pred_original_sample = step_results["pred_original_sample"]
					latents = step_results["prev_sample"]
					latent_tex = step_results["prev_tex"]
					pred_original_sample_app = step_results_app["pred_original_sample"]
					# latents_app = step_results_app["prev_sample"] # CIT - This is not used since the latents are supplied for all timesteps.

					# Composit latent foreground with random color background
					background_latents = [self.color_latents[color] for color in background_colors]
					composited_tensor = composite_rendered_view(self.scheduler, background_latents, latents, masks, t)
					latents = composited_tensor.type(latents.dtype)

					intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))
					intermediate_results_app.append((latents_app[i].to("cpu"), pred_original_sample_app.to("cpu")))
				else:
					step_results = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
					step_results_app = self.scheduler.step(noise_pred_app, t, latents_app[i], **extra_step_kwargs, return_dict=True)

					pred_original_sample = step_results["pred_original_sample"]
					latents = step_results["prev_sample"]
					latent_tex = None
					pred_original_sample_app = step_results_app["pred_original_sample"]
					# latents_app = step_results_app["prev_sample"] # CIT - This is not used since the latents are supplied for all timesteps.

					intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))
					intermediate_results_app.append((latents_app[i].to("cpu"), pred_original_sample_app.to("cpu")))

				del noise_pred, noise_pred_app, result_groups, result_groups_app
					


				# 9. Update pipeline settings after one step:
				# 9.1. Annealing ControlNet scale
				if (1-t/num_timesteps) < control_guidance_start[0]:
					controlnet_conditioning_scale = initial_controlnet_conditioning_scale
				elif (1-t/num_timesteps) > control_guidance_end[0]:
					controlnet_conditioning_scale = controlnet_conditioning_end_scale
				else:
					alpha = ((1-t/num_timesteps) - control_guidance_start[0]) / (control_guidance_end[0] - control_guidance_start[0])
					controlnet_conditioning_scale = alpha * initial_controlnet_conditioning_scale + (1-alpha) * controlnet_conditioning_end_scale

				# 9.2. Shuffle background colors; only black and white used after certain timestep
				if (1-t/num_timesteps) < shuffle_background_change:
					background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
				elif (1-t/num_timesteps) < shuffle_background_end:
					background_colors = [random.choice(["black","white"]) for i in range(len(self.camera_poses))]
				else:
					background_colors = background_colors



				# 10. Logging at "log_interval" intervals and last step
				# Choose to uses color approximation or vae decoding
				if i % log_interval == log_interval-1 or t == 1:
					if view_fast_preview:
						decoded_results = []
						for latent_images in intermediate_results[-1]: # TODO why does it runs twice? tell lidor, he is curious
							images = latent_preview(latent_images.to(self._execution_device))
							images = np.concatenate([img for img in images], axis=1)
							decoded_results.append(images)
						result_image = np.concatenate(decoded_results, axis=0)
						numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")
					else:
						decoded_results = []
						for latent_images in intermediate_results[-1]:
							images = decode_latents(self.vae, latent_images.to(self._execution_device))
							images = np.concatenate([img for img in images], axis=1)
							decoded_results.append(images)
						result_image = np.concatenate(decoded_results, axis=0)
						numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")

					if not t < (1-multiview_diffusion_end)*num_timesteps:
						if tex_fast_preview:
							tex = latent_tex.clone()
							texture_color = latent_preview(tex[None, ...])
							numpy_to_pil(texture_color)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.jpg")
						else:
							self.uvp_rgb.to(self._execution_device)
							result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, pred_original_sample)
							numpy_to_pil(result_tex_rgb_output)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")
							self.uvp_rgb.to("cpu")

				self.uvp.to("cpu")

				# call the callback, if provided
				if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
					progress_bar.update()
					if callback is not None and i % callback_steps == 0:
						callback(i, t, latents)

				# Signal the program to skip or end
				import select
				import sys
				if select.select([sys.stdin],[],[],0)[0]:
					userInput = sys.stdin.readline().strip()
					if userInput == "skip":
						return None
					elif userInput == "end":
						exit(0)

		
		self.uvp.to(self._execution_device)
		self.uvp_rgb.to(self._execution_device)
		result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, latents)
		self.uvp.save_mesh(f"{self.result_dir}/textured.obj", result_tex_rgb.permute(1,2,0))


		self.uvp_rgb.set_texture_map(result_tex_rgb)
		textured_views = self.uvp_rgb.render_textured_views()
		textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1,...]
		textured_views_rgb = textured_views_rgb.permute(1,2,0).cpu().numpy()[None,...]
		v = numpy_to_pil(textured_views_rgb)[0]
		v.save(f"{self.result_dir}/textured_views_rgb.jpg")
		# display(v)

		# Offload last model to CPU
		if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
			self.final_offload_hook.offload()

		self.uvp.to("cpu")
		self.uvp_rgb.to("cpu")

		return result_tex_rgb, textured_views, v