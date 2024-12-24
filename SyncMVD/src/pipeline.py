import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from IPython.display import display
import numpy as np
import math
import random
import torch
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
    replace_example_docstring,
)

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.models.attention_processor import Attention, AttentionProcessor

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from .renderer.project import UVProjection as UVP


from .syncmvd.attention import SamplewiseAttnProcessor2_0, replace_attention_processors
from .syncmvd.prompt import *
from .syncmvd.step import step_tex
from .utils import *


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Background colors
color_constants = {
    "black": [-1, -1, -1],
    "white": [1, 1, 1],
    "maroon": [0, -1, -1],
    "red": [1, -1, -1],
    "olive": [0, 0, -1],
    "yellow": [1, 1, -1],
    "green": [-1, 0, -1],
    "lime": [-1, 1, -1],
    "teal": [-1, 0, 0],
    "aqua": [-1, 1, 1],
    "navy": [-1, -1, 0],
    "blue": [-1, -1, 1],
    "purple": [0, -1, 0],
    "fuchsia": [1, -1, 1],
}
color_names = list(color_constants.keys())


# Used to generate depth or normal conditioning images
@torch.no_grad()
def get_conditioning_images(
    uvp, output_size, render_size=512, blur_filter=5, cond_type="normal"
):
    verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(
        image_size=render_size
    )
    masks = normals[..., 3][:, None, ...]
    masks = Resize((output_size // 8,) * 2, antialias=True)(masks)
    normals_transforms = Compose(
        [
            Resize(
                (output_size,) * 2,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            GaussianBlur(blur_filter, blur_filter // 3 + 1),
        ]
    )

    if cond_type == "normal":
        view_normals = uvp.decode_view_normal(normals).permute(0, 3, 1, 2) * 2 - 1
        conditional_images = normals_transforms(view_normals)
    # Some problem here, depth controlnet don't work when depth is normalized
    # But it do generate using the unnormalized form as below
    elif cond_type == "depth":
        view_depths = uvp.decode_normalized_depth(depths).permute(0, 3, 1, 2)
        conditional_images = normals_transforms(view_depths)

    return conditional_images, masks


# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t):
    composited_images = []
    for i, (background, foreground, mask) in enumerate(
        zip(backgrounds, foregrounds, masks)
    ):
        if t > 0:
            alphas_cumprod = scheduler.alphas_cumprod[t]
            noise = torch.normal(0, 1, background.shape, device=background.device)
            background = (1 - alphas_cumprod) * noise + alphas_cumprod * background
        composited = foreground * mask + background * (1 - mask)
        composited_images.append(composited)
    composited_tensor = torch.stack(composited_images)
    return composited_tensor


# Split into micro-batches to use less memory in each unet prediction
# But need more investigation on reducing memory usage
# Assume it has no possitive effect and use a large "max_batch_size" to skip splitting
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
    if len(group) > 0:
        group_sets.append((group, ref_group))

    group_metas = []
    for group, ref_group in group_sets:
        in_mask = sorted(list(group | ref_group))
        out_mask = []
        group_attention_masks = []
        for idx in in_mask:
            if idx in group:
                out_mask.append(in_mask.index(idx))
            group_attention_masks.append(
                [in_mask.index(idxx) for idxx in attention_mask[idx] if idxx in in_mask]
            )
        ref_attention_mask = [in_mask.index(idx) for idx in ref_view]
        group_metas.append(
            [in_mask, out_mask, group_attention_masks, ref_attention_mask]
        )

    return group_metas


"""

	MultiView-Diffusion Stable-Diffusion Pipeline
	Modified from a Diffusers StableDiffusionControlNetPipeline
	Just mimic the pipeline structure but did not follow any API convention

"""


# start of the run
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
        #Denoising Diffusion Probabilistic Models.
        #the scheduler controlling the timesteps and the amount of noise added or removed at each step of the diffusion process
        self.scheduler = DDPMScheduler.from_config(self.scheduler.config)
        # allow offload the model to save momory
        self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
        self.enable_model_cpu_offload()
        #VAE -  change image to latent, Slicing - split image to patches
        self.enable_vae_slicing()
        #image processor probably in this case for tranpforming image to latent
        #vae_scale_factor - might be used to adjust the resolution or dimensions of the images 
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
    def setup_output_dirs(self, logging_config):
        output_dir = logging_config["output_dir"]
        self.result_dir = f"{output_dir}/results"
        self.intermediate_dir = f"{output_dir}/intermediate"
        dirs = [output_dir, self.result_dir, self.intermediate_dir]
 		#validate that all target directories exist/create one if not
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)
    
    
    def setup_cameras(self, camera_azims, camera_centers, top_cameras, ref_views, max_batch_size):
        # Define the cameras for rendering
        self.camera_poses = []
        self.attention_mask = []
        self.centers = camera_centers
		#camera_azims ==> list of azimuth angles for different camera positions.
        cam_count = len(camera_azims)
        front_view_diff = 360
        back_view_diff = 360
        front_view_idx = 0
        back_view_idx = 0
        for i, azim in enumerate(camera_azims):
            #convert Azim to be 0-360
            if azim < 0:
                azim += 360
            self.camera_poses.append((0, azim))
            #keep neighbour cameras to ensuring smooth transitions between views
            self.attention_mask.append([(cam_count + i - 1) % cam_count, i, (i + 1) % cam_count])
         	#find front and back for multiple purposes: 
			# front: for alignment, reference in attention mechanisms, or for ensuring that the main features of the object are prominently captured.
			#back of the object: back of the object is properly captured and can be used for tasks that require a comprehensive view of the object from all angles.
			#Find the fronview camera: closest to 0,   
            if abs(azim) < front_view_diff:
                front_view_idx = i
                front_view_diff = abs(azim)
 			#Find the back view camera: closest to 180,
            if abs(azim - 180) < back_view_diff:
                back_view_idx = i
                back_view_diff = abs(azim - 180)
		# Add two additional cameras for painting the top surfaces
		# Adding top-view cameras provides better coverage of the object, especially the top surfaces, which might not be fully visible from the original horizontal camera positions.
        if top_cameras:
            self.camera_poses.append((30, 0))
            self.camera_poses.append((30, 180))
            self.attention_mask.append([front_view_idx, cam_count]) #neighbour of top front
            self.attention_mask.append([back_view_idx, cam_count + 1]) #neighbour of top back
		# Reference view for attention (all views attend the the views in this list)
		# A forward view will be used if not specified
        if len(ref_views) == 0:
            ref_views = [front_view_idx]
 		# Calculate in-group attention mask
		#split camera to smaller group for better memory usage
        self.group_metas = split_groups(self.attention_mask, max_batch_size, ref_views)

    def setup_uv_projection(self, mesh_path, mesh_transform, mesh_autouv, texture_size, latent_size, render_rgb_size, texture_rgb_size):
 		# Set up pytorch3D for projection between screen space and UV space
		# uvp is for latent and uvp_rgb for rgb color
		# Initializes a UVP object with specific rendering and texture parameters.
       	self.uvp = UVP(texture_size=texture_size, render_size=latent_size, sampling_mode="nearest", channels=4, device=self._execution_device)
        if mesh_path.lower().endswith(".obj"):
            self.uvp.load_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
        elif mesh_path.lower().endswith(".glb"):
            self.uvp.load_glb_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
 		#Configures camera positions and render settings
        self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=self.centers, camera_distance=4.0)
        self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size, sampling_mode="nearest", channels=3, device=self._execution_device)
 		#Cloning the mesh ensures that both uvp and uvp_rgb work with the same 3D model, 
		#   but possibly with different texture and rendering settings.
	    self.uvp_rgb.mesh = self.uvp.mesh.clone()
 		#cosine maps, which are used to calculate angles and lighting effects.
        self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=self.centers, camera_distance=4.0)
        _, _, _, cos_maps, _, _ = self.uvp_rgb.render_geometry()
 		#important for shading and lighting calculations, 
		#   ensuring that the rendered textures correctly reflect 
        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)
        del _, cos_maps
        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")
    
    
    def setup_color_latents(self, latent_size):
 		#preparing color images and encoding them into latents using a Variational Autoencoder (VAE)
		#creates a tensor of color images from predefined color constants.
        color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(-1, 3, 1, 1).to(dtype=self.text_encoder.dtype, device=self._execution_device)
 		#expands the color images to the desired latent size
        color_images = torch.ones((1, 1, latent_size * 8, latent_size * 8), device=self._execution_device, dtype=self.text_encoder.dtype) * color_images
 		#normalizes the color images to the range [0, 1].
        color_images = ((0.5 * color_images) + 0.5)
   		#normalized color images into latent representations 
        color_latents = encode_latents(self.vae, color_images)
 		#stores the encoded latents in a dictionary for easy access.
        self.color_latents = {color[0]: color[1] for color in zip(color_names, [latent for latent in color_latents])}
		#save GPU memory by ove to CPU
        self.vae = self.vae.to("cpu")
    
    def initialize_pipeline(
			self,
			mesh_path=None,
			mesh_transform=None,
			mesh_autouv=None,
			camera_azims=None,
			camera_centers=None,
			top_cameras=True,
			ref_views=[],
			latent_size=None,
			render_rgb_size=None,
			texture_size=None,
			texture_rgb_size=None,

			max_batch_size=24,
			logging_config=None,
		):
        self.setup_output_dirs(logging_config)
        self.setup_cameras(camera_azims, camera_centers, top_cameras, ref_views, max_batch_size)
        self.setup_uv_projection(mesh_path, mesh_transform, mesh_autouv, texture_size, latent_size, render_rgb_size, texture_rgb_size)
        self.setup_color_latents(latent_size)
        print("Done Initialization")

    def create_prompt_embedding_groups(
        self, do_classifier_free_guidance, prompt_embed_dict, negative_prompt_embed_dict
    ):
        """Creates groups of prompt embeddings for positive and negative prompts.

        This function generates embeddings for multiple camera poses by mixing prompt embeddings
        according to azimuth angles. It separates positive and negative prompts into groups
        to prevent cross-attention between them during processing.

        Args:
            do_classifier_free_guidance (bool): Flag to determine if negative prompts should be included
            prompt_embed_dict (dict): Dictionary containing positive prompt embeddings
            negative_prompt_embed_dict (dict): Dictionary containing negative prompt embeddings

        Returns:
            dict: A dictionary containing prompt embedding groups with keys:
                - 'positive': Tensor of positive prompt embeddings for all camera poses
                - 'negative': Tensor of negative prompt embeddings (if classifier free guidance is enabled)
                
        Shape:
            - Output['positive']: (num_poses, embedding_dim)
            - Output['negative']: (num_poses, embedding_dim) if do_classifier_free_guidance is True
        """
        # mix prompt embeds according to azim angle
        # TODO: what exsexli is it- assamption function to use to get this exect vew
        positive_prompt_embeds = [
            azim_prompt(prompt_embed_dict, pose) for pose in self.camera_poses
        ]
        positive_prompt_embeds = torch.stack(positive_prompt_embeds, axis=0)

        negative_prompt_embeds = [
            azim_neg_prompt(negative_prompt_embed_dict, pose)
            for pose in self.camera_poses
        ]
        negative_prompt_embeds = torch.stack(negative_prompt_embeds, axis=0)
        """
            Use groups to manage prompt and results
            Make sure negative and positive prompt does not perform attention together
        """
        prompt_embeds_groups = {"positive": positive_prompt_embeds}
        if do_classifier_free_guidance:
            prompt_embeds_groups["negative"] = negative_prompt_embeds
        return prompt_embeds_groups

    def calculate_controlnet_conditioning_scale(self, timestep_index: int, controlnet_timestep_weights: list, base_conditioning_scale: Union[float, List[float]]) -> Union[float, List[float]]:
        """
        Calculates the effective conditioning scale for ControlNet at a given timestep.

        This function adjusts the base conditioning scale based on timestep-specific weights.
        It handles both single and multiple ControlNet models.

        Args:
            timestep_index (int): Current timestep index in the denoising process
            controlnet_timestep_weights (list): List of weights per timestep, indicating how much to apply ControlNet
            base_conditioning_scale (float or List[float]): Base conditioning scale(s) for ControlNet(s)

        Returns:
            float or List[float]: Adjusted conditioning scale(s) for the current timestep
            
        Example:
            For single ControlNet:
                base_scale = 0.7
                weight = 0.8
                effective_scale = 0.7 * 0.8 = 0.56
                
            For multiple ControlNets:
                base_scales = [0.7, 0.8] 
                weights = [0.8, 0.9]
                effective_scales = [0.56, 0.72]
        """
        
        if isinstance(controlnet_timestep_weights[timestep_index], list):
            # Multiple ControlNets case - multiply each base scale by its corresponding weight
            return [
                base * weight 
                for base, weight in zip(base_conditioning_scale, controlnet_timestep_weights[timestep_index])
            ]
        else:
            # Single ControlNet case
            base_scale = base_conditioning_scale[0] if isinstance(base_conditioning_scale, list) else base_conditioning_scale
            return base_scale * controlnet_timestep_weights[timestep_index]

    def group_by_metas(self, something):
        """
        Groups tensor data based on predefined group metadata.

        This method takes a tensor and splits it into groups according to the indices
        stored in self.group_metas. Each group is created by selecting elements from
        the input tensor using the corresponding indices.

        Args:
            something (torch.Tensor): Input tensor to be grouped.

        Returns:
            list[torch.Tensor]: A list of tensors, where each tensor contains the grouped
                elements according to the corresponding indices in self.group_metas.

        Example:
            If self.group_metas = [[0,1], [2,3]] and something is a tensor with 4 elements,
            the output will be a list of 2 tensors containing elements [0,1] and [2,3]
            respectively from the input tensor.
        """
        return [
            torch.index_select(
                something,
                dim=0,
                index=torch.tensor(meta[0], device=self._execution_device),
            )
            for meta in self.group_metas
        ]

    def process_controlnet_batch(
        self,
        latent_model_input: torch.Tensor,
        prompt_embeds: torch.Tensor, 
        t: torch.Tensor,
        conditioning_images: torch.Tensor,
        controlnet_keep: float,
        controlnet_conditioning_scale: float,
        guess_mode: bool,
        i: int,
    ) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
        """Process a batch through ControlNet to get conditioning features.

        This method processes input batches through ControlNet to generate conditioning features
        for guided image generation. It handles batch splitting and processing according to
        group metadata.

        Returns:
            Tuple containing:
                - List of down-block residual sample lists
                - List of mid-block residual samples

        Note:
            The method:
            1. Calculates conditioning scale based on iteration
            2. Groups inputs into micro-batches
            3. Processes each micro-batch through ControlNet
            4. Concatenates results across batches
        """
        down_block_res_samples_list = []
        mid_block_res_sample_list = []

        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds

        cond_scale = self.calculate_controlnet_conditioning_scale(
            i, controlnet_keep, controlnet_conditioning_scale
        )

        # Split into micro-batches according to group meta info
        model_input_batches = self.group_by_metas(control_model_input)
        prompt_embeds_batches = self.group_by_metas(controlnet_prompt_embeds)
        conditioning_images_batches = self.group_by_metas(conditioning_images)
        for (
            model_input_batch,
            prompt_embeds_batch,
            conditioning_images_batch,
        ) in zip(
            model_input_batches,
            prompt_embeds_batches,
            conditioning_images_batches,
        ):#my be the part of the linear interpolation
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
        """ For the ith element of down_block_res_samples, concat the ith element of all mini-batch result """
        model_input_batches = prompt_embeds_batches = conditioning_images_batches = None
        return down_block_res_samples_list, mid_block_res_sample_list

    def process_noise_pred(
        self,
        latent_model_input,
        prompt_embeds,
        t,
        num_timesteps,
        ref_attention_end,
        cross_attention_kwargs,
        down_block_res_samples_list,
        mid_block_res_sample_list,
    ):
        """Process and predict noise for diffusion model with reference attention.

        This method handles noise prediction in the diffusion process, managing reference attention
        and batch processing for the UNet model.

        Args:
            latent_model_input (torch.Tensor): Input latent representations for the model.
            prompt_embeds (torch.Tensor): Embedded prompt conditioning.
            t (int): Current timestep in the diffusion process.
            num_timesteps (int): Total number of timesteps in the diffusion process.
            ref_attention_end (float): Threshold for when to apply reference attention (0-1).
            cross_attention_kwargs (dict): Additional arguments for cross attention.
            down_block_res_samples_list (list): Residual samples from downsampling blocks.
            mid_block_res_sample_list (list): Residual samples from middle blocks.

        Returns:
            torch.Tensor: Concatenated noise predictions for the batch.

        Note:
            - Handles batching using group metadata
            - Dynamically switches attention processing based on timestep
            - Manages reference attention weights and masks
            - Processes each batch through UNet with residual connections
        """
        noise_pred_list = []
        model_input_batches = self.group_by_metas(latent_model_input)
        prompt_embeds_batches = self.group_by_metas(prompt_embeds)
        for (
            model_input_batch,
            prompt_embeds_batch,
            down_block_res_samples_batch,
            mid_block_res_sample_batch,
            meta,
        ) in zip(
            model_input_batches,
            prompt_embeds_batches,
            down_block_res_samples_list,
            mid_block_res_sample_list,
            self.group_metas,
        ):
            if t > num_timesteps * (1 - ref_attention_end):
                replace_attention_processors(
                    self.unet,
                    SamplewiseAttnProcessor2_0,
                    attention_mask=meta[2],
                    ref_attention_mask=meta[3],
                    ref_weight=1,
                )
            else:
                replace_attention_processors(
                    self.unet,
                    SamplewiseAttnProcessor2_0,
                    attention_mask=meta[2],
                    ref_attention_mask=meta[3],
                    ref_weight=0,
                )#make the change of action

            noise_pred = self.unet(
                model_input_batch,
                t,
                encoder_hidden_states=prompt_embeds_batch,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples_batch,
                mid_block_additional_residual=mid_block_res_sample_batch,
                return_dict=False,
            )[0]
            #do the actual atention
            noise_pred_list.append(noise_pred)
        #not shure but probably the other new part
        noise_pred_list = [
            torch.index_select(
                noise_pred,
                dim=0,
                index=torch.tensor(meta[1], device=self._execution_device),
            )
            for noise_pred, meta in zip(noise_pred_list, self.group_metas)
        ]
        return torch.cat(noise_pred_list, dim=0)

    def process_batch_with_controlnet(
        
        self,
        prompt_tag,
        guess_mode,
        controlnet_keep,
        controlnet_conditioning_scale,
        i,
        t,
        latents,
        num_timesteps,
        ref_attention_end,
        cross_attention_kwargs,
        dbres_sizes_list,
        mbres_size_list,
        conditioning_images,
        prompt_embeds,
        latent_model_input,
    ):
        """Process a batch of latents with ControlNet conditioning.

        This method handles the processing of image batches using ControlNet, managing both positive
        and negative prompts in guess mode, and coordinating the noise prediction process.

        Returns:
            torch.Tensor: Predicted noise residual for the input latents.

        Notes:
            - For positive prompts or when not in guess mode, performs ControlNet inference
            - In guess mode, stores residual sizes for later use
            - For negative prompts in guess mode, creates zero tensors matching stored sizes
            - Manages memory by clearing intermediate tensors after processing"""
        if prompt_tag == "positive" or not guess_mode:
            # controlnet(s) inference
            down_block_res_samples_list ,mid_block_res_sample_list = self.process_controlnet_batch(
                latent_model_input,
                prompt_embeds,
                t,
                conditioning_images,
                controlnet_keep,
                controlnet_conditioning_scale,
                guess_mode,
                i,
            )

            if guess_mode:#see this part and what this do
                for dbres in down_block_res_samples_list:
                    dbres_sizes = []
                    for res in dbres:
                        dbres_sizes.append(res.shape)
                    dbres_sizes_list.append(dbres_sizes)

                for mbres in mid_block_res_sample_list:
                    mbres_size_list.append(mbres.shape)

        else:  # TODO-will we get here
            # Infered ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            # We copy the tensor shapes from a conditional batch
            down_block_res_samples_list = []
            mid_block_res_sample_list = []
            for dbres_sizes in dbres_sizes_list:
                down_block_res_samples_list.append(
                    [
                        torch.zeros(
                            shape,
                            device=self._execution_device,
                            dtype=latents.dtype,
                        )
                        for shape in dbres_sizes
                    ]
                )
            for mbres in mbres_size_list:
                mid_block_res_sample_list.append(
                    torch.zeros(
                        mbres,
                        device=self._execution_device,
                        dtype=latents.dtype,
                    )
                )
            dbres_sizes_list = []
            mbres_size_list = []

        """
        predict the noise residual, split into mini-batches
        Downblock res samples has n samples, we split each sample into m batches
        and re group them into m lists of n mini batch samples.
        """
        noise_pred = self.process_noise_pred(
            latent_model_input,
            prompt_embeds,
            t,
            num_timesteps,
            ref_attention_end,
            cross_attention_kwargs,
            down_block_res_samples_list,
            mid_block_res_sample_list,
        )  # yam:may be can be removed
        down_block_res_samples_list = None
        mid_block_res_sample_list = None
        noise_pred_list = None
        model_input_batches = prompt_embeds_batches = down_block_res_samples_batches = (
            mid_block_res_sample_batches
        ) = None
        return noise_pred


    def calculate_controlnet_scale(self, t, num_timesteps, control_guidance_start, initial_controlnet_conditioning_scale, control_guidance_end, controlnet_conditioning_end_scale):
        """
        Calculates the control guidance scale based on the current timestep.

        This method implements a linear interpolation of the control guidance scale
        between start and end points of the diffusion process.

        Returns:
            float: The calculated control guidance scale for the current timestep.
                - Returns initial_controlnet_conditioning_scale if before start point
                - Returns controlnet_conditioning_end_scale if after end point
                - Returns linearly interpolated value between start and end points
        """
        if (1 - t / num_timesteps) < control_guidance_start[0]:
            return (
                initial_controlnet_conditioning_scale
            )
        elif (1 - t / num_timesteps) > control_guidance_end[0]:
             return controlnet_conditioning_end_scale
        else:
            alpha = ((1 - t / num_timesteps) - control_guidance_start[0]) / (
                control_guidance_end[0] - control_guidance_start[0]
            )
            return (
                alpha * initial_controlnet_conditioning_scale
                + (1 - alpha) * controlnet_conditioning_end_scale
            )
    def get_background_colors(
        self,
        t,
        num_timesteps,
        shuffle_background_change,
        shuffle_background_end,
        background_colors,
    ):
        """
        Determines background colors for camera views based on training progress.

        This function manages the background color selection during training, implementing
        a progressive transition strategy from diverse colors to binary (black/white) 
        to fixed colors.

        Returns:
            list: List of background colors, one for each camera pose

        Example:
            >>> pipeline.get_background_colors(500, 1000, 0.7, 0.3, ['red', 'blue'])
            ['green', 'yellow']  # When in random color phase
        """
        if (1 - t / num_timesteps) < shuffle_background_change:
            return [
                random.choice(list(color_constants.keys()))
                for i in range(len(self.camera_poses))
            ]
        elif (1 - t / num_timesteps) < shuffle_background_end:
            return [
                random.choice(["black", "white"]) for i in range(len(self.camera_poses))
            ]
        else:
            return background_colors
     
    def save_intermediate_visualization(self, intermediate_results, view_fast_preview, i, t, multiview_diffusion_end, num_timesteps, tex_fast_preview, latent_tex, pred_original_sample):
        """
        Saves intermediate visualization results during the diffusion process.

        This method processes and saves both view and texture visualization results at each step.
        For views, it can either save fast preview or fully decoded images.
        For textures, it saves either preview or full RGB textures based on the preview flag.

        Saves:
            - View images as JPG files named 'step_XX.jpg'
            - Texture images as JPG/PNG files named 'texture_XX.jpg/png'
            where XX is the step number

        Notes:
            - Images are saved in the directory specified by self.intermediate_dir
            - Texture saving only occurs after multiview diffusion end threshold
        """
        decoded_results = []
        for latent_images in intermediate_results[-1]:
            if view_fast_preview:
                images = latent_preview(
                    latent_images.to(self._execution_device)
                )
            else:
                images = decode_latents(
                    self.vae, latent_images.to(self._execution_device)
                )              
            images = np.concatenate([img for img in images], axis=1)
            decoded_results.append(images)
        result_image = np.concatenate(decoded_results, axis=0)
        numpy_to_pil(result_image)[0].save(
            f"{self.intermediate_dir}/step_{i:02d}.jpg"
        )
        if not t < (1 - multiview_diffusion_end) * num_timesteps:
            if tex_fast_preview:
                tex = latent_tex.clone()
                texture_color = latent_preview(tex[None, ...])
                numpy_to_pil(texture_color)[0].save(
                    f"{self.intermediate_dir}/texture_{i:02d}.jpg"
                )
            else:
                self.uvp_rgb.to(self._execution_device)
                result_tex_rgb, result_tex_rgb_output = get_rgb_texture(
                    self.vae, self.uvp_rgb, pred_original_sample
                )
                numpy_to_pil(result_tex_rgb_output)[0].save(
                    f"{self.intermediate_dir}/texture_{i:02d}.png"
                )
                self.uvp_rgb.to("cpu")
    
    def perform_denoising_step(self, t, multiview_diffusion_end, num_timesteps, latents, latent_tex, noise_pred, background_colors, masks, current_exp, extra_step_kwargs, intermediate_results):
        """
        Performs a denoising step in the diffusion process, handling both texture-based and regular denoising.

        Returns:
            tuple: Contains:
                - latents (torch.Tensor): Updated latent representations
                - latent_tex (torch.Tensor): Updated texture latents
                - intermediate_results (list): Updated list of intermediate results
                - step_results (dict): Results from the stepping function
        """
        if t > (1 - multiview_diffusion_end) * num_timesteps:
            step_results = step_tex(
                scheduler=self.scheduler,
                uvp=self.uvp,
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                texture=latent_tex,
                return_dict=True,
                main_views=[],
                exp=current_exp,
                **extra_step_kwargs,
            )

            pred_original_sample = step_results["pred_original_sample"]
            latents = step_results["prev_sample"]
            latent_tex = step_results["prev_tex"]

            # Composit latent foreground with random color background
            background_latents = [
                self.color_latents[color] for color in background_colors
            ]
            composited_tensor = composite_rendered_view(
                self.scheduler, background_latents, latents, masks, t
            )
            latents = composited_tensor.type(latents.dtype)

            intermediate_results.append(
                (latents.to("cpu"), pred_original_sample.to("cpu"))
            )
        else:
            step_results = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=True
            )

            pred_original_sample = step_results["pred_original_sample"]
            latents = step_results["prev_sample"]
            latent_tex = None
            intermediate_results.append(
                (latents.to("cpu"), pred_original_sample.to("cpu"))
            )
        return latents, latent_tex, intermediate_results, step_results
    
    
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
        mesh_autouv=False,
        camera_azims=None,
        camera_centers=None,
        top_cameras=True,
        texture_size=1536,
        render_rgb_size=1024,
        texture_rgb_size=1024,
        multiview_diffusion_end=0.8,
        exp_start=0.0,
        exp_end=6.0,
        shuffle_background_change=0.4,
        shuffle_background_end=0.99,  # 0.4
        use_directional_prompt=True,
        ref_attention_end=0.2,
        logging_config=None,
        cond_type="depth",
    ):

       self.initialize_pipeline(
				mesh_path=mesh_path,
				mesh_transform=mesh_transform,
				mesh_autouv=mesh_autouv,
				camera_azims=camera_azims,
				camera_centers=camera_centers,
				top_cameras=top_cameras,
				ref_views=[],
				latent_size=height//8,
				render_rgb_size=render_rgb_size,
				texture_size=texture_size,
				texture_rgb_size=texture_rgb_size,

				max_batch_size=max_batch_size,

				logging_config=logging_config
			)


		# configuration of logging and preview
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
			#Control guidance allows for precise control over the generation process 
			#   by using additional conditioning information. 
			#   It helps in achieving more accurate and desired outputs, especially in complex generation tasks.
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

		#Global pooling conditions typically involve aggregating information across the entire input to make more informed predictions or decisions.
		global_pool_conditions = (
			controlnet.config.global_pool_conditions
			if isinstance(controlnet, ControlNetModel)
			else controlnet.nets[0].config.global_pool_conditions
		)
		#In guess mode, the model might attempt to generate or infer certain aspects of the output based on incomplete or ambiguous input conditions
		guess_mode = controlnet_guess_mode or global_pool_conditions


		# 3. Encode input prompt
		#The negative prompt provided by the user to guide the model on what to avoid generating.
		prompt, negative_prompt = prepare_directional_prompt(prompt, negative_prompt)

		#Low-Rank Adaptation (LoRA) use for adapting large pre-trained models to new tasks 
		#  by introducing low-rank decompositions into the model's weight matrices. 
		#  It reduces computational costs and memory usage while retaining the model's performance,
		text_encoder_lora_scale = (
			cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
		)
		#Encoding the prompts into embedding (semantic meaning of the data)
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
		#Divides the encoded embeddings into negative and positive prompt embeddings. and generate dictionary
		negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
		prompt_embed_dict = dict(zip(direction_names, [emb for emb in prompt_embeds]))
		negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in negative_prompt_embeds]))

		# (4. Prepare image) This pipeline use internal conditional images from Pytorch3D
		self.uvp.to(self._execution_device)
		conditioning_images, masks = get_conditioning_images(self.uvp, height, cond_type=cond_type)
		#converts the data type of the conditioning images to match the data type of the
		conditioning_images = conditioning_images.type(prompt_embeds.dtype)
		#normalizes the conditioning images to a range of [0, 1],
		cond = (conditioning_images/2+0.5).permute(0,2,3,1).cpu().numpy()
		cond = np.concatenate([img for img in cond], axis=1)
		numpy_to_pil(cond)[0].save(f"{self.intermediate_dir}/cond.jpg")

		# 5. Prepare timesteps
		self.scheduler.set_timesteps(num_inference_steps, device=device)
		timesteps = self.scheduler.timesteps

		# 6. Prepare latent variables
		num_channels_latents = self.unet.config.in_channels
		latents = self.prepare_latents(
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
		noise_views = self.uvp.render_textured_views()
		foregrounds = [view[:-1] for view in noise_views]
		masks = [view[-1:] for view in noise_views]
		composited_tensor = composite_rendered_view(self.scheduler, latents, foregrounds, masks, timesteps[0]+1)
		latents = composited_tensor.type(latents.dtype)
		self.uvp.to("cpu")


		# 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

		# 7.1 Create tensor stating which controlnets to keep
		controlnet_keep = []
		#This list comprehension calculates the keep values 
		#   for each pair of control_guidance_start and control_guidance_end at the current timestep i
		#    indicates whether the control should be "kept" (applied) or "ignored" (not applied)
		#    at that particular timestep.
		for i in range(len(timesteps)):
			keeps = [
				1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
				for s, e in zip(control_guidance_start, control_guidance_end)
			]
			controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        # if
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        intermediate_results = []
        background_colors = [
            random.choice(list(color_constants.keys()))
            for i in range(len(self.camera_poses))
        ]
        dbres_sizes_list = []
        mbres_size_list = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                prompt_embeds_groups = self.create_prompt_embedding_groups(
                    do_classifier_free_guidance,
                    prompt_embed_dict,
                    negative_prompt_embed_dict,
                )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                result_groups = {}

                for prompt_tag, prompt_embeds in prompt_embeds_groups.items():
                    result_groups[prompt_tag] = self.process_batch_with_controlnet(
                        prompt_tag,
                        guess_mode,
                        controlnet_keep,
                        controlnet_conditioning_scale,
                        i,
                        t,
                        latents,
                        num_timesteps,
                        ref_attention_end,
                        cross_attention_kwargs,
                        dbres_sizes_list,
                        mbres_size_list,
                        conditioning_images,
                        prompt_embeds,
                        latent_model_input,
                    )
                positive_noise_pred = result_groups["positive"]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred = result_groups["negative"] + guidance_scale * (
                        positive_noise_pred - result_groups["negative"]
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )

                self.uvp.to(self._execution_device)
                # compute the previous noisy sample x_t -> x_t-1
                # Multi-View step or individual step
                current_exp = (
                    (exp_end - exp_start) * i / num_inference_steps
                ) + exp_start
                latents, latent_tex, intermediate_results, step_results=self.perform_denoising_step(t, multiview_diffusion_end, num_timesteps, latents, latent_tex, noise_pred, background_colors, masks, current_exp, extra_step_kwargs, intermediate_results)
                del noise_pred, result_groups

                # Update pipeline settings after one step:
                # 1. Annealing ControlNet scale
                controlnet_conditioning_scale = self.calculate_controlnet_scale(t, num_timesteps, control_guidance_start, initial_controlnet_conditioning_scale, control_guidance_end, controlnet_conditioning_end_scale)

                # 2. Shuffle background colors; only black and white used after certain timestep
                background_colors = self.get_background_colors(t,num_timesteps,shuffle_background_change,shuffle_background_end,background_colors)

                # Logging at "log_interval" intervals and last step
                # Choose to uses color approximation or vae decoding
                if i % log_interval == log_interval - 1 or t == 1:
                    self.save_intermediate_visualization(intermediate_results, view_fast_preview, i, t, multiview_diffusion_end, num_timesteps, tex_fast_preview, latent_tex, pred_original_sample):

                self.uvp.to("cpu")

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                # Signal the program to skip or end
                import select
                import sys

                if select.select([sys.stdin], [], [], 0)[0]:
                    userInput = sys.stdin.readline().strip()
                    if userInput == "skip":
                        return None
                    elif userInput == "end":
                        exit(0)

        self.uvp.to(self._execution_device)
        self.uvp_rgb.to(self._execution_device)
        result_tex_rgb, result_tex_rgb_output = get_rgb_texture(
            self.vae, self.uvp_rgb, latents
        )
        self.uvp.save_mesh(
            f"{self.result_dir}/textured.obj", result_tex_rgb.permute(1, 2, 0)
        )

        self.uvp_rgb.set_texture_map(result_tex_rgb)
        textured_views = self.uvp_rgb.render_textured_views()
        textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1, ...]
        textured_views_rgb = (
            textured_views_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
        )
        v = numpy_to_pil(textured_views_rgb)[0]
        v.save(f"{self.result_dir}/textured_views_rgb.jpg")
        # display(v)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")

        return result_tex_rgb, textured_views, v
