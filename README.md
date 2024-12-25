# Main Structure

Here we represent the structure of both works for future reference.

### General Concepts

* When running classifier free guidance (CFG), it's customary to duplicate the date in the form of `torch.cat([latents] * 2)` so that one copy is "guided" and the other isn't.

---
- - -

## Cross Image Attention

### run.py

File used to actually run the project. Takes configurations from a config.yaml file.
THe running instructions are specified in the usage section of the readme file:

To generate an image, you can simply run the `run.py` script. For example,
```
python run.py \
--app_image_path /path/to/appearance/image.png \
--struct_image_path /path/to/structure/image.png \
--output_path /path/to/output/images.png \
--domain_name [domain the objects are taken from (e.g., animal, building)] \
--use_masked_adain True \
--contrast_strength 1.67 \
--swap_guidance_scale 3.5 \
```
Notes:
- To perform the inversion, if no prompt is specified explicitly, we will use the prompt `"A photo of a [domain_name]"`
- If `--use_masked_adain` is set to `True` (its default value), then `--domain_name` must be given in order 
  to compute the masks using the self-segmentation technique.
  - In cases where the domains are not well-defined, you can also set `--use_masked_adain` to `False` and 
    no `domain_name` is required.
- You can set `--load_latents` to `True` to load the latents from a file instead of inverting the input images every time. 
  - This is useful if you want to generate multiple images with the same structure but different appearances.

Most interestingly, `run.py` calls the `model.pipe(...)` function, so we will talk about the file which defines it next.
Some of the important pipe parameters are:
* `guidance_scale=1.0` - implies no CFG
* `callback=model.get_adain_callback()` - sets use of the AdaIN module
* `swap_guidance_scale=3.5` (from config file) - sets the strength of the "specialized" CFG
---
### appearance_transfer_model.py

This file defines the transfer model, and borrows its .pipe function from the CrossImageAttentionDiffusionPipeline class defined in models/`stable_diffusion.py`.
Specifically, the model parameters are set by the `get_stable_diffusion_model()` function from `utils.model_utils.py`, however since the file has no more features I won't dwell on it.
It also defines a new class `AttentionProcessor` used in `register_attention_control(...)`. This class applies changes to the standard model to accomodate for new features, such as:
* Applying attention masking
* Performing the cross-image attention in specified timesteps for each of the `up`-steps of the UNet
The registration function then applies this modification to each layer in the network via the `register_recr(...)` function.

---

### models/stable_diffusion.py

As the original writer stated, this file defines a modification of `StableDiffusionPipeline` as provided by the `diffusers` module, by auaugmenting it to include the modified UNet model and other things.
The main function is `__call__(...)` as it defines the pipeline function, and includes the following steps:
* 0 - 5: Preprocessing, where notably step 3 encodes the prompt and step 5 encodes the latent variables. 
* 6: Ommitted.
* 7: Main denoising loop (per timestep `t`), containing several substeps:
  * Getting a noise prediction for swamp and no-swap. This is done via calling the UNet model in `models.unet_2d_condition.py`. Doesn't have anything particularly special.
  * Perform one of the following:
    * If CFG enabled - perform it.
    * Otherwise, perform a cross-image step and swap CFG. Note that the swap CFG strength increases over the loop process.
  * Perform DDPM step using the latents and noise prediction.
  * Use AdaIN.
* 8: Post-processing.

The DDPM function doesn't contain anything interesting beyond the standard DDPM theory and so I will not elaborate here.

---
- - -

## SyncMVD

### run_experiment.py

File used to run the project.
The instructions as coppied from the readme file:
To install, first clone the repository and install the basic dependencies
```bash
git clone https://github.com/LIU-Yuxin/SyncMVD.git
cd SyncMVD
conda create -n syncmvd python=3.8
conda activate syncmvd
pip install -r requirements.txt
```
Then install PyTorch3D through the following URL (change the respective Python, CUDA and PyTorch version in the link for the binary compatible with your setup), or install according to official [installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt200/download.html
```
The pretrained models will be downloaded automatically on demand, including:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [lllyasviel/control_v11f1p_sd15_depth](lllyasviel/control_v11f1p_sd15_depth)
- [lllyasviel/control_v11p_sd15_normalbae](https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae) 

The important function to mind here is the `__call__(...)` of `StableSyncMVDPipeline`, which we will tackle now.

 -----

### src/pipeline.py

Main pipeline object, inheriting from `StableDiffusionControlNetPipeline` of the `diffusers` module. Also contains some other functions we might touch on later.

><u>Note:</u> `split_groups(...)` is noted to have no positive effects. We should check whether we should just remove it altogether.

One of the arguments that the pipeline gets is `controlnet_guess_mode`. In general, guess mode is usually invoked for ControlNet whenever we don't actually know what to condition on or what we expect to get. However, since this is not the case, this argument is set to `False`.

In `initialize_pipepline(...)`:
1. Initialize camera positions. Aside from the side cameras, there are also 2 top cameras, and front and back cameras. In particular, the front camera view is used as the reference for the reference-based attention later.
2. Set up the UV mappings. As noted by the annotation, `self.uvp` is used for the latent texture and `self.uvp_rgb` for the color texture.

After that, the rest of the `__call__(...)` function is called:
* 0 - 5: Preprocessing, out of which step 4 sets up the conditional images, i.e. the depth images given by the mesh to represent its geometry in space.
* 6: Preparation of the latent variables from the prompt embeddings.
* 7 - 7.1: More setup for arguments and ControlNet. Unclear.
* 8: Main denoising loop, using the following steps for timestep `t`. Note that for each timestep, a different background color is chosen for the images, so to not have correlations to the background:
  * Set up positive and negative embeds. This is kind of a contrastive learning tool.
  * For each prompt:
    * Generate `down_block_res_sample_list` and `mid_block_res_sample_list`:
      * If the prompt is positive - feed the input, prompt embeddings and conditioning (depth) images into ControlNet.
      * Otherwise, add zero-valued tensors to both lists.
    * Replace attention processors for those in `src/syncmovd/attention.py`, which we'll touch on later.
    * Predict the noise using UNet.
  * Perform CFG.
  * Perform the denoising step using the predicted noise. The step is defined in `src/syncmvd/step.py`. The specific case of the step function is defined by if `t` is larger than some argument, i.e. at the start of the denoising process.
  * Step postprocessing, including parameter changes in ControlNet to the conditioning scale and shuffling the background color.

-----

### src/syncmvd/attention.py

Alongside `replace_attention_processors(...)` which has a self-indicative name, the file contains `SamplewiseAttnProcessor2_0` which is used to implement dot-product attention as needed for SyncMVD. The `__call__(...)` function includes the following:
* Setting up queries from the hidden state.
* Setting up the keys and values from the encoder hidden states.
* Perform the cross-view attention according to the attention mask and with the reference view.
* Perform linear projection and dropout (?) and add results to the residual hidden states.

- - -

### src/syncmvd/step.py

Performs the DDPM step, guided by the following points:
* 1: Compute $\alpha$ and $\beta$ variables according to DDPM theory.
* 2 - 3: Compute $x_0$.
* 4 - 5: Compute `pred_tex` as $x_{t-1}\sim\mathcal N({\mu_{t}, \mathbb{1}\epsilon})$ via interpolation from $x_0$ and $x_t$.
* 6: Add noise around the precited mean $x_{t-1}$.

If I understand correctly, assigning the weighted average via visibility masks in UV space happens using `uvp.bake_texture(...)`.