# Main Structure

Here we represent the structure of both works for future reference.

### General Concepts

* When running classifier free guidance (CFG), it's customary to duplicate the date in the form of `torch.cat([latents] * 2)` so that one copy is "guided" and the other isn't.

## Cross Image Attention

### run.py

File used to actually run the project. Takes configurations from a config.yaml file.
THe running instructions are specified in the usage section of the readme file:

<p align="center">
<img src="docs/general_results.jpg" width="90%"/>  
<br>
Sample appearance transfer results obtained by our cross-image attention technique.
</p>

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

## appearance_transfer_model.py

This file defines the transfer model, and borrows its .pipe function from the CrossImageAttentionDiffusionPipeline class defined in models/`stable_diffusion.py`.
Specifically, the model parameters are set by the `get_stable_diffusion_model()` function from `utils.model_utils.py`, however since the file has no more features I won't dwell on it.
It also defines a new class `AttentionProcessor` used in `register_attention_control(...)`. This class applies changes to the standard model to accomodate for new features, such as:
* Applying attention masking
* Performing the cross-image attention in specified timesteps for each of the `up`-steps of the UNet
The registration function then applies this modification to each layer in the network via the `register_recr(...)` function.

## models/stable_diffusion.py

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

