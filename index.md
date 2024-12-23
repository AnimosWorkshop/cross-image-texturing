# Main Structure

Here we represent the structure of both works for future reference.

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

## appearance_transfer_model.py

This file defines the transfer model, and borrows its .pipe function from the CrossImageAttentionDiffusionPipeline class defined in models/`stable_diffusion.py`.
It also defines a new class in `register_attention_control(...)`, called `AttentionProcessor`, the importance of which I'm sure we will learn later.

## models/stable_diffusion.py

As the original writer stated, this file defines a modification of `StableDiffusionPipeline` as provided by the `diffusers` module, by auaugmenting it to include the modified UNet model and other things.