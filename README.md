# Cross-Image Attention for Zero-Shot Appearance Transfer (SIGGRAPH 2024)

> Yuval Alaluf*, Daniel Garibi*, Or Patashnik, Hadar Averbuch-Elor, Daniel Cohen-Or  
> Tel Aviv University  
> \* Denotes equal contribution  
>
> Recent advancements in text-to-image generative models have demonstrated a remarkable ability to capture a deep semantic understanding of images. In this work, we leverage this semantic knowledge to transfer the visual appearance between objects that share similar semantics but may differ significantly in shape. To achieve this, we build upon the self-attention layers of these generative models and introduce a cross-image attention mechanism that implicitly establishes semantic correspondences across images. Specifically, given a pair of images ––– one depicting the target structure and the other specifying the desired appearance ––– our cross-image attention combines the queries corresponding to the structure image with the keys and values of the appearance image. This operation, when applied during the denoising process, leverages the established semantic correspondences to generate an image combining the desired structure and appearance. In addition, to improve the output image quality, we harness three mechanisms that either manipulate the noisy latent codes or the model's internal representations throughout the denoising process. Importantly, our approach is zero-shot, requiring no optimization or training. Experiments show that our method is effective across a wide range of object categories and is robust to variations in shape, size, and viewpoint between the two input images.

<a href="https://arxiv.org/abs/2311.03335"><img src="https://img.shields.io/badge/arXiv-2311.03335-b31b1b.svg" height=22.5></a>
<a href="https://garibida.github.io/cross-image-attention/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=20.5></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yuvalalaluf/cross-image-attention)

<p align="center">
<img src="docs/teaser.jpg" width="90%"/>  
<br>
Given two images depicting a source structure and a target appearance, our method generates an image merging the structure of one image with the appearance of the other in a zero-shot manner.
</p>


## Description  
Official implementation of our Cross-Image Attention and Appearance Transfer paper.


## Environment
Our code builds on the requirement of the `diffusers` library. To set up their environment, please run:
```
conda env create -f environment/environment.yaml
conda activate cross_image
```

## Usage  
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


### Demo Notebook 
<p align="center">
<img src="docs/grids.jpg" width="90%"/>  
<br>
Additional appearance transfer results obtained by our cross-image attention technique.
</p>

We also provide a notebook to run in Google Colab, please see `notebooks/demo.ipynb`.


## HuggingFaceDemo :hugs:
We also provide a simple HuggingFace demo to run our method on your own images.   
Check it out [here](https://huggingface.co/spaces/yuvalalaluf/cross-image-attention)!


## Acknowledgements 
This code builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library. In addition, we 
borrow code from the following repositories: 
- [Edit-Friendly DDPM Inversion](https://github.com/inbarhub/DDPM_inversion) for inverting the input images.
- [Prompt Mixing](https://github.com/orpatashnik/local-prompt-mixing) for computing the masks used in our AdaIN operation.
- [FreeU](https://github.com/ChenyangSi/FreeU) for improving the general generation quality of Stable Diffusion.


## Citation
If you use this code for your research, please cite the following work: 
```
@misc{alaluf2023crossimage,
      title={Cross-Image Attention for Zero-Shot Appearance Transfer}, 
      author={Yuval Alaluf and Daniel Garibi and Or Patashnik and Hadar Averbuch-Elor and Daniel Cohen-Or},
      year={2023},
      eprint={2311.03335},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# SyncMVD

Official [Pytorch](https://pytorch.org/) & [Diffusers](https://github.com/huggingface/diffusers) implementation of the paper:



**[Text-Guided Texturing by Synchronized Multi-View Diffusion](https://arxiv.org/pdf/2311.12891)**

<!-- Authors: Yuxin Liu, Minshan Xie, Hanyuan Liu, Tien-Tsin Wong -->

<img src=assets/teaser.jpg width=768>

**SyncMVD** can generate texture for a 3D object from a text prompt using a **Sync**hronized **M**ulti-**V**iew **D**iffusion approach.
The method shares the denoised content among different views in each denoising step to ensure texture consistency and avoid seams and fragmentation (fig a).



<table style="table-layout: fixed; width: 100%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
  <tr>
  <td>
    <img src=assets/gif/batman.gif width="170">
  </td>
  <td>
    <img src=assets/gif/david.gif width="170">
  </td>
  <td>
    <img src=assets/gif/teapot.gif width="170">
  </td>
  <td>
    <img src=assets/gif/vangogh.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"Photo of Batman, sitting on a rock."</td>
    <td style="font-family:courier">"Publicity photo of a 60s movie, full color."</td>
    <td style="font-family:courier">"A photo of a beautiful chintz glided teapot."</td>
    <td style="font-family:courier">"A beautiful oil paint of a stone building in Van Gogh style."</td>
  </tr>
   <tr>
  <td>
    <img src=assets/gif/gloves.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/link.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/house.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/luckycat.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A photo of a robot hand with mechanical joints."</td>
    <td style="font-family:courier">"Photo of link in the legend of zelda, photo-realistic, unreal 5."</td>
    <td style="font-family:courier">"Photo of a lowpoly fantasy house from warcraft game, lawn."</td>
    <td style="font-family:courier">"Blue and white pottery style lucky cat with intricate patterns."</td>
  </tr>

  <tr>
  <td>
    <img src=assets/gif/chair.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/Moai.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/sneakers.gif width="170">
  </td>
  <td>
    <img src=assets/gif/dragon.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A photo of an beautiful embroidered seat with royal patterns"</td>
    <td style="font-family:courier">"Photo of James Harden."</td>
    <td style="font-family:courier">"A photo of a gray and black Nike Airforce high top sneakers."</td>
    <td style="font-family:courier">"A photo of a Chinese dragon sculpture, glazed facing, vivid colors."</td>
  </tr>

  <tr>
  <td>
    <img src=assets/gif/guardian.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/horse.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/jackiechan.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/knight.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A muscular man wearing grass hula skirt."</td>
    <td style="font-family:courier">"Photo of a horse."</td>
    <td style="font-family:courier">"A Jackie Chan figure."</td>
    <td style="font-family:courier">"A photo of a demon knight, flame in eyes, warcraft style."</td>
  </tr>

</table>

## Installation :wrench:
The program is developed and tested on Linux system with Nvidia GPU. If you find compatibility issues on Windows platform, you can also consider using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

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

## Data :floppy_disk:
The current program based on [PyTorch3D](https://github.com/facebookresearch/pytorch3d) library requires a input `.obj` mesh with `.mtl` material and related textures to read the original UV mapping of the object, which may require manual cleaning. Alternatively the program also support auto unwrapping based on [XAtlas](https://github.com/jpcy/xatlas) to load mesh that does not met the above requirements. The program also supports loading `.glb` mesh, but it may not be stable as its a PyTorch3D experiment feature.

To avoid unexpected artifact, the object being textured should avoid flipped face normals and overlapping UV, and keep the number of triangle faces within around 40,000. You can try [Blender](https://www.blender.org/) for manual mesh cleaning and processing, or its python scripting for automation.

You can try out the method with the following pre-processed meshes and configs:
- [Face - "Portrait photo of Kratos, god of war."](data/face/config.yaml) (by [2on](https://sketchfab.com/3d-models/face-ffde29cb64584cf1a939ac2b58d0a931))
- [Sneaker - "A photo of a camouflage military boot."](data/sneaker/config.yaml) (by [gianpego](https://sketchfab.com/3d-models/air-jordan-1-1985-2614cef9a3724ec5852144446fbb726f))

## Inference :rocket:
```bash
python run_experiment.py --config {your config}.yaml
```
Refer to [config.py](src/configs.py) for the list of arguments and settings you can adjust. You can change these settings by including them in a `.yaml` config file or passing the related arguments in command line; values specified in command line will overwrite those in config files.

When no output path is specified, the generated result will be placed in the same folder as the config file by default.

## License :scroll:
The program licensed under [MIT License](LICENSE).

## Citation :memo:
```bibtex
@article{liu2023text,
  title={Text-Guided Texturing by Synchronized Multi-View Diffusion},
  author={Liu, Yuxin and Xie, Minshan and Liu, Hanyuan and Wong, Tien-Tsin},
  journal={arXiv preprint arXiv:2311.12891},
  year={2023}
}
```
