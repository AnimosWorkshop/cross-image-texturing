from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionXLPipeline
from diffusers.utils import load_image

# suppress partial model loading warning
logging.set_verbosity_error()

import os
from PIL import Image, ImageFilter
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import torchvision.transforms as T
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_timesteps(scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == 'sdxl1.0':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "bdsqlsz/stable-diffusion-v1-5"
        elif self.sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
            self.use_depth = True
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        if 'xl' in model_key:
            self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, scheduler=self.scheduler).to("cuda")
            
            # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", # revision="fp16",
            #                                         torch_dtype=torch.float16).to(self.device)
            # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", # revision="fp16",
            #                                                 torch_dtype=torch.float16).to(self.device)
            # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", # revision="fp16",
            #                                                 torch_dtype=torch.float16).to(self.device)
            # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        else:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                device = device)
            self.vae = pipe.vae.to(device)
            self.unet = pipe.unet.to(device)
            self.text_encoder = pipe.text_encoder.to(device)
            self.scheduler = pipe.scheduler
            self.tokenizer = pipe.tokenizer
            self.device = device
            self.controlnet = pipe.controlnet.to(device)
            self.pipeline = pipe


            # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", revision="fp16",
            #                                         torch_dtype=torch.float16).to(self.device)
            # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
            #                                                 torch_dtype=torch.float16).to(self.device)
            # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
            #                                                 torch_dtype=torch.float16).to(self.device)
            # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')

        self.inversion_func = self.cn_ddim_inversion

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        # blur image
        # blurred_image = image_pil.filter(ImageFilter.GaussianBlur(radius=9))
        # blurred_image.save('very_blurred_image.png')
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    # def prepare_image_for_cn(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        

    @torch.no_grad()
    def step_with_cn(self, cond_image, cond_batch, latent, t):
        # CIT - we now use the controlnet to generate the residuals                
        latent_model_input = torch.cat([latent] * 2)
        control_model_input = latent_model_input

        control_model_cond_image_input = self.pipeline.prepare_image(
                image=cond_image,
                width=None,
                height=None,
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=torch.float16,
            )
        
        
        down_block_res_samples, mid_block_res_sample = self.controlnet( #TODO initialize self.controlnet in __init__
            control_model_input,
            t,
            encoder_hidden_states=cond_batch,
            controlnet_cond=control_model_cond_image_input,
            conditioning_scale=1, # we don't support negative prompts
            guess_mode=False, 
            return_dict=False,
        )
        
        # CIT - passing the cn residuals to the unet
        #TODO bug: unet is called several times (wtf) latent_model_input.shape[0] is changed from 2 to 4 somehow, and the bug is that eps results in a tensor with shape[0] = 4 which is incompatible with the rest of the module.
        eps = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=cond_batch,
            # cross_attention_kwargs=cross_attention_kwargs, #TODO might just be None
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        
        return eps


    @torch.no_grad()
    def cn_ddim_inversion(self, cond_image, cond_embeds, latent, save_path, save_latents=True,
                                timesteps_to_save=None):
        """
        cond_image: image to condition on. should be PIL image
        cond_embeds: text embeddings that result from self.get_text_embeds
        latent: encoded latents of the image to invert
        """
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond_embeds.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                
                eps = self.step_with_cn(cond_image, cond_batch, latent, t)                    
                    
                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond_image, cond_embeds, save_path, save_latents=False, timesteps_to_save=None):
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                    cond_batch = cond_embeds.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    # eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample
                    eps = self.step_with_cn(cond_image, cond_batch, x, t)                    

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps

            if save_latents:
                torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x

    @torch.no_grad()
    def extract_latents(self, cond_image_path, num_steps, data_path, save_path, timesteps_to_save,
                        inversion_prompt='', extract_reverse=False):
        self.scheduler.set_timesteps(num_steps)

        # cond_embeds = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0) # GOTCHA
        cond_embeds = self.get_text_embeds(inversion_prompt, "")
        
        image = self.load_img(data_path)

        
        latent = self.encode_imgs(image)

        cond_image = Image.open(cond_image_path)
        inverted_x = self.inversion_func(cond_image, cond_embeds, latent, save_path, save_latents=not extract_reverse,
                                         timesteps_to_save=timesteps_to_save)
        latent_reconstruction = self.ddim_sample(inverted_x, cond_image, cond_embeds, save_path, save_latents=extract_reverse,
                                                 timesteps_to_save=timesteps_to_save)
        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        return rgb_reconstruction  # , latent_reconstruction

def run(opt):
    # timesteps to save
    device = 'cuda'
    if opt.sd_version == '2.1':
        model_key = "stabilityai/stable-diffusion-2-1-base"
    elif opt.sd_version == 'sdxl1.0':
        model_key = "stabilityai/stable-diffusion-xl-base-1.0"
    elif opt.sd_version == '2.0':
        model_key = "stabilityai/stable-diffusion-2-base"
    elif opt.sd_version == '1.5':
        model_key = "runwayml/stable-diffusion-v1-5"
    elif opt.sd_version == 'depth':
        model_key = "stabilityai/stable-diffusion-2-depth"
    
    seed_everything(opt.seed)

    extraction_path_prefix = "_reverse" if opt.extract_reverse else "_forward"
    save_path = os.path.join(opt.save_dir + extraction_path_prefix, os.path.splitext(os.path.basename(opt.data_path))[0])

    os.makedirs(save_path, exist_ok=True)

    model = Preprocess(device, sd_version=opt.sd_version, hf_key=None)


        # toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # toy_scheduler.set_timesteps(opt.save_steps)
        # timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=opt.save_steps,
        #                                                        strength=1.0,
        #                                                        device=device)

    recon_image = model.extract_latents(cond_image_path=opt.cond_image_path,
                                        data_path=opt.data_path,
                                        num_steps=opt.steps,
                                        save_path=save_path,
                                        timesteps_to_save=None,
                                        inversion_prompt=opt.inversion_prompt,
                                        extract_reverse=opt.extract_reverse)

    T.ToPILImage()(recon_image[0]).save(os.path.join(save_path, f'recon.jpg'))

class LatentExtractor:
    def __init__(self, device='cuda', sd_version='2.1'):
        self.device = device
        self.sd_version = sd_version

        self.model = Preprocess(self.device, sd_version=self.sd_version, hf_key=None)

    def run(self, opt):
        seed_everything(opt.seed)

        extraction_path_prefix = "_reverse" if opt.extract_reverse else "_forward"
        save_path = os.path.join(opt.save_dir + extraction_path_prefix, os.path.splitext(os.path.basename(opt.data_path))[0])

        os.makedirs(save_path, exist_ok=True)

        recon_image = self.model.extract_latents(
            data_path=opt.data_path,
            num_steps=opt.steps,
            save_path=save_path,
            timesteps_to_save=None,
            inversion_prompt=opt.inversion_prompt,
            extract_reverse=opt.extract_reverse
        )

        T.ToPILImage()(recon_image[0]).save(os.path.join(save_path, f'recon.jpg'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond_image_path', type=str)
    parser.add_argument('--data_path', type=str,
                        default='style_images/clouds.jpg')
    parser.add_argument('--save_dir', type=str, default='latents')
    parser.add_argument('--sd_version', type=str, choices=['1.5', '2.0', '2.1', 'sdxl1.0'],# formerly 2.1 by default
                        help="stable diffusion version")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30)
    # parser.add_argument('--save-steps', type=int, default=1000)
    parser.add_argument('--inversion_prompt', type=str, default='a photo of a')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    opt = parser.parse_args()
    run(opt)
