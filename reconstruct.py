import torch
from src.cit_utils import load_size
from invert_all import pipe, cfg
from PIL import Image

path_of_latent = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/inverted/inverted_4.pt"
path_of_condition_photo = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/face_cond_4.jpg"
input_image_prompt = "Portrait photo of Kratos, god of war."

def save_step_callback(step, timestep, latents):
    # Decode latents to image
    print("hi")
    images = pipe.decode_latents(latents)
    for i, img in enumerate(images):
        pil_image = Image.fromarray((img * 255).astype("uint8"))
        pil_image.save(f"/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/tmp_recon/step_{step:03d}_image_{i}.png")

def main():
    inverted_latents = torch.load(path_of_latent)
    condition_photo = load_size(path_of_condition_photo, size=512)
    
    result = pipe(
        input_image_prompt,
        image=[condition_photo],
        latents=inverted_latents[-1][None].to(torch.float32), 
        num_inference_steps=cfg.num_timesteps,
        guidance_scale=cfg.swap_guidance_scale,
        
        callback=save_step_callback,
    )
    
    # Save the final image
    final_image = result.images[0]
    final_image.save("reconstructed1.png")

if __name__ == "__main__":
    main()