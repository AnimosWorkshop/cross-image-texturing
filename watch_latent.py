from invert_all import pipe
from src.cit_utils import show_latents, lidor_dir
import torch


# def watch_latent():
#     latent_path = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/inverted/noisy/inverted_4.pt"
#     latent = torch.load(latent_path)
#     show_latents(latent, pipe.vae, lidor_dir)
    
def rearrange_inverted():
    inverted_path = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/inverted"
    dest_path = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/inverted/noisy"
    for i in range(10):
        inverted = torch.load(f"{inverted_path}/inverted_{i}.pt")
        torch.save(inverted[0], f"{dest_path}/inverted_{i}.pt")
    
if __name__ == "__main__":
    # rearrange_inverted()
    # watch_latent()
    show_latents("/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/data/latents_app.pt", pipe.vae)
    