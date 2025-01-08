import configargparse
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional


# Belongs to SyncMVD
def parse_config():
    parser = configargparse.ArgumentParser(
                        prog='Multi-View Diffusion',
                        description='Generate texture given mesh and texture prompt',
                        epilog='Refer to https://arxiv.org/abs/2311.12891 for more details')
    # File Config
    parser.add_argument('--config', type=str, required=True, is_config_file=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--mesh_app', type=str, required=True)
    parser.add_argument('--tex_app', type=str, required=True)
    parser.add_argument('--mtl_app', type=str, required=True)
    parser.add_argument('--mesh_config_relative', action='store_true', help="Search mesh file relative to the config path instead of current working directory")
    parser.add_argument('--output', type=str, default=None, help="If not provided, use the parent directory of config file for output")
    parser.add_argument('--prefix', type=str, default='MVD')
    parser.add_argument('--use_mesh_name', action='store_true')
    parser.add_argument('--timeformat', type=str, default='%d%b%Y-%H%M%S', help='Setting to None will not use time string in output directory')
    # Diffusion Config
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, default='oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=15.5, help='Recommend above 12 to avoid blurriness')
    parser.add_argument('--seed', type=int, default=0)
    # ControlNet Config
    parser.add_argument('--cond_type', type=str, default='depth', help='Support depth and normal, less multi-face in normal mode, but some times less details')
    parser.add_argument('--guess_mode', action='store_true')
    parser.add_argument('--conditioning_scale', type=float, default=0.7)
    parser.add_argument('--conditioning_scale_end', type=float, default=0.9, help='Gradually increasing conditioning scale for better geometry alignment near the end')
    parser.add_argument('--control_guidance_start', type=float, default=0.0)
    parser.add_argument('--control_guidance_end', type=float, default=0.99)
    parser.add_argument('--guidance_rescale', type=float, default=0.0, help='Not tested')
    # Multi-View Config
    parser.add_argument('--latent_view_size', type=int, default=96, help='Larger resolution, less aliasing in latent images; quality may degrade if much larger trained resolution of networks')
    parser.add_argument('--latent_tex_size', type=int, default=512, help='Originally 1536 in paper, use lower resolution save VRAM')
    parser.add_argument('--rgb_view_size', type=int, default=1536)
    parser.add_argument('--rgb_tex_size', type=int, default=1024)
    parser.add_argument('--camera_azims', type=int, nargs="*", default=[-180, -135, -90, -45, 0, 45, 90, 135], help='Place the cameras at the listed azim angles')
    parser.add_argument('--no_top_cameras', action='store_true', help='Two cameras added to paint the top surface')
    parser.add_argument('--mvd_end', type=float, default=0.8, help='Time step to stop texture space aggregation')
    parser.add_argument('--mvd_exp_start', type=float, default=0.0, help='Initial exponent for weighted texture space aggregation, low value encourage consistency')
    parser.add_argument('--mvd_exp_end', type=float, default=6.0, help='End exponent for weighted texture space aggregation, high value encourage sharper results')
    parser.add_argument('--ref_attention_end', type=float, default=0.2, help='Lower->better quality; higher->better harmonization')
    parser.add_argument('--shuffle_bg_change', type=float, default=0.4, help='Use only black and white background after certain timestep')
    parser.add_argument('--shuffle_bg_end', type=float, default=0.8, help='Don\'t shuffle background after certain timestep. background color may bleed onto object')
    parser.add_argument('--mesh_scale', type=float, default=1.0, help='Set above 1 to enlarge object in camera views')
    parser.add_argument('--keep_mesh_uv', action='store_true', help='Don\'t use Xatlas to unwrap UV automatically')
    # Logging Config
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--view_fast_preview', action='store_true', help='Use color transformation matrix instead of decoder to log view images')
    parser.add_argument('--tex_fast_preview', action='store_true', help='Use color transformation matrix instead of decoder to log texture images')
    options = parser.parse_args()

    return options


# Belongs to CIA
class Range(NamedTuple):
    start: int
    end: int


# Belongs to CIA
@dataclass
class RunConfig:
    # Appearance image path - empty as we have it by code
    app_image_path: Path = ""
    # Struct image path - empty as we have it by code
    struct_image_path: Path = ""
    # Domain name (e.g., buildings, animals)
    domain_name: Optional[str] = None
    # Output path
    output_path: Path = Path('./output')
    # Random seed
    seed: int = 42
    # Input prompt for inversion (will use domain name as default)
    prompt: Optional[str] = None
    # Number of timesteps
    num_timesteps: int = 100
    # Whether to use a binary mask for performing AdaIN
    use_masked_adain: bool = True
    # Timesteps to apply cross-attention on 64x64 layers
    cross_attn_64_range: Range = Range(start=10, end=90)
    # Timesteps to apply cross-attention on 32x32 layers
    cross_attn_32_range: Range = Range(start=10, end=70)
    # Timesteps to apply AdaIn
    adain_range: Range = Range(start=20, end=100)
    # Swap guidance scale
    swap_guidance_scale: float = 3.5
    # Attention contrasting strength
    contrast_strength: float = 1.67
    # Object nouns to use for self-segmentation (will use the domain name as default)
    object_noun: Optional[str] = None
    # Whether to load previously saved inverted latent codes
    load_latents: bool = True
    # Number of steps to skip in the denoising process (used value from original edit-friendly DDPM paper)
    skip_steps: int = 32


    def __init__(self, prompt):
        self.prompt = prompt


    def __post_init__(self):
        # save_name = f'app={self.app_image_path.stem}---struct={self.struct_image_path.stem}'
        save_name = f'app=app---struct=struct'
        # self.output_path = self.output_path / self.domain_name / save_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Handle the domain name, prompt, and object nouns used for masking, etc.
        if self.use_masked_adain and self.domain_name is None:
            raise ValueError("Must provide --domain_name and --prompt when using masked AdaIN")
        if not self.use_masked_adain and self.domain_name is None:
            self.domain_name = "object"
        if self.prompt is None:
            self.prompt = f"A photo of a {self.domain_name}"
        if self.object_noun is None:
            self.object_noun = self.domain_name

        # Define the paths to store the inverted latents to
        self.latents_path = Path(self.output_path) / "latents"
        self.latents_path.mkdir(parents=True, exist_ok=True)
        self.app_latent_save_path = self.latents_path / f"{self.app_image_path.stem}.pt"
        self.struct_latent_save_path = self.latents_path / f"{self.struct_image_path.stem}.pt"
