import os
import sys
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import trange

from .slerp import slerp
from .str_utils import str2num

from . import dnnlib
from . import torch_utils
# from . import legacy
sys.modules["dnnlib"] = dnnlib
sys.modules["torch_utils"] = torch_utils
# sys.modules["legacy"] = legacy

import folder_paths
from comfy.utils import PROGRESS_BAR_ENABLED, ProgressBar
from comfy.model_management import get_torch_device

# set the models directory
if "stylegan" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "stylegan")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["stylegan"]
folder_paths.folder_names_and_paths["stylegan"] = (current_paths, folder_paths.supported_pt_extensions)

class LoadStyleGAN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_file": (folder_paths.get_filename_list("stylegan"), ),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN",)
    FUNCTION = "load_stylegan"
    CATEGORY = "StyleGAN"
    
    def load_stylegan(self, stylegan_file):
        with open(folder_paths.get_full_path("stylegan", stylegan_file), 'rb') as f:
            G = pickle.load(f)['G_ema'].to(get_torch_device())
        return (G,)

class GenerateStyleGANLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                # "class_label": ("INT", {"default": -1, "min": -1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024}),                
                "psi": ("FLOAT", {"default": 0.7, "min": -1.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN"
    
    def generate_latent(self, stylegan_model, seed, batch_size, psi):
        if seed < 0xffffffff:
            # legacy seed compatible with sd-webui-gan-generator
            z = np.random.RandomState(seed).randn(batch_size, stylegan_model.z_dim)
            z = torch.tensor(z, dtype=torch.float32).to(get_torch_device())
        else:
            torch.manual_seed(seed)
            z = torch.randn([batch_size, stylegan_model.z_dim]).to(get_torch_device())


        w = []
        w_avg = stylegan_model.mapping.w_avg
        for i in range(batch_size):
            _w = stylegan_model.mapping(z, None)
            _w = w_avg + (_w - w_avg) * psi
            w.append(_w)
        
        return (torch.cat(w, dim=0), )

class StyleGANSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "noise_mode": (['const', 'random'],),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "StyleGAN"
    
    def generate_image(self, stylegan_model, stylegan_latent, noise_mode):
        imgs = []
        batch_size = stylegan_latent.size(0)
        pbar = None
        if PROGRESS_BAR_ENABLED and batch_size > 1:
            pbar = ProgressBar(batch_size)
        for i in trange(batch_size):
            img = stylegan_model.synthesis(stylegan_latent[i].unsqueeze(0), noise_mode=noise_mode)
            img = torch.permute(img, (0, 2, 3, 1)) # BCHW -> BHWC
            img = torch.clip(img / 2 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
            imgs.append(img)
            if pbar is not None:
                pbar.update(1)
        
        imgs = torch.cat(imgs, dim=0)
        return (imgs, )

class StyleGANInversion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "image": ("IMAGE", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "num_steps": ("INT", {"default": 1000, "min": 1}),
                "w_avg_samples": ("INT", {"default": 10000, "min": 1, "max": 100000}),
                "initial_learning_rate": ("FLOAT", {"default": 0.1, "min": 0.00001, "max": 1.0, "step": 0.00001}),
                "initial_noise_factor": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "lr_rampdown_length": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lr_rampup_length": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_ramp_length": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "regularize_noise_weight": ("FLOAT", {"default": 1e5, "min": 0.0, "max": 1e7}),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT", "STYLEGAN_LATENT")
    RETURN_NAMES = ("training_latents", "final_latent")
    FUNCTION = "train_inversion"
    CATEGORY = "StyleGAN"
    
    def train_inversion(
        self,
        stylegan_model,
        image,
        seed,
        num_steps,
        w_avg_samples,
        initial_learning_rate,
        initial_noise_factor,
        lr_rampdown_length,
        lr_rampup_length,
        noise_ramp_length,
        regularize_noise_weight
        ):
        
        device = get_torch_device()
        img_resolution = stylegan_model.img_resolution
        target_image = torch.permute(image[...,:3], (0, 3, 1, 2)) # BHWC -> BCHW, RGB only
        if target_image.shape != (stylegan_model.img_channels, img_resolution, img_resolution):
            target_image = F.interpolate(target_image, size=(img_resolution, img_resolution), mode='area')
        target_image = target_image[0] * 255
        
        from .projector import project
        
        projected_w_steps = project(
            stylegan_model,
            target_image,
            num_steps = num_steps,
            w_avg_samples = w_avg_samples,
            seed = seed,
            initial_learning_rate       = initial_learning_rate,
            initial_noise_factor        = initial_noise_factor,
            lr_rampdown_length          = lr_rampdown_length,
            lr_rampup_length            = lr_rampup_length,
            noise_ramp_length           = noise_ramp_length,
            regularize_noise_weight     = regularize_noise_weight,
            device                      = device,
            )
        
        return (projected_w_steps, projected_w_steps[-1].unsqueeze(0))

class BlendStyleGANLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_1": ("STYLEGAN_LATENT", ),
                "latent_2": ("STYLEGAN_LATENT", ),
                "blend": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.001}),
                "mode": (["slerp", "lerp"],),
                "mask": (["total (0xFFFF)", "coarse (0xFF00)", "mid (0x0FF0)", "fine (0x00FF)", "alt1 (0xF0F0)", "alt2 (0x0F0F)", "alt3 (0xF00F)"],)
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, latent_1, latent_2, blend, mode, mask):
        if latent_1.shape != latent_2.shape:
            raise Exception(f"latent_1 shape {latent_1.shape} and latent_2 shape {latent_2.shape} do not match!")

        z = latent_1.clone() # transfer onto L image as default

        if mask == 0xFFFF:
            blend = self.jmap(blend, -1.0, 1.0, 0.0, 1.0) # make unipolar
        else:
            if blend > 0: # transfer L onto R
                z = latent_2.clone()
            else: # transfer R onto L
                blend = abs(blend)
                latent_1,latent_2 = latent_2,latent_1 # swap L and R

        mask = self.num2mask( str2num(mask) )

        m = slerp if mode == "slerp" else torch.lerp
        z[:,mask,:] = m(latent_1[:,mask,:], latent_2[:,mask,:], blend)

        return (z,)

    # @classmethod
    def jmap(self, sourceValue, sourceRangeMin, sourceRangeMax, targetRangeMin, targetRangeMax) -> float:
        if sourceRangeMax == sourceRangeMin:
            raise ValueError("mapping from a range of zero will produce NaN!")
        return targetRangeMin + ((targetRangeMax - targetRangeMin) * (sourceValue - sourceRangeMin)) / (sourceRangeMax - sourceRangeMin)

    # @classmethod
    def num2mask(self, num: int) -> np.ndarray:
        return np.array([x=='1' for x in bin(num)[2:].zfill(16)], dtype=bool)



class BatchAverageStyleGANLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT", ),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, stylegan_latent):
        w = torch.mean(stylegan_latent, dim=0, keepdim=True)
        std, mean = torch.std_mean(w)
        w = (w - mean) / std
        
        return (w, )

class StyleGANLatentFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "index": ("INT", {"default": 0, "min": 0}),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, stylegan_latent, index):
        clipped_index = min(index, stylegan_latent.size(0) - 1)
        w = stylegan_latent[clipped_index].unsqueeze(0).detach().clone()
        
        return (w, )

NODE_CLASS_MAPPINGS = {
    "LoadStyleGAN": LoadStyleGAN,
    "GenerateStyleGANLatent": GenerateStyleGANLatent,
    "StyleGANSampler": StyleGANSampler,
    "BlendStyleGANLatents": BlendStyleGANLatents,
    "BatchAverageStyleGANLatents": BatchAverageStyleGANLatents,
    "StyleGANLatentFromBatch": StyleGANLatentFromBatch,
    "StyleGANInversion": StyleGANInversion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStyleGAN": "Load StyleGAN Model",
    "GenerateStyleGANLatent": "Generate StyleGAN Latent",
    "StyleGANSampler": "StyleGAN Sampler",
    "BlendStyleGANLatents": "Blend StyleGAN Latents (lerp or slerp)",
    "BatchAverageStyleGANLatents": "Batch Average StyleGAN Latents",
    "StyleGANLatentFromBatch": "StyleGAN Latent From Batch",
    "StyleGANInversion": "StyleGAN Inversion",
}