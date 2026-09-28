import os
import sys
import json
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import trange
from safetensors import safe_open
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .slerp import slerp
from .str_utils import str2num, tensor2str, str2tensor

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

# TODO: example workflows
# - Generating images from random latent vectors
# - Generating variants of an image by moving the latent vector in a random direction
# - Interpolating between two images by averaging their latent vectors
# - !!Completing "image analogies" like A:B::C:D (the latent vector of D is calculated as C+B-A)

LATENT_METADATA_KEY = "stylegan_latent"

class LoadStyleGANLatentImg:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "stylegan_image": (sorted(files), {"image_upload": True}),
            },
        }
    RETURN_TYPES = ("IMAGE", "STYLEGAN_LATENT")
    FUNCTION = "load_latent_image"
    CATEGORY = "StyleGAN"

    def load_latent_image(self, stylegan_image):
        image_path = folder_paths.get_annotated_filepath(stylegan_image)
        img = Image.open(image_path)

        encoded = img.info.get(LATENT_METADATA_KEY)
        if encoded is None:
            raise ValueError(f"{stylegan_image} has no '{LATENT_METADATA_KEY}' metadata; it wasn't saved with SaveStyleGANLatentImg")
        latent = str2tensor(encoded).to(get_torch_device())

        image = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image, latent)

class SaveStyleGANLatentImg:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_image": ("IMAGE",),
                "stylegan_latent": ("STYLEGAN_LATENT",),
                "filename_prefix": ("STRING", {"default": "StyleGAN"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_latent_image"
    CATEGORY = "StyleGAN"
    OUTPUT_NODE = True

    def save_latent_image(self, stylegan_latent, stylegan_image, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, stylegan_image[0].shape[1], stylegan_image[0].shape[0])

        results = []
        for batch_number, image in enumerate(stylegan_image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            index = min(batch_number, stylegan_latent.size(0) - 1)
            latent = stylegan_latent[index:index + 1]
            metadata.add_text(LATENT_METADATA_KEY, tensor2str(latent))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1

        return {"ui": {"images": results}}

class StyleGANLatentToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"
    CATEGORY = "StyleGAN/extra"

    def encode(self, stylegan_latent):
        return (tensor2str(stylegan_latent),)

class StringToStyleGANLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent_string": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "decode"
    CATEGORY = "StyleGAN/extra"

    def decode(self, stylegan_latent_string):
        return (str2tensor(stylegan_latent_string).to(get_torch_device()),)
def load_stylegan_safetensors(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        weights = {key: f.get_tensor(key) for key in f.keys()}

    arch = metadata["arch"]
    init_kwargs = json.loads(metadata["init_kwargs"])
    if arch == "stylegan3":
        from . import networks_stylegan3 as networks
    elif arch == "stylegan2":
        from . import networks_stylegan2 as networks
    else:
        raise ValueError(f"Unknown StyleGAN architecture in safetensors metadata: {arch}")

    G = networks.Generator(**init_kwargs)
    G.load_state_dict(weights)
    return G.eval()

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
        path = folder_paths.get_full_path("stylegan", stylegan_file)
        if path.endswith(".safetensors"):
            G = load_stylegan_safetensors(path)
        else:
            with open(path, 'rb') as f:
                G = pickle.load(f)['G_ema']
        return (G.to(get_torch_device()),)

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
            },
            "optional": {
                "noise_mode": (["const", "random",], {"default": "const"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE","STYLEGAN_LATENT",)
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
        return (imgs, stylegan_latent, )

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
    "LoadStyleGANLatentImg": LoadStyleGANLatentImg,
    "SaveStyleGANLatentImg": SaveStyleGANLatentImg,
    "LoadStyleGAN": LoadStyleGAN,
    "GenerateStyleGANLatent": GenerateStyleGANLatent,
    "StyleGANSampler": StyleGANSampler,
    "BlendStyleGANLatents": BlendStyleGANLatents,
    "BatchAverageStyleGANLatents": BatchAverageStyleGANLatents,
    "StyleGANLatentFromBatch": StyleGANLatentFromBatch,
    "StyleGANInversion": StyleGANInversion,
    "StyleGANLatentToString": StyleGANLatentToString,
    "StringToStyleGANLatent": StringToStyleGANLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStyleGANLatentImg": "Load StyleGAN Latent from Image Metadata",
    "SaveStyleGANLatentImg": "Save StyleGAN Latent to Image Metadata",
    "LoadStyleGAN": "Load StyleGAN Model",
    "GenerateStyleGANLatent": "Generate StyleGAN Latent",
    "StyleGANSampler": "StyleGAN Sampler",
    "BlendStyleGANLatents": "Blend StyleGAN Latents (lerp or slerp)",
    "BatchAverageStyleGANLatents": "Batch Average StyleGAN Latents",
    "StyleGANLatentFromBatch": "StyleGAN Latent From Batch",
    "StyleGANInversion": "StyleGAN Inversion",
    "StyleGANLatentToString": "StyleGAN Latent to String",
    "StringToStyleGANLatent": "String to StyleGAN Latent",
}