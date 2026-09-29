import os
import sys
import json
import inspect
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.weak import WeakTensorKeyDictionary
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

from .convert_to_safetensors import save_stylegan_safetensors

import folder_paths
from comfy.utils import PROGRESS_BAR_ENABLED, ProgressBar
from comfy.model_management import get_torch_device

# set the models directory
if "stylegan" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "stylegan")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["stylegan"]
folder_paths.folder_names_and_paths["stylegan"] = (current_paths, folder_paths.supported_pt_extensions)

# direction bundles live in their own folder, separate from model checkpoints,
# so the two don't show up mixed together in either node's file picker
if "stylegan_directions" not in folder_paths.folder_names_and_paths:
    current_direction_paths = [os.path.join(folder_paths.models_dir, "stylegan_directions")]
else:
    current_direction_paths, _ = folder_paths.folder_names_and_paths["stylegan_directions"]
folder_paths.folder_names_and_paths["stylegan_directions"] = (current_direction_paths, {".safetensors"})

# TODO: example workflows
# - Generating images from random latent vectors
# - Generating variants of an image by moving the latent vector in a random direction
# - Interpolating between two images by averaging their latent vectors
# - !!Completing "image analogies" like A:B::C:D (the latent vector of D is calculated as C+B-A)

LATENT_METADATA_KEY = "stylegan_latent"
MODEL_METADATA_KEY = "stylegan_model_file"

# --- Auto-embed latent metadata into any SaveImage/PreviewImage output --------
#
# Rather than requiring SaveStyleGANLatentImg specifically, register the latent
# against the in-memory image batch tensor's identity the moment StyleGANSampler
# produces it, then patch core SaveImage/PreviewImage to splice our metadata
# chunk into whatever file they just wrote, if (and only if) that exact tensor
# object is one we recognize. No regeneration, no graph-walking: this only
# fires when the pixels really are the direct, unmodified sampler output (any
# processing node in between produces a new tensor object, breaking identity,
# which is also the correct behavior -- the latent may no longer match).
#
# Weak-keyed by design: a strong-ref registry would pin GPU memory for whatever
# images happened to still be in it, fighting ComfyUI's own execution cache
# (which is what actually decides how long a tensor needs to stay alive).
_latent_registry = WeakTensorKeyDictionary()

def _register_latent_for_images(image_tensor, latent_tensor, model_file=None):
    _latent_registry[image_tensor] = (latent_tensor.detach().cpu(), model_file)

def _lookup_latent_for_images(image_tensor):
    return _latent_registry.get(image_tensor, (None, None))

def _embed_latent_in_png(path, latent, model_file, compress_level):
    img = Image.open(path)
    metadata = PngInfo()
    for k, v in img.info.items():
        if isinstance(v, str):
            metadata.add_text(k, v)
    metadata.add_text(LATENT_METADATA_KEY, tensor2str(latent))
    if model_file:
        metadata.add_text(MODEL_METADATA_KEY, model_file)
    img.save(path, pnginfo=metadata, compress_level=compress_level)

def _patch_core_save_nodes():
    import nodes as comfy_core_nodes
    from comfy.cli_args import args as comfy_args

    if getattr(comfy_core_nodes.SaveImage, "_stylegan_latent_patched", False):
        return  # already patched (e.g. module re-imported under a dev auto-reloader)

    original_save_images = comfy_core_nodes.SaveImage.save_images
    if "images" not in inspect.signature(original_save_images).parameters:
        raise RuntimeError("SaveImage.save_images no longer takes an 'images' parameter; ComfyUI's save-node API has changed")

    def patched_save_images(self, images, *args, **kwargs):
        result = original_save_images(self, images, *args, **kwargs)
        latent, model_file = _lookup_latent_for_images(images)
        if latent is not None and not comfy_args.disable_metadata:
            try:
                entries = result.get("ui", {}).get("images", [])
                type_to_dir = {
                    "output": folder_paths.get_output_directory(),
                    "temp": folder_paths.get_temp_directory(),
                    "input": folder_paths.get_input_directory(),
                }
                compress_level = getattr(self, "compress_level", 4)
                for i, entry in enumerate(entries):
                    idx = min(i, latent.size(0) - 1)
                    directory = type_to_dir.get(entry.get("type"), folder_paths.get_output_directory())
                    path = os.path.join(directory, entry.get("subfolder", ""), entry["filename"])
                    _embed_latent_in_png(path, latent[idx:idx + 1], model_file, compress_level)
            except Exception as e:
                print(f"StyleGAN: couldn't auto-embed latent metadata: {e}")
        return result

    patched_save_images._stylegan_latent_patched = True
    comfy_core_nodes.SaveImage.save_images = patched_save_images
    comfy_core_nodes.SaveImage._stylegan_latent_patched = True

try:
    _patch_core_save_nodes()
except Exception as e:
    print(f"StyleGAN: couldn't patch SaveImage for latent auto-embedding: {e}")

class LoadStyleGANLatentImg:
    @classmethod
    def INPUT_TYPES(s):
        # Include output/ and temp/, not just input/: a saved StyleGAN image with
        # its latent metadata is just as likely to still be sitting in output/ as
        # to have been copied into input/, and ComfyUI's prompt validation rejects
        # any value not in this list regardless of where the file actually lives.
        # Filter to images that actually carry our metadata, not every image in
        # those folders -- output/ in particular tends to be full of unrelated
        # generations, and PNG metadata is cheap to check (PIL doesn't decode
        # pixel data just to read .info).
        candidates = set()
        for directory in (folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory()):
            candidates.update(f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))
        candidates = folder_paths.filter_files_content_types(sorted(candidates), ["image"])

        files = []
        for f in candidates:
            path = folder_paths.get_annotated_filepath(f) if os.path.isfile(os.path.join(folder_paths.get_input_directory(), f)) else None
            if path is None:
                for directory in (folder_paths.get_output_directory(), folder_paths.get_temp_directory()):
                    candidate_path = os.path.join(directory, f)
                    if os.path.isfile(candidate_path):
                        path = candidate_path
                        break
            try:
                with Image.open(path) as img:
                    if LATENT_METADATA_KEY in img.info:
                        files.append(f)
            except Exception:
                pass

        return {
            "required": {
                "stylegan_image": (files, {"image_upload": True}),
            },
        }
    RETURN_TYPES = ("IMAGE", "STYLEGAN_LATENT", "STRING")
    RETURN_NAMES = ("image", "stylegan_latent", "model_file")
    FUNCTION = "load_latent_image"
    CATEGORY = "StyleGAN"

    def load_latent_image(self, stylegan_image):
        image_path = folder_paths.get_annotated_filepath(stylegan_image)
        if not os.path.isfile(image_path):
            # Browsing to a file outside input/ (e.g. picking one from the output
            # gallery) doesn't reliably annotate the widget value with "[output]"/
            # "[temp]" for custom nodes the way it does for the built-in LoadImage,
            # so get_annotated_filepath falls back to input/ and misses it. Fall
            # back to searching the other folders by basename before giving up.
            basename = os.path.basename(stylegan_image)
            for directory in (folder_paths.get_output_directory(), folder_paths.get_temp_directory()):
                candidate = os.path.join(directory, basename)
                if os.path.isfile(candidate):
                    image_path = candidate
                    break
            else:
                raise FileNotFoundError(f"Could not find '{stylegan_image}' in input, output, or temp directories")
        img = Image.open(image_path)

        encoded = img.info.get(LATENT_METADATA_KEY)
        if encoded is None:
            raise ValueError(f"{stylegan_image} has no '{LATENT_METADATA_KEY}' metadata; it wasn't produced by a StyleGANSampler output saved with SaveImage/PreviewImage/SaveStyleGANLatentImg")
        latent = str2tensor(encoded).to(get_torch_device())
        model_file = img.info.get(MODEL_METADATA_KEY, "")

        image = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        _register_latent_for_images(image, latent, model_file or None)

        return (image, latent, model_file)

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

        _, model_file = _lookup_latent_for_images(stylegan_image)

        results = []
        for batch_number, image in enumerate(stylegan_image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            index = min(batch_number, stylegan_latent.size(0) - 1)
            latent = stylegan_latent[index:index + 1]
            metadata.add_text(LATENT_METADATA_KEY, tensor2str(latent))
            if model_file:
                metadata.add_text(MODEL_METADATA_KEY, model_file)

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
            cache_path = os.path.splitext(path)[0] + ".safetensors"
            if os.path.isfile(cache_path):
                G = load_stylegan_safetensors(cache_path)
            else:
                with open(path, 'rb') as f:
                    G = pickle.load(f)['G_ema']
                try:
                    save_stylegan_safetensors(G, cache_path)
                except Exception as e:
                    print(f"StyleGAN: couldn't cache {cache_path} as safetensors: {e}")
        G = G.to(get_torch_device())
        G.stylegan_source_file = stylegan_file
        return (G,)

class LoadStyleGANDirections:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_directions_file": (folder_paths.get_filename_list("stylegan_directions"), ),
            },
            "optional": {
                "direction_name": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STYLEGAN_DIRECTIONS",)
    FUNCTION = "load_directions"
    CATEGORY = "StyleGAN/directions"

    def load_directions(self, stylegan_directions_file, direction_name=""):
        path = folder_paths.get_full_path("stylegan_directions", stylegan_directions_file)
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if direction_name:
                if direction_name not in keys:
                    raise ValueError(f"'{direction_name}' not found in {stylegan_directions_file}. Available: {', '.join(keys)}")
                directions = f.get_tensor(direction_name).unsqueeze(0)
            else:
                metadata = f.metadata() or {}
                order = json.loads(metadata["component_order"]) if "component_order" in metadata else sorted(keys)
                directions = torch.stack([f.get_tensor(k) for k in order], dim=0)
        return (directions.to(get_torch_device()),)

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
        _register_latent_for_images(imgs, stylegan_latent, getattr(stylegan_model, "stylegan_source_file", None))
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

        mask_num = str2num(mask)
        if mask_num == 0xFFFF:
            blend = self.jmap(blend, -1.0, 1.0, 0.0, 1.0) # make unipolar
        else:
            if blend > 0: # transfer L onto R
                z = latent_2.clone()
            else: # transfer R onto L
                blend = abs(blend)
                latent_1,latent_2 = latent_2,latent_1 # swap L and R

        mask = self.num2mask( mask_num )

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



MASK_OPTIONS = ["total (0xFFFF)", "coarse (0xFF00)", "mid (0x0FF0)", "fine (0x00FF)", "alt1 (0xF0F0)", "alt2 (0x0F0F)", "alt3 (0xF00F)"]

class DiscoverGANSpaceDirections:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "num_samples": ("INT", {"default": 5000, "min": 100, "max": 1000000}),
                "num_components": ("INT", {"default": 10, "min": 1, "max": 512}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STYLEGAN_DIRECTIONS",)
    FUNCTION = "discover"
    CATEGORY = "StyleGAN/directions"

    def discover(self, stylegan_model, num_samples, num_components, seed):
        from .ganspace import sample_ganspace_directions
        device = get_torch_device()
        directions, _mean = sample_ganspace_directions(stylegan_model, num_samples, num_components, seed, device)
        return (directions,)

class DiscoverSeFaDirections:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "num_components": ("INT", {"default": 10, "min": 1, "max": 512}),
            },
        }

    RETURN_TYPES = ("STYLEGAN_DIRECTIONS",)
    FUNCTION = "discover"
    CATEGORY = "StyleGAN/directions"

    def discover(self, stylegan_model, num_components):
        from .sefa import compute_sefa_directions
        directions = compute_sefa_directions(stylegan_model, num_components)
        return (directions,)

class ApplyStyleGANDirection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "directions": ("STYLEGAN_DIRECTIONS", ),
                "component_index": ("INT", {"default": 0, "min": 0}),
                "strength": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.05}),
                "mask": (MASK_OPTIONS,),
            },
        }

    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "apply"
    CATEGORY = "StyleGAN/directions"

    def apply(self, stylegan_latent, directions, component_index, strength, mask):
        from .str_utils import num2mask
        idx = min(component_index, directions.shape[0] - 1)
        direction = directions[idx].to(stylegan_latent.device, stylegan_latent.dtype)
        num_ws = stylegan_latent.shape[1]
        layer_mask = num2mask(str2num(mask), num_ws=num_ws)

        z = stylegan_latent.clone()
        z[:, layer_mask, :] = z[:, layer_mask, :] + strength * direction

        return (z,)

class StyleGANDirectionSweep:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "directions": ("STYLEGAN_DIRECTIONS", ),
                "component_index": ("INT", {"default": 0, "min": 0}),
                "min_strength": ("FLOAT", {"default": -3.0, "min": -50.0, "max": 50.0, "step": 0.05}),
                "max_strength": ("FLOAT", {"default": 3.0, "min": -50.0, "max": 50.0, "step": 0.05}),
                "steps": ("INT", {"default": 7, "min": 2, "max": 64}),
                "mask": (MASK_OPTIONS,),
            },
            "optional": {
                "noise_mode": (["const", "random"], {"default": "const"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sweep"
    CATEGORY = "StyleGAN/directions"

    def sweep(self, stylegan_model, stylegan_latent, directions, component_index, min_strength, max_strength, steps, mask, noise_mode="const"):
        from .str_utils import num2mask
        idx = min(component_index, directions.shape[0] - 1)
        direction = directions[idx].to(stylegan_latent.device, stylegan_latent.dtype)
        num_ws = stylegan_latent.shape[1]
        layer_mask = num2mask(str2num(mask), num_ws=num_ws)
        base = stylegan_latent[0:1].detach().clone()

        imgs = []
        latents = []
        pbar = ProgressBar(steps) if PROGRESS_BAR_ENABLED and steps > 1 else None
        for i in trange(steps):
            t = min_strength + (max_strength - min_strength) * i / (steps - 1)
            z = base.clone()
            z[:, layer_mask, :] = z[:, layer_mask, :] + t * direction

            img = stylegan_model.synthesis(z, noise_mode=noise_mode)
            img = torch.permute(img, (0, 2, 3, 1))  # BCHW -> BHWC
            img = torch.clip(img / 2 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
            imgs.append(img)
            latents.append(z)
            if pbar is not None:
                pbar.update(1)

        imgs = torch.cat(imgs, dim=0)
        _register_latent_for_images(imgs, torch.cat(latents, dim=0), getattr(stylegan_model, "stylegan_source_file", None))
        return (imgs,)

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
    "LoadStyleGANDirections": LoadStyleGANDirections,
    "DiscoverGANSpaceDirections": DiscoverGANSpaceDirections,
    "DiscoverSeFaDirections": DiscoverSeFaDirections,
    "ApplyStyleGANDirection": ApplyStyleGANDirection,
    "StyleGANDirectionSweep": StyleGANDirectionSweep,
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
    "LoadStyleGANDirections": "Load StyleGAN Directions",
    "DiscoverGANSpaceDirections": "Discover GANSpace Directions",
    "DiscoverSeFaDirections": "Discover SeFa Directions",
    "ApplyStyleGANDirection": "Apply StyleGAN Direction",
    "StyleGANDirectionSweep": "StyleGAN Direction Sweep",
}