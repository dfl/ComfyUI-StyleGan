"""Offline direction discovery: GANSpace / SeFa on a StyleGAN checkpoint -> .safetensors.

Usage:
    python discover_directions.py model.safetensors --method sefa
    python discover_directions.py model.pkl --method ganspace --num-samples 5000
    python discover_directions.py model.safetensors --method sefa --sweep 0,1,2 --sweep-out sweep.png

Loads a StyleGAN generator directly (no ComfyUI required) and runs the
requested unsupervised direction-discovery method. Discovery itself only
touches G.mapping (GANSpace) or the synthesis affine weights (SeFa), so it
doesn't need the compiled CUDA/MPS synthesis ops that ComfyUI/setup.md
describe; only --sweep (which renders actual images) does.

Saves each component as its own named tensor (component_00, component_01,
...) in one .safetensors file, with the discovery order recorded in
metadata. Load it back into a ComfyUI workflow with the
LoadStyleGANDirections node: omit direction_name to get the whole batch
(for StyleGANDirectionSweep-style exploration), or pass one to pick a
single direction once you've identified what it does.
"""
import argparse
import json
import sys
from pathlib import Path

if "dnnlib" not in sys.modules:
    # Only needed standalone; when imported from nodes.py, dnnlib/torch_utils
    # are already set up there, and doing this again would import a second,
    # distinct copy of torch_utils.persistence.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dnnlib
    import torch_utils
    sys.modules["dnnlib"] = dnnlib
    sys.modules["torch_utils"] = torch_utils

import pickle
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ganspace import sample_ganspace_directions
from sefa import compute_sefa_directions

DEFAULT_STRENGTH_RANGE = {"sefa": (-10.0, 10.0), "ganspace": (-3.0, 3.0)}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_stylegan(path: Path):
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            weights = {key: f.get_tensor(key) for key in f.keys()}
        arch = metadata["arch"]
        init_kwargs = json.loads(metadata["init_kwargs"])
        if arch == "stylegan3":
            import networks_stylegan3 as networks
        elif arch == "stylegan2":
            import networks_stylegan2 as networks
        else:
            raise ValueError(f"Unknown StyleGAN architecture in safetensors metadata: {arch}")
        G = networks.Generator(**init_kwargs)
        G.load_state_dict(weights)
    else:
        with open(path, "rb") as f:
            G = pickle.load(f)["G_ema"]
    return G.eval()


def save_directions(directions, out_path, method):
    order = [f"component_{i:02d}" for i in range(directions.shape[0])]
    tensors = {name: directions[i].contiguous().cpu() for i, name in enumerate(order)}
    metadata = {
        "method": method,
        "num_components": str(directions.shape[0]),
        "w_dim": str(directions.shape[1]),
        "component_order": json.dumps(order),
    }
    save_file(tensors, str(out_path), metadata=metadata)


def render_sweep(G, directions, indices, device, min_strength, max_strength, steps):
    from PIL import Image
    import numpy as np

    z = torch.randn(1, G.z_dim, device=device)
    latent = G.mapping(z, None)  # [1, num_ws, w_dim]

    rows = []
    for idx in indices:
        direction = directions[idx].to(device)
        frames = []
        for i in range(steps):
            t = min_strength + (max_strength - min_strength) * i / (steps - 1)
            w = latent + t * direction
            img = G.synthesis(w, noise_mode="const")
            img = torch.clip(img / 2 + 0.5, 0, 1)[0]
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(Image.fromarray(img))
        rows.append(frames)

    w, h = rows[0][0].size
    grid = Image.new("RGB", (w * steps, h * len(rows)))
    for r, frames in enumerate(rows):
        for c, frame in enumerate(frames):
            grid.paste(frame, (c * w, r * h))
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--method", choices=["sefa", "ganspace"], required=True)
    parser.add_argument("--num-components", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=5000, help="GANSpace only")
    parser.add_argument("--seed", type=int, default=0, help="GANSpace only")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sweep", type=str, default=None, help="comma-separated component indices to preview, e.g. 0,1,2")
    parser.add_argument("--sweep-out", type=Path, default=None)
    parser.add_argument("--sweep-min", type=float, default=None)
    parser.add_argument("--sweep-max", type=float, default=None)
    parser.add_argument("--sweep-steps", type=int, default=7)
    args = parser.parse_args()

    device = get_device()
    G = load_stylegan(args.model).to(device)

    if args.method == "sefa":
        directions = compute_sefa_directions(G, args.num_components)
    else:
        directions, _mean = sample_ganspace_directions(G, args.num_samples, args.num_components, args.seed, device)

    # default: ComfyUI/models/stylegan_directions/, a sibling of the model's own
    # models/stylegan/ folder, so LoadStyleGANDirections' file picker doesn't mix
    # direction bundles in with model checkpoints
    default_out = args.model.parent.parent / "stylegan_directions" / f"{args.model.stem}_{args.method}_directions.safetensors"
    out_path = args.out or default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_directions(directions, out_path, args.method)
    print(f"{args.model.name}: {args.method}, {directions.shape[0]} components -> {out_path.name}")

    if args.sweep:
        indices = [int(x) for x in args.sweep.split(",")]
        min_s = args.sweep_min if args.sweep_min is not None else DEFAULT_STRENGTH_RANGE[args.method][0]
        max_s = args.sweep_max if args.sweep_max is not None else DEFAULT_STRENGTH_RANGE[args.method][1]
        grid = render_sweep(G, directions, indices, device, min_s, max_s, args.sweep_steps)
        sweep_out = args.sweep_out or out_path.with_suffix(".png")
        grid.save(sweep_out)
        print(f"Sweep preview (components {indices}) -> {sweep_out.name}")


if __name__ == "__main__":
    main()
