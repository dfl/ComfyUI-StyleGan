"""Offline converter: StyleGAN .pkl checkpoints -> .safetensors.

Usage:
    python convert_to_safetensors.py <model.pkl> [<model2.pkl> ...]

Unpickles G_ema (trusted local file), records its exact constructor
kwargs (`init_kwargs`, captured automatically by torch_utils.persistence
for every persistent class) and the architecture family, then saves the
state_dict as safetensors with that metadata embedded. This avoids
guessing hyperparameters from tensor shapes: reconstruction later uses
the same init_kwargs the model was originally built with.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dnnlib
import torch_utils
sys.modules["dnnlib"] = dnnlib
sys.modules["torch_utils"] = torch_utils

import pickle
from safetensors.torch import save_model


def detect_arch(G):
    if hasattr(G.synthesis, "input"):
        return "stylegan3"
    if hasattr(G.synthesis, "b4"):
        return "stylegan2"
    raise ValueError("Unrecognized StyleGAN architecture (neither stylegan2 nor stylegan3 synthesis network)")


def convert(pkl_path: Path):
    safetensors_path = pkl_path.with_suffix(".safetensors")
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)["G_ema"]

    arch = detect_arch(G)
    init_kwargs = json.loads(json.dumps(dict(G.init_kwargs)))  # drop EasyDict/numpy types

    save_model(G, str(safetensors_path), metadata={
        "arch": arch,
        "init_kwargs": json.dumps(init_kwargs),
    })
    print(f"{pkl_path.name}: {arch}, {len(init_kwargs)} init_kwargs -> {safetensors_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl_files", nargs="+", type=Path)
    args = parser.parse_args()
    for pkl_path in args.pkl_files:
        convert(pkl_path)
