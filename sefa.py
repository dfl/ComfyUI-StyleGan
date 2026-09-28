import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_sefa_directions(G, num_components):
    # SeFa (Shen & Zhou): eigen-decompose the generator's style-modulation affine weights.
    # Matched by class name, not isinstance: a .pkl-loaded G reconstructs its classes
    # dynamically (torch_utils.persistence), so those class objects aren't identical to
    # the ones imported here, and isinstance would silently miss them on that load path.
    # This also naturally excludes StyleGAN3's SynthesisInput.affine, which is a
    # different (geometric-transform) affine that happens to share the attribute name.
    weights = []
    device = None
    for m in G.synthesis.modules():
        if type(m).__name__ in ("SynthesisLayer", "ToRGBLayer"):
            device = m.affine.weight.device
            weights.append(m.affine.weight.cpu())  # [channels, w_dim]

    if not weights:
        raise ValueError("No SynthesisLayer/ToRGBLayer affine weights found on this generator")

    # torch.linalg.eigh isn't implemented on MPS; compute on CPU (the matrix
    # is only w_dim x w_dim, so this is cheap regardless of the model's device)
    A = torch.cat(weights, dim=0).float()  # [total_channels, w_dim]
    M = A.t() @ A  # [w_dim, w_dim], symmetric PSD
    eigvals, eigvecs = torch.linalg.eigh(M)

    order = torch.argsort(eigvals.abs(), descending=True)
    top = order[:num_components]
    directions = eigvecs[:, top].t()  # [num_components, w_dim]

    return F.normalize(directions, dim=-1).to(device)
