import torch

@torch.no_grad()
def sample_ganspace_directions(G, num_samples, num_components, seed, device, batch_size=500):
    # GANSpace (Harkonen et al.): PCA over sampled W vectors.
    # Returns directions scaled to "1 sigma" units, matching the paper's convention.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    ws = []
    remaining = num_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        z = torch.randn(n, G.z_dim, generator=generator).to(device)
        w = G.mapping(z, None)  # [n, num_ws, w_dim], broadcast across layers before any truncation
        ws.append(w[:, 0, :].cpu())
        remaining -= n

    W = torch.cat(ws, dim=0)  # [num_samples, w_dim]
    mean = W.mean(dim=0)
    Wc = W - mean

    U, S, V = torch.pca_lowrank(Wc, q=num_components, center=False, niter=4)
    stdevs = S / (Wc.shape[0] - 1) ** 0.5
    directions = V.t() * stdevs.unsqueeze(1)  # [num_components, w_dim]

    return directions.to(device), mean.to(device)
