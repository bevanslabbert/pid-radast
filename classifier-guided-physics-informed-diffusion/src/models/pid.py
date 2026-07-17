import torch
import torch.nn.functional as F


def estimate_x0(x_t, noise_pred, alphas_cumprod, t):
    """Recover a clean-image estimate x_0 from the noisy image x_t and the predicted noise.

    DDPM forward process: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
    Rearranging gives: x_0 = (x_t - sqrt(1 - alpha_bar) * noise_pred) / sqrt(alpha_bar)
    """
    alpha_bar = alphas_cumprod[t].view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting
    x_0 = (x_t - (1.0 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
    return x_0.clamp(-1.0, 1.0)  # keep in valid normalised range


def symmetry_loss(x_0_pred):
    """Penalise deviation from bilateral (point) symmetry about the image centre.

    Radio galaxy jets are approximately point-symmetric about the AGN core.
    Symmetry is measured as the average of the H-flip and V-flip MSE residuals,
    which together enforce 180-degree rotational (point) symmetry.
    """
    # penalise horizontal asymmetry: image should match its left-right mirror
    loss_h = F.mse_loss(x_0_pred, torch.flip(x_0_pred, dims=[-1]))
    # penalise vertical asymmetry: image should match its top-bottom mirror
    loss_v = F.mse_loss(x_0_pred, torch.flip(x_0_pred, dims=[-2]))
    return (loss_h + loss_v) * 0.5


def nonnegativity_loss(x_0_pred):
    """Penalise sub-zero pixel values in the predicted clean image.

    In symmetric-log-SNR normalisation, 0.0 in normalised space maps to
    zero Jy/beam in physical space — so any value below 0.0 is unphysical
    negative flux. ReLU(-x) is zero where the constraint is already satisfied
    and grows linearly where it is violated.
    """
    return F.relu(-x_0_pred).mean()


_concentration_mask_cache = {}


def _center_mask(size: int, inner_frac: float, device) -> torch.Tensor:
    """Boolean (1, 1, size, size) disk mask, True within inner_frac of the half-width from center."""
    key = (size, inner_frac, device)
    if key not in _concentration_mask_cache:
        coords = torch.arange(size, device=device) - (size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        radius = (yy ** 2 + xx ** 2).sqrt()
        mask = (radius <= inner_frac * (size / 2.0)).unsqueeze(0).unsqueeze(0)
        _concentration_mask_cache[key] = mask
    return _concentration_mask_cache[key]


def concentration_index(x_0_pred, inner_frac: float = 0.25) -> torch.Tensor:
    """Fraction of total absolute flux falling within a central disk of the image.

    Used to enforce the Fanaroff-Riley morphological definition: FR-I sources
    are core-brightened (jet brightness peaks near the AGN core and fades
    outward -> high concentration), FR-II sources are edge-brightened (bright
    hotspots/lobes far from the core -> low concentration).
    """
    size = x_0_pred.shape[-1]
    mask = _center_mask(size, inner_frac, x_0_pred.device)
    flux = x_0_pred.abs()
    inner_flux = (flux * mask).sum(dim=(1, 2, 3))
    total_flux = flux.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    return inner_flux / total_flux


def concentration_loss(x_0_pred, labels, inner_frac: float = 0.25, margin: float = 0.5) -> torch.Tensor:
    """Push generated images' flux concentration toward their class's FR morphology.

    Label 0 (FR-I, core-brightened): penalise concentration falling below `margin`.
    Label 1 (FR-II, edge-brightened): penalise concentration rising above `margin`.
    Hinge loss: zero once the sample is on the correct side of the margin.
    """
    c = concentration_index(x_0_pred, inner_frac)
    is_fri = labels == 0
    loss_fri = F.relu(margin - c[is_fri])
    loss_frii = F.relu(c[~is_fri] - margin)
    terms = torch.cat([loss_fri, loss_frii])
    return terms.mean() if terms.numel() > 0 else torch.zeros((), device=x_0_pred.device)


def physics_loss(x_0_pred, lambda_sym: float) -> torch.Tensor:
    """Symmetry penalty only."""
    return lambda_sym * symmetry_loss(x_0_pred)


def sample_pid_zeros(model, scheduler, class_emb, num_samples, num_classes, device,
                     shape=(1, 150, 150), guidance_scale=7.5):
    """Generate class-0 (FR-I) samples using CFG, then apply physics projection."""
    model.eval()

    # conditional (class 0) and unconditional (null class) embeddings for CFG
    cond_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    uncond_labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)
    cond_emb = class_emb(cond_labels).unsqueeze(1)    # (B, 1, D)
    uncond_emb = class_emb(uncond_labels).unsqueeze(1) # (B, 1, D)

    scheduler.set_timesteps(50)
    images = torch.randn((num_samples, *shape), device=device)

    for t in scheduler.timesteps:
        # batch cond + uncond in one forward pass to halve kernel launches
        model_input = torch.cat([images] * 2)
        combined_emb = torch.cat([uncond_emb, cond_emb])

        with torch.no_grad():
            output = model(model_input, t, encoder_hidden_states=combined_emb).sample
            noise_pred_uncond, noise_pred_cond = output.chunk(2)

            # CFG: push prediction away from uncond towards cond
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            images = scheduler.step(noise_pred, t, images).prev_sample

    return images


def sample_pid_ones(model, scheduler, class_emb, num_samples, num_classes, device,
                    shape=(1, 150, 150), guidance_scale=7.5):
    """Generate class-1 (FR-II) samples using CFG, then apply physics projection."""
    model.eval()

    cond_labels = torch.ones(num_samples, dtype=torch.long, device=device)
    uncond_labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)
    cond_emb = class_emb(cond_labels).unsqueeze(1)    # (B, 1, D)
    uncond_emb = class_emb(uncond_labels).unsqueeze(1) # (B, 1, D)

    scheduler.set_timesteps(50)
    images = torch.randn((num_samples, *shape), device=device)

    for t in scheduler.timesteps:
        model_input = torch.cat([images] * 2)
        combined_emb = torch.cat([uncond_emb, cond_emb])

        with torch.no_grad():
            output = model(model_input, t, encoder_hidden_states=combined_emb).sample
            noise_pred_uncond, noise_pred_cond = output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            images = scheduler.step(noise_pred, t, images).prev_sample

    return images
