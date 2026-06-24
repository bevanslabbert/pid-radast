import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def generate_class_samples(unet, scheduler, class_emb, num_classes, num_samples, device,
                            shape=(1, 150, 150), guidance_scale=7.5):
    """Generate CFG images for class 0 and class 1 in a single call.

    Returns:
        gen_0: Tensor (num_samples, *shape) — FR-I images
        gen_1: Tensor (num_samples, *shape) — FR-II images

    Generates both classes before returning so callers can reuse the same
    tensors for FID, KID, and PDF without running the denoising loop twice.
    """
    unet.eval()

    def _sample(class_idx):
        cond_labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
        uncond_labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)
        cond_emb = class_emb(cond_labels).unsqueeze(1)
        uncond_emb = class_emb(uncond_labels).unsqueeze(1)

        scheduler.set_timesteps(50)
        images = torch.randn((num_samples, *shape), device=device)
        for t in scheduler.timesteps:
            model_input = torch.cat([images] * 2)
            combined_emb = torch.cat([uncond_emb, cond_emb])
            with torch.no_grad():
                out = unet(model_input, t, encoder_hidden_states=combined_emb).sample
                noise_uncond, noise_cond = out.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                images = scheduler.step(noise_pred, t, images).prev_sample
        return images

    gen_0 = _sample(0)
    gen_1 = _sample(1)
    return gen_0, gen_1


def _to_uint8_rgb(t):
    """Convert (B, 1, H, W) in [-1, 1] to (B, 3, H, W) uint8 for Inception metrics."""
    t = t.repeat(1, 3, 1, 1)
    t = (t + 1.0) / 2.0
    t = (t * 255).clamp(0, 255)
    return t.to(torch.uint8)


def compute_fid_kid(gen_0, gen_1, valloader, device):
    """Compute FID and KID from pre-generated image tensors vs real validation images.

    Args:
        gen_0: Tensor (B, 1, H, W) in [-1, 1] — generated class-0 images
        gen_1: Tensor (B, 1, H, W) in [-1, 1] — generated class-1 images
        valloader: DataLoader of real (images, labels) pairs
        device: torch.device

    Returns:
        (fid_score: float, kid_mean: float)

    KID is preferred over FID on small datasets like MiraBest (~1200 images)
    because it is an unbiased estimator. Both use Inception pool3 features.
    """
    n_fake = gen_0.shape[0] + gen_1.shape[0]
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(feature=2048, normalize=False,
                                  subset_size=min(n_fake, 50)).to(device)

    with torch.no_grad():
        for real_imgs, _ in valloader:
            real_u8 = _to_uint8_rgb(real_imgs.to(device))
            fid.update(real_u8, real=True)
            kid.update(real_u8, real=True)

        fake_u8 = _to_uint8_rgb(torch.cat([gen_0, gen_1], dim=0).to(device))
        fid.update(fake_u8, real=False)
        kid.update(fake_u8, real=False)

    fid_score = fid.compute().item()
    kid_mean, _ = kid.compute()
    return fid_score, kid_mean.item()


def compute_pixel_pdf(gen_0, gen_1, valloader, num_classes, result_dir, epoch, n_bins=100):
    """Compare pixel intensity PDFs of generated vs real validation images.

    Pixel values in normalised log-SNR space map directly to physical flux
    density (Jy/beam), so comparing their distributions is a domain-appropriate
    complement to Inception-feature metrics like FID/KID.

    Wasserstein-1 is computed from CDFs — no scipy dependency.

    Args:
        gen_0: Tensor (B, 1, H, W) in [-1, 1] — generated class-0 images
        gen_1: Tensor (B, 1, H, W) in [-1, 1] — generated class-1 images
        valloader: DataLoader of real (images, labels) pairs
        num_classes: int
        result_dir: str — directory to save the per-epoch PDF comparison plot
        epoch: int — used in the plot title and filename

    Returns:
        Mean Wasserstein-1 distance across both classes (lower = better).
    """
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = float(bins[1] - bins[0])

    real_pixels = {c: [] for c in range(num_classes)}
    for imgs, labels in valloader:
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                real_pixels[c].append(imgs[mask].cpu().numpy().flatten())

    gen_pixels = {
        0: gen_0.cpu().numpy().flatten(),
        1: gen_1.cpu().numpy().flatten(),
    }

    color_pairs = {0: ('tab:blue', 'tab:cyan'), 1: ('tab:red', 'tab:orange')}
    class_names = {0: 'FR-I', 1: 'FR-II'}

    w_dists = []
    fig, axes = plt.subplots(1, num_classes, figsize=(7 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]
    fig.suptitle(f'Pixel Intensity PDF — Real vs Generated (Epoch {epoch})', fontsize=13)

    for c in range(num_classes):
        ax = axes[c]
        real_col, gen_col = color_pairs.get(c, ('tab:blue', 'tab:cyan'))
        cls_name = class_names.get(c, f'Class {c}')

        real_arr = np.concatenate(real_pixels[c]) if real_pixels[c] else np.array([])
        if len(real_arr) == 0:
            continue

        real_hist, _ = np.histogram(real_arr, bins=bins, density=True)
        gen_hist, _ = np.histogram(gen_pixels[c], bins=bins, density=True)

        w_dist = float(np.sum(np.abs(np.cumsum(real_hist) - np.cumsum(gen_hist))) * bin_width ** 2)
        w_dists.append(w_dist)

        ax.plot(bin_centers, real_hist, color=real_col, linewidth=2, label=f'Real {cls_name}')
        ax.plot(bin_centers, gen_hist, color=gen_col, linewidth=2, linestyle='--',
                label=f'Generated {cls_name}')
        ax.set_title(f'{cls_name}  (W={w_dist:.4f})')
        ax.set_xlabel('Pixel intensity (normalised)')
        ax.set_ylabel('Probability density')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.savefig(os.path.join(result_dir, f'pixel_pdf_epoch_{epoch}.png'), dpi=150)
    plt.close()

    return float(np.mean(w_dists)) if w_dists else float('nan')
