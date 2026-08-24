"""EDM (Karras et al. 2022) diffusion baseline, architecture-matched to
Vicanek Martinez et al. (2024, A&A) "Simulating images of radio galaxies
with diffusion models" (arXiv:2410.07794) -- for a fair architectural
comparison against this project's DDPM-based diffusion/CGD/PID models.

Differences from this project's DDPM models, following the paper:
- Continuous noise level sigma instead of 1000 discrete timesteps.
- Karras preconditioning (c_skip/c_out/c_in/c_noise) wrapping the raw network.
- Noise sampled as ln(sigma) ~ N(P_mean, P_std^2) with P_mean=-2.5, P_std=1.8.
- Deterministic Heun 2nd-order ODE sampler (25 steps) instead of 50-step
  ancestral DDPM sampling.
- EMA of model weights (decay 0.9999), used for sampling/evaluation.

Adapted to this project's 150x150 resolution with 4 U-Net resolution levels
(paper uses 80x80 with 3 levels, which doesn't evenly divide 150); channel
width (128 base, mult 1/1/2/2) and attention placement (inner two levels)
follow the paper's stated configuration as closely as the resolution allows.
"""
import copy

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel

SIGMA_DATA = 0.5
# 4 down/up-sampling resolution levels require input dims divisible by 8;
# 150 (this project's resolution) isn't, so the U-Net operates on a
# reflect-padded 152x152 internally and every output is cropped back to
# 150x150 -- see _pad_to_multiple/_crop_to below.
_PAD_MULTIPLE = 8


def build_edm_components(cfg: dict, device):
    num_classes = cfg['data']['num_classes']
    input_size = cfg['data']['input_size']
    padded_size = input_size + (_PAD_MULTIPLE - input_size % _PAD_MULTIPLE) % _PAD_MULTIPLE

    unet = UNet2DModel(
        sample_size=padded_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim=32,
        # Index `num_classes` is the null/unconditional class, matching this
        # codebase's existing classifier-free-guidance convention.
        num_class_embeds=num_classes + 1,
    ).to(device)

    lr = float(cfg['training']['learning_rate'])
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    ema = EMA(unet, decay=float(cfg['training'].get('ema_decay', 0.9999)))

    return unet, ema, optimizer


class EMA:
    """Exponential moving average of model weights (paper: decay rate 0.9999).
    Sampling/evaluation uses `ema.shadow`, matching the paper's practice of
    reporting results from EMA weights rather than raw training weights.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def _edm_scalings(sigma, sigma_data=SIGMA_DATA):
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_noise = 0.25 * sigma.log()
    return c_skip, c_out, c_in, c_noise


def _pad_to_multiple(x, multiple=_PAD_MULTIPLE):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, h, w)
    top, left = pad_h // 2, pad_w // 2
    x_padded = F.pad(x, (left, pad_w - left, top, pad_h - top), mode='reflect')
    return x_padded, (top, left, h, w)


def _crop_to(x, pad_info):
    top, left, h, w = pad_info
    return x[..., top:top + h, left:left + w]


def edm_denoise(unet, x, sigma, class_labels):
    """D_theta(x; sigma | c) = c_skip*x + c_out*F_theta(c_in*x; c_noise, c)."""
    sigma = sigma.view(-1, 1, 1, 1)
    c_skip, c_out, c_in, c_noise = _edm_scalings(sigma)

    x_padded, pad_info = _pad_to_multiple(c_in * x)
    F_x = unet(x_padded, c_noise.flatten(), class_labels=class_labels).sample
    F_x = _crop_to(F_x, pad_info)

    return c_skip * x + c_out * F_x


def edm_loss(unet, images, labels, num_classes, label_dropout, device,
             P_mean=-2.5, P_std=1.8, sigma_data=SIGMA_DATA):
    """EDM training loss (Karras et al. 2022), matching the paper's noise
    distribution (P_mean=-2.5, P_std=1.8) and weighted-L2 objective (the
    paper's `c_out(sigma)^-2` weighting is folded into this `weight` term).
    """
    batch = images.shape[0]

    drop_mask = torch.rand(labels.shape, device=device) < label_dropout
    training_labels = labels.clone()
    training_labels[drop_mask] = num_classes

    log_sigma = P_mean + P_std * torch.randn(batch, device=device)
    sigma = log_sigma.exp()
    sigma_ = sigma.view(-1, 1, 1, 1)

    weight = (sigma_ ** 2 + sigma_data ** 2) / (sigma_ * sigma_data) ** 2
    noisy = images + torch.randn_like(images) * sigma_

    D_x = edm_denoise(unet, noisy, sigma, training_labels)
    loss = (weight * (D_x - images) ** 2).mean()
    return loss, {}


@torch.no_grad()
def edm_heun_sampler(unet, num_samples, shape, class_idx, num_classes, device, *,
                      num_steps=25, sigma_min=2e-3, sigma_max=80.0, rho=7.0,
                      guidance_scale=3.0):
    """Deterministic Heun 2nd-order ODE sampler (Karras et al. 2022, Algorithm 1),
    with classifier-free guidance, matching the paper's inference settings
    (25 steps, sigma_max=80, sigma_min=2e-3, rho=7).

    `guidance_scale` here is `1 + omega` in the paper's notation -- their
    D~ = (1+omega)*D(cond) - omega*D(uncond) is algebraically identical to the
    standard CFG form `D(uncond) + guidance_scale*(D(cond) - D(uncond))` used
    below, and to this codebase's existing DDPM classifier-free guidance.
    """
    unet.eval()
    step_idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigmas = (sigma_max ** (1 / rho) + step_idx / (num_steps - 1) *
              (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float64)])

    cond_labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
    uncond_labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)

    def denoise_cfg(x, sigma_scalar):
        sigma_b = sigma_scalar.repeat(num_samples)
        d_cond = edm_denoise(unet, x, sigma_b, cond_labels)
        if guidance_scale == 1.0:
            return d_cond
        d_uncond = edm_denoise(unet, x, sigma_b, uncond_labels)
        return d_uncond + guidance_scale * (d_cond - d_uncond)

    x = (torch.randn((num_samples, *shape), device=device) * sigmas[0].to(torch.float32))

    for i in range(num_steps):
        sigma_cur = sigmas[i].to(torch.float32)
        sigma_next = sigmas[i + 1].to(torch.float32)

        denoised = denoise_cfg(x, sigma_cur.unsqueeze(0))
        d_cur = (x - denoised) / sigma_cur
        x_next = x + (sigma_next - sigma_cur) * d_cur

        if sigma_next > 0:
            denoised_next = denoise_cfg(x_next, sigma_next.unsqueeze(0))
            d_next = (x_next - denoised_next) / sigma_next
            x_next = x + (sigma_next - sigma_cur) * 0.5 * (d_cur + d_next)

        x = x_next

    return x.clamp(-1.0, 1.0)


def generate_class_samples_edm(unet, num_classes, num_samples, device,
                                shape=(1, 150, 150), guidance_scale=3.0, num_steps=25):
    """Generate EDM Heun-sampled images for class 0 and class 1, mirroring
    `src.utils.metrics.generate_class_samples`'s (gen_0, gen_1) interface so
    the same FID/KID/pixel-PDF metric functions can be reused unchanged.
    """
    gen_0 = edm_heun_sampler(unet, num_samples, shape, 0, num_classes, device,
                              guidance_scale=guidance_scale, num_steps=num_steps)
    gen_1 = edm_heun_sampler(unet, num_samples, shape, 1, num_classes, device,
                              guidance_scale=guidance_scale, num_steps=num_steps)
    return gen_0, gen_1
