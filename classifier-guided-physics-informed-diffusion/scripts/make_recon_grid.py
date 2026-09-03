"""CRUMB VQ-VAE reconstruction figures for the EDM comparison deck.

Each source strip in ../vqvae/output/**/*_reconstructions.png is a vertical
stack of (input | reconstruction) pairs. This produces:
  - recon_overview.png : one example pair per model, 2x2
  - recon_<model>.png  : up to N example pairs for one model, 2 columns
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "..", "vqvae", "output")
OUTDIR = os.path.join(ROOT, "results/edm_baseline")

STRIPS = [
    ("crumb", "Real CRUMB", os.path.join(V, "crumb_reconstructions.png"), 8),
    ("edm", "EDM-generated", os.path.join(V, "edm_outputs/edm_reconstructions.png"), 32),
    ("ddpm", "DDPM-generated", os.path.join(V, "diffusion_outputs/diffusion_reconstructions.png"), 16),
    ("cgd", "CGD-generated", os.path.join(V, "cgd_outputs/cgd_reconstructions.png"), 32),
]
HEADER = 60  # px of "Original / Reconstruction" caption at the top of each strip


def pairs(path, npairs):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    rowh = (h - HEADER) / npairs
    return [im.crop((0, round(HEADER + i * rowh), w, round(HEADER + (i + 1) * rowh)))
            for i in range(npairs)]


def montage(imgs, title, out, ncols=2):
    nrows = math.ceil(len(imgs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.7 * nrows))
    axes = axes.reshape(nrows, ncols)
    for k, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if k < len(imgs):
            ax.imshow(imgs[k])
        else:
            ax.axis("off")
    fig.suptitle(title + "   -   each tile: input (left)  vs  VQ-VAE reconstruction (right)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


# ---- overview: first pair of each model
first = [(lbl, pairs(p, n)[0]) for _, lbl, p, n in STRIPS]
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
for ax, (lbl, img) in zip(axes.flat, first):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(lbl, fontsize=13, fontweight="bold")
fig.suptitle("CRUMB VQ-VAE: input (left of each tile)  vs  reconstruction (right)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "recon_overview.png"), dpi=130)
plt.close(fig)
print("wrote recon_overview.png")

# ---- per-model: up to 8 evenly-spaced example pairs
N = 8
for key, lbl, path, npairs in STRIPS:
    ps = pairs(path, npairs)
    step = max(1, len(ps) // N)
    sel = ps[::step][:N]
    montage(sel, lbl, os.path.join(OUTDIR, f"recon_{key}.png"))
