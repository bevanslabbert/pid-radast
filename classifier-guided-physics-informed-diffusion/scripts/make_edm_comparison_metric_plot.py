"""Combined FID / KID / pixel-PDF-Wasserstein vs epoch for EDM baseline,
DDPM diffusion, and CGD -- one figure for the comparison deck."""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/edm_baseline/metric_comparison.png")


def from_log(*paths):
    """Parse one or more slurm logs (e.g. a run plus its resume) and
    concatenate their epoch/FID/KID/PDF series, keyed on epoch so a
    re-evaluated epoch from a resume overwrites the earlier value."""
    by_epoch = {}
    for path in paths:
        t = open(os.path.join(ROOT, path)).read()
        ep = [int(x) for x in re.findall(r"at epoch (\d+)", t)]
        fid = [float(x) for x in re.findall(r"FID: ([\d.]+)", t)]
        kid = [float(x) for x in re.findall(r"KID: ([\d.]+)", t)]
        pdf = [float(x) for x in re.findall(r"W-dist: ([\d.]+)", t)]
        n = min(map(len, [ep, fid, kid, pdf]))
        for i in range(n):
            by_epoch[ep[i]] = (fid[i], kid[i], pdf[i])
    eps = sorted(by_epoch)
    fid = [by_epoch[e][0] for e in eps]
    kid = [by_epoch[e][1] for e in eps]
    pdf = [by_epoch[e][2] for e in eps]
    return np.array(eps), np.array(fid), np.array(kid), np.array(pdf)


series = {
    "EDM baseline": (from_log("slurm-912853.out", "slurm-913384.out"), "#c44"),
    "DDPM (diffusion)": (from_log("slurm-912851.out", "slurm-913386.out"), "#48a"),
    "CGD": (from_log("slurm-912852.out"), "#2a2"),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, idx, ttl, log in zip(axes, [1, 2, 3], ["FID", "KID", "pixel-PDF Wasserstein"], [0, 0, 1]):
    for name, ((ep, fid, kid, pdf), c) in series.items():
        y = [fid, kid, pdf][idx - 1]
        ax.plot(ep, y, "o-", ms=3, lw=1.4, color=c, label=name)
    ax.set_title(ttl + "  (lower is better)", fontsize=11)
    ax.set_xlabel("epoch")
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
