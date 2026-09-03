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


def from_log(path):
    t = open(os.path.join(ROOT, path)).read()
    ep = [int(x) for x in re.findall(r"at epoch (\d+)", t)]
    fid = [float(x) for x in re.findall(r"FID: ([\d.]+)", t)]
    kid = [float(x) for x in re.findall(r"KID: ([\d.]+)", t)]
    pdf = [float(x) for x in re.findall(r"W-dist: ([\d.]+)", t)]
    n = min(map(len, [ep, fid, kid, pdf]))
    return np.array(ep[:n]), np.array(fid[:n]), np.array(kid[:n]), np.array(pdf[:n])


import json

def from_json(path):
    m = json.load(open(os.path.join(ROOT, path)))
    ep = np.array(m["fid_epochs"])
    return ep, np.array(m["fid"]), np.array(m["kid"]), np.array(m["pdf"], dtype=float)


series = {
    "EDM baseline": (from_json("results/edm_baseline/20260825_191230_untagged_824642/metrics.json"), "#c44"),
    "DDPM (diffusion)": (from_log("slurm-315339.out"), "#48a"),
    "CGD": (from_log("slurm-321045.out"), "#2a2"),
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
