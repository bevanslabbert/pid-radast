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


# Each model maps to a list of per-seed runs (a run = one or more logs,
# e.g. an original run plus its resume). More than one seed per model
# will grow as the 5-seed sweep (2026-09-20) completes.
models = {
    "EDM baseline": (
        [["slurm-909739.out", "slurm-912853.out", "slurm-913384.out"],
         ["slurm-921086.out"],
         ["slurm-921087.out"],
         ["slurm-921088.out"],
         ["slurm-921089.out"]],
        "#c44",
    ),
    "DDPM (diffusion)": (
        [["slurm-912851.out", "slurm-913386.out"],
         ["slurm-921090.out"],
         ["slurm-921091.out"],
         ["slurm-921092.out"]],
        "#48a",
    ),
    "CGD": (
        [["slurm-321045.out", "slurm-912852.out", "slurm-921074.out"],
         ["slurm-921094.out"],
         ["slurm-921095.out", "slurm-13877273.out"],
         ["slurm-921096.out", "slurm-13877274.out"],
         ["slurm-921097.out", "slurm-13877275.out"]],
        "#2a2",
    ),
}

series = {name: ([from_log(*paths) for paths in seed_logs], color)
          for name, (seed_logs, color) in models.items()}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
panels = [("FID", 0, False), ("KID", 1, False), ("pixel-PDF Wasserstein", 2, True)]
for ax, (ttl, col, log) in zip(axes, panels):
    for name, (runs, c) in series.items():
        # mean (and std band) across seeds at each evaluated epoch
        per_ep = {}
        for ep, *metrics in runs:
            for e, v in zip(ep.tolist(), metrics[col]):
                per_ep.setdefault(e, []).append(v)
        eps = np.array(sorted(per_ep))
        vals = [np.array(per_ep[e]) for e in eps]
        mean = np.array([v.mean() for v in vals])
        n = len(runs)
        ax.plot(eps, mean, "-", lw=2.2, color=c, label=f"{name}  (n={n})")
        if n > 1:
            std = np.array([v.std(ddof=1) if len(v) > 1 else 0.0 for v in vals])
            lo = np.clip(mean - std, mean * 0.05, None) if log else mean - std
            ax.fill_between(eps, lo, mean + std, color=c, alpha=0.15, lw=0)
    ax.set_title(ttl, fontsize=13)
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_xlim(0, 400)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11, frameon=False)
fig.text(0.5, 0.005, "Lower is better.  Line = mean across seeds, band = ±1 std.", ha="center", fontsize=10, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.9))
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
