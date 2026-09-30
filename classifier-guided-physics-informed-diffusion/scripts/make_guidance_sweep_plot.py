"""Class accuracy and FID vs CFG guidance scale, for EDM / DDPM / CGD.

Reads results/generation_classifier_eval/eval_{edm,ddpm,cgd}_cfg{s}/classifier_eval_metrics.json
(scripts/evaluate_generated_samples.py with --guidance-scale s) and writes
results/edm_baseline/guidance_sweep.png for the comparison deck.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "results/generation_classifier_eval")
OUT = os.path.join(ROOT, "results/edm_baseline/guidance_sweep.png")

SCALES = ["1.0", "3.0", "7.5"]
REAL_ACC = 78.1  # eval_fits_bs8_seed42 on real crumb_fits test images

# Same colours as metric_comparison.png; markers so identity isn't colour-alone.
models = {
    "EDM baseline": ("edm", "#c44", "o"),
    "DDPM (diffusion)": ("ddpm", "#48a", "s"),
    "CGD": ("cgd", "#2a2", "^"),
}


def load(key, scale):
    with open(os.path.join(EVAL, f"eval_{key}_cfg{scale}/classifier_eval_metrics.json")) as f:
        return next(iter(json.load(f).values()))


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
panels = [
    ("Class accuracy (%)", lambda r: 100 * np.array(r["class_accuracy_per_seed"])),
    ("FID  (lower is better)", lambda r: np.array(r["fid_per_seed"])),
]
x = np.array([float(s) for s in SCALES])
for ax, (ttl, get) in zip(axes, panels):
    for name, (key, c, m) in models.items():
        runs = [get(load(key, s)) for s in SCALES]
        mean = np.array([v.mean() for v in runs])
        std = np.array([v.std(ddof=1) for v in runs])
        ax.errorbar(x, mean, yerr=std, color=c, marker=m, ms=8, lw=2, capsize=4,
                    label=f"{name}  (n={len(runs[0])})")
    ax.set_title(ttl, fontsize=13)
    ax.set_xlabel("CFG guidance scale", fontsize=11)
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].axhline(REAL_ACC, color="#888", ls="--", lw=1.5)
axes[0].text(7.4, REAL_ACC - 3.5, f"real CRUMB test images ({REAL_ACC}%)", ha="right", fontsize=10, color="#555")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11, frameon=False)
fig.text(0.5, 0.005, "Mean ± 1 sample std across seeds; 128 samples per class per seed. "
         "Native settings: EDM 3.0, DDPM/CGD 7.5.", ha="center", fontsize=10, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.9))
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
