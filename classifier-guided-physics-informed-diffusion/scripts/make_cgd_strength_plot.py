"""CGD class accuracy and FID vs classifier-guidance strength (robust guidance classifier).

Reads results/generation_classifier_eval/eval_cgd_cfg3.0_robcls{s}/classifier_eval_metrics.json
(scripts/evaluate_generated_samples.py, CFG 3.0, --classifier-scale s, guided by
checkpoints/robust_classification/guide_fits) and the unguided DDPM reference
(eval_ddpm_cfg3.0), and writes results/edm_baseline/cgd_guidance_strength.png for the deck.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "results/generation_classifier_eval")
OUT = os.path.join(ROOT, "results/edm_baseline/cgd_guidance_strength.png")

STRENGTHS = [0, 1, 3, 10, 30, 100]
BEST = 3
REAL_ACC, REAL_FRI = 78.1, 84.6  # scoring classifier on real crumb_fits test images


def load(name):
    with open(os.path.join(EVAL, name, "classifier_eval_metrics.json")) as f:
        return next(iter(json.load(f).values()))


runs = [load(f"eval_cgd_cfg3.0_robcls{s}") for s in STRENGTHS]
ddpm = load("eval_ddpm_cfg3.0")
x = np.arange(len(STRENGTHS))  # evenly spaced categories; strengths span 0-100


def stats(values, scale=1.0):
    v = np.array(values) * scale
    return v.mean(), v.std(ddof=1)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
series = [
    ("class accuracy", lambda r: r["class_accuracy_per_seed"], "#2a2", "o"),
    ("FR-I accuracy", lambda r: r["per_class_accuracy_per_seed"]["FR-I"], "#c44", "s"),
]
for label, get, color, marker in series:
    m, s = zip(*[stats(get(r), 100) for r in runs])
    axes[0].errorbar(x, m, yerr=s, color=color, marker=marker, ms=8, lw=2, capsize=4, label=f"CGD {label}")
axes[0].axhline(stats(ddpm["class_accuracy_per_seed"], 100)[0], color="#48a", ls="--", lw=1.5, label="DDPM unguided, class acc")
axes[0].axhline(REAL_FRI, color="#888", ls=":", lw=1.5, label=f"real CRUMB FR-I ({REAL_FRI}%)")
axes[0].set_title("Class accuracy (%)", fontsize=13)

m, s = zip(*[stats(r["fid_per_seed"]) for r in runs])
axes[1].errorbar(x, m, yerr=s, color="#2a2", marker="o", ms=8, lw=2, capsize=4, label="CGD")
axes[1].axhline(stats(ddpm["fid_per_seed"])[0], color="#48a", ls="--", lw=1.5, label="DDPM unguided")
axes[1].set_title("FID  (lower is better)", fontsize=13)

for ax in axes:
    ax.axvspan(STRENGTHS.index(BEST) - 0.3, STRENGTHS.index(BEST) + 0.3, color="#2a2", alpha=0.08, lw=0)
    ax.set_xticks(x, [str(s) for s in STRENGTHS])
    ax.set_xlabel("classifier-guidance strength (classifier_scale)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

fig.text(0.5, 0.005, f"CFG 3.0, 5 CGD seeds (mean ± 1 sample std), 128 samples/class/seed, same starting noise at every strength. "
         f"Shaded: chosen strength {BEST}.", ha="center", fontsize=10, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
