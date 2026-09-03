"""Render a real-CRUMB ground-truth sample grid matching the style of the
models' comparison_epoch_*.png (2x2 FR-I | 2x2 FR-II), for the EDM
comparison deck. Uses the same symmetric-log-SNR normalisation the
generative models train on (MiraBestFITS._normalise)."""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FITS_DIR = os.path.join(ROOT, "src/datasets/crumb/fits")
OUT = os.path.join(ROOT, "results/edm_baseline/crumb_groundtruth_samples.png")

# indices picked to show a range of morphologies; deterministic
FRI = ["100_162.947_+055.386.fits", "100_209.019_+012.133.fits",
       "100_233.541_+034.894.fits", "100_199.959_+025.530.fits"]
FRII = ["200_175.116_+012.052.fits", "200_192.935_+008.941.fits",
        "200_221.046_+026.031.fits", "200_146.761_+009.728.fits"]


def _raw(name):
    with fits.open(os.path.join(FITS_DIR, name)) as h:
        return np.nan_to_num(h[0].data.astype(np.float32))


# dataset-level stats, matching MiraBestFITS._compute_stats
_all = glob.glob(os.path.join(FITS_DIR, "*.fits"))
_nrms, _plog = [], []
for f in np.random.default_rng(0).choice(_all, min(400, len(_all)), replace=False):
    d = _raw(os.path.basename(f))
    b = np.concatenate([d[:20].ravel(), d[-20:].ravel(), d[:, :20].ravel(), d[:, -20:].ravel()])
    nr = np.std(b) + 1e-8
    _nrms.append(nr)
    _plog.append(np.log1p(np.abs(d / nr).max()))
NOISE = float(np.median(_nrms))
PEAK = float(np.median(_plog))


def load_norm(name):
    snr = _raw(name) / NOISE
    sym = np.sign(snr) * np.log1p(np.abs(snr))
    return np.clip(sym / PEAK, -1.0, 1.0)


fig, axes = plt.subplots(2, 4, figsize=(10, 5))
fig.suptitle("Real CRUMB (ground truth)      "
             "Class 0 FR-I  |  Class 1 FR-II", fontsize=13)
for ax, name in zip(axes.flat, FRI[:2] + FRII[:2] + FRI[2:] + FRII[2:]):
    ax.imshow(load_norm(name), cmap="gray", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
# reorder so left 2 cols = FR-I, right 2 cols = FR-II
for ax in axes.flat:
    ax.remove()
order = [FRI[0], FRI[1], FRII[0], FRII[1], FRI[2], FRI[3], FRII[2], FRII[3]]
gs = fig.add_gridspec(2, 4, hspace=0.05, wspace=0.05)
for i, name in enumerate(order):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ax.imshow(load_norm(name), cmap="gray", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
