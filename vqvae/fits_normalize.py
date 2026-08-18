"""Symmetric log-SNR normalisation, kept identical to the diffusion pipeline's
MiraBestFITS dataset (classifier-guided-physics-informed-diffusion/src/datasets/mirabest/MiraBestFITS.py)
so the VAE sees real and diffusion-generated CRUMB FITS images on the same
scale the diffusion model was trained on and denormalises back to.

Reads the same cached fits_stats.json the diffusion pipeline's crumb_fits
loader produces for src/datasets/crumb/fits, rather than recomputing stats
independently.
"""
import json
import os

import numpy as np
from astropy.io import fits
from PIL import Image


def load_fits_stats(fits_dir):
    stats_path = os.path.join(fits_dir, "fits_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"{stats_path} not found. Run the diffusion pipeline's crumb_fits "
            "data loader (MiraBestFITS) at least once first so dataset stats "
            "are cached -- the VAE reuses those stats rather than computing "
            "its own, so the two stay on the same scale."
        )
    with open(stats_path) as f:
        stats = json.load(f)
    return stats["median_noise_rms"], stats["median_peak_log"]


def normalise(data, noise_rms, peak_log):
    snr = data / noise_rms
    sym_log = np.sign(snr) * np.log1p(np.abs(snr))
    return np.clip(sym_log / peak_log, -1.0, 1.0)


def denormalise(data, noise_rms, peak_log):
    sym_log = data * peak_log
    snr = np.sign(sym_log) * np.expm1(np.abs(sym_log))
    return snr * noise_rms


def load_and_normalise_fits(filenames, fits_dir, dim, noise_rms, peak_log):
    """Load FITS files, normalise at native resolution (matching MiraBestFITS's
    normalise-then-resize order), then resize to (dim, dim). Output range is
    [-1, 1].
    """
    images = []
    for fname in filenames:
        with fits.open(os.path.join(fits_dir, fname)) as h:
            data = np.nan_to_num(np.array(h[0].data, dtype=np.float32), nan=0.0)
        normed = normalise(data, noise_rms, peak_log)
        resized = np.array(Image.fromarray(normed).resize((dim, dim)), dtype=np.float32)
        images.append(resized.reshape(dim, dim, 1))
    return np.array(images, dtype=np.float32)
