"""Compute trivial-baseline MSE/NCC (zero image, mean training image) to
give a reference point for judging whether the VQ-VAE's reconstruction
scores are actually 'good'.
"""
import os

import numpy as np

from sourcestamp_utils import loadImg
from metrics_utils import basic_MSE, score_NCC
from train_crumb_vqvae import CRUMB_FITS_DIR, load_crumb_images, SEED, IMG_DIM
from apply_vqvae_to_diffusion_output import DIFFUSION_FITS_DIR, load_and_normalise, FINAL_PATTERN


def report(name, originals, baseline):
    mse = np.array([basic_MSE(baseline[i, :, :, 0], originals[i, :, :, 0]) for i in range(len(originals))])
    ncc = np.array([score_NCC(baseline[i, :, :, 0], originals[i, :, :, 0]) for i in range(len(originals))])
    print(f"{name}: MSE={mse.mean():.6f}±{mse.std():.6f}  NCC={ncc.mean():.4f}±{ncc.std():.4f}")


def main():
    images, filenames = load_crumb_images(CRUMB_FITS_DIR)
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(images))
    n_val = max(1, int(len(images) * 0.1))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    train_images = images[train_idx]
    val_images = images[val_idx]

    zero_baseline = np.zeros_like(val_images)
    mean_image = train_images.mean(axis=0, keepdims=True)
    mean_baseline = np.repeat(mean_image, len(val_images), axis=0)

    print("=== Real held-out CRUMB images ===")
    report("Zero-image baseline", val_images, zero_baseline)
    report("Mean-image baseline", val_images, mean_baseline)

    diff_files = sorted(f for f in os.listdir(DIFFUSION_FITS_DIR) if FINAL_PATTERN.match(f))
    diff_images = load_and_normalise(diff_files, DIFFUSION_FITS_DIR)
    zero_baseline_d = np.zeros_like(diff_images)
    mean_baseline_d = np.repeat(mean_image, len(diff_images), axis=0)

    print("=== Diffusion-generated images ===")
    report("Zero-image baseline", diff_images, zero_baseline_d)
    report("Mean-image baseline", diff_images, mean_baseline_d)


if __name__ == "__main__":
    main()
