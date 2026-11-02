"""Score every seed's generated FITS samples from one evaluate_generated_samples.py
run (--save-fits) with the CRUMB-trained VQ-VAE, and write per-seed and per-model
reconstruction MSE / NCC to <eval-dir>/vqvae_metrics.json.

Expects <eval-dir>/fits_<model>_<tag>/generated_fits/generated_class*_*.fits, the
layout evaluate_generated_samples.py writes. Uses the same normalisation, weights
and metrics as train_crumb_vqvae.py, so scores are comparable with its held-out
real-CRUMB reference (output/crumb_reconstruction_metrics.txt).
"""
import argparse
import json
import os
import re

import numpy as np

from metrics_utils import basic_MSE, score_NCC
from vq_vae_modified import get_vqvae
from fits_normalize import load_fits_stats, load_and_normalise_fits

CRUMB_FITS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "classifier-guided-physics-informed-diffusion",
    "src", "datasets", "crumb", "fits",
)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "output", "vqvae_crumb.weights.h5")
IMG_DIM = 128
LATENT_DIM = 16
NUM_EMBEDDINGS = 256

# model types of evaluate_generated_samples.py, longest first so prefixes don't shadow them
SEED_DIR = re.compile(
    r"^fits_(diffusion_classifier_guided|classifier_guided_diffusion|edm_baseline|diffusion)_(.+)$"
)
FITS_FILE = re.compile(r"^generated_class(\d+)_\d+\.fits$")


def score_dir(autoencoder, fits_dir, noise_rms, peak_log):
    filenames = sorted(f for f in os.listdir(fits_dir) if FITS_FILE.match(f))
    images = load_and_normalise_fits(filenames, fits_dir, IMG_DIM, noise_rms, peak_log)
    recons = autoencoder.predict(images, verbose=0)
    mse = np.array([basic_MSE(recons[i, :, :, 0], images[i, :, :, 0]) for i in range(len(images))])
    ncc = np.array([score_NCC(recons[i, :, :, 0], images[i, :, :, 0]) for i in range(len(images))])
    return len(images), mse, ncc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True,
                        help="Output dir of an evaluate_generated_samples.py run made with --save-fits.")
    args = parser.parse_args()

    noise_rms, peak_log = load_fits_stats(CRUMB_FITS_DIR)
    autoencoder = get_vqvae(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS)
    autoencoder.load_weights(WEIGHTS_PATH)

    results = {}
    for entry in sorted(os.listdir(args.eval_dir)):
        match = SEED_DIR.match(entry)
        fits_dir = os.path.join(args.eval_dir, entry, "generated_fits")
        if not match or not os.path.isdir(fits_dir):
            continue
        n, mse, ncc = score_dir(autoencoder, fits_dir, noise_rms, peak_log)
        model = results.setdefault(match.group(1), {"tags": [], "n_per_seed": [],
                                                    "mse_per_seed": [], "ncc_per_seed": []})
        model["tags"].append(match.group(2))
        model["n_per_seed"].append(n)
        model["mse_per_seed"].append(float(mse.mean()))
        model["ncc_per_seed"].append(float(ncc.mean()))
        print(f"[{entry}] n={n}  MSE={mse.mean():.6f}  NCC={ncc.mean():.4f}")

    if not results:
        raise SystemExit(f"No fits_<model>_<tag>/generated_fits dirs found in {args.eval_dir}")

    for name, model in results.items():
        for metric in ("mse", "ncc"):
            values = np.array(model[f"{metric}_per_seed"])
            model[f"{metric}_mean"] = float(values.mean())
            # sample std across seeds, matching the other dissertation tables
            model[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        print(f"{name}: MSE={model['mse_mean']:.6f} +/- {model['mse_std']:.6f}  "
              f"NCC={model['ncc_mean']:.4f} +/- {model['ncc_std']:.4f}  ({len(model['tags'])} seeds)")

    out_path = os.path.join(args.eval_dir, "vqvae_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
