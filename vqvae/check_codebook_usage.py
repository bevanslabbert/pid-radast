"""Diagnose VQ-VAE codebook collapse: how many of the num_embeddings codes
are actually used when encoding the dataset, and how skewed is usage.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vq_vae_modified import get_vqvae
from fits_normalize import load_fits_stats, load_and_normalise_fits

CRUMB_FITS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "classifier-guided-physics-informed-diffusion",
    "src", "datasets", "crumb", "fits",
)

IMG_DIM = 128
LATENT_DIM = 16
NUM_EMBEDDINGS = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits-dir", default=CRUMB_FITS_DIR)
    parser.add_argument("--weights", default=None,
                         help="Defaults to output/vqvae_<dataset>.weights.h5 "
                              "inferred from --fits-dir's parent directory name.")
    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.normpath(os.path.dirname(args.fits_dir)))
    weights_path = args.weights or os.path.join(
        os.path.dirname(__file__), "output", f"vqvae_{dataset_name}.weights.h5")

    filenames = sorted(f for f in os.listdir(args.fits_dir) if f.endswith(".fits"))
    noise_rms, peak_log = load_fits_stats(args.fits_dir)
    images = load_and_normalise_fits(filenames, args.fits_dir, IMG_DIM, noise_rms, peak_log)

    print(f"Loading VQ-VAE weights from {weights_path}")
    autoencoder = get_vqvae(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS)
    autoencoder.load_weights(weights_path)

    encoder = autoencoder.get_layer("encoder")
    vq_layer = encoder.get_layer("vector_quantizer")
    pre_vq_model = keras.Model(encoder.input, encoder.layers[-2].output)

    pre_vq = pre_vq_model.predict(images, verbose=0)
    flattened = tf.reshape(pre_vq, [-1, LATENT_DIM])
    indices = vq_layer.get_code_indices(flattened).numpy()

    counts = np.bincount(indices, minlength=NUM_EMBEDDINGS)
    used = np.count_nonzero(counts)
    total = counts.sum()
    top10 = np.argsort(counts)[::-1][:10]

    print(f"{len(images)} images -> {total} latent positions")
    print(f"Codes used: {used}/{NUM_EMBEDDINGS} ({100 * used / NUM_EMBEDDINGS:.1f}%)")
    print(f"Top-10 code usage share: {100 * counts[top10].sum() / total:.1f}%")
    print("Most-used codes (index: count):")
    for idx in top10:
        print(f"  {idx}: {counts[idx]} ({100 * counts[idx] / total:.2f}%)")


if __name__ == "__main__":
    main()
