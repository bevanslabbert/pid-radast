"""Train the VQ-VAE (vq_vae_modified.py) on CRUMB FITS cutouts (FIRST-survey
radio galaxy images) and reconstruct a held-out sample to see how it performs.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from metrics_utils import basic_MSE, score_NCC
from vq_vae_modified import VQVAETrainer
from fits_normalize import load_fits_stats, load_and_normalise_fits

# Default FITS directory (any directory with a cached fits_stats.json works --
# same "root" convention as the diffusion pipeline's MiraBestFITS(root=...));
# pass --fits-dir to point at a different dataset, e.g. .../mirabest/fits.
CRUMB_FITS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "classifier-guided-physics-informed-diffusion",
    "src", "datasets", "crumb", "fits",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

IMG_DIM = 128
LATENT_DIM = 16
NUM_EMBEDDINGS = 256
NUM_RECON_EXAMPLES = 8
SEED = 42


def load_crumb_images(fits_dir, dim=IMG_DIM):
    """Load and normalise FITS cutouts the same way the diffusion pipeline
    does: FITS -> nan_to_num -> symmetric log-SNR normalise (MiraBestFITS's
    own transform, via fits_normalize.py, using that directory's cached
    fits_stats.json), producing an array in [-1, 1] on the same scale the
    diffusion model was trained on / denormalises its output back to.
    """
    filenames = sorted(f for f in os.listdir(fits_dir) if f.endswith(".fits"))
    noise_rms, peak_log = load_fits_stats(fits_dir)
    images = load_and_normalise_fits(filenames, fits_dir, dim, noise_rms, peak_log)
    return images, filenames


def plot_reconstructions(originals, reconstructions, filenames, out_path):
    n = len(originals)
    fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
    for i in range(n):
        axes[i, 0].imshow(originals[i, :, :, 0] / 2 + 0.5, cmap="viridis")
        axes[i, 0].set_ylabel(filenames[i], fontsize=6)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].imshow(reconstructions[i, :, :, 0] / 2 + 0.5, cmap="viridis")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Reconstruction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits-dir", default=CRUMB_FITS_DIR,
                         help="FITS directory to train on, e.g. .../src/datasets/crumb/fits "
                              "or .../src/datasets/mirabest/fits. Must contain a cached "
                              "fits_stats.json (produced by the diffusion pipeline's "
                              "MiraBestFITS loader for that same directory).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.normpath(os.path.dirname(args.fits_dir)))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Loading FITS cutouts from {args.fits_dir} ...")
    images, filenames = load_crumb_images(args.fits_dir)
    print(f"Loaded {len(images)} images, shape {images.shape}")

    indices = rng.permutation(len(images))
    n_val = max(1, int(len(images) * args.val_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_images = images[train_idx]
    val_images = images[val_idx]
    val_filenames = [filenames[i] for i in val_idx]
    print(f"Train: {len(train_images)}  Held-out val: {len(val_images)}")

    data_variance = np.var(train_images)
    trainer = VQVAETrainer(data_variance, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS)
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4))

    trainer.fit(train_images, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    autoencoder = trainer.vqvae
    autoencoder.save_weights(os.path.join(OUTPUT_DIR, f"vqvae_{dataset_name}.weights.h5"))

    print("Reconstructing held-out images...")
    reconstructions = autoencoder.predict(val_images)

    mse_scores = np.array([basic_MSE(reconstructions[i, :, :, 0], val_images[i, :, :, 0]) for i in range(len(val_images))])
    ncc_scores = np.array([score_NCC(reconstructions[i, :, :, 0], val_images[i, :, :, 0]) for i in range(len(val_images))])

    summary = (
        f"Held-out set: {len(val_images)} images\n"
        f"MSE: mean={mse_scores.mean():.6f} std={mse_scores.std():.6f}\n"
        f"NCC: mean={ncc_scores.mean():.4f} std={ncc_scores.std():.4f}\n"
    )
    print(summary)
    with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_reconstruction_metrics.txt"), "w") as f:
        f.write(summary)
        for fn, mse, ncc in zip(val_filenames, mse_scores, ncc_scores):
            f.write(f"{fn}\tMSE={mse:.6f}\tNCC={ncc:.4f}\n")

    n_examples = min(NUM_RECON_EXAMPLES, len(val_images))
    example_idx = rng.choice(len(val_images), size=n_examples, replace=False)
    plot_reconstructions(
        val_images[example_idx],
        reconstructions[example_idx],
        [val_filenames[i] for i in example_idx],
        os.path.join(OUTPUT_DIR, f"{dataset_name}_reconstructions.png"),
    )
    print(f"Saved reconstructions plot and metrics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
