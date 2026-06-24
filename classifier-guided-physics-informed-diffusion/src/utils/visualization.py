import os

import matplotlib.pyplot as plt
import torchvision


def save_comparison_grid(zero_images, one_images, epoch, result_dir):
    """Save a side-by-side class-0 / class-1 comparison PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    def _to_numpy(t):
        grid = torchvision.utils.make_grid(t, nrow=2, normalize=True, value_range=(-1, 1))
        return grid.permute(1, 2, 0).cpu().numpy()

    axes[0].imshow(_to_numpy(zero_images), cmap='gray')
    axes[0].set_title(f"Class 0 FR-I (Epoch {epoch})")
    axes[0].axis('off')
    axes[1].imshow(_to_numpy(one_images), cmap='gray')
    axes[1].set_title(f"Class 1 FR-II (Epoch {epoch})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'comparison_epoch_{epoch}.png'))
    plt.close()


def save_training_plot(epochs, losses, val_losses, result_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.plot(epochs, losses, color='tab:blue', linewidth=2, label='Training Loss')
    ax.plot(epochs, val_losses, color='tab:cyan', linewidth=2, linestyle=':', label='Validation Loss')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    plt.title('Diffusion Training: MSE Loss Trends')
    fig.tight_layout()
    os.makedirs(result_dir, exist_ok=True)
    plot_path = os.path.join(result_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Loss graph saved to {plot_path}")


def save_generative_metrics_plot(fid_epochs, fid_history, kid_history, result_dir):
    if not fid_epochs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Generative Quality Metrics (lower is better)', fontsize=14)

    axes[0].plot(fid_epochs, fid_history, color='tab:purple', linewidth=2, marker='o')
    axes[0].set_title('FID')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('FID score')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(fid_epochs, kid_history, color='tab:orange', linewidth=2, marker='o')
    axes[1].set_title('KID')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('KID mean')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plot_path = os.path.join(result_dir, 'generative_metrics.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Generative metrics plot saved to {plot_path}")


def save_pixel_pdf_history_plot(pdf_epochs, pdf_history, result_dir):
    if not pdf_epochs:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(pdf_epochs, pdf_history, color='tab:green', linewidth=2, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Wasserstein-1 distance (lower is better)')
    plt.title('Pixel PDF Fidelity over Training')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(result_dir, 'pixel_pdf_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Pixel PDF history plot saved to {path}")


def save_pid_training_plots(epochs, loss_history, val_loss_history,
                             mse_history, sym_history, neg_history,
                             compliance_epochs, pct_negative_history, sym_score_history,
                             result_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PID Training: Loss Decomposition', fontsize=14)

    axes[0, 0].plot(epochs, loss_history, color='tab:blue', linewidth=2, label='Train')
    axes[0, 0].plot(epochs, val_loss_history, color='tab:cyan', linewidth=2, linestyle=':', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    axes[0, 1].plot(epochs, mse_history, color='tab:orange', linewidth=2)
    axes[0, 1].set_title('MSE Loss (noise prediction)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    axes[1, 0].plot(epochs, sym_history, color='tab:green', linewidth=2)
    axes[1, 0].set_title('Symmetry Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    axes[1, 1].plot(epochs, neg_history, color='tab:red', linewidth=2)
    axes[1, 1].set_title('Non-negativity Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.savefig(os.path.join(result_dir, 'pid_loss_decomposition.png'), dpi=150)
    plt.close()

    if compliance_epochs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('PID Physics Compliance (generated images)', fontsize=14)

        axes[0].plot(compliance_epochs, pct_negative_history, color='tab:red', linewidth=2, marker='o')
        axes[0].set_title('% Negative Pixels')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('% of pixels < 0')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].plot(compliance_epochs, sym_score_history, color='tab:green', linewidth=2, marker='o')
        axes[1].set_title('Symmetry Score (lower = more symmetric)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE(image, flipped)')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        plt.savefig(os.path.join(result_dir, 'pid_physics_compliance.png'), dpi=150)
        plt.close()
