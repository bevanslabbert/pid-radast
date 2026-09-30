"""Why does CGD's classifier gradient barely change its samples?

Three checks on CGD's frozen guidance classifier (config/classifier_guided_diffusion.yaml
model.classifier_checkpoint):

  1. Clean real images: accuracy on the real crumb_fits val split. Near 50%, or near-
     constant predictions, means it was trained in the wrong domain (e.g. linear `crumb`
     instead of log-SNR `crumb_fits`).
  2. Noisy real images: accuracy and target-class probability on the same images after
     the DDPM forward process at increasing t. Guidance takes its gradient on noisy x_t
     at every step, so a classifier that collapses to chance as t grows gives no signal.
  3. Real sampling: runs generate_class_samples_guided on one CGD checkpoint and logs,
     per step, the classifier nudge's size relative to the CFG noise prediction
     (nudge_ratio, at classifier_scale=1) and the classifier's belief on x_t.

Writes <output>/guidance_diagnostics.json and guidance_diagnostics.png.

Usage:
    python scripts/diagnose_cgd_guidance.py --tag cls_guided_diffusion_crumb_fits_seed42
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint
from src.utils.data import get_data_loaders
from src.models.diffusion import build_diffusion_components
from src.utils.metrics import generate_class_samples_guided
from src.pipelines.train_pipeline import _load_guidance_classifier

NOISE_TIMESTEPS = [0, 25, 50, 100, 200, 400, 600, 800, 999]


def classify(classifier, time_aware, images, t):
    if time_aware:
        return classifier(images, torch.full((images.shape[0],), t, dtype=torch.long, device=images.device))
    return classifier(images)


def check_real(classifier, time_aware, scheduler, valloader, device):
    """Accuracy / mean target-class probability / FR-II prediction rate on real val images at each noise level."""
    results = []
    for t in NOISE_TIMESTEPS:
        correct, p_target, pred_1, n = 0.0, 0.0, 0.0, 0
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            if t > 0:
                noise = torch.randn_like(images)
                images = scheduler.add_noise(images, noise, torch.full_like(labels, t))
            with torch.no_grad():
                probs = F.softmax(classify(classifier, time_aware, images, t), dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            p_target += probs.gather(1, labels.unsqueeze(1)).sum().item()
            pred_1 += (preds == 1).sum().item()
            n += labels.shape[0]
        results.append({'t': t, 'accuracy': correct / n, 'p_target': p_target / n, 'frac_pred_FR-II': pred_1 / n})
        print(f"  t={t:4d}  accuracy={correct / n:.3f}  p_target={p_target / n:.3f}  predicted FR-II={pred_1 / n:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', default='cls_guided_diffusion_crumb_fits_seed42')
    parser.add_argument('--num-samples', type=int, default=16, help="Samples per class for the sampling check.")
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--guidance-classifier', default=None,
                        help="Override the guidance classifier checkpoint dir (default: config model.classifier_checkpoint).")
    parser.add_argument('--guidance-classifier-type', default=None, choices=['classification', 'robust_classification'])
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='results/cgd_guidance_diagnostics')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config('config/classifier_guided_diffusion.yaml')
    config['data']['num_classes'] = args.num_classes
    if args.guidance_classifier is not None:
        config['model']['classifier_checkpoint'] = args.guidance_classifier
    if args.guidance_classifier_type is not None:
        config['model']['classifier_type'] = args.guidance_classifier_type
    classifier, time_aware = _load_guidance_classifier(config, device)
    unet, scheduler, class_emb, _ = build_diffusion_components(config, {}, device)
    _, valloader, _, _ = get_data_loaders('crumb_fits', None, batch_size=32)
    print(f"Guidance classifier: {config['model']['classifier_checkpoint']} "
          f"({config['model']['classifier_type']}, time-aware={time_aware})")

    print("\n[1+2] Guidance classifier on real crumb_fits val images, clean (t=0) and noised:")
    real = check_real(classifier, time_aware, scheduler, valloader, device)

    print(f"\n[3] Sampling {args.tag} at CFG {args.guidance_scale}, classifier_scale 1.0:")
    ckpt = load_checkpoint(f'checkpoints/classifier_guided_diffusion/{args.tag}', device)
    unet.load_state_dict(ckpt['model_state_dict'])
    class_emb.load_state_dict(ckpt['class_emb_state_dict'])
    unet.to(device).eval()
    steps = []
    generate_class_samples_guided(
        unet, scheduler, class_emb, args.num_classes, args.num_samples, device,
        classifier=classifier, classifier_time_aware=time_aware, shape=(1, 150, 150),
        guidance_scale=args.guidance_scale, classifier_scale=1.0, diagnostics=steps,
    )
    for d in steps[::5]:
        print(f"  class={d['class']}  t={d['t']:4d}  nudge_ratio={d['nudge_ratio']:.2e}  "
              f"p_target={d['p_target']:.3f}  frac_pred_target={d['frac_pred_target']:.2f}")
    ratios = np.array([d['nudge_ratio'] for d in steps])
    print(f"  nudge_ratio over all steps: median={np.median(ratios):.2e}  max={ratios.max():.2e}")

    with open(os.path.join(args.output, 'guidance_diagnostics.json'), 'w') as f:
        json.dump({'guidance_classifier': config['model']['classifier_checkpoint'], 'cgd_tag': args.tag,
                   'guidance_scale': args.guidance_scale, 'real_images': real, 'sampling_steps': steps}, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot([r['t'] for r in real], [r['accuracy'] for r in real], 'o-', color='#48a')
    axes[0].axhline(0.5, color='#888', ls='--', lw=1)
    axes[0].set(title='Guidance classifier on noised real images', xlabel='noise timestep t', ylabel='accuracy')
    for c, name, col in [(0, 'FR-I', '#c44'), (1, 'FR-II', '#2a2')]:
        ds = [d for d in steps if d['class'] == c]
        axes[1].plot([d['t'] for d in ds], [d['nudge_ratio'] for d in ds], 'o-', ms=4, color=col, label=name)
        axes[2].plot([d['t'] for d in ds], [d['p_target'] for d in ds], 'o-', ms=4, color=col, label=name)
    axes[1].set(title='|classifier nudge| / |CFG noise pred|  (scale 1)', xlabel='sampling timestep t', yscale='log')
    axes[2].set(title='Guidance classifier p(target) on x_t', xlabel='sampling timestep t', ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].legend(frameon=False)
    for ax in axes[1:]:
        ax.invert_xaxis()  # sampling runs from t=999 down to 0
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, 'guidance_diagnostics.png'), dpi=150)
    print(f"\nSaved to {args.output}/")


if __name__ == '__main__':
    main()
