"""Compare classifier accuracy on the full CRUMB test set vs. the subset of
CRUMB test images whose sky coordinates match a MiraBest source.

CRUMB filenames encode coordinates as 'RA_+/-Dec.png', e.g.
    ./CombinedCat/PNG/Scaled_Final/000.880_+000.468.png
MiraBest filenames encode them as 'class_RA+/-Dec_z_size.png', e.g.
    ./MiraBest_data/100_162.947+055.386_0.0739_0250.34.png

Matching (RA, Dec) within a small tolerance identifies which CRUMB test
images are the same physical sources as MiraBest galaxies (verified
separately: matched coordinates agree to ~1e-5 deg and pixel content is
near-identical). Restricting to the CRUMB *test* split (not train) keeps
the comparison fair — the classifier never saw these images during training.

For each scope (full crumb test set / mirabest-matched subset) this script
reports:
  1. A full deterministic pass over every image in the scope (no sampling,
     so this number is exactly reproducible).
  2. `--repeats` runs of the original random-100-sample sanity-check
     methodology (with replacement, matching scripts/sanity_check_classifier.py),
     to check how much the previously reported ~40% figure was sampling noise.

Usage:
    python scripts/test_classifier_on_mirabest_subset.py --config config/classification.yaml
"""
import argparse
import glob
import os
import pickle
import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint
from src.datasets.crumb.CRUMB import CRUMB

CHECKPOINT_DIR = 'checkpoints'
CLASS_NAMES = ['FR-I', 'FR-II']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CRUMB_COORD_RE = re.compile(r'([+-]?\d+\.\d+)_([+-]\d+\.\d+)\.png$')
MIRABEST_COORD_RE = re.compile(r'_(\d+\.\d+)([+-]\d+\.\d+)_')

COORD_TOLERANCE_DEG = 0.01  # ~36 arcsec


def load_mirabest_coords(mirabest_batches_dir):
    """Read raw MiraBest pickle batches directly (bypassing MiraBest.py, which
    discards filenames) to recover the (RA, Dec) of every MiraBest source."""
    coords = []
    files = sorted(glob.glob(os.path.join(mirabest_batches_dir, 'data_batch_*')))
    files.append(os.path.join(mirabest_batches_dir, 'test_batch'))
    for fp in files:
        with open(fp, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
        for fn in entry['filenames']:
            m = MIRABEST_COORD_RE.search(fn)
            if m:
                coords.append((float(m.group(1)), float(m.group(2))))
    return coords


def build_model(num_classes, dropout_p, device):
    model = resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(model.fc.in_features, num_classes),
    )
    checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/classification', device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def evaluate_full(dataset, model, device, label):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1).cpu()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total else float('nan')
    print(f"[{label}] FULL deterministic pass: {correct}/{total} correct ({acc:.1%})")
    return acc


def evaluate_random_samples(dataset, model, device, label, num_samples, seed):
    rng = random.Random(seed)
    correct = 0
    with torch.no_grad():
        for _ in range(num_samples):
            idx = rng.randrange(len(dataset))
            image, target = dataset[idx]
            logits = model(image.unsqueeze(0).to(device))
            pred = int(torch.argmax(logits, dim=1).item())
            correct += int(pred == target)
    acc = correct / num_samples
    print(f"[{label}] random {num_samples}-sample run (seed={seed}): "
          f"{correct}/{num_samples} correct ({acc:.1%})")
    return acc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/classification.yaml")
    parser.add_argument("--mirabest-batches-dir", default="src/datasets/mirabest/batches/batches")
    parser.add_argument("--tolerance-deg", type=float, default=COORD_TOLERANCE_DEG)
    parser.add_argument("--num-samples", type=int, default=100, help="Samples per random-sample run")
    parser.add_argument("--repeats", type=int, default=5, help="Number of random-sample runs per scope")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"On device {device}")

    eval_transform = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    crumb_test = CRUMB(root='./batches', train=False, download=True, transform=eval_transform)

    # Full crumb test set, hybrid class excluded (matches get_data_loaders / training pipeline).
    full_indices = [i for i, t in enumerate(crumb_test.targets) if t != 2]

    # Subset of the (non-hybrid) test set whose coordinates match a MiraBest source.
    mirabest_coords = load_mirabest_coords(args.mirabest_batches_dir)
    tol = args.tolerance_deg
    matched_indices = []
    for i in full_indices:
        m = CRUMB_COORD_RE.search(crumb_test.filenames[i])
        if not m:
            continue
        ra, dec = float(m.group(1)), float(m.group(2))
        for mra, mdec in mirabest_coords:
            if abs(ra - mra) < tol and abs(dec - mdec) < tol:
                matched_indices.append(i)
                break

    print(f"CRUMB test set: {len(full_indices)} non-hybrid images total, "
          f"{len(matched_indices)} matched to a MiraBest source "
          f"(tolerance={tol} deg)\n")

    full_crumb_set = Subset(crumb_test, full_indices)
    mirabest_subset = Subset(crumb_test, matched_indices)

    num_classes = len(CLASS_NAMES)
    dropout_p = float(cfg['model'].get('dropout', 0.2))
    model = build_model(num_classes, dropout_p, device)

    results = {}
    for label, ds in (("full_crumb", full_crumb_set), ("mirabest_subset", mirabest_subset)):
        print(f"--- {label} (n={len(ds)}) ---")
        results[f"{label}_full_pass"] = evaluate_full(ds, model, device, label)
        run_accs = []
        for r in range(args.repeats):
            run_accs.append(
                evaluate_random_samples(ds, model, device, label, args.num_samples, seed=r)
            )
        results[f"{label}_random_runs"] = run_accs
        mean_acc = sum(run_accs) / len(run_accs)
        print(f"[{label}] mean over {args.repeats} random-sample runs: {mean_acc:.1%}\n")

    print("=== Summary ===")
    for label in ("full_crumb", "mirabest_subset"):
        full_acc = results[f"{label}_full_pass"]
        run_accs = results[f"{label}_random_runs"]
        print(f"{label}: full-set accuracy={full_acc:.1%}  "
              f"random-run accuracies={[f'{a:.1%}' for a in run_accs]}")


if __name__ == "__main__":
    main()
