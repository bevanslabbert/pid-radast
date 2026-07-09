"""Compare classifier accuracy across CRUMB's four parent-dataset subsets
(MiraBest, FR-DEEP, AT17, MiraBest Hybrid) and the full set, for both the
train and test splits, in a single run.

CRUMB's own `complete_labels` field (loaded by src/datasets/crumb/CRUMB.py,
4 columns: MiraBest, FR-DEEP, AT17, MiraBest Hybrid; -1 = "source absent from
this parent catalogue") is used directly to assign each image to the parent
dataset it originated from, following CRUMB's own priority order documented
in its builder notebook (github.com/fmporter/CRUMB/CRUMB_builder.ipynb):
MiraBest > FR-DEEP > AT17 > MiraBest Hybrid (i.e. a source counts as
"MiraBest" if present in MiraBest at all, even if it's also present in one of
the other three).

Motivation: prior investigation (results/2026-07-09/mirabest_classifier_vs_crumb_dataset/findings.md)
found CRUMB's test-split basic labels are decorrelated from CRUMB's own
`complete_labels` metadata specifically for the MiraBest-derived subset. This
script checks whether the classifier's accuracy drop on CRUMB's test set is
uniform across all four parent subsets, or concentrated in one.

Usage:
    python scripts/test_classifier_on_all_crumb_subsets.py --config config/classification.yaml
    # Score the test split against corrected labels instead of CRUMB's raw
    # (test_batch-corrupted) ones -- see scripts/derive_corrected_crumb_test_labels.py:
    python scripts/test_classifier_on_all_crumb_subsets.py \\
        --config config/classification.yaml \\
        --corrected-labels-csv results/2026-07-09/mirabest_classifier_vs_crumb_dataset/crumb_test_corrected_labels.csv
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint
from src.datasets.crumb.CRUMB import CRUMB

CHECKPOINT_DIR = 'checkpoints'
CLASS_NAMES = ['FR-I', 'FR-II']
CLASS_NAMES_RAW = ['FRI', 'FRII', 'Hyb']  # matches scripts/derive_corrected_crumb_test_labels.py output
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CRUMB's own priority order for assigning a source to a parent dataset
# (github.com/fmporter/CRUMB/CRUMB_builder.ipynb): MiraBest > FR-DEEP > AT17 > Hybrid.
PARENT_NAMES = ['MiraBest', 'FR-DEEP', 'AT17', 'Hybrid']


def build_model(num_classes, dropout_p, device, tag=None):
    model = resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(model.fc.in_features, num_classes),
    )
    ckpt_dir = f'{CHECKPOINT_DIR}/classification' + (f'/{tag}' if tag else '')
    checkpoint = load_checkpoint(ckpt_dir, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_corrected_labels(csv_path):
    """Read scripts/derive_corrected_crumb_test_labels.py output: filename -> class idx."""
    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES_RAW)}
    overrides = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            corrected = row['corrected_label']
            if corrected == 'UNKNOWN':
                continue
            overrides[row['filename']] = name_to_idx[corrected]
    return overrides


def parent_source(complete_label):
    for col, name in enumerate(PARENT_NAMES):
        if complete_label[col] != -1:
            return name
    return 'Unknown'


def evaluate_full(dataset, model, device, label):
    if len(dataset) == 0:
        print(f"[{label}] n=0, skipping")
        return float('nan'), 0, 0
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1).cpu()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total
    print(f"[{label}] {correct}/{total} correct ({acc:.1%})")
    return acc, correct, total


def build_subsets(crumb_dataset):
    """Group non-hybrid-basic-label indices by parent dataset of origin."""
    groups = {name: [] for name in PARENT_NAMES}
    full_indices = []
    for i, (target, complete_label) in enumerate(zip(crumb_dataset.targets, crumb_dataset.complete_labels)):
        if target == 2:  # exclude hybrid basic label, matches training pipeline
            continue
        full_indices.append(i)
        groups[parent_source(complete_label)].append(i)
    return full_indices, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/classification.yaml")
    parser.add_argument("--corrected-labels-csv", default=None,
                         help="Path to scripts/derive_corrected_crumb_test_labels.py output. "
                              "If given, the test split is scored against these labels instead "
                              "of CRUMB's own (corrupted) test_batch labels.")
    parser.add_argument("--tag", default=None,
                         help="Load checkpoints/classification/<tag> instead of the untagged "
                              "checkpoints/classification default.")
    args = parser.parse_args()

    corrected = load_corrected_labels(args.corrected_labels_csv) if args.corrected_labels_csv else None

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

    num_classes = len(CLASS_NAMES)
    dropout_p = float(cfg['model'].get('dropout', 0.2))
    model = build_model(num_classes, dropout_p, device, tag=args.tag)

    summary = {}
    for split_name, is_train in (("train", True), ("test", False)):
        print(f"\n===== CRUMB {split_name} split =====")
        crumb_ds = CRUMB(root='./batches', train=is_train, download=True, transform=eval_transform)

        if not is_train and corrected is not None:
            n_overridden = 0
            for i, fn in enumerate(crumb_ds.filenames):
                if fn in corrected:
                    crumb_ds.targets[i] = corrected[fn]
                    n_overridden += 1
            print(f"Overrode {n_overridden}/{len(crumb_ds.filenames)} test labels "
                  f"from {args.corrected_labels_csv}")

        full_indices, groups = build_subsets(crumb_ds)

        print(f"Full (non-hybrid): {len(full_indices)}")
        for name in PARENT_NAMES:
            print(f"  {name}: {len(groups[name])}")

        scopes = [("full", full_indices)] + [(name, groups[name]) for name in PARENT_NAMES]
        for scope_name, indices in scopes:
            subset = Subset(crumb_ds, indices)
            acc, correct, total = evaluate_full(subset, model, device, f"{split_name}/{scope_name}")
            summary[(split_name, scope_name)] = (acc, correct, total)

    print("\n===== Summary =====")
    test_col = "test (corrected)" if corrected is not None else "test (raw)"
    header = f"{'scope':10s} {'train':>18s} {test_col:>18s}"
    print(header)
    for scope_name in ["full"] + PARENT_NAMES:
        train_acc, train_c, train_t = summary[("train", scope_name)]
        test_acc, test_c, test_t = summary[("test", scope_name)]
        train_str = f"{train_c}/{train_t} ({train_acc:.1%})" if train_t else "n/a"
        test_str = f"{test_c}/{test_t} ({test_acc:.1%})" if test_t else "n/a"
        print(f"{scope_name:10s} {train_str:>18s} {test_str:>18s}")


if __name__ == "__main__":
    main()
