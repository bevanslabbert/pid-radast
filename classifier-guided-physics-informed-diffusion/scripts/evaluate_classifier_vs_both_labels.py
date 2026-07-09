"""For each MiraBest-matched CRUMB image (from mirabest_crumb_mapping.csv),
run the classifier and record its prediction alongside BOTH the CRUMB label
and the MiraBest label, so we can directly check whether the model tracks
true (MiraBest) morphology even when CRUMB's own label disagrees with it.

Usage:
    python scripts/evaluate_classifier_vs_both_labels.py --config config/classification.yaml --scope train
    python scripts/evaluate_classifier_vs_both_labels.py --config config/classification.yaml --scope test
"""
import argparse
import csv
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint
from src.datasets.crumb.CRUMB import CRUMB

CHECKPOINT_DIR = 'checkpoints'
CLASS_NAMES = ['FR-I', 'FR-II']  # index 0 = FRI/100, index 1 = FRII/200
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CRUMB.py strips the first 31 chars off every filename in __init__, so
# ds.filenames won't match the untrimmed filenames stored in the mapping
# CSV. Match by coordinate (parsed from the filename suffix, which survives
# the trimming) instead of by exact filename string.
CRUMB_COORD_RE = re.compile(r'([+-]?\d+\.\d+)_([+-]\d+\.\d+)\.png$')

CRUMB_TO_IDX = {'FRI': 0, 'FRII': 1}
MIRABEST_TO_IDX = {'100': 0, '200': 1}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/classification.yaml")
    parser.add_argument("--mapping-csv", default="results/mirabest_classifier_vs_crumb_dataset/mirabest_crumb_mapping.csv")
    parser.add_argument("--scope", choices=["train", "test", "both"], default="train",
                         help="Which CRUMB split (of the matched image) to evaluate")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    output_csv = args.output_csv or (
        f"results/mirabest_classifier_vs_crumb_dataset/predictions_vs_both_labels_{args.scope}.csv"
    )

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

    with open(args.mapping_csv) as f:
        mapping_rows = list(csv.DictReader(f))

    scopes = ['train', 'test'] if args.scope == 'both' else [args.scope]
    mapping_rows = [r for r in mapping_rows if r['crumb_split'] in scopes and r['crumb_filename']]
    print(f"Evaluating {len(mapping_rows)} matched images (crumb_split in {scopes})")

    # Build (RA, Dec) -> dataset index for each needed split, loaded once.
    # Matched by coordinate rather than filename string since CRUMB.py
    # strips a filename prefix in __init__ (see CRUMB_COORD_RE comment above).
    datasets = {}
    coord_to_idx = {}
    for split in scopes:
        ds = CRUMB(root='./batches', train=(split == 'train'), download=True, transform=eval_transform)
        datasets[split] = ds
        idx_map = {}
        for i, fn in enumerate(ds.filenames):
            m = CRUMB_COORD_RE.search(fn)
            if m:
                idx_map[(float(m.group(1)), float(m.group(2)))] = i
        coord_to_idx[split] = idx_map

    num_classes = len(CLASS_NAMES)
    dropout_p = float(cfg['model'].get('dropout', 0.2))
    model = build_model(num_classes, dropout_p, device)

    results = []
    with torch.no_grad():
        for row in mapping_rows:
            split = row['crumb_split']
            coord = (float(row['crumb_ra']), float(row['crumb_dec']))
            idx = coord_to_idx[split][coord]
            image, _ = datasets[split][idx]

            logits = model(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            pred = int(torch.argmax(probs).item())
            confidence = float(probs[pred].item())

            crumb_idx = CRUMB_TO_IDX[row['crumb_label']]
            mirabest_idx = MIRABEST_TO_IDX[row['mirabest_label']]

            results.append({
                'crumb_filename': row['crumb_filename'],
                'mirabest_filename': row['mirabest_filename'],
                'crumb_split': split,
                'crumb_label': row['crumb_label'],
                'mirabest_label': row['mirabest_label'],
                'labels_agree': row['labels_agree'],
                'prediction': CLASS_NAMES[pred],
                'confidence': round(confidence, 4),
                'correct_vs_crumb': pred == crumb_idx,
                'correct_vs_mirabest': pred == mirabest_idx,
            })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = len(results)
    acc_vs_crumb = sum(r['correct_vs_crumb'] for r in results) / n
    acc_vs_mirabest = sum(r['correct_vs_mirabest'] for r in results) / n

    agree_rows = [r for r in results if r['labels_agree'] == 'True']
    disagree_rows = [r for r in results if r['labels_agree'] == 'False']

    print(f"\nn={n}")
    print(f"Accuracy vs CRUMB label:    {acc_vs_crumb:.1%}")
    print(f"Accuracy vs MiraBest label: {acc_vs_mirabest:.1%}")

    if disagree_rows:
        follows_mirabest = sum(r['correct_vs_mirabest'] for r in disagree_rows) / len(disagree_rows)
        follows_crumb = sum(r['correct_vs_crumb'] for r in disagree_rows) / len(disagree_rows)
        print(f"\nOn the {len(disagree_rows)} images where CRUMB/MiraBest labels disagree:")
        print(f"  model prediction matches MiraBest: {follows_mirabest:.1%}")
        print(f"  model prediction matches CRUMB:    {follows_crumb:.1%}")

    print(f"\nWrote {output_csv}")


if __name__ == "__main__":
    main()
