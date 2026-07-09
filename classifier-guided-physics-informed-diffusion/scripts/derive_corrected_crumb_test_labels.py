"""Reconstruct accurate FRI/FRII/Hyb labels for CRUMB's test split.

Background (results/2026-07-09/mirabest_classifier_vs_crumb_dataset/findings.md):
CRUMB's test_batch `labels` field is decorrelated from its own `complete_labels`
metadata (each source's per-parent-catalogue code) for a large fraction of
sources, across all four parent datasets -- not just the MiraBest subset. The
train batches don't have this problem: there, `complete_labels` cleanly
predicts `labels` via CRUMB's own code->class rule (see CRUMB_builder.ipynb),
at ~94-100% purity per code, matching the known ~5% baseline label noise.

This script exploits that: it learns a majority-vote code->class mapping per
(parent dataset, code) from the *train* split (trustworthy), then applies
that mapping to the *test* split's `complete_labels` to reconstruct a label
per test image -- ignoring test's own corrupted `labels` field entirely.

Usage:
    python scripts/derive_corrected_crumb_test_labels.py --config config/classification.yaml
Writes:
    results/mirabest_classifier_vs_crumb_dataset/crumb_test_corrected_labels.csv
"""
import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.crumb.CRUMB import CRUMB

PARENT_NAMES = ['MiraBest', 'FR-DEEP', 'AT17', 'Hybrid']
CLASS_NAMES = ['FRI', 'FRII', 'Hyb']

OUT_PATH = 'results/mirabest_classifier_vs_crumb_dataset/crumb_test_corrected_labels.csv'


def parent_source(complete_label):
    for col, name in enumerate(PARENT_NAMES):
        if complete_label[col] != -1:
            return name, int(complete_label[col])
    return 'Unknown', None


def learn_code_to_label(train_ds):
    tally = collections.defaultdict(collections.Counter)
    for target, complete_label in zip(train_ds.targets, train_ds.complete_labels):
        src, code = parent_source(complete_label)
        if src == 'Unknown':
            continue
        tally[(src, code)][target] += 1

    mapping = {}
    print("Learned (parent, code) -> label mapping from TRAIN split:")
    for key in sorted(tally, key=lambda k: (k[0], k[1])):
        counts = tally[key]
        total = sum(counts.values())
        majority_label, majority_count = counts.most_common(1)[0]
        purity = majority_count / total
        mapping[key] = majority_label
        print(f"  {key}: {dict(counts)} -> {CLASS_NAMES[majority_label]} "
              f"(purity={purity:.1%}, n={total})")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    train_ds = CRUMB(root='./batches', train=True, download=True, transform=None)
    test_ds = CRUMB(root='./batches', train=False, download=True, transform=None)

    mapping = learn_code_to_label(train_ds)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_corrected = 0
    n_unmapped = 0
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'crumb_raw_label', 'parent_source', 'parent_code',
                          'corrected_label', 'changed'])
        for fn, raw_target, complete_label in zip(test_ds.filenames, test_ds.targets, test_ds.complete_labels):
            src, code = parent_source(complete_label)
            key = (src, code)
            corrected = mapping.get(key)
            if corrected is None:
                n_unmapped += 1
                corrected_name = 'UNKNOWN'
                changed = ''
            else:
                changed = corrected != raw_target
                n_corrected += int(changed)
                corrected_name = CLASS_NAMES[corrected]
            writer.writerow([fn, CLASS_NAMES[raw_target], src, code, corrected_name, changed])

    print(f"\nWrote {args.out}")
    print(f"{n_corrected}/{len(test_ds)} test labels changed by correction "
          f"({n_unmapped} sources had no (parent, code) seen in train, left as raw label)")


if __name__ == "__main__":
    main()
