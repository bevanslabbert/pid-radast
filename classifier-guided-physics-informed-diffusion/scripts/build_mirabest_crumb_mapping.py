"""For every MiraBest image, find its matching CRUMB image (same sky
coordinates, parsed from filenames) and write a CSV listing both sources'
split, label, coordinate offset, pixel similarity, and whether the FR-I/FR-II
label agrees between the two catalogues.

Usage:
    python scripts/build_mirabest_crumb_mapping.py --output results/mirabest_classifier_vs_crumb_dataset/mirabest_crumb_mapping.csv
"""
import argparse
import csv
import glob
import pickle
import re

import numpy as np

CRUMB_COORD_RE = re.compile(r'([+-]?\d+\.\d+)_([+-]\d+\.\d+)\.png$')
MIRABEST_COORD_RE = re.compile(r'_(\d+\.\d+)([+-]\d+\.\d+)_')

CRUMB_LABEL_NAMES = ['FRI', 'FRII', 'Hyb']
MIRABEST_LABEL_NAMES = ['100', '200']

COORD_TOLERANCE_DEG = 0.01  # ~36 arcsec


def load_crumb(batches_dir):
    """Returns list of dicts: split, filename, ra, dec, label, data."""
    records = []
    for split, files in (
        ('train', sorted(glob.glob(f'{batches_dir}/data_batch_*'))),
        ('test', [f'{batches_dir}/test_batch']),
    ):
        for fp in files:
            with open(fp, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
            for fn, lab, img in zip(entry['filenames'], entry['labels'], entry['data']):
                m = CRUMB_COORD_RE.search(fn)
                if not m:
                    continue
                records.append({
                    'split': split,
                    'filename': fn,
                    'ra': float(m.group(1)),
                    'dec': float(m.group(2)),
                    'label': lab,
                    'data': np.asarray(img),
                })
    return records


def load_mirabest(batches_dir):
    records = []
    files = sorted(glob.glob(f'{batches_dir}/data_batch_*'))
    for split, flist in (('train', files), ('test', [f'{batches_dir}/test_batch'])):
        for fp in flist:
            with open(fp, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
            for fn, lab, img in zip(entry['filenames'], entry['labels'], entry['data']):
                m = MIRABEST_COORD_RE.search(fn)
                if not m:
                    continue
                records.append({
                    'split': split,
                    'filename': fn,
                    'ra': float(m.group(1)),
                    'dec': float(m.group(2)),
                    'label': lab,
                    'data': np.asarray(img),
                })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirabest-batches-dir', default='src/datasets/mirabest/batches/batches')
    parser.add_argument('--crumb-batches-dir', default='batches/CRUMB_batches')
    parser.add_argument('--tolerance-deg', type=float, default=COORD_TOLERANCE_DEG)
    parser.add_argument('--output', default='results/mirabest_classifier_vs_crumb_dataset/mirabest_crumb_mapping.csv')
    args = parser.parse_args()

    mirabest = load_mirabest(args.mirabest_batches_dir)
    crumb = load_crumb(args.crumb_batches_dir)
    print(f"MiraBest images: {len(mirabest)}   CRUMB images: {len(crumb)}")

    rows = []
    unmatched = 0
    for mb in mirabest:
        best = None
        best_dist = None
        for cr in crumb:
            d = ((mb['ra'] - cr['ra']) ** 2 + (mb['dec'] - cr['dec']) ** 2) ** 0.5
            if d < args.tolerance_deg and (best is None or d < best_dist):
                best = cr
                best_dist = d

        if best is None:
            unmatched += 1
            rows.append({
                'mirabest_filename': mb['filename'],
                'mirabest_split': mb['split'],
                'mirabest_ra': mb['ra'],
                'mirabest_dec': mb['dec'],
                'mirabest_label': MIRABEST_LABEL_NAMES[mb['label']],
                'crumb_filename': '',
                'crumb_split': '',
                'crumb_ra': '',
                'crumb_dec': '',
                'crumb_label': '',
                'coord_offset_deg': '',
                'pixel_mean_abs_diff': '',
                'labels_agree': '',
            })
            continue

        crumb_class = CRUMB_LABEL_NAMES[best['label']]
        mb_class = MIRABEST_LABEL_NAMES[mb['label']]
        # FRI<->100, FRII<->200
        agree = (crumb_class == 'FRI' and mb_class == '100') or \
                (crumb_class == 'FRII' and mb_class == '200')
        pixel_diff = float(np.mean(np.abs(mb['data'].astype(float) - best['data'].astype(float))))

        rows.append({
            'mirabest_filename': mb['filename'],
            'mirabest_split': mb['split'],
            'mirabest_ra': mb['ra'],
            'mirabest_dec': mb['dec'],
            'mirabest_label': mb_class,
            'crumb_filename': best['filename'],
            'crumb_split': best['split'],
            'crumb_ra': best['ra'],
            'crumb_dec': best['dec'],
            'crumb_label': crumb_class,
            'coord_offset_deg': round(best_dist, 8),
            'pixel_mean_abs_diff': round(pixel_diff, 6),
            'labels_agree': agree,
        })

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matched = len(rows) - unmatched
    disagreements = sum(1 for r in rows if r['labels_agree'] is False)
    print(f"Matched: {matched}/{len(mirabest)}   Unmatched: {unmatched}")
    print(f"Label disagreements among matched: {disagreements}/{matched}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
