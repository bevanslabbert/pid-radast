"""Downloads real FITS cutouts for every CRUMB source that survives the
label-correction pipeline (src/datasets/crumb/CRUMB.py), so CRUMB can be
loaded like MiraBest's FITS dataset (src/datasets/mirabest/MiraBestFITS.py)
instead of only via its own pre-scaled 8-bit PNG batches.

Mirrors the CRUMB_builder.ipynb's own image-acquisition method exactly:
SkyView cutouts from the VLA FIRST (1.4 GHz) survey, at each source's RA/Dec
(which CRUMB already encodes in its own PNG filenames, e.g.
"000.880_+000.468.png").

Output filenames follow MiraBestFITS's own convention (`_parse_label`):
"<100|200>_<RA>_<±Dec>.fits" -- 1xx = FR-I, 2xx = FR-II. Written straight
into src/datasets/mirabest/fits/'s sibling directory (src/datasets/crumb/fits/)
so MiraBestFITS can load it unmodified by pointing `root` there.

Usage:
    python scripts/download_crumb_fits.py [--limit N]
"""
import argparse
import os
import sys
import urllib.request

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.crumb.CRUMB import (
    CRUMB,
    correct_crumb_test_labels,
    correct_crumb_train_labels_from_mirabest,
    correct_crumb_train_labels_by_majority_vote,
    is_at17_only_source,
)

OUT_DIR = 'src/datasets/crumb/fits'
SURVEY = 'VLA FIRST (1.4 GHz)'
PIXELS = 300


def _corrected_sources():
    """Reproduces the exact correction + filter pipeline get_data_loaders
    applies for dataset='crumb' (src/utils/data.py), returning a flat list
    of (filename, binary_label) for every retained train+test source."""
    train = CRUMB(root='./batches', train=True, download=True, transform=None)
    test = CRUMB(root='./batches', train=False, download=True, transform=None)

    _, matched = correct_crumb_train_labels_from_mirabest(train)
    correct_crumb_train_labels_by_majority_vote(train, skip_indices=matched)
    correct_crumb_test_labels(train, test)

    sources = []
    for dataset in (train, test):
        for filename, target, complete_label in zip(dataset.filenames, dataset.targets, dataset.complete_labels):
            if target == 2 or is_at17_only_source(complete_label):
                continue
            sources.append((filename, target))
    return sources


def _parse_ra_dec(filename):
    stem = filename.rsplit('/', 1)[-1].removesuffix('.png')
    ra_str, dec_str = stem.split('_', 1)
    return stem, float(ra_str), float(dec_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                         help='Only process the first N sources (for testing)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    sources = _corrected_sources()
    print(f"{len(sources)} corrected, non-hybrid, non-AT17-only sources to fetch")
    if args.limit is not None:
        sources = sources[:args.limit]

    n_ok = n_skip = n_fail = 0
    failures = []

    for i, (filename, target) in enumerate(sources):
        stem, ra, dec = _parse_ra_dec(filename)
        label_prefix = '100' if target == 0 else '200'
        out_path = os.path.join(OUT_DIR, f'{label_prefix}_{stem}.fits')

        if os.path.exists(out_path):
            n_skip += 1
            continue

        try:
            coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
            location = SkyView.get_image_list(position=coords, survey=SURVEY, pixels=PIXELS)
            if not location:
                raise RuntimeError('no image returned by SkyView')
            urllib.request.urlretrieve(location[0], out_path)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            failures.append((filename, str(e)))
            print(f"[{i+1}/{len(sources)}] FAILED {filename}: {e}")

        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(sources)}] ok={n_ok} skip={n_skip} fail={n_fail}")

    print(f"\nDone. ok={n_ok} skip={n_skip} fail={n_fail} total={len(sources)}")
    if failures:
        fail_path = os.path.join(OUT_DIR, 'download_failures.txt')
        with open(fail_path, 'w') as f:
            for filename, err in failures:
                f.write(f'{filename}\t{err}\n')
        print(f"Failure list written to {fail_path}")


if __name__ == '__main__':
    main()
