"""Convert MiraBest FITS files to uint8 grayscale PNGs.

Reads each .fits file from src/datasets/mirabest/fits/, applies the same
sym-log-SNR normalisation used by MiraBestFITS (values clamped to [-1, 1]),
maps to [0, 255] uint8, and saves to src/datasets/mirabest/png/ with the
same filename stem.

Run from the project root:
    python scripts/convert_fits_to_png.py
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.mirabest.MiraBestFITS import MiraBestFITS

FITS_DIR = 'src/datasets/mirabest/fits'
PNG_DIR  = 'src/datasets/mirabest/png'


def main():
    os.makedirs(PNG_DIR, exist_ok=True)

    # Load without any spatial transform so we get the raw normalised 300x300 tensor
    dataset = MiraBestFITS(root=FITS_DIR, transform=None)

    for i in range(len(dataset)):
        tensor, label = dataset[i]          # tensor: (1, H, W) in [-1, 1]
        arr = tensor.squeeze(0).numpy()     # (H, W)
        arr_uint8 = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

        stem = os.path.splitext(dataset.files[i])[0]
        out_path = os.path.join(PNG_DIR, stem + '.png')
        Image.fromarray(arr_uint8, mode='L').save(out_path)

        print(f"[{i+1}/{len(dataset)}] {dataset.files[i]} -> {stem}.png  (label={label})")

    print(f"\nDone. {len(dataset)} PNGs written to {PNG_DIR}/")


if __name__ == '__main__':
    main()
