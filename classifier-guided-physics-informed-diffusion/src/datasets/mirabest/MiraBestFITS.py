import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits


# Filename prefix → binary label mapping
# 1xx = FR-I  → 0
# 2xx = FR-II → 1
# 3xx = bent-tailed → excluded
def _parse_label(filename):
    prefix = int(filename.split('_')[0])
    if 100 <= prefix <= 199:
        return 0
    elif 200 <= prefix <= 299:
        return 1
    else:
        return None  # exclude


class MiraBestFITS(Dataset):
    """MiraBest dataset loaded directly from FITS files.

    Reads 300x300 float32 radio images from FITS files. The binary class
    label (FR-I = 0, FR-II = 1) is parsed from the filename prefix:
      1xx → FR-I (0), 2xx → FR-II (1), 3xx → excluded.

    Normalisation uses dataset-level median noise_rms and median peak_log so
    that the scaling can be inverted when writing generated FITS files.
    Stats are computed once and cached to <root>/fits_stats.json.

    Args:
        root (str): Directory containing the .fits files.
        indices (list[int] | None): If provided, use only these sample indices.
        transform (callable | None): Tensor transform applied after normalisation.
    """

    STATS_FILE = 'fits_stats.json'

    def __init__(self, root, indices=None, transform=None):
        self.root = root
        self.transform = transform

        all_files = sorted(
            f for f in os.listdir(root) if f.endswith('.fits')
        )

        self.files = []
        self.targets = []

        for fname in all_files:
            label = _parse_label(fname)
            if label is not None:
                self.files.append(fname)
                self.targets.append(label)

        if indices is not None:
            self.files = [self.files[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]

        self.median_noise_rms, self.median_peak_log = self._load_or_compute_stats()

    # ------------------------------------------------------------------
    def _load_or_compute_stats(self):
        stats_path = os.path.join(self.root, self.STATS_FILE)

        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            print(f"Loaded FITS stats: noise_rms={stats['median_noise_rms']:.6f}, peak_log={stats['median_peak_log']:.6f}")
            return stats['median_noise_rms'], stats['median_peak_log']

        return self._compute_and_save_stats(stats_path)

    def _compute_and_save_stats(self, stats_path):
        print(f"Computing dataset stats across {len(self.files)} FITS files...")
        noise_rms_list = []
        peak_log_list = []

        for fname in self.files:
            path = os.path.join(self.root, fname)
            with fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float32)
            data = np.nan_to_num(data, nan=0.0)
            noise_rms = float(np.std(data)) + 1e-8
            snr = data / noise_rms
            peak_log = float(np.log1p(np.abs(snr).max()) + 1e-8)
            noise_rms_list.append(noise_rms)
            peak_log_list.append(peak_log)

        median_noise_rms = float(np.median(noise_rms_list))
        median_peak_log = float(np.median(peak_log_list))

        stats = {
            'median_noise_rms': median_noise_rms,
            'median_peak_log': median_peak_log,
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Saved FITS stats: noise_rms={median_noise_rms:.6f}, peak_log={median_peak_log:.6f}")
        return median_noise_rms, median_peak_log

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        label = self.targets[idx]

        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)  # (300, 300)

        data = np.nan_to_num(data, nan=0.0)
        data = self._normalise(data)

        # Always return a (1, H, W) tensor; transform operates on tensors
        data = torch.from_numpy(data).unsqueeze(0)

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        snr = data / self.median_noise_rms
        sym_log = np.sign(snr) * np.log1p(np.abs(snr))
        return np.clip(sym_log / self.median_peak_log, -1.0, 1.0)

    def denormalise(self, data: np.ndarray) -> np.ndarray:
        """Invert normalisation to recover approximate Jy/beam values.

        Args:
            data: numpy array in [-1, 1]
        Returns:
            numpy array in approximate Jy/beam
        """
        sym_log = data * self.median_peak_log
        snr = np.sign(sym_log) * np.expm1(np.abs(sym_log))
        return snr * self.median_noise_rms

    @staticmethod
    def write_fits(data: np.ndarray, path: str):
        """Write a 2D numpy array to a FITS file with a minimal header.

        Args:
            data: 2D numpy array (H, W) in Jy/beam
            path: output file path
        """
        header = fits.Header()
        header['SIMPLE'] = True
        header['BITPIX'] = -32
        header['NAXIS'] = 2
        header['NAXIS1'] = data.shape[1]
        header['NAXIS2'] = data.shape[0]
        header['BUNIT'] = 'JY/BEAM'
        header['COMMENT'] = 'Synthetically generated by diffusion model'
        hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
        hdu.writeto(path, overwrite=True)

    @property
    def classes(self):
        return ['FR-I', 'FR-II']
