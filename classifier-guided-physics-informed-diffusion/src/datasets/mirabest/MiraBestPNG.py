import os
from PIL import Image
from torch.utils.data import Dataset


def _parse_label(filename):
    prefix = int(filename.split('_')[0])
    if 100 <= prefix <= 199:
        return 0
    elif 200 <= prefix <= 299:
        return 1
    return None


class MiraBestPNG(Dataset):
    """MiraBest PNG dataset converted from FITS via scripts/convert_fits_to_png.py.

    Returns PIL images so standard torchvision transforms (ToTensor, Normalize,
    RandomRotation, etc.) can be applied directly.

    Args:
        root (str): Directory containing the .png files.
        transform (callable | None): Transform applied to each PIL image.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        all_files = sorted(f for f in os.listdir(root) if f.endswith('.png'))

        self.files = []
        self.targets = []
        for fname in all_files:
            label = _parse_label(fname)
            if label is not None:
                self.files.append(fname)
                self.targets.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        image = Image.open(path)   # mode='L', uint8
        label = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        return ['FR-I', 'FR-II']
