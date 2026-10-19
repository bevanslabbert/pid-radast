import os
from PIL import Image
from torch.utils.data import Dataset


class MGCLSPNG(Dataset):
    """MGCLS (MeerKAT Galaxy Cluster Legacy Survey) PNG cutouts.

    Expects one subdirectory per class under `root` (e.g. typical/, exotic/,
    unclassified/). The label of each image is the index of its class
    directory within `classes`, so the class set and order are chosen by the
    caller rather than inferred.

    Returns PIL images so standard torchvision transforms (ToTensor, Normalize,
    RandomRotation, etc.) can be applied directly.

    Args:
        root (str): Stretch directory, e.g. src/datasets/mgcls/png/asinh.
        classes (list[str]): Class subdirectory names; label = list index.
        transform (callable | None): Transform applied to each PIL image.
    """

    def __init__(self, root, classes, transform=None):
        self.root = root
        self.transform = transform
        self._classes = list(classes)

        self.files = []
        self.targets = []
        for label, class_name in enumerate(self._classes):
            class_dir = os.path.join(root, class_name)
            for fname in sorted(f for f in os.listdir(class_dir) if f.endswith('.png')):
                self.files.append(os.path.join(class_name, fname))
                self.targets.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        image = Image.open(path).convert('L')
        label = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        return self._classes
