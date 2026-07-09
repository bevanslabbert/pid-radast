from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class CRUMB(data.Dataset):
    """
    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``CRUMB.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = 'CRUMB_batches'
    url = "http://www.jb.man.ac.uk/research/MiraBest/CRUMB/CRUMB_batches.tar.gz"
    filename = "CRUMB_batches.tar.gz"
    tgz_md5 = 'a33c0564b99d66fb825e224a0392bc78'
    train_list = [
                  ['data_batch_1', '004e97220b29da803cf67e762ade4b52'],
                  ['data_batch_2', 'a05122141382c3ccec5d5c717a582b16'],
                  ['data_batch_3', 'aada5e8eab52732b3d171b158081bfa7'],
                  ['data_batch_4', 'ebc353fb9059dbeb44da28a50e6092bc'],
                  ['data_batch_5', '5d9459f61a710b27b3a790d3686fb14d'],
                  ['data_batch_6', '965c62bfff96acf83245e68ca42e0c10'],
                  ]

    test_list = [
                 ['test_batch', '0cd9c3869700b720f4adcadba79d793c'],
                 ]
    meta = {
                'filename': 'batches.meta',
                'key': 'label_names',
                'md5': '58f77558538ea5cd398fea6300201332',
                }

    def __init__(self, root, labels='basic', train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.filenames = []
        self.complete_labels = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                for i in range(300):
                    entry['filenames'][i] = entry['filenames'][i][31:]

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                    self.filenames.extend(entry['filenames'])
                    self.complete_labels.extend(entry['complete_labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                    self.filenames.extend(entry['filenames'])
                    self.complete_labels.extend(entry['complete_labels'])

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def correct_crumb_test_labels(train_dataset, test_dataset):
    """Fix CRUMB's test-split `labels`, which are decorrelated from CRUMB's own
    `complete_labels` metadata for a large fraction of test sources across all
    four parent datasets (MiraBest, FR-DEEP, AT17, MiraBest Hybrid) -- see
    results/2026-07-09/mirabest_classifier_vs_crumb_dataset/findings.md.

    In the train split, `complete_labels` cleanly predicts `labels` (94-100%
    purity per (parent, code) pair, matching CRUMB's own builder-notebook rule).
    This learns that majority-vote (parent, code) -> class mapping from
    `train_dataset` and uses it to overwrite `test_dataset.targets` in place,
    ignoring the test split's own corrupted `labels`. Sources whose (parent,
    code) never appears in train are left unchanged.

    Returns the number of test targets that were changed.
    """
    from collections import Counter, defaultdict

    def parent_and_code(complete_label):
        for col in range(len(complete_label)):
            if complete_label[col] != -1:
                return col, complete_label[col]
        return None, None

    tally = defaultdict(Counter)
    for target, complete_label in zip(train_dataset.targets, train_dataset.complete_labels):
        key = parent_and_code(complete_label)
        if key[0] is None:
            continue
        tally[key][target] += 1
    mapping = {key: counter.most_common(1)[0][0] for key, counter in tally.items()}

    n_changed = 0
    for i, complete_label in enumerate(test_dataset.complete_labels):
        key = parent_and_code(complete_label)
        corrected = mapping.get(key)
        if corrected is not None and corrected != test_dataset.targets[i]:
            test_dataset.targets[i] = corrected
            n_changed += 1
    return n_changed
