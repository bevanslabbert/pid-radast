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


def is_at17_only_source(complete_label):
    """True if a source appears only in the AT17 parent catalogue (not
    MiraBest, FR-DEEP, or MiraBest Hybrid) -- see
    findings/crumb_test_label_corruption.md. CRUMB_builder.ipynb's
    label-assignment branch for these sources checks the wrong column
    (`label_vector[i, 1]`, FR-DEEP, instead of `label_vector[i, 2]`, AT17),
    so the condition is structurally always false and every AT17-only source
    is unconditionally labeled FRI regardless of its true AT17 code. There's
    no known-correct relabeling (the intended AT17 threshold rule isn't
    recoverable from the buggy branch alone), so these sources should be
    excluded from training/evaluation rather than trusted. Affects 210/2100
    sources (10%) of the full CRUMB dataset.

    `complete_label` columns are [MiraBest, FR-DEEP, AT17, Hybrid]; `-1`
    means "source absent from this parent catalogue".
    """
    mirabest, fr_deep, at17, hybrid = complete_label[0], complete_label[1], complete_label[2], complete_label[3]
    return at17 != -1 and mirabest == -1 and fr_deep == -1 and hybrid == -1


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


def correct_crumb_train_labels_from_mirabest(
        train_dataset,
        mapping_csv='results/mirabest_classifier_vs_crumb_dataset/mirabest_crumb_mapping.csv'):
    """Overwrite CRUMB train-split `labels` with MiraBest's own canonical
    label wherever a train image is a MiraBest source -- see
    findings/crumb_test_label_corruption.md.

    Unlike the test split (whose `labels` field is decorrelated from its own
    `complete_labels` metadata), CRUMB's train split has no index-alignment
    bug -- but its labels still disagree with MiraBest's on ~5.3% of the
    ~665 train images that are also MiraBest sources (confirmed via
    `build_mirabest_crumb_mapping.py`'s RA/Dec coordinate match, tolerance
    0.01 deg). The classifier's own predictions on those disagreement cases
    track MiraBest's label 88.6% of the time, so MiraBest is treated as
    ground truth for this overlap subset rather than CRUMB's own label.

    `mapping_csv` is `results/mirabest_classifier_vs_crumb_dataset/mirabest_crumb_mapping.csv`,
    produced by `scripts/build_mirabest_crumb_mapping.py`. Sources with no
    MiraBest match (the ~1135 non-overlap train images) are left unchanged.

    Returns a tuple `(n_changed, matched_indices)`: the number of train
    targets that were changed, and the set of train indices with a MiraBest
    match (whether or not the label actually changed) -- callers should
    treat these as ground-truth-confirmed and skip them in any subsequent,
    lower-confidence correction pass (e.g.
    `correct_crumb_train_labels_by_majority_vote`).
    """
    import csv

    label_to_idx = {'FRI': 0, 'FRII': 1, 'Hyb': 2, '100': 0, '200': 1}

    filename_to_label = {}
    with open(mapping_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row['crumb_split'] != 'train' or not row['crumb_filename']:
                continue
            basename = row['crumb_filename'].rsplit('/', 1)[-1]
            filename_to_label[basename] = label_to_idx[row['mirabest_label']]

    n_changed = 0
    matched_indices = set()
    for i, filename in enumerate(train_dataset.filenames):
        corrected = filename_to_label.get(filename)
        if corrected is None:
            continue
        matched_indices.add(i)
        if corrected != train_dataset.targets[i]:
            train_dataset.targets[i] = corrected
            n_changed += 1
    return n_changed, matched_indices


def correct_crumb_train_labels_by_majority_vote(train_dataset, skip_indices=frozenset()):
    """Self-consistency correction for the CRUMB train sources that have no
    MiraBest counterpart to check against (~850 of ~1550 post-filter train
    images) -- see findings/crumb_test_label_corruption.md.

    Reuses the same logic as `correct_crumb_test_labels`: learns a
    majority-vote `(parent, code) -> label` mapping from `train_dataset`
    itself (train's own `complete_labels` predict its `labels` at 94-100%
    purity per group) and overwrites any train target that disagrees with
    its group's majority. Unlike the MiraBest ground-truth fix, this is a
    self-consistency vote, not independent ground truth -- lower confidence,
    used only to fill the gap where no ground truth exists.

    Two exclusions are required for this to be safe:
    - `skip_indices` (typically the indices already matched against MiraBest
      ground truth by `correct_crumb_train_labels_from_mirabest`) are left
      untouched, so the higher-confidence ground-truth label always wins.
    - AT17-only sources (`is_at17_only_source`) are excluded from both the
      vote's tally and the correction targets. That group is unconditionally
      mislabeled FRI by a CRUMB_builder.ipynb bug, so its "majority" label
      *is* the bug -- voting would entrench it rather than fix it. These
      sources are excluded from training/eval entirely elsewhere.

    Returns the number of train targets that were changed.
    """
    from collections import Counter, defaultdict

    def parent_and_code(complete_label):
        for col in range(len(complete_label)):
            if complete_label[col] != -1:
                return col, complete_label[col]
        return None, None

    tally = defaultdict(Counter)
    for target, complete_label in zip(train_dataset.targets, train_dataset.complete_labels):
        if is_at17_only_source(complete_label):
            continue
        key = parent_and_code(complete_label)
        if key[0] is None:
            continue
        tally[key][target] += 1
    mapping = {key: counter.most_common(1)[0][0] for key, counter in tally.items()}

    n_changed = 0
    for i, complete_label in enumerate(train_dataset.complete_labels):
        if i in skip_indices or is_at17_only_source(complete_label):
            continue
        key = parent_and_code(complete_label)
        corrected = mapping.get(key)
        if corrected is not None and corrected != train_dataset.targets[i]:
            train_dataset.targets[i] = corrected
            n_changed += 1
    return n_changed
