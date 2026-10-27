import os
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from typing import Tuple
from torch.utils.data import random_split
import torchvision.transforms as transforms
from src.datasets.crumb.CRUMB import CRUMB, correct_crumb_test_labels, correct_crumb_train_labels_from_mirabest, correct_crumb_train_labels_by_majority_vote, is_at17_only_source
from src.datasets.mirabest.MiraBest import MiraBest
from src.datasets.mirabest.MiraBestFITS import MiraBestFITS
from src.datasets.mirabest.MiraBestPNG import MiraBestPNG
from src.datasets.mgcls.MGCLSPNG import MGCLSPNG
import torchvision
import numpy as np


def _seed_worker(worker_id):
    """DataLoader workers only auto-inherit torch's RNG from the main
    process; `random`/`numpy` (which torchvision's Random* transforms use
    internally) are left at each worker's fork-inherited state otherwise,
    breaking run-to-run reproducibility even with a fixed seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loaders(dataset, transform, batch_size=2, val_split=0.2, eval_transform=None,
                     stretch='asinh', classes=('typical', 'exotic'), balance=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns trainloader, valloader, and testloader.
    - transform is applied to the training loader (with augmentation).
    - eval_transform, if provided, is applied to val and test loaders (deterministic).
      Falls back to transform when not given.
    - Train/val split uses a fixed seed so results are reproducible across runs.
    - stretch / classes / balance only apply to mgcls_png (PNG stretch directory,
      the class subdirectories to load with label = index in classes, and whether
      to oversample minority classes in the train loader).
    """
    _eval = eval_transform if eval_transform is not None else transform

    print(f"Getting data loader {dataset}")
    if dataset.lower() == 'crumb':
        # Load training data twice so train and val can have different transforms
        full_train_aug  = CRUMB(root='./batches', train=True,  download=True, transform=transform)
        full_train_eval = CRUMB(root='./batches', train=True,  download=True, transform=_eval)
        full_test_set   = CRUMB(root='./batches', train=False, download=True, transform=_eval)

        # Correct train labels first (MiraBest ground truth), since
        # correct_crumb_test_labels learns its (parent, code) -> label
        # mapping from full_train_eval's targets -- it should learn from the
        # cleaned-up labels, not the ones still containing the ~5.3% overlap
        # disagreement with MiraBest.
        n_train_corrected_aug,  mirabest_matched_aug  = correct_crumb_train_labels_from_mirabest(full_train_aug)
        n_train_corrected_eval, mirabest_matched_eval = correct_crumb_train_labels_from_mirabest(full_train_eval)
        print(f"Corrected {n_train_corrected_eval}/{len(full_train_eval)} CRUMB train labels "
              f"using MiraBest ground truth for overlapping sources")

        # For train sources with no MiraBest counterpart (the majority --
        # skip_indices excludes the ones already ground-truth-corrected
        # above), fall back to a self-consistency majority vote learned from
        # train's own (parent, code) -> label distribution.
        n_vote_corrected_aug  = correct_crumb_train_labels_by_majority_vote(full_train_aug,  skip_indices=mirabest_matched_aug)
        n_vote_corrected_eval = correct_crumb_train_labels_by_majority_vote(full_train_eval, skip_indices=mirabest_matched_eval)
        print(f"Corrected {n_vote_corrected_eval}/{len(full_train_eval)} CRUMB train labels "
              f"using self-consistency majority vote (non-MiraBest-overlap sources)")

        n_corrected = correct_crumb_test_labels(full_train_eval, full_test_set)
        print(f"Corrected {n_corrected}/{len(full_test_set)} CRUMB test labels "
              f"using train-derived (parent, code) -> label mapping")

        # Filter out hybrid sources (class 2) and AT17-only sources
        # (unconditionally mislabeled FRI by a CRUMB_builder.ipynb bug -- see
        # findings/crumb_test_label_corruption.md)
        n_at17_train = sum(1 for cl in full_train_aug.complete_labels if is_at17_only_source(cl))
        n_at17_test  = sum(1 for cl in full_test_set.complete_labels  if is_at17_only_source(cl))
        print(f"Excluding {n_at17_train}/{len(full_train_aug)} train and "
              f"{n_at17_test}/{len(full_test_set)} test AT17-only sources (mislabeled FRI)")

        binary_indices = [i for i, t in enumerate(full_train_aug.targets)
                          if t != 2 and not is_at17_only_source(full_train_aug.complete_labels[i])]
        test_indices   = [i for i, t in enumerate(full_test_set.targets)
                          if t != 2 and not is_at17_only_source(full_test_set.complete_labels[i])]

        val_size   = int(len(binary_indices) * val_split)
        train_size = len(binary_indices) - val_size

        # Fixed seed for reproducible splits across runs
        generator = torch.Generator().manual_seed(42)
        train_idx, val_idx = random_split(binary_indices, [train_size, val_size], generator=generator)

        trainloader = DataLoader(Subset(full_train_aug,  list(train_idx)), batch_size=batch_size, shuffle=True,  num_workers=2, worker_init_fn=_seed_worker)
        valloader   = DataLoader(Subset(full_train_eval, list(val_idx)),   batch_size=batch_size, shuffle=False, num_workers=2)
        testloader  = DataLoader(Subset(full_test_set,   test_indices),    batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, valloader, testloader

    if dataset.lower() == 'mirabest':
        full_train_aug  = MiraBest(root='./batches', train=True,  download=True, transform=transform)
        full_train_eval = MiraBest(root='./batches', train=True,  download=True, transform=_eval)
        testset         = MiraBest(root='./batches', train=False, download=True, transform=_eval)

        total_train = len(full_train_aug)
        val_size    = int(total_train * val_split)
        train_size  = total_train - val_size

        generator = torch.Generator().manual_seed(42)
        train_idx, val_idx = random_split(list(range(total_train)), [train_size, val_size], generator=generator)

        trainloader = DataLoader(Subset(full_train_aug,  list(train_idx)), batch_size=batch_size, shuffle=True,  num_workers=2, worker_init_fn=_seed_worker)
        valloader   = DataLoader(Subset(full_train_eval, list(val_idx)),   batch_size=batch_size, shuffle=False, num_workers=2)
        testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        show_batch(trainloader)

        return trainloader, valloader, testloader

    if dataset.lower() == 'mirabest_fits':
        fits_dir = 'src/datasets/mirabest/fits'
        fits_train_transform = transforms.Compose([
            transforms.Resize(213, antialias=True),
            transforms.RandomRotation(180),
            transforms.CenterCrop(150),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        fits_eval_transform = transforms.Compose([
            transforms.Resize(150, antialias=True),
            transforms.CenterCrop(150),
        ])

        train_dataset = MiraBestFITS(root=fits_dir, transform=fits_train_transform)
        eval_dataset  = MiraBestFITS(root=fits_dir, transform=fits_eval_transform)

        total = len(train_dataset)
        val_size   = int(total * val_split)
        test_size  = int(total * 0.1)
        train_size = total - val_size - test_size

        indices = list(range(total))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx   = indices[train_size:train_size + val_size]
        test_idx  = indices[train_size + val_size:]

        trainloader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=2, worker_init_fn=_seed_worker)
        valloader   = DataLoader(Subset(eval_dataset,  val_idx),   batch_size=batch_size, shuffle=False, num_workers=2)
        testloader  = DataLoader(Subset(eval_dataset,  test_idx),  batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, valloader, testloader, train_dataset

    if dataset.lower() == 'crumb_fits':
        fits_dir = 'src/datasets/crumb/fits'
        fits_train_transform = transforms.Compose([
            transforms.Resize(213, antialias=True),
            transforms.RandomRotation(180),
            transforms.CenterCrop(150),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        fits_eval_transform = transforms.Compose([
            transforms.Resize(150, antialias=True),
            transforms.CenterCrop(150),
        ])

        train_dataset = MiraBestFITS(root=fits_dir, transform=fits_train_transform)
        eval_dataset  = MiraBestFITS(root=fits_dir, transform=fits_eval_transform)

        total = len(train_dataset)
        val_size   = int(total * val_split)
        test_size  = int(total * 0.1)
        train_size = total - val_size - test_size

        indices = list(range(total))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx   = indices[train_size:train_size + val_size]
        test_idx  = indices[train_size + val_size:]

        trainloader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=2, worker_init_fn=_seed_worker)
        valloader   = DataLoader(Subset(eval_dataset,  val_idx),   batch_size=batch_size, shuffle=False, num_workers=2)
        testloader  = DataLoader(Subset(eval_dataset,  test_idx),  batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, valloader, testloader, train_dataset

    if dataset.lower() == 'mirabest_fits_png':
        png_dir = 'src/datasets/mirabest/png'
        train_dataset = MiraBestPNG(root=png_dir, transform=transform)
        eval_dataset  = MiraBestPNG(root=png_dir, transform=_eval)

        total = len(train_dataset)
        val_size   = int(total * val_split)
        test_size  = int(total * 0.1)
        train_size = total - val_size - test_size

        indices = list(range(total))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx   = indices[train_size:train_size + val_size]
        test_idx  = indices[train_size + val_size:]

        trainloader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=2, worker_init_fn=_seed_worker)
        valloader   = DataLoader(Subset(eval_dataset,  val_idx),   batch_size=batch_size, shuffle=False, num_workers=2)
        testloader  = DataLoader(Subset(eval_dataset,  test_idx),  batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, valloader, testloader

    if dataset.lower() == 'mgcls_png':
        png_dir = os.path.join('src/datasets/mgcls/png', stretch)
        train_dataset = MGCLSPNG(root=png_dir, classes=classes, transform=transform)
        eval_dataset  = MGCLSPNG(root=png_dir, classes=classes, transform=_eval)

        # Stratified per class: exotic has only ~30 images, so a plain shuffle
        # could leave val/test with almost none of them.
        rng = np.random.default_rng(42)
        targets = np.array(train_dataset.targets)
        train_idx, val_idx, test_idx = [], [], []
        for label in range(len(classes)):
            class_indices = np.flatnonzero(targets == label)
            rng.shuffle(class_indices)
            n_val  = int(len(class_indices) * val_split)
            n_test = int(len(class_indices) * 0.1)
            n_train = len(class_indices) - n_val - n_test
            train_idx += class_indices[:n_train].tolist()
            val_idx   += class_indices[n_train:n_train + n_val].tolist()
            test_idx  += class_indices[n_train + n_val:].tolist()

        print(f"MGCLS ({stretch}) classes {list(classes)}: "
              f"{np.bincount(targets, minlength=len(classes)).tolist()} images per class")

        if balance:
            # typical:exotic is ~70:1, so an unweighted loader lets models ignore
            # exotic entirely. Draw each train image with probability inversely
            # proportional to its class size so every class is ~equally likely
            # per batch; exotic repeats are varied by the random augmentations.
            train_targets = targets[train_idx]
            class_counts = np.bincount(train_targets, minlength=len(classes))
            sample_weights = torch.as_tensor(1.0 / class_counts[train_targets], dtype=torch.double)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)
            print(f"Balancing MGCLS train loader: {class_counts.tolist()} train images per class "
                  f"oversampled to ~equal frequency")
            trainloader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, sampler=sampler, num_workers=2, worker_init_fn=_seed_worker)
        else:
            trainloader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=2, worker_init_fn=_seed_worker)
        valloader   = DataLoader(Subset(eval_dataset,  val_idx),   batch_size=batch_size, shuffle=False, num_workers=2)
        testloader  = DataLoader(Subset(eval_dataset,  test_idx),  batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, valloader, testloader

    raise ValueError(f"Dataset '{dataset}' is not supported!")

def get_data(dataset,
             transform=transforms.Compose([
                 transforms.ToTensor(),  # to range [0,1]
                 transforms.Normalize([0.5], [0.5])  # 0 centers
             ])):
    """
        returns data sets
    """
    if dataset.lower() == 'crumb':
        # Generate trainloader and testloader
        trainset = CRUMB(root='./batches', train=True,
                         download=True, transform=transform)
        testset = CRUMB(root='./batches', train=False,
                        download=True, transform=transform)

        return trainset, testset

    if dataset.lower() == 'mirabest':
        trainset = MiraBest(root='./batches', train=True,
                            download=True, transform=transform)
        testset = MiraBest(root='./batches', train=False,
                           download=True, transform=transform)

        return trainset, testset

    raise ValueError(
        f'Value {dataset} does not exist in list of known datasets!')

def show_batch(dataloader, num_images=4):
    # 1. Grab a single batch
    images, labels = next(iter(dataloader))
    
    # 2. Limit the number of images to show
    images = images[:num_images]

   # 2. PRINT DIMENSIONS
    # Shape is [Batch Size, Channels, Height, Width]
    print(f"Channels (3 for RGB, 1 for Gray): {images.shape[1]}")
    print(f"Height: {images.shape[2]} pixels")
    print(f"Width: {images.shape[3]} pixels") 

    # 3. Un-normalize: If your transform used Mean=0.5, Std=0.5 
    # (common for diffusion), we need to bring it back to [0, 1]
    images = images / 2 + 0.5     
    
    # 4. Make a grid
    grid = torchvision.utils.make_grid(images)
    
    # 5. Convert from Tensor (C, H, W) to Numpy (H, W, C) for Matplotlib
    np_img = grid.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(f"Labels: {labels[:num_images].tolist()}")
    plt.axis('off')
    # plt.show()