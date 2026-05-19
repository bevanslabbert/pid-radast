import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from typing import Tuple
from torch.utils.data import random_split
import torchvision.transforms as transforms
from src.datasets.crumb.CRUMB import CRUMB
from src.datasets.mirabest.MiraBest import MiraBest
from src.datasets.mirabest.MiraBestFITS import MiraBestFITS
from src.datasets.mirabest.MiraBestPNG import MiraBestPNG
import torchvision
import numpy as np

def get_data_loaders(dataset, transform, batch_size=2, val_split=0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns trainloader, valloader, and testloader.
    - Splits the training data into train and validation sets.
    """
    print(f"Getting data loader {dataset}")
    if dataset.lower() == 'crumb':
        # ---- Load full training and test sets ----
        full_train_set = CRUMB(root='./batches', train=True, download=True, transform=transform)
        full_test_set = CRUMB(root='./batches', train=False, download=True, transform=transform)

        # ---- Filter out hybrid sources (class 2) to align with MiraBest binary classes ----
        train_indices = [i for i, t in enumerate(full_train_set.targets) if t != 2]
        test_indices = [i for i, t in enumerate(full_test_set.targets) if t != 2]

        binary_train_set = Subset(full_train_set, train_indices)
        binary_test_set = Subset(full_test_set, test_indices)

        # ---- Create train/val split ----
        val_size = int(len(binary_train_set) * val_split)
        train_size = len(binary_train_set) - val_size

        train_subset, val_subset = random_split(binary_train_set, [train_size, val_size])

        # ---- DataLoaders ----
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        testloader = DataLoader(binary_test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, valloader, testloader

    if dataset.lower() == 'mirabest':
        full_train_set = MiraBest(root='./batches', train=True, download=True, transform=transform)
        testset = MiraBest(root='./batches', train=False, download=True, transform=transform)

        total_train_size = len(full_train_set)
        val_size = int(total_train_size * val_split)
        train_size = total_train_size - val_size

        train_subset, val_subset = random_split(full_train_set, [train_size, val_size])

        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        show_batch(trainloader)

        return trainloader, valloader, testloader

    if dataset.lower() == 'mirabest_fits':
        fits_dir = 'src/datasets/mirabest/fits'
        # FITS images are already float tensors normalized to [-1, 1].
        # Use a tensor-compatible transform (resize + spatial augmentations).
        fits_transform = transforms.Compose([
            # Upscale to ceil(150 * sqrt(2)) = 213 so that a 150x150 centre crop
            # contains only real image content after any rotation angle.
            transforms.Resize(213, antialias=True),
            transforms.RandomRotation(180),
            transforms.CenterCrop(150),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        full_dataset = MiraBestFITS(root=fits_dir, transform=fits_transform)

        total = len(full_dataset)
        val_size = int(total * val_split)
        test_size = int(total * 0.1)
        train_size = total - val_size - test_size

        indices = list(range(total))
        # Fixed seed for reproducible splits
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx   = indices[train_size:train_size + val_size]
        test_idx  = indices[train_size + val_size:]

        trainloader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=2)
        valloader   = DataLoader(Subset(full_dataset, val_idx),   batch_size=batch_size, shuffle=False, num_workers=2)
        testloader  = DataLoader(Subset(full_dataset, test_idx),  batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, valloader, testloader, full_dataset

    if dataset.lower() == 'mirabest_fits_png':
        png_dir = 'src/datasets/mirabest/png'
        full_dataset = MiraBestPNG(root=png_dir, transform=transform)

        total = len(full_dataset)
        val_size = int(total * val_split)
        test_size = int(total * 0.1)
        train_size = total - val_size - test_size

        indices = list(range(total))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx   = indices[train_size:train_size + val_size]
        test_idx  = indices[train_size + val_size:]

        trainloader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=2)
        valloader   = DataLoader(Subset(full_dataset, val_idx),   batch_size=batch_size, shuffle=False, num_workers=2)
        testloader  = DataLoader(Subset(full_dataset, test_idx),  batch_size=batch_size, shuffle=False, num_workers=2)

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