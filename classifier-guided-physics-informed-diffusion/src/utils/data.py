import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Tuple
from torch.utils.data import random_split
import torchvision.transforms as transforms
from src.datasets.mirabest.MiraBest import MiraBest
import torchvision
import numpy as np

def get_data_loaders(dataset, transform, batch_size=2, val_split=0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns trainloader, valloader, and testloader.
    - Splits the training data into train and validation sets.
    """
    print(f"Getting data loader {dataset}")
    if dataset.lower() == 'mirabest':
        # ---- Load full training and test sets ----
        full_train_set = MiraBest(root='./batches', train=True, download=True, transform=transform)
        testset = MiraBest(root='./batches', train=False, download=True, transform=transform)

        # ---- Create train/val split ----
        total_train_size = len(full_train_set)
        val_size = int(total_train_size * val_split)
        train_size = total_train_size - val_size

        train_subset, val_subset = random_split(full_train_set, [train_size, val_size])

        # ---- DataLoaders ----
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        show_batch(trainloader)

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
    if dataset.lower() == 'mirabest':
        # Generate trainloader and testloader
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