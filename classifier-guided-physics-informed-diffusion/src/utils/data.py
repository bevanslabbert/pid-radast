import torch
from torch.utils.data import DataLoader
from typing import Tuple
from torch.utils.data import random_split
import torchvision.transforms as transforms
from src.datasets.mirabest.MiraBest import MiraBest

def get_data_loaders(dataset, transform, batch_size=2, val_split=0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns trainloader, valloader, and testloader.
    - Splits the training data into train and validation sets.
    """
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
    match dataset.lower():
        case 'mirabest':
            # Generate trainloader and testloader
            trainset = MiraBest(root='./batches', train=True,
                                download=True, transform=transform)
            testset = MiraBest(root='./batches', train=False,
                               download=True, transform=transform)

            return trainset, testset
        case _:
            raise ValueError(
                f'Value {dataset} does not exist in list of known datasets!')
