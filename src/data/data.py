from mirabest.MiraBest import MiraBest
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torchvision

__all__ = ['get_data', 'show_image']

# Returns tuple of train and test data as trainloader, already processed and ready for nn feeding
def get_dataloaders(dataset, transform, batch_size=2) -> Tuple[DataLoader, DataLoader]:
    match dataset.lower():
        case 'mirabest':
            # Generate trainloader and testloader
            trainset = MiraBest(root='./batches', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            testset = MiraBest(root='./batches', train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

            # TODO: Pre-process the data before returning it
            return trainloader, testloader
        case _:
            raise ValueError(f'Value {dataset} does not exist in list of known datasets!')

# returns data sets
def get_data(dataset):
    match dataset.lower():
        case 'mirabest':
            # Generate trainloader and testloader
            trainset = MiraBest(root='./batches', train=True, download=True, transform=transform)
            testset = MiraBest(root='./batches', train=False, download=True, transform=transform)

            return trainset, testset
        case _:
            raise ValueError(f'Value {dataset} does not exist in list of known datasets!')

def show_image(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5 # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
