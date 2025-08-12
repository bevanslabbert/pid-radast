import sys
from torchvision import datasets, transforms
from data import get_data_loaders
import os
import numpy as np
from ray import tune
from torch import nn
import torch
import matplotlib.pyplot as plt
from parameter_optimization.optimize import optimize_parameters, get_best_config
from models.classification_model import ClassificationModel
from models.diffusion_model import UNet

project_root = "/Users/bevanslabbert/Documents/GitHub/pid-radast"
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_noisy_images(image, noise_schedule, steps=[0, 100, 500, 999]):
    """
    Visualize the noisy images at selected timesteps.
    """
    fig, axes = plt.subplots(1, len(steps), figsize=(15, 5))
    for i, t in enumerate(steps):
        noisy_image = forward_diffusion_process(image, t, noise_schedule)
        axes[i].imshow(noisy_image.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Timestep {t}")
        axes[i].axis("off")
    plt.show()


def linear_beta_schedule(timesteps):
    """
    Generate a linear beta schedule.

    Args:
        timesteps (int): Number of timesteps in the schedule.

    Returns:
        torch.Tensor: A tensor of beta values.
    """
    beta_start = 1e-4  # Smallest beta value
    beta_end = 2e-2    # Largest beta value
    return torch.linspace(beta_start, beta_end, timesteps)


def optimize_classification_model():
    config = {
        'lr': tune.loguniform(1e-4, 1e-2),
        'optimizer_class': tune.choice([torch.optim.AdamW, torch.optim.Adam]),
        'model_class': ClassificationModel,
        'criterion_class': nn.CrossEntropyLoss,
        'dataset': 'MiraBest',
        'batch_size': tune.choice([8, 16])
    }

    # Setting a global seed for reproducibility
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    set_seed(42)
    results = optimize_parameters(config)


def forward_diffusion_process(x, t, noise_schedule):
    """
    Adds noise to an image at a specific timestep.

    Args:
        x (torch.Tensor): Original image tensor.
        t (int): Timestep index.
        noise_schedule (torch.Tensor): Beta schedule tensor.

    Returns:
        torch.Tensor: Noisy image.
    """
    beta_t = noise_schedule[t].view(-1, 1, 1, 1)  # make broadcastable
    noise = torch.randn_like(x)  # Random Gaussian noise
    noised_img = torch.sqrt(1 - beta_t) * x
    another_noised_img = torch.sqrt(beta_t) * noise
    return noised_img + another_noised_img


def reverse_diffusion_step(x, t, noise_schedule):
    """
    Simulates one step of reverse diffusion.

    Args:
        x (torch.Tensor): Noisy image tensor.
        t (int): Timestep index.
        noise_schedule (torch.Tensor): Beta schedule tensor.

    Returns:
        torch.Tensor: Less noisy image.
    """
    beta_t = noise_schedule[t]
    noise = torch.randn_like(x)
    return (x - torch.sqrt(beta_t) * noise) / torch.sqrt(1 - beta_t)


def plot_noisy_images(image, noise_schedule, steps=[0, 100, 500, 999]):
    """
    Visualize the noisy images at selected timesteps.
    """
    fig, axes = plt.subplots(1, len(steps), figsize=(15, 5))
    for i, t in enumerate(steps):
        noisy_image = forward_diffusion_process(image, t, noise_schedule)
        axes[i].imshow(noisy_image.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Timestep {t}")
        axes[i].axis("off")
    plt.show()


def diffusion_model_training():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),           # Resize images to 64x64
        transforms.ToTensor(),                 # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])

    epochs = 10
    timesteps = 1000
    beta_schedule = linear_beta_schedule(timesteps)

    # Model, optimizer, and device setup
    model = UNet(in_channels=1, out_channels=3).to(device)
    trainloader, testloader = get_data_loaders('mirabest', transform, 32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 10
    timesteps = 1000
    beta_schedule = linear_beta_schedule(timesteps)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, _) in enumerate(trainloader):
            images = images.to(device)

            # Forward diffusion: Add noise
            t = torch.randint(0, timesteps, (images.size(0),)).to(
                device)  # Random timestep for each image
            noisy_images = forward_diffusion_process(images, t, beta_schedule)
            noise = torch.randn_like(images)  # True noise

            # UNet prediction
            predicted_noise = model(noisy_images)

            # Compute loss
            loss = criterion(predicted_noise, noise)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(trainloader)
                                           } - Loss: {loss.item():.4f}")


def main():

    set_seed(42)
    optimize_classification_model()


if __name__ == "__main__":
    main()
