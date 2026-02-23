import torch
import numpy as np
import argparse
import os
from datetime import datetime
import torchvision.transforms as transforms
from src.utils.config import load_config
from src.pipelines.optimize_parameters_pipeline import optimize_parameters
from src.pipelines.train_pipeline import train_model
from src.pipelines.test_pipeline import test_model
from src.utils.data import get_data_loaders
from src.utils.common import clear_gpu_memory
from src.utils.checkpoint import load_checkpoint
from torchvision.utils import make_grid, save_image

# Setting a global seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():

    clear_gpu_memory()

    parser = argparse.ArgumentParser(description="Run experiments modularly.")
    subparsers = parser.add_subparsers(dest="command")

    model_help = "[robust_classification | classification | diffusion ] The model type to perform the current action on"
    config_help = "[string] Path to .yaml file to use for config [default: config/<model>.yaml]"
    resume_help = "[True | False] Whether to resume training from last saved epoch"
    checkpoint_help = "[True | False] Whether to save the training checkpoints"
    dataset_help = "[mirabest] Which dataset to evaluate"

    # --- Optimize command ---
    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--model", required=True, help=model_help)
    optimize_parser.add_argument("--checkpoint", required=False)
    optimize_parser.add_argument("--config", required=True)
    optimize_parser.add_argument("--dataset", help=dataset_help)

    # --- Train command ---
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True, help=model_help)
    train_parser.add_argument("--config", help=config_help)
    train_parser.add_argument("--resume", help=resume_help)
    train_parser.add_argument("--checkpoint", help=checkpoint_help)
    train_parser.add_argument("--dataset", help=dataset_help)

    # --- Test command ---
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model", required=True, help=model_help)
    test_parser.add_argument("--checkpoint", required=False, help=checkpoint_help)
    test_parser.add_argument("--config", help=config_help)
    test_parser.add_argument("--dataset", help=dataset_help)


    args = parser.parse_args()

    if not args.config:
        args.config = f"config/{args.model}.yaml"

    cfg = load_config(args.config)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"On device {device}")
    set_seed(cfg["seed"])

    classification_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Grayscale(num_output_channels=3),  # Convert 1 channel → 3 channels
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    diffusion_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.Grayscale(num_output_channels=1),  # greyscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # debug
    trainloader, valloader, testloader = get_data_loaders(
        args.dataset or "Mirabest",
        transform = diffusion_transform,
        batch_size=cfg['data']['batch_size']
    )

    unique_labels = set()

    for _, labels in trainloader:
        # Convert tensor labels to a list of Python integers and add to set
        unique_labels.update(labels.tolist())

    # Sort them for clarity
    sorted_labels = sorted(list(unique_labels))

    print(f"Total unique classes found: {len(sorted_labels)}")
    print(f"Label IDs: {sorted_labels}")

    # Ensure the results/model_type directory exists for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_directory = f'results/{args.model}/run_{timestamp}'
    os.makedirs(result_directory, exist_ok=True)

    if args.command == "optimize":
        optimize_parameters(args.model, cfg, result_directory)
    elif args.command == "train":
        model = train_model(args.model, cfg, trainloader, valloader, testloader, device, result_directory, resume=args.resume, checkpoint=args.checkpoint)
        test_model(model_type=args.model, model=model, config=cfg, testloader=testloader, device=device, result_directory=result_directory)
    elif args.command == "test":
        # TODO: need to get the trained model
        test_model(model_type=args.model, config=cfg, testloader=testloader, device=device, result_directory=result_directory)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()