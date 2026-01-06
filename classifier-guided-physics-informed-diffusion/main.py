import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
from src.utils.config import load_config
from src.pipelines.optimize_parameters_pipeline import optimize_parameters
from src.pipelines.train_pipeline import train_model
from src.pipelines.test_pipeline import test_model
from src.pipelines.test_pipeline import evaluate_model_performance
from src.utils.data import get_data_loaders

# Setting a global seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():

    parser = argparse.ArgumentParser(description="Run experiments modularly.")
    subparsers = parser.add_subparsers(dest="command")

    # --- Optimize command ---
    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--model", required=True)
    optimize_parser.add_argument("--checkpoint", required=False)
    optimize_parser.add_argument("--config", required=True)
    optimize_parser.add_argument("--dataset", help="Name of dataset to use")

    # --- Train command ---
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True, help="Model type: classifier, robust_classifier, diffusion, integrated_diffusion")
    train_parser.add_argument("--config", required=True, help="Path to config file")
    train_parser.add_argument("--resume", help="Optional checkpoint path to resume")
    train_parser.add_argument("--dataset", help="Name of dataset to use")

    # --- Test command ---
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model", required=True)
    test_parser.add_argument("--checkpoint", required=False)
    test_parser.add_argument("--config", required=True)
    test_parser.add_argument("--dataset", help="Name of dataset to use")


    args = parser.parse_args()
    cfg = load_config(args.config)

    # Set device to GPU if available
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"On device {device}")
    set_seed(cfg["seed"])


    trainloader, valloader, testloader = get_data_loaders(
        args.dataset or "Mirabest",
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Grayscale(num_output_channels=3),  # Convert 1 channel → 3 channels
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        batch_size=cfg['data']['batch_size']
    )

    if args.command == "optimize":
        optimize_parameters(args.model, cfg)
    elif args.command == "train":
        model = train_model(args.model, cfg, trainloader, valloader, device, resume=args.resume)
        evaluate_model_performance(args.model, model, cfg, testloader, device)
    elif args.command == "test":
        test_model(args.model, cfg, testloader, device, args.checkpoint)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()