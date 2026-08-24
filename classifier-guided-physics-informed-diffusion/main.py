import torch
import numpy as np
import random
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

# Setting a global seed for reproducibility. torchvision's Random* transforms
# (RandomRotation, RandomHorizontalFlip, RandomVerticalFlip) draw from Python's
# `random` module, not torch's RNG, so it must be seeded too. cudnn is forced
# deterministic since its default autotuned algorithm selection is otherwise a
# GPU-side source of run-to-run variance even with a fixed seed.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_result_directory(model, tag, seed=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = tag if tag else "untagged"
    jobid = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or "local"
    seed_part = f"_seed{seed}" if seed is not None else ""
    return f'results/{model}/{timestamp}_{tag_part}_{jobid}{seed_part}'

def main():

    clear_gpu_memory()

    parser = argparse.ArgumentParser(description="Run experiments modularly.")
    subparsers = parser.add_subparsers(dest="command")

    model_help = "[classification | robust_classification | diffusion | pid | classifier_guided_diffusion | robust_classifier_guided_diffusion | edm_baseline] The model type to perform the current action on"
    config_help = "[string] Path to .yaml file to use for config [default: config/<model>.yaml]"
    resume_help = "[True | False] Whether to resume training from last saved epoch"
    checkpoint_help = "[True | False] Whether to save the training checkpoints"
    tag_help = "[string] Tag for this model's checkpoint dir (checkpoints/<model>/<tag>) so it isn't overwritten by other runs"

    # --- Optimize command ---
    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--model", required=True, help=model_help)
    optimize_parser.add_argument("--checkpoint", required=False)
    optimize_parser.add_argument("--config", required=False)

    # --- Train command ---
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True, help=model_help)
    train_parser.add_argument("--config", help=config_help)
    train_parser.add_argument("--resume", help=resume_help)
    train_parser.add_argument("--checkpoint", help=checkpoint_help)
    train_parser.add_argument("--runs", type=int, default=1, help="[int] Number of independent runs (each uses a different seed)")
    train_parser.add_argument("--seed", type=int, help="[int] Base seed for run 0; run i uses seed+i (overrides config seed)")
    train_parser.add_argument("--tag", help=tag_help)

    # --- Test command ---
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model", required=True, help=model_help)
    test_parser.add_argument("--checkpoint", required=False, help=checkpoint_help)
    test_parser.add_argument("--config", help=config_help)
    test_parser.add_argument("--tag", help=tag_help)


    args = parser.parse_args()

    if not args.config:
        args.config = f"config/{args.model}.yaml"

    cfg = load_config(args.config)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"On device {device}")
    set_seed(cfg["seed"])

    # ResNet18 (pretrained, ImageNet-normalized 3-channel) transform for
    # `classification` -- scored ~90% on MiraBest. Kept here in case we revert
    # from the from-scratch SimpleCNN back to ResNet18 transfer learning.
    # train_transform = transforms.Compose([
    #     transforms.Resize(213),        # upscale so corners stay filled after rotation
    #     transforms.RandomRotation(180),  # full 360° — orientation is arbitrary for radio galaxies
    #     transforms.CenterCrop(150),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # eval_transform = transforms.Compose([
    #     transforms.Resize(150),
    #     transforms.CenterCrop(150),
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    if args.model == 'classification':
        # Radio galaxy orientation is arbitrary either way, so vertical flip
        # is as valid as horizontal -- free extra augmentation for a ~1.4k
        # image training set.
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(213),
            transforms.RandomRotation(180),
            transforms.CenterCrop(150),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # Upscale to ceil(150 * sqrt(2)) = 213 so that a 150x150 centre crop
            # contains only real image content after any rotation angle.
            transforms.Resize(213),
            transforms.RandomRotation(180),
            transforms.CenterCrop(150),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    result = get_data_loaders(
        cfg['data']['dataset'],
        transform=train_transform,
        eval_transform=eval_transform,
        batch_size=cfg['data']['batch_size']
    )
    # mirabest_fits returns a 4th value (the dataset object) for FITS inverse scaling
    if len(result) == 4:
        trainloader, valloader, testloader, fits_dataset = result
    else:
        trainloader, valloader, testloader = result
        fits_dataset = None

    total_images = len(trainloader.dataset)

    total_batches = len(trainloader)

    batch_size = trainloader.batch_size
    unique_labels = set()

    for _, labels in trainloader:
        # Convert tensor labels to a list of Python integers and add to set
        unique_labels.update(labels.tolist())

    # Sort them for clarity
    sorted_labels = sorted(list(unique_labels))

    cfg['data']['num_classes'] = len(sorted_labels)

    print(f"Total unique classes found: {len(sorted_labels)}")
    print(f"Label IDs: {sorted_labels}")
    print(f"Total images in dataset: {total_images + len(valloader.dataset) + len(testloader.dataset)}")
    print(f"Total batches: {total_batches} (at batch size {batch_size})")

    if args.command == "optimize":
        result_directory = make_result_directory(args.model, getattr(args, "tag", None))
        os.makedirs(result_directory, exist_ok=True)
        optimize_parameters(args.model, cfg, trainloader, valloader, device, result_directory,
                            dataset=fits_dataset)
    elif args.command == "train":
        num_runs = args.runs if args.runs is not None else 1
        base_seed = args.seed if args.seed is not None else cfg["seed"]

        for run_idx in range(num_runs):
            seed = base_seed + run_idx
            set_seed(seed)

            result_directory = make_result_directory(args.model, args.tag, seed=seed if num_runs > 1 else None)
            os.makedirs(result_directory, exist_ok=True)

            # With multiple runs in one invocation, each seed needs its own checkpoint
            # dir or later runs would silently overwrite earlier ones' weights.
            run_tag = f"{args.tag}_seed{seed}" if (num_runs > 1 and args.tag) else args.tag

            print(f"\n{'='*60}")
            print(f"Run {run_idx + 1}/{num_runs}  |  seed={seed}  |  tag={run_tag}  |  results -> {result_directory}")
            print(f"{'='*60}\n")

            model = train_model(args.model, cfg, trainloader, valloader, testloader, device, result_directory, resume=args.resume, checkpoint=args.checkpoint, dataset=fits_dataset, tag=run_tag)
            test_model(model_type=args.model, model=model, config=cfg, testloader=testloader, device=device, result_directory=result_directory, tag=run_tag)
    elif args.command == "test":
        result_directory = make_result_directory(args.model, args.tag)
        os.makedirs(result_directory, exist_ok=True)
        # TODO: need to get the trained model
        test_model(model_type=args.model, config=cfg, testloader=testloader, device=device, result_directory=result_directory, tag=args.tag)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()