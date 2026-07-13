"""Pull a random test-set image, classify it with the trained classification
checkpoint, and save the image (with true/predicted labels) to a PNG.

Manual sanity check for the reported test accuracy — lets you eyeball whether
individual predictions actually line up with the image content.

Usage:
    python scripts/sanity_check_classifier.py --config config/classification.yaml
    python scripts/sanity_check_classifier.py --config config/classification.yaml --dataset mirabest --num-samples 5
"""
import argparse
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint
from src.utils.data import get_data_loaders
from src.models.simple_cnn import SimpleCNN

CHECKPOINT_DIR = 'checkpoints'
CLASS_NAMES = ['FR-I', 'FR-II']


def build_model(num_classes, device, tag=None):
    model = SimpleCNN(num_classes=num_classes)
    ckpt_dir = f'{CHECKPOINT_DIR}/classification' + (f'/{tag}' if tag else '')
    checkpoint = load_checkpoint(ckpt_dir, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def unnormalize(tensor):
    return (tensor.cpu() * 0.5 + 0.5).clamp(0, 1)


def classify_random_sample(test_dataset, model, device, output_dir, sample_idx):
    idx = random.randrange(len(test_dataset))
    image, true_label = test_dataset[idx]

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        pred_label = int(torch.argmax(probs).item())
        confidence = float(probs[pred_label].item())

    display_img = unnormalize(image)[0]

    correct = pred_label == true_label
    title = (
        f"True: {CLASS_NAMES[true_label]}  |  Pred: {CLASS_NAMES[pred_label]} "
        f"({confidence:.1%})  |  {'CORRECT' if correct else 'WRONG'}"
    )

    plt.figure(figsize=(4, 4))
    plt.imshow(display_img, cmap='gray')
    plt.title(title, fontsize=9, color='green' if correct else 'red')
    plt.axis('off')

    fname = f"sample_{sample_idx:02d}_idx{idx}_true-{CLASS_NAMES[true_label]}_pred-{CLASS_NAMES[pred_label]}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"[{sample_idx}] test_idx={idx}  true={CLASS_NAMES[true_label]}  "
          f"pred={CLASS_NAMES[pred_label]}  confidence={confidence:.1%}  "
          f"{'CORRECT' if correct else 'WRONG'}  -> {out_path}")

    return correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/classification.yaml")
    parser.add_argument("--dataset", default=None, help="Override cfg['data']['dataset'] (e.g. mirabest, crumb)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of random test images to classify")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for picking images (default: nondeterministic)")
    parser.add_argument("--output", default="results/sanity_check")
    parser.add_argument("--tag", default=None,
                         help="Load checkpoints/classification/<tag> instead of the untagged "
                              "checkpoints/classification default.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cfg = load_config(args.config)
    if args.dataset is not None:
        cfg['data']['dataset'] = args.dataset

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"On device {device}")
    print(f"Dataset: {cfg['data']['dataset']}")

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    result = get_data_loaders(
        cfg['data']['dataset'],
        transform=eval_transform,
        eval_transform=eval_transform,
        batch_size=1,
    )
    testloader = result[2]
    test_dataset = testloader.dataset

    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes, device, tag=args.tag)

    num_correct = 0
    for i in range(args.num_samples):
        num_correct += classify_random_sample(test_dataset, model, device, args.output, i)

    print(f"\n{num_correct}/{args.num_samples} correct on this random sample.")


if __name__ == "__main__":
    main()
