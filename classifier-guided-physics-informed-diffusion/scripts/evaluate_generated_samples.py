"""Classifier-based generation quality metric.

Loads a classifier checkpoint that was trained ONLY for evaluation (never used
as a guidance signal during training -- using the same weights that guided
classifier_guided_diffusion's generation to also score it would be circular),
generates samples from each of diffusion / classifier_guided_diffusion /
edm_baseline's checkpoints across multiple seeds, classifies them, and reports:

  - class_accuracy: fraction of generated class-c images the classifier
    predicts as class c (does the model actually generate what it was asked to?)
  - mean_confidence: mean softmax probability assigned to the intended class

Aggregated as mean +/- std across seeds per model type.

Usage:
    python scripts/evaluate_generated_samples.py \
        --diffusion-tags exp_seed42 exp_seed43 exp_seed44 exp_seed45 exp_seed46 \
        --cgd-tags exp_seed42 exp_seed43 exp_seed44 exp_seed45 exp_seed46 \
        --edm-tags exp_seed42 exp_seed43 exp_seed44 exp_seed45 exp_seed46 \
        --classifier-tag eval \
        --num-samples 64
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint
from src.models.simple_cnn import SimpleCNN
from src.models.diffusion import build_diffusion_components
from src.models.edm import build_edm_components, generate_class_samples_edm
from src.utils.metrics import generate_class_samples, generate_class_samples_guided

CHECKPOINT_DIR = 'checkpoints'
CLASS_NAMES = ['FR-I', 'FR-II']


def load_eval_classifier(num_classes, device, tag):
    model = SimpleCNN(num_classes=num_classes)
    ckpt_dir = f'{CHECKPOINT_DIR}/classification' + (f'/{tag}' if tag else '')
    checkpoint = load_checkpoint(ckpt_dir, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model


def classify_generated(classifier, gen_0, gen_1, device):
    """Returns (accuracy, mean_confidence) for a single (gen_0, gen_1) pair."""
    images = torch.cat([gen_0, gen_1], dim=0).to(device)
    labels = torch.cat([
        torch.zeros(gen_0.shape[0], dtype=torch.long),
        torch.ones(gen_1.shape[0], dtype=torch.long),
    ]).to(device)

    with torch.no_grad():
        logits = classifier(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    accuracy = (preds == labels).float().mean().item()
    confidence = probs[torch.arange(labels.shape[0]), labels].mean().item()
    return accuracy, confidence


def generate_diffusion(config, tag, num_classes, num_samples, device, shape):
    unet, scheduler, class_emb, _ = build_diffusion_components(config, {}, device)
    ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/diffusion/{tag}', device)
    unet.load_state_dict(ckpt['model_state_dict'])
    class_emb.load_state_dict(ckpt['class_emb_state_dict'])
    unet.to(device).eval()

    guidance_scale = float(config['training'].get('guidance_scale', 7.5))
    with torch.no_grad():
        return generate_class_samples(
            unet, scheduler, class_emb, num_classes, num_samples, device,
            shape=shape, guidance_scale=guidance_scale,
        )


def generate_cgd(config, tag, num_classes, num_samples, device, shape):
    from src.pipelines.train_pipeline import _load_guidance_classifier

    unet, scheduler, class_emb, _ = build_diffusion_components(config, {}, device)
    ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/classifier_guided_diffusion/{tag}', device)
    unet.load_state_dict(ckpt['model_state_dict'])
    class_emb.load_state_dict(ckpt['class_emb_state_dict'])
    unet.to(device).eval()

    guidance_scale = float(config['training'].get('guidance_scale', 7.5))
    classifier_scale = float(config['training'].get('classifier_scale', 1.0))
    guidance_classifier, time_aware = _load_guidance_classifier(config, device)

    with torch.no_grad():
        return generate_class_samples_guided(
            unet, scheduler, class_emb, num_classes, num_samples, device,
            classifier=guidance_classifier, classifier_time_aware=time_aware,
            shape=shape, guidance_scale=guidance_scale, classifier_scale=classifier_scale,
        )


def generate_edm(config, tag, num_classes, num_samples, device, shape):
    unet, ema, _ = build_edm_components(config, device)
    ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/edm_baseline/{tag}', device)
    unet.load_state_dict(ckpt['model_state_dict'])
    ema.load_state_dict(ckpt['ema_state_dict'])

    guidance_scale = float(config['training'].get('guidance_scale', 3.0))
    num_sampling_steps = int(config['training'].get('num_sampling_steps', 25))
    return generate_class_samples_edm(
        ema.shadow, num_classes, num_samples, device,
        shape=shape, guidance_scale=guidance_scale, num_steps=num_sampling_steps,
    )


MODEL_SPECS = {
    'diffusion': ('config/diffusion.yaml', generate_diffusion),
    'classifier_guided_diffusion': ('config/classifier_guided_diffusion.yaml', generate_cgd),
    'edm_baseline': ('config/edm_baseline.yaml', generate_edm),
}


def evaluate_model(model_type, tags, classifier, num_classes, num_samples, device, shape):
    config_path, generate_fn = MODEL_SPECS[model_type]
    config = load_config(config_path)

    accuracies, confidences = [], []
    for tag in tags:
        gen_0, gen_1 = generate_fn(config, tag, num_classes, num_samples, device, shape)
        acc, conf = classify_generated(classifier, gen_0, gen_1, device)
        print(f"  [{model_type}/{tag}]  class_accuracy={acc:.4f}  mean_confidence={conf:.4f}")
        accuracies.append(acc)
        confidences.append(conf)

    return {
        'tags': tags,
        'class_accuracy_per_seed': accuracies,
        'mean_confidence_per_seed': confidences,
        'class_accuracy_mean': float(np.mean(accuracies)),
        'class_accuracy_std': float(np.std(accuracies)),
        'mean_confidence_mean': float(np.mean(confidences)),
        'mean_confidence_std': float(np.std(confidences)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--diffusion-tags', nargs='+', default=[])
    parser.add_argument('--cgd-tags', nargs='+', default=[])
    parser.add_argument('--edm-tags', nargs='+', default=[])
    parser.add_argument('--classifier-tag', default='eval',
                         help="checkpoints/classification/<tag> -- must be trained separately "
                              "from any classifier used to guide classifier_guided_diffusion.")
    parser.add_argument('--num-samples', type=int, default=64, help="Generated samples per class per seed.")
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--input-size', type=int, default=150)
    parser.add_argument('--output', default='results/generation_classifier_eval')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"On device {device}")

    classifier = load_eval_classifier(args.num_classes, device, args.classifier_tag)
    shape = (1, args.input_size, args.input_size)

    results = {}
    for model_type, tags in [
        ('diffusion', args.diffusion_tags),
        ('classifier_guided_diffusion', args.cgd_tags),
        ('edm_baseline', args.edm_tags),
    ]:
        if not tags:
            continue
        print(f"\nEvaluating {model_type} ({len(tags)} seeds)...")
        results[model_type] = evaluate_model(
            model_type, tags, classifier, args.num_classes, args.num_samples, device, shape,
        )

    print("\n=== Summary (class_accuracy: predicted-as-intended-class rate) ===")
    for model_type, r in results.items():
        print(f"{model_type:30s}  acc={r['class_accuracy_mean']:.4f} +/- {r['class_accuracy_std']:.4f}"
              f"   conf={r['mean_confidence_mean']:.4f} +/- {r['mean_confidence_std']:.4f}")

    out_path = os.path.join(args.output, 'classifier_eval_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
