"""Classifier-based generation quality metric.

Loads a classifier checkpoint that was trained ONLY for evaluation (never used
as a guidance signal during training -- using the same weights that guided
classifier_guided_diffusion's generation to also score it would be circular),
generates samples from each of diffusion / classifier_guided_diffusion /
edm_baseline's checkpoints across multiple seeds, classifies them, and reports:

  - class_accuracy: fraction of generated class-c images the classifier
    predicts as class c (does the model actually generate what it was asked to?)
  - mean_confidence: mean softmax probability assigned to the intended class
  - per-class accuracy (FR-I / FR-II), so a model that collapses onto one class
    shows up as ~1.0 / ~0.0 rather than hiding behind a 0.5 overall accuracy
  - FID / KID of the same samples vs the real crumb_fits validation split (the
    split training-time FID uses), so class fidelity and realism are measured
    on identical images

--guidance-scale overrides every model's CFG scale (config training.guidance_scale:
EDM 3.0, DDPM/CGD 7.5) so models can be compared at matched guidance.
--classifier-scale overrides CGD's classifier-gradient strength (config
training.classifier_scale, 1.0); 0 disables the gradient. Other models ignore it.
--guidance-classifier / --guidance-classifier-type override which frozen classifier
guides CGD (config model.classifier_checkpoint / model.classifier_type).
--guided-diffusion-tags applies that same inference-time guidance to DDPM (`diffusion`)
checkpoints, separating the guidance method from which checkpoints it is applied to.

Samples are generated in micro-batches of --batch-size per class (CFG doubles each
forward pass, so large single batches OOM on an 11.9GB GPU) until --num-samples
per class is reached.

Aggregated as mean +/- std across seeds per model type.

Usage:
    python scripts/evaluate_generated_samples.py \
        --diffusion-tags exp_seed42 exp_seed43 exp_seed44 exp_seed45 exp_seed46 \
        --cgd-tags exp_seed42 exp_seed43 exp_seed44 exp_seed45 exp_seed46 \
        --edm-tags exp_seed42 exp_seed43 exp_seed44 exp_seed45 exp_seed46 \
        --classifier-tag eval \
        --num-samples 128 --batch-size 16 \
        --guidance-scale 3.0
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
from src.utils.metrics import generate_class_samples, generate_class_samples_guided, compute_fid_kid
from src.utils.data import get_data_loaders

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
    """Returns (accuracy, mean_confidence, per_class_accuracy) for a (gen_0, gen_1) pair."""
    images = torch.cat([gen_0, gen_1], dim=0).to(device)
    labels = torch.cat([
        torch.zeros(gen_0.shape[0], dtype=torch.long),
        torch.ones(gen_1.shape[0], dtype=torch.long),
    ]).to(device)

    with torch.no_grad():
        logits = classifier(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    correct = (preds == labels).float()
    accuracy = correct.mean().item()
    confidence = probs[torch.arange(labels.shape[0]), labels].mean().item()
    per_class = [correct[labels == c].mean().item() for c in range(len(CLASS_NAMES))]
    return accuracy, confidence, per_class


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


def _generate_classifier_guided(checkpoint_subdir, config, tag, num_classes, num_samples, device, shape):
    from src.pipelines.train_pipeline import _load_guidance_classifier

    unet, scheduler, class_emb, _ = build_diffusion_components(config, {}, device)
    ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/{checkpoint_subdir}/{tag}', device)
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


def generate_cgd(config, tag, num_classes, num_samples, device, shape):
    return _generate_classifier_guided('classifier_guided_diffusion', config, tag, num_classes, num_samples, device, shape)


def generate_diffusion_classifier_guided(config, tag, num_classes, num_samples, device, shape):
    """DDPM checkpoints sampled with CGD's inference-time classifier guidance (same UNet architecture)."""
    return _generate_classifier_guided('diffusion', config, tag, num_classes, num_samples, device, shape)


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
    'diffusion_classifier_guided': ('config/diffusion.yaml', generate_diffusion_classifier_guided),
    'edm_baseline': ('config/edm_baseline.yaml', generate_edm),
}


def generate_in_batches(generate_fn, config, tag, num_classes, num_samples, batch_size, device, shape):
    gen_0, gen_1 = [], []
    remaining = num_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        g0, g1 = generate_fn(config, tag, num_classes, n, device, shape)
        gen_0.append(g0.detach().cpu())
        gen_1.append(g1.detach().cpu())
        remaining -= n
    return torch.cat(gen_0), torch.cat(gen_1)


def evaluate_model(model_type, tags, classifier, valloader, num_classes, num_samples, batch_size,
                   guidance_scale, classifier_scale, guidance_classifier, guidance_classifier_type, device, shape):
    config_path, generate_fn = MODEL_SPECS[model_type]
    config = load_config(config_path)
    config['data']['num_classes'] = num_classes
    if guidance_scale is not None:
        config['training']['guidance_scale'] = guidance_scale
    if classifier_scale is not None:
        config['training']['classifier_scale'] = classifier_scale
    if guidance_classifier is not None:
        config['model']['classifier_checkpoint'] = guidance_classifier
    if guidance_classifier_type is not None:
        config['model']['classifier_type'] = guidance_classifier_type

    accuracies, confidences, per_class_accs, fids, kids = [], [], [], [], []
    for tag in tags:
        gen_0, gen_1 = generate_in_batches(
            generate_fn, config, tag, num_classes, num_samples, batch_size, device, shape,
        )
        acc, conf, per_class = classify_generated(classifier, gen_0, gen_1, device)
        fid, kid = compute_fid_kid(gen_0, gen_1, valloader, device)
        per_class_str = '  '.join(f"{n}={a:.4f}" for n, a in zip(CLASS_NAMES, per_class))
        print(f"  [{model_type}/{tag}]  class_accuracy={acc:.4f}  mean_confidence={conf:.4f}  ({per_class_str})"
              f"  fid={fid:.2f}  kid={kid:.4f}")
        accuracies.append(acc)
        confidences.append(conf)
        per_class_accs.append(per_class)
        fids.append(fid)
        kids.append(kid)

    return {
        'tags': tags,
        'guidance_scale': float(config['training']['guidance_scale']),
        'classifier_scale': config['training'].get('classifier_scale'),
        'guidance_classifier': config['model'].get('classifier_checkpoint'),
        'fid_per_seed': fids,
        'kid_per_seed': kids,
        'class_accuracy_per_seed': accuracies,
        'mean_confidence_per_seed': confidences,
        'per_class_accuracy_per_seed': {
            name: [pc[i] for pc in per_class_accs] for i, name in enumerate(CLASS_NAMES)
        },
        'class_accuracy_mean': float(np.mean(accuracies)),
        'class_accuracy_std': float(np.std(accuracies)),
        'mean_confidence_mean': float(np.mean(confidences)),
        'mean_confidence_std': float(np.std(confidences)),
        'fid_mean': float(np.mean(fids)),
        'fid_std': float(np.std(fids)),
        'kid_mean': float(np.mean(kids)),
        'kid_std': float(np.std(kids)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--diffusion-tags', nargs='+', default=[])
    parser.add_argument('--cgd-tags', nargs='+', default=[])
    parser.add_argument('--edm-tags', nargs='+', default=[])
    parser.add_argument('--guided-diffusion-tags', nargs='+', default=[],
                        help="DDPM (diffusion) checkpoints sampled with classifier guidance; "
                             "needs --guidance-classifier / --guidance-classifier-type.")
    parser.add_argument('--classifier-tag', default='eval',
                         help="checkpoints/classification/<tag> -- must be trained separately "
                              "from any classifier used to guide classifier_guided_diffusion.")
    parser.add_argument('--num-samples', type=int, default=128, help="Generated samples per class per seed.")
    parser.add_argument('--batch-size', type=int, default=16, help="Samples per class per generation call (OOM guard).")
    parser.add_argument('--seed', type=int, default=0, help="Sampling noise seed, for reproducible scores.")
    parser.add_argument('--guidance-scale', type=float, default=None,
                        help="Override every model's CFG scale (default: each model's config value).")
    parser.add_argument('--classifier-scale', type=float, default=None,
                        help="Override CGD's classifier-gradient strength (default: config value, 1.0). 0 disables it.")
    parser.add_argument('--guidance-classifier', default=None,
                        help="Override CGD's guidance classifier checkpoint dir (default: config model.classifier_checkpoint).")
    parser.add_argument('--guidance-classifier-type', default=None, choices=['classification', 'robust_classification'],
                        help="Architecture of --guidance-classifier (default: config model.classifier_type).")
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--input-size', type=int, default=150)
    parser.add_argument('--output', default='results/generation_classifier_eval')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"On device {device}")

    classifier = load_eval_classifier(args.num_classes, device, args.classifier_tag)
    shape = (1, args.input_size, args.input_size)
    # Real reference set for FID/KID: the crumb_fits validation split training-time FID uses.
    _, valloader, _, _ = get_data_loaders('crumb_fits', None, batch_size=32)

    results = {}
    for model_type, tags in [
        ('diffusion', args.diffusion_tags),
        ('classifier_guided_diffusion', args.cgd_tags),
        ('edm_baseline', args.edm_tags),
        ('diffusion_classifier_guided', args.guided_diffusion_tags),
    ]:
        if not tags:
            continue
        print(f"\nEvaluating {model_type} ({len(tags)} seeds)...")
        results[model_type] = evaluate_model(
            model_type, tags, classifier, valloader, args.num_classes, args.num_samples, args.batch_size,
            args.guidance_scale, args.classifier_scale, args.guidance_classifier, args.guidance_classifier_type,
            device, shape,
        )
        results[model_type]['classifier_tag'] = args.classifier_tag

    print("\n=== Summary (class_accuracy: predicted-as-intended-class rate) ===")
    for model_type, r in results.items():
        print(f"{model_type:30s}  acc={r['class_accuracy_mean']:.4f} +/- {r['class_accuracy_std']:.4f}"
              f"   conf={r['mean_confidence_mean']:.4f} +/- {r['mean_confidence_std']:.4f}"
              f"   fid={r['fid_mean']:.2f} +/- {r['fid_std']:.2f}   kid={r['kid_mean']:.4f} +/- {r['kid_std']:.4f}"
              f"   (cfg={r['guidance_scale']})")

    out_path = os.path.join(args.output, 'classifier_eval_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
