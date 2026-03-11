# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project trains and evaluates three model types on the **MiraBest** radio galaxy dataset (binary classification: FR-I vs FR-II morphology):
1. **classification** — fine-tuned ResNet50 classifier
2. **robust_classification** — adversarially robust classifier (`TimeDependentResNet`) trained with PGD attacks + curriculum diffusion noise
3. **diffusion** — class-conditional image generator (`UNet2DConditionModel` + DDPM) using classifier-free guidance

## Commands

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
python main.py train --model classification
python main.py train --model robust_classification
python main.py train --model diffusion
# Resume from checkpoint:
python main.py train --model diffusion --resume True --checkpoint True
# Custom config:
python main.py train --model diffusion --config config/diffusion.yaml
```

### Testing
```bash
python main.py test --model classification
python main.py test --model diffusion
```

### Hyperparameter Optimization (Ray Tune)
```bash
python main.py optimize --model classifier --config config/classification.yaml
```

### HPC (CHPC Lengau cluster)
```bash
make ssh          # SSH into Lengau
make gpu_job      # Request GPU node (1 GPU, 9 CPUs, 4h walltime)
make cpu_job      # Request CPU node
```

## Architecture

### Entry Point
`main.py` — parses `train | test | optimize` subcommands with `--model`, `--config`, `--resume`, `--checkpoint`, `--dataset` flags. Config defaults to `config/<model>.yaml`. Results are written to `results/<model>/run_<timestamp>/`.

### Pipelines (`src/pipelines/`)
- `train_pipeline.py` — dispatches to `train_classification`, `train_robust_classification`, or `train_diffusion`. Handles checkpointing, loss plotting, and sample generation every 2 epochs (diffusion only).
- `test_pipeline.py` — loads checkpoint from `checkpoints/<model_type>/state.pt` and runs evaluation/generation.
- `optimize_parameters_pipeline.py` — Ray Tune Bayesian search; results stored in `tuning_results/`.

### Models (`src/models/`)
- `classification_model.py` — simple CNN (legacy, not used in main training flow; actual training uses ResNet50 directly)
- `time_dependent_resnet.py` — `TimeDependentResNet`: ResNet50 backbone with sinusoidal timestep embedding added to features, enabling the classifier to condition on diffusion noise level

### Diffusion Model (defined inline in `train_pipeline.py`)
- `UNet2DConditionModel` (HuggingFace diffusers) with cross-attention conditioning on class embeddings
- `DDPMScheduler` with 1000 training timesteps
- Class embedding: `nn.Embedding(num_classes + 1, 256)` where index `num_classes` is the null/unconditional class
- 15% label dropout during training enables classifier-free guidance (CFG, scale=7.5) at inference

### Data (`src/utils/data.py`, `src/datasets/mirabest/`)
- `MiraBest` dataset auto-downloads to `./batches/` (CIFAR-style batches)
- 80/20 train/val split from training set; separate test split
- Classification transform: 224×224 RGB, ImageNet normalization
- Diffusion transform: 150×150 grayscale, pad→rotate→crop, normalize `mean=0.5 std=0.5`

### Checkpoints (`src/utils/checkpoint.py`)
- Saved to `checkpoints/<model_type>/state.pt`
- Diffusion checkpoints include: `model_state_dict`, `optimizer_state_dict`, `class_emb_state_dict`, `epoch`, `loss_history`, `val_loss_history`, `rng_state`

### Robust Classifier Training (`src/utils/augmentation.py`)
- `pgd_attack_early_stop`: PGD adversarial attack on `(x, t)` — stops early if all samples are fooled
- `get_noisy_image`: adds diffusion noise at timestep `t` using DDPM forward process
- `get_max_timestep`: curriculum schedule that linearly increases max noise level over training epochs

## Key Implementation Notes

- The `main.py` currently hardcodes `diffusion_transform` for data loading regardless of `--model`; classification training uses the diffusion transform. This may be intentional for consistency or a known issue.
- Checkpoint save logic in `train_pipeline.py` uses `not checkpoint == None or not resume == None` (note: `or` not `and`), meaning checkpointing happens whenever either flag is set.
- `src/models/classification_model.py` (`ClassificationModel`) is not used by the main training pipeline; actual classification uses `resnet50` directly in `train_pipeline.py`.
- The `evaluators/` directory is empty.
