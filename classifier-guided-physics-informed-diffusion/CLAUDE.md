# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project trains and evaluates three model types on radio galaxy datasets (binary classification: FR-I vs FR-II morphology):
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
`main.py` — parses `train | test | optimize` subcommands with `--model`, `--config`, `--resume`, `--checkpoint` flags. Config defaults to `config/<model>.yaml`. Results are written to `results/<model>/run_<timestamp>/`. Always loads data with `diffusion_transform` regardless of model type.

### Pipelines (`src/pipelines/`)
- `train_pipeline.py` — dispatches to `train_classification`, `train_robust_classification`, or `train_diffusion`. Handles checkpointing, loss plotting, and sample generation every 5 epochs (diffusion only).
- `test_pipeline.py` — loads checkpoint from `checkpoints/<model_type>/state.pt` and runs evaluation/generation. **Note:** The diffusion test path uses a hardcoded architecture (sample_size=32, `cross_attention_dim=128`) that does not match the training architecture (sample_size=150, `cross_attention_dim=256`); this path will fail to load a real checkpoint.
- `optimize_parameters_pipeline.py` — Ray Tune Bayesian search; results stored in `tuning_results/`.

### Models (`src/models/`)
- `classification_model.py` — legacy CNN, not used in main pipeline; actual training uses `resnet50` directly in `train_pipeline.py`
- `time_dependent_resnet.py` — `TimeDependentResNet`: ResNet50 backbone with sinusoidal timestep embedding (dim=128) projected and added to extracted features, enabling noise-level conditioning

### Diffusion Model (defined inline in `train_pipeline.py`)
- `UNet2DConditionModel` (HuggingFace diffusers): input 150×150 grayscale, block channels (64,128,256,512), CrossAttn at the inner two blocks
- `DDPMScheduler` with 1000 training timesteps; inference uses 50 steps with CFG (guidance_scale=7.5)
- Class embedding: `nn.Embedding(num_classes + 1, 256)` — index `num_classes` is the null/unconditional class
- 15% label dropout during training enables classifier-free guidance at inference
- `sample_from_model_zeros` / `sample_from_model_ones` generate class-specific images with CFG; `sample_from_model` generates with random labels (no CFG)

### Datasets (`src/utils/data.py`, `src/datasets/`)
Three datasets are supported via the `dataset` key in config:

| `dataset` value | Source | Notes |
|---|---|---|
| `mirabest` | CIFAR-style batches, auto-downloads to `./batches/` | 80/20 train/val split |
| `mirabest_fits` | FITS files in `src/datasets/mirabest/fits/` | Returns 4-tuple including dataset object; stats cached to `fits_stats.json`; label from filename prefix (1xx→FR-I, 2xx→FR-II, 3xx excluded) |
| `crumb` | CIFAR-style batches, auto-downloads to `./batches/` | 80/20 train/val split |

`MiraBestFITS` normalises using symmetric log-SNR (invertible via `denormalise()`), enabling generated images to be written back as FITS files.

### Checkpoints (`src/utils/checkpoint.py`)
- Saved to `checkpoints/<model_type>/state.pt`
- Diffusion checkpoints include: `model_state_dict`, `optimizer_state_dict`, `class_emb_state_dict`, `epoch`, `loss_history`, `val_loss_history`, `epochs_range`, `fid_history`, `rng_state`, `cuda_rng_state`

### Robust Classifier Training (`src/utils/augmentation.py`)
- `pgd_attack_early_stop`: PGD on `(x_t, t)` — stops early when all samples in the batch are misclassified
- `get_noisy_image`: DDPM forward process — adds noise at timestep `t` using precomputed `alphas_cumprod`
- `get_max_timestep`: curriculum schedule that linearly increases max noise level over training epochs

## Known Issues

- `train_pipeline.py:61` has a typo: `model.loa. _state_dict(...)` — this will crash when resuming classification training.
- Checkpoint save condition differs between models: `train_classification` and `train_robust_classification` use `and` (both `--checkpoint` and `--resume` must be set), while `train_diffusion` uses `or` (either flag triggers saving).
- `train_diffusion` loads `optimizer_state_dict` twice when resuming (lines ~194 and ~204); the second load (after the if-block) is the one that takes effect.
- `test_pipeline.py` diffusion path hardcodes a mismatched architecture — it cannot load a checkpoint produced by `train_diffusion`.
- `evaluators/` directory is empty.
- `main.py` always uses `diffusion_transform` for data loading regardless of `--model`; classification models receive 150×150 grayscale instead of 224×224 RGB.

## Dissertation Change Log

After every edit, append a brief entry here so the user can track all changes for their dissertation write-up.

| Date | File | Change |
|------|------|--------|
| 2026-06-29 | `config/pid.yaml` | Set `lambda_neg` from `0.1` → `0.0`; non-negativity loss suppressed because it caused generated images to be artificially bright. `epochs` also changed from 300 → 200 (external edit). |
