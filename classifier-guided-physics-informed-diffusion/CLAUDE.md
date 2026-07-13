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
# Tag a checkpoint so it doesn't get overwritten by later runs of the same model type:
python main.py train --model classification --checkpoint True --tag acc90
```

### Testing
```bash
python main.py test --model classification
python main.py test --model diffusion
# Test a specific tagged checkpoint:
python main.py test --model classification --tag acc90
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
`main.py` — parses `train | test | optimize` subcommands with `--model`, `--config`, `--resume`, `--checkpoint`, `--tag` flags. Config defaults to `config/<model>.yaml`. Results are written to `results/<model>/run_<timestamp>/`. Always loads data with `diffusion_transform` regardless of model type.

### Pipelines (`src/pipelines/`)
- `train_pipeline.py` — dispatches to `train_classification`, `train_robust_classification`, or `train_diffusion`. Handles checkpointing, loss plotting, and sample generation every 5 epochs (diffusion only). Pass `--tag <name>` to save/load a model's own checkpoint under `checkpoints/<model_type>/<tag>/` instead of the untagged default, so a good run isn't silently overwritten by a later one. `classifier_guided_diffusion`/`robust_classifier_guided_diffusion` load a separate frozen guidance classifier controlled by `model.classifier_type` (`classification` or `robust_classification`, default `robust_classification`) and `model.classifier_checkpoint` (defaults to `checkpoints/<classifier_type>`, override to point at a tagged checkpoint) via `_load_guidance_classifier`.
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
- Val and test loaders now use `eval_transform` (deterministic: no random rotation/flip), so val accuracy is a stable, reliable metric rather than a noisy estimate inflated by random augmentations.

## Dissertation Change Log

After every edit, append a brief entry here so the user can track all changes for their dissertation write-up.

| Date | File | Change |
|------|------|--------|
| 2026-06-29 | `config/pid.yaml` | Set `lambda_neg` from `0.1` → `0.0`; non-negativity loss suppressed because it caused generated images to be artificially bright. `epochs` also changed from 300 → 200 (external edit). |
| 2026-06-29 | `main.py`, `train_pipeline.py`, `test_pipeline.py` | Fixed classifier ~50% accuracy caused by normalization mismatch: classification model now uses `classification_transform` (ImageNet mean/std, 3-channel) instead of `diffusion_transform` ([-1,1] grayscale). Removed 1-channel conv1 surgery in train and test pipelines since input is now 3-channel matching pretrained ResNet50 expectations. |
| 2026-06-29 | `test_pipeline.py` | Added `metrics.json` save with `test_accuracy` at end of classification evaluation in `test_model`. |
| 2026-06-30 | `main.py` | Fixed `classification_transform`: replaced `RandomResizedCrop(224)` (default scale 0.08–1.0, could crop 42×42px patches destroying FR-I/FR-II morphology) with `Resize(224)`. Also moved `Grayscale` to first position and removed `saturation`/`hue` from `ColorJitter` and `GaussianBlur` (both no-ops on grayscale). Root cause of ~50% classification accuracy. |
| 2026-06-30 | `main.py`, `train_pipeline.py`, `test_pipeline.py` | Reverted classification model to 1-channel input: removed `classification_transform` entirely, restored conv1 surgery on ResNet50 (average RGB pretrained weights to 1-channel), all models now use `diffusion_transform`. The 2026-06-29 fix misdiagnosed the root cause — the real issue was `RandomResizedCrop` (fixed above), not the channel count. |
| 2026-07-01 | `main.py`, `src/utils/data.py` | Fixed inflated val accuracy: added `eval_transform` (Grayscale → Resize(150) → CenterCrop(150) → ToTensor → Normalize, no random ops) applied to val and test loaders, while training loader keeps augmentation. Also fixed train/val split to use a fixed seed (42) so the split is reproducible across runs. |
| 2026-07-01 | `src/pipelines/train_pipeline.py`, `config/classification.yaml`, `main.py` | Reverted `train_classification` to the Dec 2025 baseline: plain Adam, patience=10 early stopping, no scheduler, standard ResNet50 with no conv1 surgery. Classification now uses a 3-channel ImageNet-normalised transform (Resize(150), augmentation, Grayscale(3), Normalize ImageNet stats) matching pretrained ResNet50 expectations. Diffusion/robust models keep the 1-channel 150×150 pipeline. |
| 2026-07-01 | `src/models/simple_cnn.py`, `src/pipelines/train_pipeline.py`, `main.py`, `config/classification.yaml` | Replaced ResNet50 transfer learning with `SimpleCNN`: 4 conv blocks (32→64→128→256 channels, BatchNorm+ReLU+MaxPool), global avg pool, dropout(0.5), linear head. Trained from scratch on 1-channel 150×150 input with [-1,1] normalisation. Config: lr=0.001, weight_decay=0.0001. Reason: ResNet50 ImageNet transfer learning showed persistent ~50% test accuracy due to domain mismatch with radio galaxy morphology. |
| 2026-07-06 | `ZAMLA_talk.pptx` (slide 10, "Real vs Generated: Side by Side") | Replaced the "True (MiraBest)" FR-I and FR-II panels (previously black-background CIFAR-style MiraBest images) with real examples cropped from `mirabest_true_fri_grid.png` / `mirabest_true_frii_grid.png` — actual symmetric-log-SNR normalised FITS images from `mirabest_fits`, showing the true gray/noisy background the diffusion model is actually trained on. |
| 2026-07-09 | `scripts/test_classifier_on_mirabest_subset.py` (new), `scripts/slurm/eval/job_sanity_check_mirabest_subset.sh` (new) | Added a script that isolates the CRUMB images corresponding to MiraBest sources (matched by RA/Dec parsed from filenames, tolerance 0.01°, recovers 771/770 MiraBest sources within CRUMB's 2100 images) and evaluates the classifier checkpoint only on that subset. Investigating why the classifier scores ~40% on 100 random CRUMB samples — this isolates whether the drop is specific to CRUMB's non-MiraBest sources or affects MiraBest sources too. |
| 2026-07-09 | `main.py`, `src/pipelines/train_pipeline.py`, `src/pipelines/test_pipeline.py`, `scripts/sanity_check_classifier.py`, `scripts/evaluate_classifier_vs_both_labels.py`, `scripts/test_classifier_on_mirabest_subset.py`, `scripts/test_classifier_on_all_crumb_subsets.py`, `scripts/slurm/train/job_*.sh`, `scripts/slurm/eval/job_*.sh`, `config/classifier_guided_diffusion.yaml`, `config/robust_classifier_guided_diffusion.yaml` | Added `--tag` to `train`/`test` so a checkpoint saves/loads at `checkpoints/<model_type>/<tag>/` instead of the untagged default, preventing a good run (e.g. a 90%-accuracy classifier) from being silently overwritten by a later one; all classifier-loading scripts and slurm job scripts (`${TAG:+--tag $TAG}`) now accept it too. Also made the classifier used to guide `classifier_guided_diffusion`/`robust_classifier_guided_diffusion` training configurable via new `model.classifier_type` (`classification` or `robust_classification`, default `robust_classification` — matches prior hardcoded behavior) and `model.classifier_checkpoint` (defaults to `checkpoints/<classifier_type>`) config keys, added `_load_guidance_classifier` helper in `train_pipeline.py`. Previously this guidance classifier was hardcoded to `TimeDependentResNet` loaded from `checkpoints/robust_classification`; `classification`-type classifiers (resnet18, no timestep input) now work too since the guidance loss always calls the classifier at t=0 on the estimated clean image. |
| 2026-07-09 | `config/*.yaml` (all six) | Removed params never read by any code path (verified by grepping `src/`, `main.py`, `scripts/` for every access pattern): top-level `experiment_name`/`device`; `model.type`/`backbone`/`pretrained`/`num_layers`/`checkpoint_dir`; `data.num_workers` (DataLoader hardcodes `num_workers=2`, or `0` in optimize); `data.input_size` and `model.dropout` from configs whose pipeline never reads them; `training.batch_size` (distinct, unused key — only `data.batch_size` is read); `training.optimizer`/`mixed_precision`/`gradient_accumulation_steps`/`output_dir`/`scheduler`/`early_stopping`/`save_best_only`/`log_interval` (diffusion optimizer is hardcoded AdamW, no scheduler/early-stopping/logging code exists); `optimization.method`; the whole `evaluation:` block (`metrics`/`save_predictions`/`output_dir` — no evaluation code reads them); and `optimization.parameters` entries that are swept but never consumed by their objective function (`guidance_scale` in the two guided-diffusion configs, `warmup_epochs`/`transition_epochs`/`trades_beta` in `robust_classification.yaml`). Also removed dead `training.embedding_dim` from `diffusion.yaml` (a duplicate of the actually-used `model.embedding_dim`). Fixed a bug found along the way in `robust_classification.yaml`: `training:` had two separate `pgd:` mapping keys, so YAML silently dropped the first (`epsilon`/`alpha`/`num_steps`) and kept only the second (`random_start`) — the tuned PGD values (ε=0.092, α=0.0012, steps=40) were never actually reaching the code, which fell back to hardcoded defaults (0.03/0.01/20). Merged into one `pgd:` block so the tuned values take effect. |
| 2026-07-13 | `config/classification.yaml` | Step 1 of countering classification-model overfitting (val loss diverges from epoch ~6 while train loss keeps falling): raised `model.dropout` 0.2 → 0.5 and `training.weight_decay` 0.001 → 0.01. Trying stronger regularization before touching early stopping, backbone freezing, or augmentation. |
| 2026-07-13 | `src/pipelines/train_pipeline.py` (`train_classification`) | Step 1 (dropout/weight_decay bump) had ~zero effect on the overfitting (best val loss 0.6104 vs 0.6086 baseline) — likely because `layer4` (millions of params) was still fully trainable against ~1.4k train images. Step 2: froze the whole ResNet18 backbone (`layer1`-`layer4`), only `fc` is now fine-tuned. Simplified the optimizer from a two-param-group Adam (backbone at `lr*0.1` + fc at `lr`) to a single `Adam(model.fc.parameters(), ...)` since there's no longer a trainable non-fc group. |
| 2026-07-13 | `src/pipelines/train_pipeline.py` (`train_classification`), `src/models/simple_cnn.py` | Step 2 (fully-frozen ResNet18) *reduced* the train/val gap but underfit — best val loss got worse (0.6272 vs 0.6104) and test accuracy dropped (71.23% vs 73.68%), since it's effectively a linear probe on generic ImageNet features with no task-specific capacity left. Step 3: switched `train_classification` from ResNet18 transfer learning to training `SimpleCNN` from scratch (all params trainable, no freezing) on the 1-channel 150×150 input, now that the CRUMB test-label bug (see 2026-07-09 entries) is fixed and no longer confounds accuracy readings. Also shrank `SimpleCNN` itself from 4 conv blocks (32→64→128→256 channels) to 2 (16→32), since ~1.4k training images is too little data to justify the deeper/wider version. The removed ResNet18 setup (frozen backbone, fc-only fine-tune, ~90% on MiraBest) is kept commented out in `train_pipeline.py` in case of a revert. |
| 2026-07-13 | `main.py` | The `SimpleCNN` switch broke `classification` training — it was still getting the ResNet18-era transform (3-channel, ImageNet-normalized), but `SimpleCNN`'s first conv expects 1 channel (`RuntimeError: expected input... to have 1 channels, but got 3`). Replaced the `classification`-specific 3-channel transform branch with the shared 1-channel `diffusion_transform`-style one (same as every other model). The old 3-channel branch is kept commented out alongside the ResNet18 model code in `train_pipeline.py`, since that ResNet18 setup scored ~90% on MiraBest and may be reverted to. |
| 2026-07-13 | `config/classification.yaml` | `crumb_scratch` run underfit badly — training loss barely moved (0.708→0.668, hovering near chance-level ln(2)=0.693) over 100 epochs, best val loss 0.6614, test accuracy 70.53% (worst yet). Root cause: `learning_rate: 0.0001344` was Ray-Tune-optimized for fine-tuning a frozen-backbone `fc` head, not for training conv filters from random init, which needs a much higher LR. Raised to `0.001`. **Correction: this edit initially went into the wrong file** — `scripts/slurm/train/job_classification_crumb.sh` was actually loading `config/classification_crumb.yaml` (a separate, duplicate config), so this change (and the earlier 2026-07-13 dropout/weight_decay bump) never took effect on the `crumb_scratch`/`crumb_scratch_lr` runs; the "~zero effect" conclusion from the dropout/weight_decay experiment was coincidentally true but for the wrong reason (nothing had changed). Fixed for real below. |
| 2026-07-13 | `config/classification_crumb.yaml` (deleted), `scripts/slurm/train/job_classification_crumb.sh` (deleted), `scripts/slurm/train/job_classification.sh` | Removed the duplicate `classification_crumb.yaml` config entirely (dataset is already selected via `data.dataset: crumb` in `classification.yaml`, no need for a separate file) and applied the intended `learning_rate: 0.001`, `weight_decay: 0.01`, `dropout: 0.5` there so they actually take effect on the next run. Also consolidated the two dataset-specific slurm job scripts (`job_classification.sh`/`job_classification_crumb.sh`) into just `job_classification.sh` — since dataset is now purely a config concern, not a job-script concern, one script covers every dataset. Gave `job_classification.sh` the crumb script's `${SEED:-42}` default. |
