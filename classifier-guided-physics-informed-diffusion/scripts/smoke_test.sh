#!/usr/bin/env bash
# Quick smoke-test: 1 epoch of each model to verify the training loop runs end-to-end.
# Run from the project root: bash scripts/smoke_test.sh
# Halt training manually once you've seen enough output (Ctrl+C).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRATCHPAD="$(mktemp -d)"
trap 'rm -rf "$SCRATCHPAD"' EXIT

source .venv/bin/activate 2>/dev/null || true

# ---------------------------------------------------------------------------
# Write 1-epoch test configs
# ---------------------------------------------------------------------------

cat > "$SCRATCHPAD/classification.yaml" <<'YAML'
experiment_name: smoke_test
seed: 42
device: cpu
data:
  dataset: crumb
  batch_size: 4
  num_workers: 0
  input_size: 224
model:
  backbone: resnet50
  pretrained: false
  num_layers: 18
  embedding_dim: 256
  dropout: 0.2
  checkpoint_dir: ./checkpoints
training:
  epochs: 1
  learning_rate: 0.0005
  weight_decay: 0.01
evaluation:
  metrics: [accuracy]
  save_predictions: false
  output_dir: ./results
YAML

cat > "$SCRATCHPAD/robust_classification.yaml" <<'YAML'
experiment_name: smoke_test
seed: 42
device: cpu
data:
  dataset: crumb
  batch_size: 4
  num_workers: 0
  input_size: 150
model:
  backbone: resnet50
  pretrained: false
  checkpoint_dir: ./checkpoints
training:
  epochs: 1
  warmup_epochs: 1
  transition_epochs: 0
  label_smoothing: 0.05
  learning_rate: 0.01
  weight_decay: 0.0005
  num_timesteps: 1000
  trades_beta: 6.0
  pgd:
    epsilon: 0.03
    alpha: 0.01
    num_steps: 2
    random_start: true
evaluation:
  metrics: [accuracy]
  save_predictions: false
  output_dir: ./results
YAML

cat > "$SCRATCHPAD/diffusion.yaml" <<'YAML'
experiment_name: smoke_test
seed: 42
device: cpu
data:
  dataset: mirabest_fits
  batch_size: 2
  num_workers: 0
  input_size: 150
model:
  backbone: resnet50
  pretrained: false
  num_layers: 18
  embedding_dim: 256
  dropout: 0.2
  checkpoint_dir: ./checkpoints
training:
  epochs: 1
  learning_rate: 0.0001
  label_dropout: 0.15
  num_train_timesteps: 1000
  optimizer: adamw
  weight_decay: 0.01
evaluation:
  metrics: [accuracy]
  save_predictions: false
  output_dir: ./results
YAML

cat > "$SCRATCHPAD/pid.yaml" <<'YAML'
experiment_name: smoke_test
seed: 42
device: cpu
data:
  dataset: mirabest_fits
  batch_size: 2
  num_workers: 0
  input_size: 150
model:
  backbone: resnet50
  pretrained: false
  num_layers: 18
  embedding_dim: 256
  dropout: 0.2
  checkpoint_dir: ./checkpoints
training:
  epochs: 1
  learning_rate: 0.0001
  label_dropout: 0.15
  num_train_timesteps: 1000
  optimizer: adamw
  weight_decay: 0.01
  lambda_sym: 0.05
  lambda_neg: 0.1
evaluation:
  metrics: [accuracy]
  save_predictions: false
  output_dir: ./results
YAML

cat > "$SCRATCHPAD/classifier_guided_diffusion.yaml" <<'YAML'
experiment_name: smoke_test
seed: 42
device: cpu
data:
  dataset: mirabest_fits
  batch_size: 2
  num_workers: 0
  input_size: 150
model:
  backbone: resnet50
  pretrained: false
  num_layers: 18
  embedding_dim: 256
  dropout: 0.2
  checkpoint_dir: ./checkpoints
  pretrained_checkpoint: checkpoints/diffusion
training:
  epochs: 1
  learning_rate: 0.00005
  label_dropout: 0.15
  num_train_timesteps: 1000
  optimizer: adamw
  weight_decay: 0.01
  lambda_cls: 0.1
evaluation:
  metrics: [accuracy]
  save_predictions: false
  output_dir: ./results
YAML

cat > "$SCRATCHPAD/robust_classifier_guided_diffusion.yaml" <<'YAML'
experiment_name: smoke_test
seed: 42
device: cpu
data:
  dataset: mirabest_fits
  batch_size: 2
  num_workers: 0
  input_size: 150
model:
  backbone: resnet50
  pretrained: false
  num_layers: 18
  embedding_dim: 256
  dropout: 0.2
  checkpoint_dir: ./checkpoints
  pretrained_checkpoint: checkpoints/diffusion
training:
  epochs: 1
  learning_rate: 0.00005
  label_dropout: 0.15
  num_train_timesteps: 1000
  optimizer: adamw
  weight_decay: 0.01
  lambda_cls: 0.1
evaluation:
  metrics: [accuracy]
  save_predictions: false
  output_dir: ./results
YAML

sep() { echo; echo "================================================================"; echo "  $1"; echo "================================================================"; }

# ---------------------------------------------------------------------------
# 1. classification
# ---------------------------------------------------------------------------
sep "[1/6] classification"
python main.py train --model classification --config "$SCRATCHPAD/classification.yaml"
echo "PASS: classification"

# ---------------------------------------------------------------------------
# 2. robust_classification
# ---------------------------------------------------------------------------
sep "[2/6] robust_classification"
python main.py train --model robust_classification --config "$SCRATCHPAD/robust_classification.yaml"
echo "PASS: robust_classification"

# Save a dummy checkpoint so classifier-guided models can load the frozen classifier.
# (The real checkpoint save needs --checkpoint + --resume; for the smoke test we
# save the just-trained weights directly.)
python - <<'PY'
import torch, os, sys
sys.path.insert(0, '.')
from src.models.time_dependent_resnet import TimeDependentResNet
m = TimeDependentResNet(2, pretrained=False)
os.makedirs('checkpoints/robust_classification', exist_ok=True)
torch.save({'model_state_dict': m.state_dict(), 'epoch': 0, 'config': {}},
           'checkpoints/robust_classification/state.pt')
print("Dummy robust_classification checkpoint written to checkpoints/robust_classification/state.pt")
PY

# ---------------------------------------------------------------------------
# 3. diffusion
# ---------------------------------------------------------------------------
sep "[3/6] diffusion"
python main.py train --model diffusion --config "$SCRATCHPAD/diffusion.yaml" --checkpoint True
echo "PASS: diffusion"

# ---------------------------------------------------------------------------
# 4. pid
# ---------------------------------------------------------------------------
sep "[4/6] pid"
python main.py train --model pid --config "$SCRATCHPAD/pid.yaml"
echo "PASS: pid"

# ---------------------------------------------------------------------------
# 5. classifier_guided_diffusion
# ---------------------------------------------------------------------------
sep "[5/6] classifier_guided_diffusion"
python main.py train --model classifier_guided_diffusion \
    --config "$SCRATCHPAD/classifier_guided_diffusion.yaml"
echo "PASS: classifier_guided_diffusion"

# ---------------------------------------------------------------------------
# 6. robust_classifier_guided_diffusion
# ---------------------------------------------------------------------------
sep "[6/6] robust_classifier_guided_diffusion"
python main.py train --model robust_classifier_guided_diffusion \
    --config "$SCRATCHPAD/robust_classifier_guided_diffusion.yaml"
echo "PASS: robust_classifier_guided_diffusion"

sep "ALL 6 SMOKE TESTS PASSED"
