#!/bin/bash
# Phase 2: submit the 2 classifier-guided models (6 jobs total, 3 seeds each).
# Run this AFTER checkpoints/diffusion and checkpoints/robust_classification exist
# from the phase 1 runs (submit_baselines.sh).
#
# Usage: bash scripts/submit_guided.sh

set -e

SEEDS=(42 43 44)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$ROOT/logs"

if [ ! -f "$ROOT/checkpoints/diffusion/state.pt" ]; then
    echo "ERROR: checkpoints/diffusion/state.pt not found."
    echo "Wait for the diffusion job from submit_comparison.sh to finish first."
    exit 1
fi

if [ ! -f "$ROOT/checkpoints/robust_classification/state.pt" ]; then
    echo "ERROR: checkpoints/robust_classification/state.pt not found."
    echo "Wait for the robust_classification job from submit_comparison.sh to finish first."
    exit 1
fi

echo "Submitting classifier_guided_diffusion jobs..."
for SEED in "${SEEDS[@]}"; do
    qsub -v SEED=$SEED "$SCRIPT_DIR/job_classifier_guided_diffusion.pbs"
    echo "  submitted classifier_guided_diffusion seed=$SEED"
done

echo "Submitting robust_classifier_guided_diffusion jobs..."
for SEED in "${SEEDS[@]}"; do
    qsub -v SEED=$SEED "$SCRIPT_DIR/job_robust_classifier_guided_diffusion.pbs"
    echo "  submitted robust_classifier_guided_diffusion seed=$SEED"
done

echo ""
echo "6 jobs submitted. Monitor with: qstat -u $USER"
