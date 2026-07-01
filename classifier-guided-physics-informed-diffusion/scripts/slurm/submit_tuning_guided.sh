#!/bin/bash
# Submit hyperparameter tuning jobs for the 2 classifier-guided diffusion models.
# Run AFTER checkpoints/diffusion/state.pt and checkpoints/robust_classification/state.pt
# exist (produced by the full training runs from submit_baselines.sh).
#
# Run from the project root: bash scripts/slurm/submit_tuning_guided.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

mkdir -p "$ROOT/logs"

if [ ! -f "$ROOT/checkpoints/diffusion/state.pt" ]; then
    echo "ERROR: checkpoints/diffusion/state.pt not found."
    echo "Run the full diffusion training first: bash scripts/slurm/submit_baselines.sh"
    exit 1
fi

if [ ! -f "$ROOT/checkpoints/robust_classification/state.pt" ]; then
    echo "ERROR: checkpoints/robust_classification/state.pt not found."
    echo "Run the full robust_classification training first: bash scripts/slurm/submit_baselines.sh"
    exit 1
fi

echo "Submitting tuning job: classifier_guided_diffusion (commit=${COMMIT})..."
sbatch \
    --job-name="tune_cgd_${COMMIT}" \
    --output="$ROOT/logs/tune_cgd_${COMMIT}_%j.out" \
    --error="$ROOT/logs/tune_cgd_${COMMIT}_%j.err" \
    --chdir="$ROOT" \
    "$SCRIPT_DIR/job_tune_classifier_guided_diffusion.sh"

echo "Submitting tuning job: robust_classifier_guided_diffusion..."
sbatch \
    --job-name="tune_rcgd_${COMMIT}" \
    --output="$ROOT/logs/tune_rcgd_${COMMIT}_%j.out" \
    --error="$ROOT/logs/tune_rcgd_${COMMIT}_%j.err" \
    --chdir="$ROOT" \
    "$SCRIPT_DIR/job_tune_robust_classifier_guided_diffusion.sh"

echo ""
echo "2 tuning jobs submitted. Monitor with: squeue -u $USER"
echo "Results will be written to results/<model>/run_<timestamp>/best_params.yaml"
