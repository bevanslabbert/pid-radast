#!/bin/bash
# Submit hyperparameter tuning jobs for the 4 independent model types.
# These can run in parallel — no checkpoint dependencies between them.
#
# Run from the project root: bash scripts/slurm/submit_tuning.sh
#
# Once these finish, copy the best values from
#   results/<model>/run_<timestamp>/best_params.yaml
# into the corresponding config/<model>.yaml, then run the full training:
#   bash scripts/slurm/submit_baselines.sh
#
# After diffusion and robust_classification checkpoints exist, tune the
# guided models with: bash scripts/slurm/submit_tuning_guided.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

mkdir -p "$ROOT/logs"

echo "Submitting tuning job: classification..."
sbatch \
    --job-name="tune_classification" \
    --output="$ROOT/logs/tune_classification_%j.out" \
    --error="$ROOT/logs/tune_classification_%j.err" \
    --chdir="$ROOT" \
    "$SCRIPT_DIR/job_tune_classification.sh"

echo "Submitting tuning job: robust_classification..."
sbatch \
    --job-name="tune_robustcls" \
    --output="$ROOT/logs/tune_robustcls_%j.out" \
    --error="$ROOT/logs/tune_robustcls_%j.err" \
    --chdir="$ROOT" \
    "$SCRIPT_DIR/job_tune_robust_classification.sh"

echo "Submitting tuning job: diffusion..."
sbatch \
    --job-name="tune_diffusion" \
    --output="$ROOT/logs/tune_diffusion_%j.out" \
    --error="$ROOT/logs/tune_diffusion_%j.err" \
    --chdir="$ROOT" \
    "$SCRIPT_DIR/job_tune_diffusion.sh"

echo "Submitting tuning job: pid..."
sbatch \
    --job-name="tune_pid" \
    --output="$ROOT/logs/tune_pid_%j.out" \
    --error="$ROOT/logs/tune_pid_%j.err" \
    --chdir="$ROOT" \
    "$SCRIPT_DIR/job_tune_pid.sh"

echo ""
echo "4 tuning jobs submitted. Monitor with: squeue -u $USER"
echo "Results will be written to results/<model>/run_<timestamp>/best_params.yaml"
echo ""
echo "Once diffusion and robust_classification training checkpoints exist,"
echo "tune the guided models with: bash scripts/slurm/submit_tuning_guided.sh"
