#!/bin/bash
# Phase 1: submit the 4 independent baseline models (12 jobs total, 3 seeds each).
# Run from the project root: bash scripts/slurm/submit_baselines.sh
#
# Once diffusion and robust_classification checkpoints are saved,
# run submit_guided.sh to queue the classifier-guided models.

set -e

SEEDS=(42 43 44)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

mkdir -p "$ROOT/logs"

echo "Submitting classification jobs..."
for SEED in "${SEEDS[@]}"; do
    sbatch \
        --job-name="classification_seed${SEED}" \
        --output="$ROOT/logs/classification_seed${SEED}_%j.out" \
        --error="$ROOT/logs/classification_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,PROJECT_DIR=$ROOT \
        "$SCRIPT_DIR/job_classification.sh"
    echo "  submitted classification seed=$SEED"
done

echo "Submitting robust_classification jobs..."
for SEED in "${SEEDS[@]}"; do
    sbatch \
        --job-name="robustcls_seed${SEED}" \
        --output="$ROOT/logs/robustcls_seed${SEED}_%j.out" \
        --error="$ROOT/logs/robustcls_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,PROJECT_DIR=$ROOT \
        "$SCRIPT_DIR/job_robust_classification.sh"
    echo "  submitted robust_classification seed=$SEED"
done

echo "Submitting diffusion jobs..."
for SEED in "${SEEDS[@]}"; do
    sbatch \
        --job-name="diffusion_seed${SEED}" \
        --output="$ROOT/logs/diffusion_seed${SEED}_%j.out" \
        --error="$ROOT/logs/diffusion_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,PROJECT_DIR=$ROOT \
        "$SCRIPT_DIR/job_diffusion.sh"
    echo "  submitted diffusion seed=$SEED"
done

echo "Submitting pid jobs..."
for SEED in "${SEEDS[@]}"; do
    sbatch \
        --job-name="pid_seed${SEED}" \
        --output="$ROOT/logs/pid_seed${SEED}_%j.out" \
        --error="$ROOT/logs/pid_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,PROJECT_DIR=$ROOT \
        "$SCRIPT_DIR/job_pid.sh"
    echo "  submitted pid seed=$SEED"
done

echo ""
echo "12 jobs submitted. Monitor with: squeue -u $USER"
echo "Once diffusion and robust_classification are done, run: bash scripts/slurm/submit_guided.sh"
