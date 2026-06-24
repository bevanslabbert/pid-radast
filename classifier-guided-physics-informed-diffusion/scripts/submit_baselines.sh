#!/bin/bash
# Phase 1: submit the 4 independent models (12 jobs total, 3 seeds each).
# Run this first. Once diffusion and robust_classification checkpoints are
# saved, run submit_guided.sh to queue the classifier-guided models.
#
# Usage: bash scripts/submit_baselines.sh

set -e

SEEDS=(42 43 44)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$ROOT/logs"

echo "Submitting classification jobs..."
for SEED in "${SEEDS[@]}"; do
    qsub -v SEED=$SEED "$SCRIPT_DIR/job_classification.pbs"
    echo "  submitted classification seed=$SEED"
done

echo "Submitting robust_classification jobs..."
for SEED in "${SEEDS[@]}"; do
    qsub -v SEED=$SEED "$SCRIPT_DIR/job_robust_classification.pbs"
    echo "  submitted robust_classification seed=$SEED"
done

echo "Submitting diffusion jobs..."
for SEED in "${SEEDS[@]}"; do
    qsub -v SEED=$SEED "$SCRIPT_DIR/job_diffusion.pbs"
    echo "  submitted diffusion seed=$SEED"
done

echo "Submitting pid jobs..."
for SEED in "${SEEDS[@]}"; do
    qsub -v SEED=$SEED "$SCRIPT_DIR/job_pid.pbs"
    echo "  submitted pid seed=$SEED"
done

echo ""
echo "12 jobs submitted. Monitor with: qstat -u $USER"
echo "Once diffusion and robust_classification are done, run: bash scripts/submit_guided.sh"
