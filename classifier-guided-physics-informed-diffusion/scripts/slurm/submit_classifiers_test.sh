#!/bin/bash
# Quick 5-epoch test of classification and robust_classification (single seed).
# Run from the project root: bash scripts/slurm/submit_classifiers_test.sh

set -e

SEED=42
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

mkdir -p "$ROOT/logs"

echo "Submitting classification test (seed=$SEED, 5 epochs)..."
sbatch \
    --job-name="classification_test" \
    --output="$ROOT/logs/classification_test_%j.out" \
    --error="$ROOT/logs/classification_test_%j.err" \
    --chdir="$ROOT" \
    --export=ALL,SEED=$SEED \
    "$SCRIPT_DIR/job_classification.sh"

echo "Submitting robust_classification test (seed=$SEED, 5 epochs)..."
sbatch \
    --job-name="robustcls_test" \
    --output="$ROOT/logs/robustcls_test_%j.out" \
    --error="$ROOT/logs/robustcls_test_%j.err" \
    --chdir="$ROOT" \
    --export=ALL,SEED=$SEED \
    "$SCRIPT_DIR/job_robust_classification.sh"

echo ""
echo "2 test jobs submitted. Monitor with: squeue -u \$USER"
