#!/bin/bash
# Submit classification and robust_classification training jobs (3 seeds each).
# Run from the project root: bash scripts/slurm/submit_classifiers.sh

set -e

SEEDS=(42 43 44)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

mkdir -p "$ROOT/logs"

echo "Submitting classification jobs (commit=${COMMIT})..."
for SEED in "${SEEDS[@]}"; do
    sbatch \
        --job-name="classification_seed${SEED}_${COMMIT}" \
        --output="$ROOT/logs/classification_seed${SEED}_${COMMIT}_%j.out" \
        --error="$ROOT/logs/classification_seed${SEED}_${COMMIT}_%j.err" \
        --chdir="$ROOT" \
        --export=ALL,SEED=$SEED \
        "$SCRIPT_DIR/job_classification.sh"
    echo "  submitted classification seed=$SEED"
done

# echo "Submitting robust_classification jobs..."
# for SEED in "${SEEDS[@]}"; do
#     sbatch \
#         --job-name="robustcls_seed${SEED}" \
#         --output="$ROOT/logs/robustcls_seed${SEED}_%j.out" \
#         --error="$ROOT/logs/robustcls_seed${SEED}_%j.err" \
#         --chdir="$ROOT" \
#         --export=ALL,SEED=$SEED \
#         "$SCRIPT_DIR/job_robust_classification.sh"
#     echo "  submitted robust_classification seed=$SEED"
# done

echo ""
# echo "6 jobs submitted. Monitor with: squeue -u \$USER"
echo "3 jobs submitted. Monitor with: squeue -u \$USER"
