#!/bin/bash
# Submit 5 independent runs for each model in the comparison.
# Each run uses a different seed (42-46) and saves results + final weights
# to results/<model>/run_<timestamp>_seed<N>/.
#
# Usage: bash scripts/submit_comparison.sh
#
# Run robust_classification first — diffusion and pid can run in parallel
# but don't depend on the classifier. If you later wire up
# classifier-guided diffusion you'll need the robust_classification
# weights first.

set -e

SEEDS=(42 43 44)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$ROOT/logs"

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
echo "All 15 jobs submitted. Monitor with: qstat -u $USER"
