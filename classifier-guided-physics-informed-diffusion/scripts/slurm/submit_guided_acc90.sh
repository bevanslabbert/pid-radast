#!/bin/bash
# Classifier-guided diffusion, guided by the tagged acc90 classification checkpoint
# (config/classifier_guided_diffusion.yaml -> model.classifier_checkpoint).
# 3 base seeds x 5 sequential runs each = 15 independent trainings, spaced far
# enough apart that --runs 5 never produces overlapping seeds across jobs.
#
# Prerequisite: checkpoints/classification/acc90/state.pt must exist.
# Run from the project root: bash scripts/slurm/submit_guided_acc90.sh

set -e

BASE_SEEDS=(42 1042 2042)
RUNS=5
TAG=acc90guided
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

mkdir -p "$ROOT/logs"

if [ ! -f "$ROOT/checkpoints/classification/acc90/state.pt" ]; then
    echo "ERROR: checkpoints/classification/acc90/state.pt not found."
    echo "Tag your existing classifier checkpoint first, e.g.:"
    echo "  mkdir -p checkpoints/classification/acc90 && mv checkpoints/classification/state.pt checkpoints/classification/acc90/state.pt"
    exit 1
fi

echo "Submitting classifier_guided_diffusion jobs (commit=${COMMIT}, tag=${TAG}, runs=${RUNS} per job)..."
for BASE in "${BASE_SEEDS[@]}"; do
    sbatch \
        --job-name="cgd_base${BASE}_${COMMIT}" \
        --output="$ROOT/logs/cgd_base${BASE}_${COMMIT}_%j.out" \
        --error="$ROOT/logs/cgd_base${BASE}_${COMMIT}_%j.err" \
        --chdir="$ROOT" \
        --time=40:00:00 \
        --export=ALL,SEED=$BASE,RUNS=$RUNS,TAG=$TAG \
        "$SCRIPT_DIR/train/job_classifier_guided_diffusion.sh"
    echo "  submitted classifier_guided_diffusion base_seed=$BASE (seeds $BASE-$((BASE+RUNS-1)))"
done

echo ""
echo "3 jobs submitted (15 runs total). Monitor with: squeue -u \$USER"
