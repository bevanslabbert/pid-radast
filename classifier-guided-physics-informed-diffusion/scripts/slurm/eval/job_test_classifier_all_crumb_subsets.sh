#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --account=b50-astro-cirg-ag
#SBATCH --qos=a01-idia-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bevanslabbert@gmail.com


module load python/3.11.15
module load cuda/11.8.0_520.61.05
source .venv/bin/activate

CORRECTED_CSV=results/2026-07-09/mirabest_classifier_vs_crumb_dataset/crumb_test_corrected_labels.csv

echo "=== 1. Baseline: classifier vs CRUMB's raw (test-split-corrupted) labels ==="
python scripts/test_classifier_on_all_crumb_subsets.py \
    --config config/classification.yaml \
    ${TAG:+--tag $TAG}

echo "=== 2. Deriving corrected test labels from train-split code->label mapping ==="
python scripts/derive_corrected_crumb_test_labels.py \
    --out ${CORRECTED_CSV}

echo "=== 3. Re-scoring: classifier vs corrected test labels ==="
python scripts/test_classifier_on_all_crumb_subsets.py \
    --config config/classification.yaml \
    --corrected-labels-csv ${CORRECTED_CSV} \
    ${TAG:+--tag $TAG}
