#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --account=b50-astro-cirg-ag
#SBATCH --qos=a01-idia-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bevanslabbert@gmail.com


module load python/3.11.15
module load cuda/11.8.0_520.61.05
source .venv/bin/activate

# Expects DIFFUSION_TAGS / CGD_TAGS / EDM_TAGS as space-separated seed tag lists,
# e.g. --export=ALL,DIFFUSION_TAGS="exp_seed42 exp_seed43 exp_seed44 exp_seed45 exp_seed46"
python scripts/evaluate_generated_samples.py \
    --diffusion-tags ${DIFFUSION_TAGS} \
    --cgd-tags ${CGD_TAGS} \
    --edm-tags ${EDM_TAGS} \
    --classifier-tag ${CLASSIFIER_TAG:-eval} \
    --num-samples ${NUM_SAMPLES:-16}
