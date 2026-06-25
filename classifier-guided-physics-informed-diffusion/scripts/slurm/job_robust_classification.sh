#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --account=b50-astro-cirg-ag
#SBATCH --qos=a01-idia-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bevanslabbert@gmail.com


module load python/3.11.15
module load cuda/11.8.0_520.61.05
source .venv/bin/activate

python main.py train \
    --model robust_classification \
    --seed ${SEED} \
    --runs 1 \
    --checkpoint True
