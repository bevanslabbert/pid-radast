#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --account=b50-astro-cirg-ag
#SBATCH --qos=a01-idia-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bevanslabbert@gmail.com


module load python/3.11.15
module load cuda/11.8.0_520.61.05
source .venv/bin/activate

# Expects DIFFUSION_TAGS / CGD_TAGS / EDM_TAGS as space-separated seed tag lists,
# e.g. --export=ALL,DIFFUSION_TAGS="exp_seed42 exp_seed43 exp_seed44 exp_seed45 exp_seed46"
# Any of the three may be left unset to evaluate a subset of models (one job per model).
# GUIDANCE_SCALE (optional) overrides every model's CFG scale for a matched-guidance comparison.
# CLASSIFIER_SCALE (optional) overrides CGD's classifier-gradient strength (0 disables it).
# GUIDANCE_CLASSIFIER / GUIDANCE_CLASSIFIER_TYPE (optional) swap CGD's guidance classifier.
python scripts/evaluate_generated_samples.py \
    ${DIFFUSION_TAGS:+--diffusion-tags $DIFFUSION_TAGS} \
    ${CGD_TAGS:+--cgd-tags $CGD_TAGS} \
    ${EDM_TAGS:+--edm-tags $EDM_TAGS} \
    ${GUIDED_DIFFUSION_TAGS:+--guided-diffusion-tags $GUIDED_DIFFUSION_TAGS} \
    --output results/generation_classifier_eval/${SLURM_JOB_NAME:-all} \
    --classifier-tag ${CLASSIFIER_TAG:-eval_fits_bs8_seed42} \
    --num-samples ${NUM_SAMPLES:-128} \
    --batch-size ${BATCH_SIZE:-16} \
    ${GUIDANCE_SCALE:+--guidance-scale $GUIDANCE_SCALE} \
    ${CLASSIFIER_SCALE:+--classifier-scale $CLASSIFIER_SCALE} \
    ${GUIDANCE_CLASSIFIER:+--guidance-classifier $GUIDANCE_CLASSIFIER} \
    ${GUIDANCE_CLASSIFIER_TYPE:+--guidance-classifier-type $GUIDANCE_CLASSIFIER_TYPE} \
    ${SAVE_GRID:+--save-grid} \
    ${SAVE_FITS:+--save-fits}
