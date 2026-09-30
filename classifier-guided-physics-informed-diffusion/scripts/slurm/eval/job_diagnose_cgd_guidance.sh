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

# Diagnoses why CGD's classifier gradient barely changes its samples.
# TAG (optional) picks the CGD checkpoint; GUIDANCE_SCALE (optional) the CFG scale.
# GUIDANCE_CLASSIFIER / GUIDANCE_CLASSIFIER_TYPE (optional) test a different guidance classifier;
# set OUTPUT too so results don't overwrite results/cgd_guidance_diagnostics/.
python scripts/diagnose_cgd_guidance.py \
    ${TAG:+--tag $TAG} \
    ${GUIDANCE_SCALE:+--guidance-scale $GUIDANCE_SCALE} \
    ${GUIDANCE_CLASSIFIER:+--guidance-classifier $GUIDANCE_CLASSIFIER} \
    ${GUIDANCE_CLASSIFIER_TYPE:+--guidance-classifier-type $GUIDANCE_CLASSIFIER_TYPE} \
    ${OUTPUT:+--output $OUTPUT}
