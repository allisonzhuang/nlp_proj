#!/bin/bash
#SBATCH --job-name=bleu-curve
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bleu_curve_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

cd ~/nlp_proj

# Args passed via sbatch: --export=MODE=ioft,LABEL="IoFT",CKPT_BASE=./checkpoints_v1/ioft
python eval_training_curve.py \
    --checkpoint_base "$CKPT_BASE" \
    --label "$LABEL" \
    --n_eval 1012 \
    --output_file "${CKPT_BASE}/training_curve.json"

echo "=== BLEU curve for $LABEL complete ==="
