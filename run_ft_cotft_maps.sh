#!/bin/bash
#SBATCH --job-name=ft-cotft-maps
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_cotft_maps_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub

cd ~/nlp_proj

python train_ft.py \
    --mode cotft_maps \
    --output_dir ./checkpoints/cotft_maps \
    --max_steps 5000 \
    --batch_size 4 \
    --grad_accum 16

echo "=== CoTFT (MAPS) training complete ==="
