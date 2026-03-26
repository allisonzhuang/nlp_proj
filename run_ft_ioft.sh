#!/bin/bash
#SBATCH --job-name=ft-ioft
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ft_ioft_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub

cd ~/nlp_proj

python train_ft.py \
    --mode ioft \
    --output_dir ./checkpoints/ioft \
    --max_steps 5000 \
    --batch_size 4 \
    --grad_accum 4

echo "=== IoFT training complete ==="
