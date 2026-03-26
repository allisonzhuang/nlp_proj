#!/bin/bash
#SBATCH --job-name=test-ft
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_ft_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub

cd ~/nlp_proj

# Quick smoke test: 10 steps of IoFT
python train_ft.py \
    --mode ioft \
    --output_dir ./checkpoints/test_ioft \
    --max_steps 10 \
    --batch_size 2 \
    --grad_accum 1 \
    --eval_size 50

echo "=== Test completed successfully ==="
