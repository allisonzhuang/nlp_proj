#!/bin/bash
#SBATCH --job-name=eval-all-ft
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_all_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

cd ~/nlp_proj

python eval_all_ft.py

echo "=== Evaluation complete ==="
