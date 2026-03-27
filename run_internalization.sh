#!/bin/bash
#SBATCH --job-name=internalize
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/internalization_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

cd ~/nlp_proj

python eval_internalization.py

echo "=== Internalization test complete ==="
