#!/bin/bash
#SBATCH --job-name=internal-v2
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/internalization_v2_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

cd ~/nlp_proj

python eval_internalization_v2.py

echo "=== Internalization test v2 complete ==="
