#!/bin/bash
#SBATCH --job-name=eval-ft
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_ft_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub

cd ~/nlp_proj

python eval_ft.py \
    --include_baseline \
    --use_comet \
    --n_eval 200 \
    --batch_size 8 \
    --output_file results_ft_eval.json

echo "=== Evaluation complete ==="
