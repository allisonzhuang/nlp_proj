#!/bin/bash
#SBATCH --job-name=thinking-1.7b
#SBATCH --partition=gpu          # Change to your H100 partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/thinking_1_7b_%j.log

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment (adjust path as needed)
source ~/nlp_proj/.venv/bin/activate

# Set HuggingFace cache
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub

cd ~/nlp_proj

echo "=== Starting Qwen3-1.7B Thinking vs Non-Thinking Evaluation ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run the evaluation with COMET
python eval_thinking_1_7b.py \
    --model Qwen/Qwen3-1.7B \
    --n_eval 50 \
    --batch_size 8 \
    --use_comet \
    --output_file results_thinking_1_7b.json

echo ""
echo "=== Evaluation complete ==="
echo "Date: $(date)"
