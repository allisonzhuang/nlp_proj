#!/bin/bash
#SBATCH --job-name=thinking-compare
#SBATCH --partition=gpu          # Change to your H100 partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/thinking_compare_%j.log

set -e

mkdir -p logs

source ~/nlp_proj/.venv/bin/activate

export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub

cd ~/nlp_proj

echo "=== Thinking vs Non-Thinking: Model Size Comparison ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ─────────────────────────────────────────────────────────────
# Run Qwen3-0.6B (re-run with COMET for fair comparison)
# ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Model: Qwen3-0.6B"
echo "=========================================="

python eval_thinking_1_7b.py \
    --model Qwen/Qwen3-0.6B \
    --n_eval 50 \
    --batch_size 16 \
    --use_comet \
    --output_file results_thinking_0_6b.json

# ─────────────────────────────────────────────────────────────
# Run Qwen3-1.7B
# ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Model: Qwen3-1.7B"
echo "=========================================="

python eval_thinking_1_7b.py \
    --model Qwen/Qwen3-1.7B \
    --n_eval 50 \
    --batch_size 8 \
    --use_comet \
    --output_file results_thinking_1_7b.json

# ─────────────────────────────────────────────────────────────
# Print combined summary
# ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "COMBINED RESULTS"
echo "=========================================="

python -c "
import json

files = ['results_thinking_0_6b.json', 'results_thinking_1_7b.json']
all_results = []

for f in files:
    try:
        with open(f) as fp:
            all_results.extend(json.load(fp))
    except FileNotFoundError:
        print(f'Warning: {f} not found')

print(f\"{'Model':<20} {'Direction':<12} {'Mode':<15} {'BLEU':>8} {'chrF++':>8} {'COMET':>8}\")
print('-' * 80)

for r in all_results:
    model = r['model'].split('/')[-1]
    direction = f\"{r['src'][:2]}→{r['tgt'][:2]}\"
    comet = r['scores'].get('comet', 'N/A')
    if isinstance(comet, float):
        comet = f'{comet:.4f}'
    print(f\"{model:<20} {direction:<12} {r['mode']:<15} {r['scores']['bleu']:>8.2f} {r['scores']['chrf']:>8.2f} {comet:>8}\")
"

echo ""
echo "=== All evaluations complete ==="
echo "Date: $(date)"
