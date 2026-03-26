# NLP Project: LLM Reasoning for Machine Translation

## Project Overview

Replication and extension of Zebaze et al. (2025) "Chain-of-Thought Reasoning In Large Reasoning Models for Machine Translation."

**Team:** Tom Boumba, Allison Zhuang, Habibi Ahmed Salem, Ruben Cardoso

---

## Experiments Completed

### Experiment 1: Thinking vs Non-Thinking Mode

**Goal:** Compare Qwen3's native reasoning (`<think>` blocks) vs direct translation.

| Model | Direction | Thinking | Non-Thinking | Notes |
|-------|-----------|----------|--------------|-------|
| Qwen3-0.6B | en→fr | 13.52 | **24.93** | Thinking broken (extraction issue) |
| Qwen3-0.6B | fr→en | **33.00** | 30.61 | Thinking helps |
| Qwen3-0.6B | en→sw | 0.76 | 0.57 | Model doesn't know Swahili |

**Finding:** Non-thinking mode outperforms on en→fr due to output extraction bugs in thinking mode.

---

### Experiment 2: In-Context Learning (Few-Shot)

**Goal:** Test how k-shot examples affect translation quality.

**Qwen3-0.6B Results:**

| Direction | k=0 | k=1 | k=3 | k=5 |
|-----------|-----|-----|-----|-----|
| en→fr | 24.93 | 22.58 | **25.75** | 25.65 |
| fr→en | 30.61 | **31.90** | 30.47 | 30.68 |

**Gemma-1B Results:**

| Direction | k=0 | k=1 | k=4 | k=8 |
|-----------|-----|-----|-----|-----|
| en→fr | 24.8 | 32.4 | **35.0** | 27.7 |
| fr→en | 24.3 | 24.4 | **27.4** | 25.1 |

**Finding:** Optimal k ≈ 1-4. Performance degrades past k=4-8 (context saturation).

---

### Experiment 3: Chain-of-Thought Prompting Strategies

**Goal:** Compare explicit CoT prompting strategies.

| Direction | Baseline | MAPS | TEaR | Self-Refine |
|-----------|----------|------|------|-------------|
| en→fr | **25.43** | 13.22 | 2.49 | 0.88 |
| fr→en | **31.56** | 29.83 | 0.97 | 28.18 |

**Finding:** Baseline wins. TEaR/Self-Refine suffer from scaffolding leakage (reasoning steps appear as output).

---

### Experiment 4: Fine-Tuning (IoFT / CoTFT)

**Goal:** Fine-tune Qwen3-0.6B on translation pairs.

| Condition | BLEU | chrF |
|-----------|------|------|
| IoFT | 3.63 | 19.82 |
| CoTFT(MAPS) | 4.07 | 20.41 |

**Finding:** Severe underperformance vs paper (~14-18 BLEU). Cause: Paper used gemma-3-4b-pt (4B params), not 0.6B.

---

## Pending Experiment: Model Scaling Analysis

**Goal:** Compare Qwen3-0.6B vs Qwen3-1.7B to answer "Is 0.6B too small?"

**What it adds:**
- Model size comparison (original contribution)
- COMET metric (required by instructor)
- Potential fix for thinking mode issues

**Scripts created:**
- `eval_thinking_1_7b.py` - Python evaluation script
- `run_thinking_1_7b.sh` - SLURM job for 1.7B only
- `run_thinking_both.sh` - SLURM job for 0.6B + 1.7B comparison

**To run:**
```bash
sbatch run_thinking_both.sh
```

---

## Key Issues Identified

| Issue | Status | Fix |
|-------|--------|-----|
| Thinking mode extraction broken | Identified | Regex cleanup in `eval_thinking_1_7b.py` |
| CoT scaffolding leakage | Identified | Need explicit "output ONLY translation" prompts |
| Missing COMET metric | In progress | Added to new evaluation script |
| Fine-tuning underperformance | Identified | Need larger model (4B) |

---

## Team Contributions

From `team_update/`:
- **ICL Analysis (Gemma-1B):** Few-shot results with qualitative analysis
- **Thinking Analysis:** CoT zero-shot vs IO direct comparison
- **Findings:** Gemma-1B output termination issues, metric improvements hide semantic errors

---

## Files Structure

```
nlp_proj/
├── results_exp1.json          # Thinking vs Non-Thinking (0.6B)
├── results_exp2.json          # ICL few-shot (0.6B)
├── results_exp3.json          # CoT prompting strategies
├── results_exp4.json          # Fine-tuning results
├── eval_thinking_1_7b.py      # NEW: Model scaling evaluation
├── run_thinking_1_7b.sh       # NEW: SLURM job (1.7B)
├── run_thinking_both.sh       # NEW: SLURM job (0.6B + 1.7B)
├── team_update/               # Team member contributions
│   ├── results_exp1 (1).json
│   ├── results_exp2_gemma_fewshot.json
│   ├── MVA_NLP_project_ICL_analysis.pdf
│   └── analyse_chatGPT (1).pdf
└── llm-reasoning-mt/          # Paper codebase
    ├── PROJECT_GUIDE.md
    ├── PROJECT_GUIDE_2.md
    └── PROJECT_GUIDE_3.md
```

---

## Next Steps

1. **Run model scaling experiment** (`sbatch run_thinking_both.sh`)
2. **Human evaluation** (50 sentences, 4 annotators, pairwise ranking)
3. **Sentence length analysis** (original contribution)
4. **Final write-up**
