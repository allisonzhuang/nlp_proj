# Project Guide: LLM Reasoning for Machine Translation

This guide provides structured advice for exploring machine translation with LLMs, covering few-shot learning, chain-of-thought prompting, thinking models, evaluation strategies, and fine-tuning.

---

## Table of Contents

1. [Few-Shot Learning Experiments](#1-few-shot-learning-experiments)
2. [CoT Prompting with Base LLMs](#2-cot-prompting-with-base-llms)
3. [Thinking Models Comparison](#3-thinking-models-comparison)
4. [Evaluation Strategy](#4-evaluation-strategy)
5. [Fine-Tuning on Colab](#5-fine-tuning-on-colab)
6. [Ideas Beyond the Paper](#6-ideas-beyond-the-paper)
7. [Practical Pitfalls to Avoid](#7-practical-pitfalls-to-avoid)
8. [Suggested Experiment Priority](#8-suggested-experiment-priority)

---

## 1. Few-Shot Learning Experiments

### What to Investigate

- Vary k ∈ {0, 1, 2, 4, 8} in-context examples
- Compare retrieval strategies: random vs. BM25 vs. embedding-based (see `comptra/retriever.py`)

### Practical Considerations for Small Models

- **Context limits**: Gemma-270m and 1B have limited context windows. With k=8 examples, you may hit length limits. Track how often truncation occurs.
- **Sensitivity**: Small models are more sensitive to example quality and ordering. Consider running multiple seeds with different example permutations.

### Translation Directions to Prioritize

| Category | Example Pairs | Why |
|----------|---------------|-----|
| High-resource | en↔de, en↔fr | Baseline performance |
| Medium-resource | en↔cs, en↔ru | WMT benchmarks available |
| Low-resource | en↔xho, en↔hau | Tests generalization; codebase focuses on Xhosa |
| Non-English | de→fr, zh→ja | Tests without English pivot |

### Hypothesis to Test

Small models may show diminishing or even negative returns past k=2-4 examples due to context confusion.

---

## 2. CoT Prompting with Base LLMs

### Key Insight

The codebase has 6 CoT templates in `comptra/prompts/translate.py`. However, for **base/pretrained** models (not instruction-tuned), vanilla "Let's think step by step" often fails because:

- Base models aren't trained to follow instructions
- They may continue generating source-language text instead of reasoning

### Recommended Approach

Structure CoT as completion rather than instruction:

```
English: {source}
Let me translate this to German step by step.
First, I identify the key phrases:
```

### Experiments to Run

1. Compare zero-shot direct vs. zero-shot CoT
2. Compare few-shot direct vs. few-shot with reasoning demonstrations
3. Measure: Does CoT increase output length without improving quality?

### Important Metric

Track the **reasoning overhead** — how many extra tokens does CoT produce, and is quality improvement proportional?

---

## 3. Thinking Models Comparison

### Models to Compare

| Type | Examples | Notes |
|------|----------|-------|
| Base | gemma-2-2b-pt, gemma-3-1b-pt | No instruction tuning |
| Instruct | gemma-2-2b-it, gemma-3-1b-it | Standard instruction-tuned |
| Thinking | DeepSeek-R1-Distill-Qwen-1.5B, QwQ | Trained with explicit reasoning |

### Experimental Design

For thinking models, compare:

1. `thinking_enabled=True` — let it reason
2. `thinking_enabled=False` — suppress `<think>` tokens, force direct answer

The codebase handles this in `comptra/utils.py` for thinking token extraction.

### Key Questions to Answer

- Do thinking models show larger gains on complex sentences (longer, more clauses)?
- Is thinking more beneficial for low-resource directions?
- What's the latency/quality tradeoff?

---

## 4. Evaluation Strategy

### Metrics Hierarchy

```
Primary (always report):
├── BLEU (sacrebleu) — interpretability, comparability
├── chrF++ — character-level, better for morphologically rich languages
└── COMET (wmt22-comet-da) — neural, correlates with human judgment

Secondary (for deeper analysis):
├── MetricX-24 — SOTA neural metric
└── COMET-QE (reference-free) — useful when references are questionable
```

### Human Evaluation Protocol

Since full annotation is expensive, use **stratified sampling**:

1. **Sample selection**: Select 50-100 sentences spanning:
   - Short/long
   - Simple/complex
   - High/low BLEU delta between systems

2. **Annotation method**: Use **pairwise comparison** (which translation is better?) rather than absolute scoring

3. **Phenomena to annotate**:
   - Fluency vs. adequacy
   - Named entity handling
   - Negation preservation
   - Number/date accuracy

### Codebase Resources

Leverage `comptra/evaluate/` — includes `metricx24/` and COMET integration.

---

## 5. Fine-Tuning on Colab

### Realistic Constraints

| Environment | VRAM | Model Size |
|-------------|------|------------|
| Colab Free (T4) | ~15GB | Up to ~3B params with 4-bit quantization |
| Colab Pro (A100) | ~40GB | Up to 7B with LoRA |

### Recommended Setup

Using the existing `train.py`:

```bash
python train.py \
  --model_name_or_path google/gemma-2-2b-pt \
  --use_peft \
  --lora_r 16 \
  --lora_alpha 32 \
  --load_in_4bit \
  --max_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_steps 1000
```

### What to Fine-Tune On

1. **IOFT** (Input-Output Fine-Tuning): Direct source→target pairs
2. **CoTFT** (CoT Fine-Tuning): Source→reasoning→target (use `paraphrase.py` to generate data)

### Key Comparison

Does CoTFT outperform IOFT for the same compute budget? The paper suggests yes, but validate on small models.

---

## 6. Ideas Beyond the Paper

| Idea | Description |
|------|-------------|
| **Reasoning ablation** | Train on CoT data, but at inference strip reasoning. Does the model internalize better representations? |
| **Cross-lingual transfer** | Fine-tune on en→de with CoT, test zero-shot on en→fr. Does reasoning transfer? |
| **Metric as reward** | Use GRPO with COMET as reward signal instead of BLEU |
| **Error taxonomy** | Categorize errors by type (lexical, syntactic, omission, addition). Do thinking models reduce specific error types? |
| **Sentence complexity** | Bin sentences by parse tree depth or clause count. Plot quality vs. complexity for each method |
| **Thinking efficiency** | For thinking models, correlate reasoning length with translation quality. Is more thinking always better? |

---

## 7. Practical Pitfalls to Avoid

### 1. Tokenization Mismatch

Ensure your metrics use detokenized outputs. SacreBLEU handles this, but MetricX may need care.

### 2. Prompt Format Leakage

Base models may output the prompt format (e.g., `[tgt]:`) as part of the translation. Post-process this.

### 3. Language Contamination

Small models may code-switch mid-translation. Track this as a failure mode.

### 4. Evaluation Set Overlap

FLORES test set may overlap with training data for some models. Use NTREX or TICO-19 as held-out validation.

### 5. Cherry-Picking

Report aggregate metrics AND variance across test sets. A model winning on FLORES may lose on NTREX.

---

## 8. Suggested Experiment Priority

Given limited compute, prioritize in this order:

### Phase 1: Baseline (~2-3 days)

- Few-shot (k=0, 1, 4) on gemma-2-2b-pt for en↔de, en↔xho
- Metrics: BLEU, chrF++, COMET

### Phase 2: CoT Analysis (~2-3 days)

- Zero-shot CoT vs. direct on same models
- Few-shot with reasoning demonstrations

### Phase 3: Thinking Models (~2 days)

- Compare gemma-3-1b-it vs. thinking model (DeepSeek-R1-Distill)
- With/without thinking ablation

### Phase 4: Fine-Tuning (~3-4 days)

- IOFT vs. CoTFT on gemma-2-2b-pt
- LoRA + 4-bit on Colab

### Phase 5: Analysis (~3-4 days)

- Human evaluation on 50-100 samples
- Error taxonomy
- Write-up

---

## Codebase Structure Reference

```
├── train.py              # SFT and GRPO training
├── paraphrase.py         # Dataset generation with reasoning strategies
├── evaluation.py         # Inference and benchmark evaluation
├── train_datasets.py     # Dataset loading/preparation
├── comptra/
│   ├── prompts/          # CoT, MAPS, SBYS, TEaR, etc.
│   ├── data/             # FLORES, NTREX, TICO-19 loaders
│   ├── evaluate/         # MetricX, COMET integration
│   └── retriever.py      # BM25, embedding-based retrieval
├── configs/              # DeepSpeed configs
└── scripts/              # Training/eval shell scripts
```

---

## Relevant Resources

- **SacreBLEU**: Standardized BLEU and chrF++ evaluation
- **MetricX-24**: `google/metricx-24-hybrid-large-v2p6-bfloat16`
- **COMET**: `Unbabel/wmt22-comet-da`
- **Paper**: [LLM Reasoning for Machine Translation](https://arxiv.org/abs/2510.11919)
