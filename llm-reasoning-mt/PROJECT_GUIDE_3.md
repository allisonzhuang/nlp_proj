# Project Guide 3: One-Day Scoped Implementation

This guide provides a realistic one-day plan addressing instructor feedback while working within compute constraints.

---

## Overview

| Experiment | Compute | Time | Status |
|------------|---------|------|--------|
| Thinking vs Non-Thinking (Qwen3-0.6B/1.7B) | HF Inference API / Colab T4 | ~2h | Priority 1 |
| ICL few-shot prompting (k=0,1,3,5) | Same | ~1h | Priority 2 |
| CoT prompting strategies (MAPS, SBYS, TEaR) | Same | ~1h | Priority 3 |
| IoFT fine-tuning (Qwen3-0.6B, ~500 steps) | Colab A100 / Vast.ai | ~3-4h | Priority 4 |
| Human evaluation (French, 50 sentences, 4 annotators) | None | ~1h | Priority 5 |

**Total: ~8-9 hours**

---

## 1. Thinking vs Non-Thinking Evaluation

### Goal
Compare thinking mode vs non-thinking mode on Qwen3-0.6B and Qwen3-1.7B.

### Language Directions
- **en→fr** (primary, for human eval)
- **fr→en** (reverse direction)

### Critical Fix: Output Extraction

Your current results show malformed outputs (`"final translation."`, incomplete text). This is an extraction problem.

```python
import re

def extract_translation(output: str, thinking_mode: bool) -> str:
    """Extract clean translation from model output."""
    if thinking_mode:
        # Remove thinking traces
        # Pattern 1: <think>...</think> tags
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
        # Pattern 2: Look for final answer after reasoning
        if 'Translation:' in output:
            output = output.split('Translation:')[-1]
        elif 'Final:' in output:
            output = output.split('Final:')[-1]

    # Clean up
    output = output.strip()
    output = re.sub(r'^[\*\#\d\.\:\-]+\s*', '', output)  # Remove markdown artifacts
    output = output.strip('"\'')

    return output
```

### Metrics to Report
| Metric | Implementation |
|--------|----------------|
| BLEU | `sacrebleu` |
| chrF++ | `sacrebleu --metric chrf` |
| COMET | `Unbabel/wmt22-comet-da` (add this!) |

### Expected Output Table

| Model | Mode | Direction | BLEU | chrF++ | COMET |
|-------|------|-----------|------|--------|-------|
| Qwen3-0.6B | Thinking | en→fr | | | |
| Qwen3-0.6B | Non-Thinking | en→fr | | | |
| Qwen3-0.6B | Thinking | fr→en | | | |
| Qwen3-0.6B | Non-Thinking | fr→en | | | |
| Qwen3-1.7B | Thinking | en→fr | | | |
| Qwen3-1.7B | Non-Thinking | en→fr | | | |
| Qwen3-1.7B | Thinking | fr→en | | | |
| Qwen3-1.7B | Non-Thinking | fr→en | | | |

---

## 2. ICL Few-Shot Prompting

### Goal
Test in-context learning with k ∈ {0, 1, 3, 5} examples.

### Setup
- Model: Qwen3-0.6B (non-thinking mode, since it performed better)
- Directions: en→fr, fr→en
- Retrieval: BM25 (use `comptra/retriever.py`)

### Prompt Template
```
Translate the following sentence from {src_lang} to {tgt_lang}.

{# k examples #}
{src_lang}: {example_src_1}
{tgt_lang}: {example_tgt_1}

...

{src_lang}: {source_sentence}
{tgt_lang}:
```

### Expected Output Table

| Direction | k=0 | k=1 | k=3 | k=5 |
|-----------|-----|-----|-----|-----|
| en→fr (BLEU) | | | | |
| en→fr (chrF++) | | | | |
| en→fr (COMET) | | | | |
| fr→en (BLEU) | | | | |
| fr→en (chrF++) | | | | |
| fr→en (COMET) | | | | |

---

## 3. CoT Prompting Strategies

### Goal
Compare reasoning-based prompting strategies (zero-shot).

### Strategies to Test

| Strategy | Description | Key Prompt Element |
|----------|-------------|-------------------|
| **Baseline** | Direct translation | "Translate X to Y" |
| **MAPS** | Multi-Aspect Prompting | Extract keywords → translate |
| **SBYS** | Sentence-by-Sentence | Break into chunks → translate each |
| **TEaR** | Translate-Explain-Refine | Translate → explain → refine |

### Critical Fix: Prevent Scaffolding Leakage

Your current results show reasoning scaffolding appearing as output (`"**Step 1 (Translate):**"`).

**Solution**: Add explicit instruction to output ONLY the final translation.

```python
MAPS_PROMPT = """Translate the following from {src} to {tgt}.

Before translating, identify:
1. Key terms and entities
2. Sentence structure
3. Potential ambiguities

Then provide ONLY the final translation with no explanations or markup.

Source: {source}
Translation:"""

TEAR_PROMPT = """Translate the following from {src} to {tgt}.

Internally:
1. Create initial translation
2. Identify potential errors
3. Refine translation

Output ONLY the final refined translation, nothing else.

Source: {source}
Translation:"""
```

### Expected Output Table

| Direction | Baseline | MAPS | SBYS | TEaR |
|-----------|----------|------|------|------|
| en→fr (BLEU) | | | | |
| en→fr (chrF++) | | | | |
| en→fr (COMET) | | | | |
| fr→en (BLEU) | | | | |
| fr→en (chrF++) | | | | |
| fr→en (COMET) | | | | |

---

## 4. IoFT Fine-Tuning

### Goal
Fine-tune Qwen3-0.6B with input-output pairs (no reasoning traces).

### Setup
| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-0.6B |
| Method | LoRA (r=16, alpha=32) |
| Quantization | 4-bit (QLoRA) |
| Steps | 500 |
| Batch size | 4 (with gradient accumulation) |
| Learning rate | 2e-4 |
| Dataset | FLORES-200 (en→fr subset, ~500 examples) |

### Training Data Format
```json
{
  "instruction": "Translate from English to French.",
  "input": "We now have 4-month-old mice that are non-diabetic that used to be diabetic.",
  "output": "Nous avons maintenant des souris de 4 mois qui ne sont pas diabétiques alors qu'elles l'étaient auparavant."
}
```

### Evaluation
Compare fine-tuned model vs zero-shot baseline on held-out test set.

| Condition | BLEU | chrF++ | COMET |
|-----------|------|--------|-------|
| Zero-shot (no FT) | | | |
| IoFT (500 steps) | | | |

### Realistic Expectations
With 500 steps on 0.6B, expect modest gains (~2-5 BLEU improvement over zero-shot). If you see degradation, check:
- Learning rate (try 1e-4)
- Overfitting (reduce steps to 300)
- Data quality (verify training examples are correct)

---

## 5. Human Evaluation

### Setup
| Aspect | Value |
|--------|-------|
| Direction | en→fr |
| Sentences | 50 |
| Annotators | 4 (you + 3 friends) |
| Method | Pairwise ranking |

### Sentence Selection
Select 50 sentences stratified by automatic metric disagreement:
- 25 where thinking mode scores higher (by COMET)
- 25 where non-thinking mode scores higher

```python
import pandas as pd

# Assuming you have a DataFrame with results
df['comet_diff'] = df['comet_thinking'] - df['comet_non_thinking']
thinking_wins = df.nlargest(25, 'comet_diff')
non_thinking_wins = df.nsmallest(25, 'comet_diff')
eval_set = pd.concat([thinking_wins, non_thinking_wins])
```

### Annotation Template

Create a Google Form or spreadsheet with:

```
Sentence ID: ___

Source (English):
"We now have 4-month-old mice that are non-diabetic that used to be diabetic."

Translation A:
"Nous avons maintenant des souris de 4 mois qui ne sont pas diabétiques alors qu'elles l'étaient auparavant."

Translation B:
"Nous avons à présent des souris de 4 mois qui ne sont pas diabétiques, qui ont été diabétiques avant."

Which translation is better?
[ ] A is better
[ ] B is better
[ ] Tie (both equally good or bad)
```

**Important**: Randomize A/B order for each sentence.

### Computing Agreement

```python
from sklearn.metrics import cohen_kappa_score
import itertools

def compute_agreement(annotations):
    """
    annotations: dict of {annotator_id: [list of judgments]}
    judgments: 0=A, 1=B, 2=Tie
    """
    annotators = list(annotations.keys())
    kappas = []

    for a1, a2 in itertools.combinations(annotators, 2):
        kappa = cohen_kappa_score(annotations[a1], annotations[a2])
        kappas.append((a1, a2, kappa))
        print(f"{a1} vs {a2}: κ = {kappa:.3f}")

    avg_kappa = sum(k[2] for k in kappas) / len(kappas)
    print(f"\nAverage pairwise κ = {avg_kappa:.3f}")

    # Interpretation
    if avg_kappa < 0.2:
        print("Interpretation: Poor agreement")
    elif avg_kappa < 0.4:
        print("Interpretation: Fair agreement")
    elif avg_kappa < 0.6:
        print("Interpretation: Moderate agreement")
    elif avg_kappa < 0.8:
        print("Interpretation: Substantial agreement")
    else:
        print("Interpretation: Almost perfect agreement")

    return kappas, avg_kappa
```

### Results Table

| Comparison | Human Preference | COMET Agreement |
|------------|------------------|-----------------|
| Thinking wins (human) | X / 50 | |
| Non-Thinking wins (human) | X / 50 | |
| Ties | X / 50 | |
| Cohen's Kappa (avg) | | |

---

## 6. One Original Analysis (Bonus)

If time permits, add **one** original contribution. Recommended (lowest effort):

### Sentence Length Analysis

```python
import matplotlib.pyplot as plt

# Bin sentences by length
df['src_length'] = df['source'].apply(lambda x: len(x.split()))
df['length_bin'] = pd.cut(df['src_length'], bins=[0, 10, 20, 30, 100],
                          labels=['Short (1-10)', 'Medium (11-20)',
                                  'Long (21-30)', 'Very Long (31+)'])

# Compare thinking vs non-thinking by length bin
for bin_name in df['length_bin'].unique():
    subset = df[df['length_bin'] == bin_name]
    print(f"\n{bin_name}:")
    print(f"  Thinking BLEU: {subset['bleu_thinking'].mean():.2f}")
    print(f"  Non-Thinking BLEU: {subset['bleu_non_thinking'].mean():.2f}")
    print(f"  Δ: {(subset['bleu_thinking'] - subset['bleu_non_thinking']).mean():.2f}")
```

**Hypothesis**: Thinking mode helps more on longer/complex sentences.

---

## 7. Deliverables Checklist

| Deliverable | Status |
|-------------|--------|
| Thinking vs Non-Thinking results table (2 models × 2 directions) | [ ] |
| ICL scaling table (k=0,1,3,5 × 2 directions) | [ ] |
| CoT strategies table (4 strategies × 2 directions) | [ ] |
| IoFT comparison (zero-shot vs fine-tuned) | [ ] |
| Human eval results (50 sentences, 4 annotators, kappa) | [ ] |
| COMET scores added to all experiments | [ ] |
| (Bonus) Sentence length analysis | [ ] |

---

## 8. Common Pitfalls to Avoid

| Issue | Solution |
|-------|----------|
| Thinking mode outputs reasoning as translation | Fix extraction regex, add explicit "output only translation" |
| CoT scaffolding leaks into output | Rewrite prompts with clear output instructions |
| Fine-tuning degrades performance | Reduce LR, fewer steps, verify data quality |
| Low inter-annotator agreement | Clarify annotation guidelines, do a practice round |
| Missing COMET metric | Install: `pip install unbabel-comet` |

---

## 9. Quick Setup Commands

```bash
# Install dependencies
pip install sacrebleu unbabel-comet transformers peft bitsandbytes

# Run COMET evaluation
comet-score -s sources.txt -t hypotheses.txt -r references.txt --model Unbabel/wmt22-comet-da

# Or in Python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [
    {"src": "Hello world", "mt": "Bonjour le monde", "ref": "Bonjour monde"}
]
output = model.predict(data, batch_size=8)
print(output.system_score)  # Corpus-level score
```

---

## Timeline

| Time | Task |
|------|------|
| Hour 1-2 | Thinking vs Non-Thinking (fix extraction, run both models) |
| Hour 3 | ICL few-shot experiments |
| Hour 4 | CoT prompting strategies (fix prompts, run) |
| Hour 5-8 | IoFT fine-tuning (setup, train, evaluate) |
| Hour 8-9 | Human evaluation (distribute, collect, compute kappa) |
| Buffer | Add COMET to all results, compile tables |

---

## What This Achieves

| Instructor Feedback | How Addressed |
|--------------------|---------------|
| "Too focused on replication" | Sentence length analysis (original), human eval with kappa |
| "Unclear metric choices" | BLEU + chrF++ + COMET (justified) |
| "Unclear language directions" | Focused on en↔fr only (team proficiency) |
| "Human eval needs refinement" | Pairwise ranking, 4 annotators, inter-annotator agreement |
