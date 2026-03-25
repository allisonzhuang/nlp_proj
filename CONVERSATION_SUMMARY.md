# Conversation Summary: LLM Reasoning for Machine Translation Project

**Exported:** 2026-03-25

---

## Project Overview

Replicating Zebaze et al. (2025) paper "LLM Reasoning for Machine Translation" with focus on:
- Thinking vs Non-Thinking modes in Qwen3 models
- In-Context Learning (ICL) with few-shot examples
- Chain-of-Thought (CoT) prompting strategies
- Fine-tuning approaches (IoFT, CoTFT)

---

## Experimental Results

### Experiment 1: Thinking vs Non-Thinking

| Direction | Mode | BLEU | chrF | Notes |
|-----------|------|------|------|-------|
| en→fr | Thinking | 13.52 | 36.01 | ⚠️ Extraction bug - outputs like `"final translation."` |
| en→fr | Non-Thinking | 24.93 | 50.49 | ✓ Matches paper (~24 BLEU) |
| fr→en | Thinking | 33.00 | 57.68 | |
| fr→en | Non-Thinking | 30.61 | 59.12 | |
| en→sw | Thinking | 0.76 | 8.54 | Model struggles with Swahili |
| en→sw | Non-Thinking | 0.57 | 7.55 | |

**Key Finding:** Thinking mode underperforms due to output extraction issues, not model capability.

### Experiment 2: ICL Few-Shot

| Direction | k=0 | k=1 | k=3 | k=5 |
|-----------|-----|-----|-----|-----|
| en→fr (BLEU) | 24.93 | 22.58 | 25.75 | 25.65 |
| en→fr (chrF) | 50.49 | 48.39 | 51.47 | 50.07 |
| fr→en (BLEU) | 30.61 | 31.90 | 30.47 | 30.68 |
| fr→en (chrF) | 59.12 | 60.23 | 58.50 | 58.41 |

**Key Finding:** k=3 performs best for en→fr; k=1 best for fr→en. Results are reasonable.

### Experiment 3: CoT Prompting Strategies

| Direction | Baseline | MAPS | TEaR | Self-Refine |
|-----------|----------|------|------|-------------|
| en→fr (BLEU) | 25.43 | 13.22 | 2.49 | 0.88 |
| en→fr (chrF) | 51.13 | 36.20 | 14.23 | 8.35 |
| fr→en (BLEU) | 31.56 | 29.83 | 0.97 | 28.18 |
| fr→en (chrF) | 60.06 | 53.19 | 5.05 | 56.70 |

**Key Finding:** CoT scaffolding leaking into output (e.g., `"**Step 1 (Translate):**"`). Prompts need fixing.

### Experiment 4: Fine-Tuning

| Condition | BLEU | chrF |
|-----------|------|------|
| IoFT | 3.63 | 19.82 |
| CoTFT(MAPS) | 4.07 | 20.41 |

**Key Finding:** Severe underperformance. Paper used gemma-3-4b-pt (4B params), not Qwen3-0.6B.

---

## Paper Analysis: Model Sizes

From Zebaze et al. (2025) paper analysis:

### For Inference/Prompting
- Qwen3 family: 0.6B, 1.7B, 4B, 8B, 14B, 32B
- Qwen3-0.6B results from paper: 24.65 BLEU (thinking), 23.33 BLEU (non-thinking) for En→Fr
- **Conclusion:** Qwen3-0.6B is NOT too small for prompting

### For Fine-Tuning
- Student model: **gemma-3-4b-pt** (4B parameters)
- Teacher model: Llama-4-Scout
- Training: 5000 steps
- **Conclusion:** Fine-tuning requires larger model than 0.6B

---

## Identified Issues and Fixes

### Issue 1: Thinking Mode Output Extraction

**Problem:** Model outputs malformed text like `"final translation."` instead of actual translations.

**Fix:**
```python
import re

def extract_translation(output: str, thinking_mode: bool) -> str:
    """Extract clean translation from model output."""
    if thinking_mode:
        # Remove thinking traces
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
        if 'Translation:' in output:
            output = output.split('Translation:')[-1]
        elif 'Final:' in output:
            output = output.split('Final:')[-1]

    # Clean up
    output = output.strip()
    output = re.sub(r'^[\*\#\d\.\:\-]+\s*', '', output)
    output = output.strip('"\'')

    return output
```

### Issue 2: CoT Scaffolding Leakage

**Problem:** Reasoning steps appear as output (e.g., `"**Step 1 (Translate):**"`).

**Fix:** Add explicit output instructions to prompts:
```python
TEAR_PROMPT = """Translate the following from {src} to {tgt}.

Internally:
1. Create initial translation
2. Identify potential errors
3. Refine translation

Output ONLY the final refined translation, nothing else.

Source: {source}
Translation:"""
```

### Issue 3: Fine-Tuning Underperformance

**Problem:** 3-4 BLEU vs paper's 14-18 BLEU.

**Causes:**
- Wrong model (Qwen3-0.6B vs gemma-3-4b-pt)
- Insufficient training (300 vs 5000 steps)

**Recommendation:** Skip fine-tuning for one-day scope, or use larger model.

---

## Project Files

| File | Description |
|------|-------------|
| `results_exp1.json` | Thinking vs Non-Thinking results |
| `results_exp2.json` | ICL few-shot results |
| `results_exp3.json` | CoT prompting results |
| `results_exp4.json` | Fine-tuning results |
| `PROJECT_GUIDE.md` | Original project guide |
| `PROJECT_GUIDE_2.md` | Instructor feedback response |
| `PROJECT_GUIDE_3.md` | One-day scoped implementation plan |
| `2510.11919v1.pdf` | Original Zebaze et al. paper |

---

## One-Day Plan (PROJECT_GUIDE_3.md)

| Priority | Experiment | Time |
|----------|------------|------|
| 1 | Thinking vs Non-Thinking (fix extraction) | ~2h |
| 2 | ICL few-shot (k=0,1,3,5) | ~1h |
| 3 | CoT prompting (fix scaffolding) | ~1h |
| 4 | IoFT fine-tuning (500 steps) | ~3-4h |
| 5 | Human evaluation (50 sentences, 4 annotators) | ~1h |

**Total: ~8-9 hours**

---

## Next Steps

1. Fix output extraction for thinking mode
2. Fix CoT prompts to prevent scaffolding leakage
3. Add COMET metric to all experiments (`pip install unbabel-comet`)
4. Re-run thinking mode and CoT experiments
5. Set up human evaluation (pairwise ranking, Cohen's Kappa)

---

## Key Metrics to Report

| Metric | Implementation |
|--------|----------------|
| BLEU | `sacrebleu` |
| chrF++ | `sacrebleu --metric chrf` |
| COMET | `Unbabel/wmt22-comet-da` |

---

## COMET Setup

```bash
pip install unbabel-comet
```

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [
    {"src": "Hello world", "mt": "Bonjour le monde", "ref": "Bonjour monde"}
]
output = model.predict(data, batch_size=8)
print(output.system_score)  # Corpus-level score
```
