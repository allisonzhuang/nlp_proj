# Project Guide 2: Addressing Feedback & Going Beyond Replication

This guide responds to the instructor feedback on the one-page report and provides concrete recommendations for strengthening the project with original contributions.

---

## Table of Contents

1. [Summary of Feedback](#1-summary-of-feedback)
2. [Concrete Metric Choices](#2-concrete-metric-choices)
3. [Language Directions & Prioritization](#3-language-directions--prioritization)
4. [ICL Demonstrations & Prompting Strategy Choices](#4-icl-demonstrations--prompting-strategy-choices)
5. [Original Ideas Beyond the Paper](#5-original-ideas-beyond-the-paper)
6. [Improved Human Evaluation Protocol](#6-improved-human-evaluation-protocol)
7. [Revised Experiment Plan](#7-revised-experiment-plan)

---

## 1. Summary of Feedback

The instructor raised four main points:

| # | Feedback | Action Required |
|---|----------|-----------------|
| 1 | Plan too focused on replication | Propose more original ideas/stress tests |
| 2 | Unclear metric choices | Specify exact metrics and justify |
| 3 | Unclear language directions, ICL shots, prompting strategies | Make concrete choices upfront |
| 4 | Human evaluation needs refinement | Focus on proficient languages, add inter-annotator agreement, consider pairwise ranking |

---

## 2. Concrete Metric Choices

### Primary Metrics (Always Report)

| Metric | Why | Implementation |
|--------|-----|----------------|
| **BLEU** | Standard, interpretable, comparable to prior work | `sacrebleu` with default tokenization |
| **chrF++** | Character-level, robust for morphologically rich languages (German, Russian) | `sacrebleu --metric chrf` |
| **COMET** | Neural metric, high correlation with human judgment | `Unbabel/wmt22-comet-da` |

### Secondary Metric (Deeper Analysis)

| Metric | Why | Implementation |
|--------|-----|----------------|
| **MetricX-24** | SOTA neural metric, reference-free option available | `google/metricx-24-hybrid-large-v2p6-bfloat16` |

### Quality Estimation (For BoA Ranking)

| Metric | Why | Implementation |
|--------|-----|----------------|
| **BLASER 2.0 QE** | Reference-free, used in the paper | As specified in your report |

### Justification

- **BLEU + chrF++**: Required for comparability with prior MT literature
- **COMET**: Captures semantic adequacy that n-gram metrics miss
- **MetricX-24**: Use selectively for final comparisons (computationally expensive)

---

## 3. Language Directions & Prioritization

### Recommended Focus (4 directions)

| Direction | Category | Justification |
|-----------|----------|---------------|
| **en→fr** | High-resource | Team proficiency for human eval |
| **fr→en** | High-resource | Reverse direction comparison |
| **en→de** | High-resource | Morphologically rich target, standard benchmark |
| **en→xho** | Low-resource | Stress test, paper's focus on Xhosa |

### Why These Choices

1. **French (en↔fr)**: At least one team member is proficient, enabling meaningful human evaluation
2. **German (en→de)**: Tests handling of compound words, case marking, word order differences
3. **Xhosa (en→xho)**: Low-resource language, tests generalization limits of small models

### Optional Extensions (If Time Permits)

| Direction | Why |
|-----------|-----|
| **en→zh** | Logographic script, very different structure |
| **de→fr** | Non-English pivot, tests direct transfer |

---

## 4. ICL Demonstrations & Prompting Strategy Choices

### Number of In-Context Examples

**Recommended values**: k ∈ {0, 1, 4}

| k | Rationale |
|---|-----------|
| 0 | Zero-shot baseline |
| 1 | Minimal demonstration |
| 4 | Sweet spot for small models before context saturation |

**Skip k=8**: Small models (Qwen3-0.6B, Gemma-3-4B) have limited context; k=8 risks truncation and diminishing returns.

### Prompting Strategies to Prioritize

Focus on **3 strategies** (not all):

| Strategy | Why Prioritize |
|----------|----------------|
| **Direct (baseline)** | Required baseline for all comparisons |
| **CoT** | Core question of the paper |
| **MAPS** | Best performing in the paper, keyword-based approach |

**Deprioritize for initial experiments**: TEaR, SBYS, CompTra, Self-Refine (can explore if time permits)

### Retrieval Strategy

**Recommend**: BM25 (simple, effective, already implemented in `comptra/retriever.py`)

Skip embedding-based retrieval initially—adds complexity without guaranteed gains for small models.

---

## 5. Original Ideas Beyond the Paper

This is the most critical section. Here are concrete original contributions to differentiate your work:

### Idea 1: CoTFT-BoA Evaluation (Already Proposed)

Your proposed CoTFT-BoA experiment is good. Strengthen it with:

- **Hypothesis**: On-the-fly strategy selection improves over fixed strategy because optimal reasoning depends on sentence characteristics
- **Analysis**: Correlate which strategy "wins" with sentence features (length, syntactic complexity, domain)

### Idea 2: Reasoning Internalization Test

**Question**: Does CoT training improve the model's internal representations, or just its output format?

**Experiment**:
```
1. Fine-tune with CoTFT (reasoning + translation)
2. At inference, force direct output (no reasoning tokens)
3. Compare to IoFT baseline

If CoTFT-direct > IoFT → reasoning is internalized
If CoTFT-direct ≈ IoFT → reasoning is just output scaffolding
```

**Why original**: Paper doesn't test whether reasoning benefits persist when reasoning is suppressed at inference.

### Idea 3: Sentence Complexity Stratification

**Question**: Do thinking models help more on complex sentences?

**Experiment**:
```
1. Compute complexity metrics for each test sentence:
   - Length (word count)
   - Parse tree depth (using spaCy)
   - Number of clauses
2. Bin sentences into: Simple / Medium / Complex
3. Report metrics separately per bin for thinking vs. non-thinking
```

**Hypothesis**: Thinking provides larger gains on complex sentences.

### Idea 4: Error Type Analysis

**Question**: What types of errors does reasoning reduce?

**Experiment**:
```
1. Sample 50 sentences where thinking mode > non-thinking mode (by COMET)
2. Sample 50 sentences where non-thinking mode > thinking mode
3. Manually categorize errors:
   - Lexical (wrong word choice)
   - Syntactic (word order, agreement)
   - Omission (missing content)
   - Addition (hallucinated content)
   - Named entities (names, numbers, dates)
```

**Why original**: Paper reports aggregate metrics but doesn't analyze *what* reasoning fixes.

### Idea 5: Reasoning Efficiency Analysis

**Question**: Is more reasoning always better?

**Experiment**:
```
1. For thinking model outputs, measure reasoning length (tokens in <think> block)
2. Correlate reasoning length with translation quality (COMET)
3. Plot: reasoning_length vs. quality_improvement (over non-thinking)
```

**Possible findings**:
- Linear relationship → more thinking = better
- Plateau → diminishing returns after N tokens
- Inverse-U → too much reasoning hurts (overthinking)

### Idea 6: Cross-Lingual Reasoning Transfer

**Question**: Does reasoning learned for one language pair transfer to another?

**Experiment**:
```
1. Fine-tune CoTFT on en→de only
2. Evaluate zero-shot on en→fr (unseen direction)
3. Compare to:
   - IoFT on en→de, tested on en→fr
   - No fine-tuning baseline on en→fr
```

**Hypothesis**: CoT reasoning is partially language-agnostic and transfers.

### Idea 7: Small Model Scaling Analysis

**Question**: How does model size interact with reasoning benefits?

**Experiment**:
```
Compare across model sizes (all same family):
- Qwen3-0.6B
- Qwen3-4B
- (Optional: Qwen3-8B if compute allows)

For each: measure Δ(thinking - non-thinking)
```

**Hypothesis**: Smaller models benefit less from reasoning (insufficient capacity to reason well).

---

## 6. Improved Human Evaluation Protocol

### Language Choice

**Focus on French (en→fr, fr→en)** where team members are proficient.

### Method: Pairwise Ranking (Recommended)

Instead of Direct Assessment (0-100 scores), use **pairwise comparison**:

```
Given:
- Source sentence (English)
- Translation A (thinking mode)
- Translation B (non-thinking mode)

Task: Which translation is better? [A / B / Tie]
```

**Advantages over DA**:
- Easier for annotators (relative judgment, not absolute)
- More reliable inter-annotator agreement
- Directly answers: "Does thinking help?"

### Sample Size & Selection

| Aspect | Recommendation |
|--------|----------------|
| **Sample size** | 100 sentences |
| **Selection** | Stratified: 50 where COMET favors thinking, 50 where COMET favors non-thinking |
| **Annotators** | 4 team members |

### Inter-Annotator Agreement

**Compute and report**:
- **Cohen's Kappa** (pairwise between annotators)
- **Fleiss' Kappa** (overall agreement across all 4)

```python
from sklearn.metrics import cohen_kappa_score
# For each annotator pair
kappa = cohen_kappa_score(annotator_1, annotator_2)
```

### Annotation Protocol

```markdown
## Instructions for Annotators

You will evaluate 100 translation pairs. For each:

1. Read the source sentence carefully
2. Read both translations (A and B) - order is randomized
3. Judge: Which translation better preserves the meaning of the source?
4. Select: A is better / B is better / Tie (both equally good or bad)

Focus on:
- Meaning preservation (adequacy)
- Grammatical correctness (fluency)
- No hallucinations or omissions

Do NOT consider:
- Minor stylistic differences
- Punctuation differences (unless meaning-changing)
```

### Output Format

Create a spreadsheet with columns:
```
sentence_id | source | translation_A | translation_B | annotator_1 | annotator_2 | annotator_3 | annotator_4
```

---

## 7. Revised Experiment Plan

### Phase 1: Baselines & ICL (Week 1)

| Experiment | Models | Directions | Metrics |
|------------|--------|------------|---------|
| Zero-shot direct | Qwen3-0.6B, Qwen3-4B, Gemma-3-4B-pt | en→fr, en→de, en→xho | BLEU, chrF++, COMET |
| Few-shot direct (k=1,4) | Same | Same | Same |

**Deliverable**: Table showing ICL scaling behavior across model sizes and language pairs.

### Phase 2: CoT Prompting (Week 1-2)

| Experiment | Models | Directions | Metrics |
|------------|--------|------------|---------|
| Zero-shot CoT | Same | en→fr, en→de | BLEU, chrF++, COMET |
| Few-shot CoT (k=1,4) | Same | Same | Same |
| MAPS prompting | Same | Same | Same |

**Deliverable**: Comparison of direct vs. CoT vs. MAPS prompting.

### Phase 3: Thinking Models (Week 2)

| Experiment | Models | Directions | Metrics |
|------------|--------|------------|---------|
| Thinking mode | Qwen3-4B (if thinking variant exists), or DeepSeek-R1-Distill-Qwen-1.5B | en→fr, en→de | BLEU, chrF++, COMET |
| Non-thinking mode (prefilled) | Same | Same | Same |

**Deliverable**: Thinking vs. non-thinking comparison with prefilled completion trick.

### Phase 4: Fine-Tuning (Week 2-3)

| Experiment | Base Model | Training Data | Directions |
|------------|------------|---------------|------------|
| IoFT | Qwen3-4B or Gemma-3-4B-pt | TopXGen dataset | en→xho |
| CoTFT | Same | Same + reasoning | Same |
| CoTFT-BoA (original) | Same | Same | Same |

**Setup**: LoRA + 4-bit quantization on Colab

**Deliverable**: IoFT vs. CoTFT vs. CoTFT-BoA comparison.

### Phase 5: Original Analyses (Week 3)

Pick 2-3 from Section 5:

| Analysis | Effort |
|----------|--------|
| Sentence complexity stratification | Low |
| Reasoning length correlation | Low |
| Error type analysis | Medium |
| Reasoning internalization test | Medium |
| Cross-lingual transfer | High |

### Phase 6: Human Evaluation (Week 3-4)

- 100 sentences, en→fr direction
- 4 annotators, pairwise ranking
- Compute inter-annotator agreement
- Compare human rankings to automatic metrics

### Phase 7: Write-Up (Week 4)

- Compile results
- Analyze where automatic metrics agree/disagree with human judgment
- Discuss findings and limitations

---

## Summary: What Makes This Project Strong

| Aspect | Replication | Original Contribution |
|--------|-------------|----------------------|
| ICL scaling | Yes | Analysis across model sizes |
| CoT prompting | Yes | MAPS comparison |
| Thinking vs. non-thinking | Yes | Sentence complexity stratification |
| Fine-tuning | Yes (IoFT, CoTFT) | **CoTFT-BoA** (your proposed extension) |
| Human evaluation | Yes | **Pairwise + inter-annotator agreement** |
| Error analysis | No | **Error type taxonomy** (original) |
| Reasoning efficiency | No | **Reasoning length analysis** (original) |

By combining solid replication with 2-3 original analyses, you demonstrate both understanding of the paper and independent thinking.

---

## Quick Reference: Codebase Entry Points

| Task | File | Key Arguments |
|------|------|---------------|
| Generate CoT data | `paraphrase.py` | `--strategy cot/maps`, `--model_name_or_path` |
| Fine-tune | `train.py` | `--use_peft`, `--lora_r`, `--load_in_4bit` |
| Evaluate | `evaluation.py` | `--model_name_or_path`, `--dataset` |
| ICL retrieval | `comptra/retriever.py` | BM25 or embedding-based |
| Prompts | `comptra/prompts/` | `translate.py`, `maps.py` |
