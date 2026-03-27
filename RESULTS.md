# Results: LLM Reasoning for Machine Translation (Replication)

Replication and extension of Zebaze et al. (2025), "LLM Reasoning for Machine Translation."

**Team:** Tom Boumba, Allison Zhuang, Habibi Ahmed Salem, Ruben Cardoso

---

## Setup

- **Student model**: gemma-3-4b-pt (LoRA fine-tuned, r=32, alpha=64)
- **Teacher model**: Llama-4-Scout-17B-16E-Instruct (pre-generated reasoning traces)
- **Language pair**: English → Xhosa (low-resource)
- **Evaluation set**: FLORES+ devtest (200 sentences for final scores, 1012 for training curves)
- **Training**: 5000 steps, batch_size=4, grad_accum=16, lr=1e-5, constant LR with 500-step warmup
- **Hardware**: 1x NVIDIA H100 80GB per job

### Training Methods

| Method | Dataset | Target |
|--------|---------|--------|
| **IoFT** | `almanach/topxgen-llama-4-scout-MAPS` | Direct translation (Xhosa) |
| **CoTFT-CoT** | `almanach/topxgen-llama-4-scout-CoT` | Reasoning chain + Xhosa |
| **CoTFT-MAPS** | `almanach/topxgen-llama-4-scout-MAPS` | Structured MAPS reasoning + Xhosa |

---

## Experiment 1: Fine-Tuning Comparison (IoFT vs CoTFT)

| Model | BLEU | chrF++ | COMET |
|-------|------|--------|-------|
| Baseline (no FT) | 1.16 | 14.27 | 0.3842 |
| IoFT | 6.52 | 31.29 | 0.6484 |
| CoTFT (CoT) | 4.32 | 28.47 | 0.6190 |
| **CoTFT (MAPS)** | **6.99** | **33.56** | **0.6655** |

**Finding:** MAPS > IoFT > CoT across all metrics, consistent with the paper. Naive CoT hurts vs IoFT (-2.2 BLEU), while structured MAPS reasoning improves (+0.47 BLEU, +2.27 chrF++).

Our absolute BLEU (~7) is lower than the paper's (~14-18), likely due to hyperparameter differences, but the relative ordering validates the paper's core claim.

---

## Experiment 2: Reasoning Internalization Test

**Question:** Does CoTFT training improve the model's internal translation ability, or does it just teach an output format?

**Method:** We prefilled `<think>` to trigger reasoning at inference and compared to direct translation. We also examined raw outputs to verify the models learned reasoning patterns.

| Model | Direct prompt | With `<think>` prefill |
|-------|--------------|----------------------|
| IoFT | 6.52 | 1.71 (garbage) |
| CoTFT-CoT | 4.32 | 0.85 (never closes think block) |
| CoTFT-MAPS | 6.99 | 0.00 (never closes think block) |

**Key observations:**
1. CoTFT models **did learn reasoning patterns** — CoT outputs `"I am analyzing the sentence structure..."` and MAPS outputs `"Here is a draft translation\n1. ..."`, matching their training data format.
2. However, the models **cannot terminate reasoning** — none produce a closing `</think>` tag. gemma-3-4b-pt lacks native thinking token support, so it generates reasoning indefinitely until hitting the token limit.
3. The benefit of MAPS comes from **higher-quality training signal**, not runtime reasoning. The teacher's structured analysis produced better translation examples, which the student absorbed into its weights.

**Conclusion:** Reasoning is partially internalized (the model learned reasoning patterns) but cannot be leveraged at inference due to the base model's lack of thinking token support. The MAPS advantage is best understood as a **data quality effect**.

---

## Experiment 3: Thinking Mode & Model Scaling (Qwen3)

**Question:** Does model size interact with thinking mode benefits?

**Method:** Compared Qwen3-0.6B and Qwen3-1.7B with thinking mode on/off on en↔fr (50 FLORES+ sentences, BLEU/chrF++/COMET).

**Important methodological note:** Initial results with `max_new_tokens=512` showed thinking mode *hurting* both models. Investigation revealed this was a **truncation artifact** — the models' think blocks exceeded 512 tokens, so the actual translation was never generated. Rerunning with `max_new_tokens=2048` reversed the finding.

| Model | Direction | Thinking | Non-Thinking | Delta |
|-------|-----------|----------|--------------|-------|
| Qwen3-0.6B | en→fr | **24.73** | 21.41 | **+3.32** |
| Qwen3-0.6B | fr→en | **31.69** | 27.81 | **+3.88** |
| Qwen3-1.7B | en→fr | 35.92 | 35.99 | -0.07 |
| Qwen3-1.7B | fr→en | **37.50** | 36.03 | **+1.47** |

**Finding:** Thinking helps the smaller model more (+3-4 BLEU for 0.6B vs ~0-1.5 BLEU for 1.7B). The 1.7B model is already capable enough to translate well directly, while the 0.6B benefits from the reasoning scratchpad. This suggests **thinking is more valuable when model capacity is limited**.

---

## Training Dynamics

### Loss Curves
All three models converge smoothly over 5000 steps. CoTFT-CoT achieves the lowest training loss (~0.25) since it fits longer reasoning traces. IoFT has the highest loss (~0.77). See `loss_curves.png`.

### BLEU Over Training Steps
Evaluated every 500 steps on the full 1012-sentence FLORES+ devtest. MAPS leads throughout after step 1000, with a notable non-monotonic dip at step 4500 (BLEU 4.01) before recovering to 6.91 at step 5000. Training loss decreases monotonically through this range, suggesting the model temporarily reorganizes internal representations. See `bleu_curves.png`.

---

## Metrics

- **BLEU**: Exact n-gram overlap (sacrebleu)
- **chrF++**: Character n-gram F-score with word n-grams (sacrebleu)
- **COMET**: Neural MT evaluation metric (Unbabel/wmt22-comet-da)

Note: The paper uses MetricX-24, which was incompatible with transformers 4.57. COMET serves as a comparable neural metric.
