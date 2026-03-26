# Results: LLM Reasoning for Machine Translation (Replication)

Replication of Zebaze et al. (2025), "LLM Reasoning for Machine Translation: Synthetic Data Generation over Thinking Tokens."

## Setup

- **Student model**: gemma-3-4b-pt (LoRA fine-tuned)
- **Teacher model**: Llama-4-Scout-17B-16E-Instruct (pre-generated reasoning traces from HuggingFace)
- **Language pair**: English → Xhosa (low-resource)
- **Evaluation set**: FLORES+ devtest (200 sentences)
- **Training**: 5000 steps, LoRA (r=32, alpha=64), batch_size=4, grad_accum=16, lr=1e-5, constant LR with 500-step warmup
- **Hardware**: 1x NVIDIA H100 80GB

### Training Datasets

| Mode | Dataset | Input | Output |
|------|---------|-------|--------|
| **IoFT** | `almanach/topxgen-llama-4-scout-MAPS` (split: MAPS) | `source` (English) | `translation` (Xhosa) |
| **CoTFT-CoT** | `almanach/topxgen-llama-4-scout-CoT` (split: CoT_T1) | `source` (English) | `target` (reasoning + Xhosa) |
| **CoTFT-MAPS** | `almanach/topxgen-llama-4-scout-MAPS` (split: MAPS) | `source` (English) | `target` (reasoning + Xhosa) |

### Fine-Tuning Methods

- **IoFT (Input-Output Fine-Tuning)**: Standard fine-tuning on English→Xhosa translation pairs. No intermediate reasoning.
- **CoTFT-CoT (Chain-of-Thought Fine-Tuning)**: Fine-tuning on English→(reasoning chain + Xhosa translation). Uses generic chain-of-thought reasoning traces.
- **CoTFT-MAPS (Modular Analysis of Parallel Structures)**: Fine-tuning on English→(structured MAPS reasoning + Xhosa translation). MAPS decomposes translation into modular linguistic analysis steps.

## Results

### Translation Quality (en→xho)

| Model | BLEU | chrF++ | COMET |
|-------|------|--------|-------|
| Baseline (no FT) | 1.16 | 14.27 | 0.3842 |
| IoFT | 6.52 | 31.29 | 0.6484 |
| CoTFT (CoT) | 4.32 | 28.47 | 0.6190 |
| **CoTFT (MAPS)** | **6.99** | **33.56** | **0.6655** |

### Training Loss (final)

| Model | Train Loss |
|-------|------------|
| IoFT | 0.766 |
| CoTFT (CoT) | 0.297 |
| CoTFT (MAPS) | 0.549 |

Training loss curves are available in `loss_curves.png`.

## Key Findings

1. **CoTFT-MAPS > IoFT > CoTFT-CoT** across all three metrics, consistent with the paper's findings.
2. **Structured reasoning (MAPS) helps**: CoTFT-MAPS outperforms IoFT by +0.47 BLEU, +2.27 chrF++, and +0.017 COMET, confirming that structured intermediate reasoning traces improve translation quality.
3. **Naive CoT hurts**: CoTFT-CoT performs worse than IoFT (-2.20 BLEU), consistent with the paper's observation that generic chain-of-thought reasoning does not help and can harm translation quality.
4. **Fine-tuning is essential**: The base gemma-3-4b-pt model produces near-zero quality Xhosa output (BLEU 1.16), with repetitive/hallucinated text and frequent English output.

## Comparison to Original Paper

Our absolute BLEU scores (~7) are lower than the paper's (~14-18). Possible explanations include differences in effective batch size, learning rate schedule details, or data preprocessing. However, the **relative ordering of methods is fully consistent** with the paper's core claim: structured reasoning traces (MAPS) improve translation quality, while generic CoT does not.

## Metrics

- **BLEU**: Exact n-gram overlap (sacrebleu)
- **chrF++**: Character n-gram F-score with word n-grams (sacrebleu)
- **COMET**: Neural MT evaluation metric (Unbabel/wmt22-comet-da). The paper uses MetricX-24, which was incompatible with our transformers version (4.57); COMET serves as a comparable neural metric.
