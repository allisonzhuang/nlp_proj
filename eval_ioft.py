"""
Evaluate IoFT checkpoint with BLEU, chrF++, and MetricX-24.
Uses the MT5ForRegression class from the paper's codebase.
"""

import sys
import os
import json
import re
import torch
import numpy as np
import transformers
from datasets import load_dataset, Dataset
from peft import PeftModel
import evaluate

# Add the paper's codebase to path for MetricX model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm-reasoning-mt", "comptra", "evaluate", "metricx24"))
from models import MT5ForRegression


def load_test_data(n_eval):
    src_ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", "xho_Latn", split="devtest")
    sources = [ex["text"] for ex in src_ds][:n_eval]
    references = [ex["text"] for ex in tgt_ds][:n_eval]
    return sources, references


def extract_translation(text):
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    markers = ['Final translation:', 'Translation:', 'Final Translation:']
    lower_text = text.lower()
    for marker in markers:
        if marker.lower() in lower_text:
            idx = lower_text.rfind(marker.lower())
            text = text[idx + len(marker):].strip()
            break
    for line in text.split('\n'):
        line = line.strip()
        if line:
            text = line
            break
    text = re.sub(r'^[\*\#\d\.\:\-]+\s*', '', text)
    text = text.strip('"\'')
    return text.strip()


def translate_batch(model, tokenizer, sources, max_new_tokens=256, batch_size=8):
    tokenizer.padding_side = 'left'
    translations = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        prompts = [f"Translate from English to Xhosa:\n{s}\nTranslation: " for s in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for row, prompt_len in zip(outputs, input_lengths):
            generated = row[int(prompt_len):]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            translations.append(extract_translation(decoded))
    return translations


def compute_metricx(hypotheses, references, batch_size=16):
    """Compute MetricX-24 scores using the paper's MT5ForRegression model."""
    model_name = "google/metricx-24-hybrid-large-v2p6-bfloat16"
    print(f"Loading MetricX-24 model: {model_name}")

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-large")
    model = MT5ForRegression.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Format inputs: "candidate: {hyp} reference: {ref}"
    inputs_text = [
        f"candidate: {h} reference: {r}"
        for h, r in zip(hypotheses, references)
    ]

    # Score each example by running encoder + decoder manually
    # to avoid causal mask issues in newer transformers
    scores = []
    model.eval()
    for text in inputs_text:
        tok = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)

        with torch.no_grad():
            # Run encoder
            encoder_outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = encoder_outputs.last_hidden_state

            # 1-token dummy decoder input
            decoder_input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

            # Run decoder with explicit cross-attention
            decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )
            sequence_output = decoder_outputs.last_hidden_state

            if model.config.tie_word_embeddings:
                sequence_output = sequence_output * (model.model_dim ** -0.5)

            lm_logits = model.lm_head(sequence_output)
            # 250089 = <extra_id_10>
            prediction = torch.clamp(lm_logits[:, 0, 250089], 0, 25)
            scores.append(float(prediction.item()))

    # MetricX: lower is better, range 0-25
    mean_score = np.mean(scores)
    print(f"MetricX-24 mean score: {mean_score:.4f} (lower is better)")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return mean_score, scores


def main():
    n_eval = 200
    base_model_name = "google/gemma-3-4b-pt"
    checkpoint_dir = "./checkpoints/ioft/final"

    print("Loading test data...")
    sources, references = load_test_data(n_eval)
    print(f"Loaded {len(sources)} test sentences (en→xho)")

    # Load IoFT model
    print(f"\nLoading IoFT model from {checkpoint_dir}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()

    # Translate
    print("Translating...")
    hypotheses = translate_batch(model, tokenizer, sources)

    # Free translation model memory
    del model, base_model
    torch.cuda.empty_cache()

    # BLEU and chrF++
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    bleu_score = bleu.compute(predictions=hypotheses, references=[[r] for r in references])["score"]
    chrf_score = chrf.compute(predictions=hypotheses, references=[[r] for r in references], word_order=2)["score"]

    print(f"\n=== IoFT Results (en→xho, n={n_eval}) ===")
    print(f"BLEU:    {bleu_score:.2f}")
    print(f"chrF++:  {chrf_score:.2f}")

    # COMET
    from comet import download_model, load_from_checkpoint
    print("Loading COMET model...")
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)
    comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    comet_output = comet_model.predict(comet_data, batch_size=8)
    comet_score = comet_output.system_score
    print(f"COMET:   {comet_score:.4f}")

    # Show samples
    print(f"\nSample translations:")
    for i in range(min(5, len(hypotheses))):
        print(f"  SRC: {sources[i][:100]}")
        print(f"  HYP: {hypotheses[i][:100]}")
        print(f"  REF: {references[i][:100]}")
        print()

    # Save
    results = {
        "model": "IoFT",
        "n_eval": n_eval,
        "bleu": round(bleu_score, 2),
        "chrf": round(chrf_score, 2),
        "comet": round(comet_score, 4),
        "translations": hypotheses[:10],
    }
    with open("results_ioft_eval.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved to results_ioft_eval.json")


if __name__ == "__main__":
    main()
