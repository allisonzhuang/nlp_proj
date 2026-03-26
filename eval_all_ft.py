"""
Evaluate all fine-tuned models (baseline, IoFT, CoTFT-CoT, CoTFT-MAPS) with BLEU, chrF++, COMET.
"""

import json
import re
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
import evaluate
from comet import download_model, load_from_checkpoint


N_EVAL = 200
BASE_MODEL = "google/gemma-3-4b-pt"
CHECKPOINTS = [
    ("Baseline (no FT)", None),
    ("IoFT", "./checkpoints/ioft/final"),
    ("CoTFT (CoT)", "./checkpoints/cotft_cot/final"),
    ("CoTFT (MAPS)", "./checkpoints/cotft_maps/final"),
]


def load_test_data(n_eval):
    src_ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", "xho_Latn", split="devtest")
    return [ex["text"] for ex in src_ds][:n_eval], [ex["text"] for ex in tgt_ds][:n_eval]


def extract_translation(text):
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    markers = ['Final translation:', 'Translation:', 'Final Translation:']
    lower = text.lower()
    for m in markers:
        if m.lower() in lower:
            idx = lower.rfind(m.lower())
            text = text[idx + len(m):].strip()
            break
    for line in text.split('\n'):
        line = line.strip()
        if line:
            text = line
            break
    text = re.sub(r'^[\*\#\d\.\:\-]+\s*', '', text)
    return text.strip('"\'').strip()


def translate(model, tokenizer, sources, max_new_tokens=256, batch_size=8):
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
        for row, pl in zip(outputs, input_lengths):
            decoded = tokenizer.decode(row[int(pl):], skip_special_tokens=True)
            translations.append(extract_translation(decoded))
    return translations


def score(hypotheses, references, sources, comet_model):
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    b = bleu.compute(predictions=hypotheses, references=[[r] for r in references])["score"]
    c = chrf.compute(predictions=hypotheses, references=[[r] for r in references], word_order=2)["score"]
    comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    comet_out = comet_model.predict(comet_data, batch_size=8)
    return round(b, 2), round(c, 2), round(comet_out.system_score, 4)


def main():
    print("Loading test data...")
    sources, references = load_test_data(N_EVAL)
    print(f"Loaded {len(sources)} sentences (en→xho)")

    print("Loading COMET model...")
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

    all_results = []

    for label, ckpt_dir in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        # Load model
        tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
        )

        if ckpt_dir is not None:
            model = PeftModel.from_pretrained(base_model, ckpt_dir)
        else:
            model = base_model
        model.eval()

        # Translate
        print("Translating...")
        hyps = translate(model, tokenizer, sources)

        # Score
        bleu, chrf, comet = score(hyps, references, sources, comet_model)
        print(f"  BLEU:   {bleu}")
        print(f"  chrF++: {chrf}")
        print(f"  COMET:  {comet}")

        # Samples
        for i in range(3):
            print(f"\n  SRC: {sources[i][:90]}")
            print(f"  HYP: {hyps[i][:90]}")
            print(f"  REF: {references[i][:90]}")

        all_results.append({
            "label": label,
            "bleu": bleu,
            "chrf": chrf,
            "comet": comet,
            "translations": hyps[:10],
        })

        del model, base_model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Fine-Tuning Results (en→xho)")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'BLEU':>8} {'chrF++':>8} {'COMET':>8}")
    print("-" * 46)
    for r in all_results:
        print(f"{r['label']:<20} {r['bleu']:>8.2f} {r['chrf']:>8.2f} {r['comet']:>8.4f}")

    with open("results_ft_all.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to results_ft_all.json")


if __name__ == "__main__":
    main()
