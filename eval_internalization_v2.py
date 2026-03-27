"""
Reasoning Internalization Test v2: Compare CoTFT models with and without
reasoning at inference time.

- Direct prompt: "Translate from English to Xhosa:\n{src}\nTranslation: "
- Reasoning prompt: same, but prefilled with "<think>\n" to trigger reasoning

If direct ≈ reasoning → reasoning is internalized (or not used)
If reasoning > direct → reasoning at inference helps (scaffolding)
If reasoning < direct → reasoning hurts (model can't reason well)
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

MODELS = [
    ("IoFT", "./checkpoints_v1/ioft/final"),
    ("CoTFT-CoT", "./checkpoints_v1/cotft_cot/final"),
    ("CoTFT-MAPS", "./checkpoints_v1/cotft_maps/final"),
]


def extract_translation(text):
    """Extract final translation, stripping any reasoning."""
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


def load_test_data(n_eval):
    src_ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", "xho_Latn", split="devtest")
    return [ex["text"] for ex in src_ds][:n_eval], [ex["text"] for ex in tgt_ds][:n_eval]


def translate(model, tokenizer, sources, prefill="", max_new_tokens=256, batch_size=8):
    """Translate with optional prefill to trigger reasoning."""
    tokenizer.padding_side = 'left'
    translations = []
    raw_outputs = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        prompts = [f"Translate from English to Xhosa:\n{s}\nTranslation: {prefill}" for s in batch]
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
            raw_outputs.append(decoded[:500])
            translations.append(extract_translation(decoded))
    return translations, raw_outputs


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

    for label, ckpt_dir in MODELS:
        tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ckpt_dir)
        model.eval()

        for mode, prefill, max_tokens in [
            ("direct", "", 256),
            ("reasoning", "<think>\n", 2048),
        ]:
            run_label = f"{label} ({mode})"
            print(f"\n{'='*60}")
            print(f"Evaluating: {run_label}")
            print(f"{'='*60}")

            hyps, raws = translate(model, tokenizer, sources,
                                   prefill=prefill, max_new_tokens=max_tokens)

            bleu, chrf, comet = score(hyps, references, sources, comet_model)
            print(f"  BLEU:   {bleu}")
            print(f"  chrF++: {chrf}")
            print(f"  COMET:  {comet}")

            # Check if model actually reasoned
            n_think = sum(1 for r in raws if '</think>' in r)
            print(f"  Outputs with </think>: {n_think}/{len(raws)}")

            for i in range(2):
                print(f"\n  SRC: {sources[i][:90]}")
                print(f"  RAW: {raws[i][:150]}")
                print(f"  HYP: {hyps[i][:90]}")
                print(f"  REF: {references[i][:90]}")

            all_results.append({
                "label": run_label,
                "model": label,
                "mode": mode,
                "bleu": bleu,
                "chrf": chrf,
                "comet": comet,
                "n_think_closed": n_think,
                "translations": hyps[:10],
                "raw_samples": raws[:5],
            })

        del model, base_model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("INTERNALIZATION TEST v2")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'BLEU':>8} {'chrF++':>8} {'COMET':>8} {'Think?':>8}")
    print("-" * 70)
    for r in all_results:
        think = f"{r['n_think_closed']}/200"
        print(f"{r['label']:<35} {r['bleu']:>8.2f} {r['chrf']:>8.2f} {r['comet']:>8.4f} {think:>8}")

    with open("results_internalization_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to results_internalization_v2.json")


if __name__ == "__main__":
    main()
