"""
Evaluate all saved checkpoints for a fine-tuning run to produce BLEU-over-steps curves.

Usage:
  python eval_training_curve.py --checkpoint_base ./checkpoints/ioft --label IoFT
  python eval_training_curve.py --checkpoint_base ./checkpoints/cotft_cot --label "CoTFT (CoT)"
  python eval_training_curve.py --checkpoint_base ./checkpoints/cotft_maps --label "CoTFT (MAPS)"
"""

import argparse
import json
import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_base", type=str, required=True,
                        help="Base checkpoint directory (e.g., ./checkpoints/ioft)")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--label", type=str, default="model")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="Number of test sentences (smaller for speed)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file (default: <checkpoint_base>/training_curve.json)")
    return parser.parse_args()


def load_test_data(n_eval):
    src_ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", "xho_Latn", split="devtest")
    sources = [ex["text"] for ex in src_ds][:n_eval]
    references = [ex["text"] for ex in tgt_ds][:n_eval]
    return sources, references


def extract_translation(text):
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    markers = ['Final translation:', 'Translation:', '[Translation]:',
               'Final Translation:', 'final translation:']
    lower_text = text.lower()
    for marker in markers:
        lower_marker = marker.lower()
        if lower_marker in lower_text:
            idx = lower_text.rfind(lower_marker)
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


def build_prompt(source):
    return f"Translate from English to Xhosa:\n{source}\nTranslation: "


def translate_batch(model, tokenizer, sources, max_new_tokens, batch_size):
    tokenizer.padding_side = 'left'
    translations = []

    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        prompts = [build_prompt(s) for s in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for row, prompt_len in zip(outputs, input_lengths):
            generated = row[int(prompt_len):]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            translations.append(extract_translation(decoded))

    return translations


def score_translations(hypotheses, references):
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    bleu_score = bleu.compute(
        predictions=hypotheses,
        references=[[r] for r in references],
    )["score"]

    chrf_score = chrf.compute(
        predictions=hypotheses,
        references=[[r] for r in references],
        word_order=2,
    )["score"]

    return round(bleu_score, 2), round(chrf_score, 2)


def find_checkpoints(checkpoint_base):
    """Find all checkpoint-XXXX directories and sort by step number."""
    checkpoints = []
    if not os.path.isdir(checkpoint_base):
        return checkpoints

    for name in os.listdir(checkpoint_base):
        match = re.match(r'checkpoint-(\d+)', name)
        if match:
            step = int(match.group(1))
            path = os.path.join(checkpoint_base, name)
            checkpoints.append((step, path))

    # Also check for "final" directory
    final_path = os.path.join(checkpoint_base, "final")
    if os.path.isdir(final_path):
        # Infer step count from the highest checkpoint
        max_step = max((s for s, _ in checkpoints), default=5000)
        checkpoints.append((max_step, final_path))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def main():
    args = get_args()

    if args.output_file is None:
        args.output_file = os.path.join(args.checkpoint_base, "training_curve.json")

    # Find all checkpoints
    checkpoints = find_checkpoints(args.checkpoint_base)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_base}")
        return

    print(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    # Load test data
    print(f"Loading test data ({args.n_eval} sentences)...")
    sources, references = load_test_data(args.n_eval)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    for step, ckpt_path in checkpoints:
        print(f"\n--- Evaluating step {step} ({ckpt_path}) ---")

        # Load base model + LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model.eval()

        hyps = translate_batch(model, tokenizer, sources, args.max_new_tokens, args.batch_size)
        bleu, chrf = score_translations(hyps, references)

        print(f"  Step {step}: BLEU={bleu}, chrF++={chrf}")

        results.append({
            "step": step,
            "bleu": bleu,
            "chrf": chrf,
            "samples": list(zip(sources[:3], hyps[:3], references[:3])),
        })

        # Free memory
        del model, base_model
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*50}")
    print(f"Training Curve: {args.label}")
    print(f"{'='*50}")
    print(f"{'Step':>8} {'BLEU':>8} {'chrF++':>8}")
    print("-" * 26)
    for r in results:
        print(f"{r['step']:>8} {r['bleu']:>8.2f} {r['chrf']:>8.2f}")

    # Save
    output = {
        "label": args.label,
        "base_model": args.base_model,
        "n_eval": args.n_eval,
        "results": results,
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output_file}")


if __name__ == "__main__":
    main()
