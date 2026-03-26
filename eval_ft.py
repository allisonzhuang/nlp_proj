"""
Evaluation script for fine-tuned models (IoFT / CoTFT).
Translates FLORES devtest en→xho and scores with BLEU, chrF++, COMET.

Usage:
  python eval_ft.py --checkpoint_dirs ./checkpoints/ioft/final ./checkpoints/cotft_cot/final ./checkpoints/cotft_maps/final
  python eval_ft.py --include_baseline  # also evaluate the base model (no fine-tuning)
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
    parser.add_argument("--checkpoint_dirs", nargs="+", default=[
        "./checkpoints/ioft/final",
        "./checkpoints/cotft_cot/final",
        "./checkpoints/cotft_maps/final",
    ])
    parser.add_argument("--base_model", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Also evaluate the base model with no fine-tuning")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Number of test sentences")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_file", type=str, default="results_ft_eval.json")
    parser.add_argument("--use_comet", action="store_true",
                        help="Include COMET scoring (slower, needs GPU memory)")
    return parser.parse_args()


def load_test_data(n_eval):
    """Load FLORES devtest en→xho."""
    src_ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", "xho_Latn", split="devtest")
    sources = [ex["text"] for ex in src_ds][:n_eval]
    references = [ex["text"] for ex in tgt_ds][:n_eval]
    return sources, references


def extract_translation(text):
    """Extract the final translation from model output, handling CoTFT reasoning traces."""
    text = text.strip()

    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Look for translation markers
    markers = ['Final translation:', 'Translation:', '[Translation]:',
               'Final Translation:', 'final translation:']
    lower_text = text.lower()
    for marker in markers:
        lower_marker = marker.lower()
        if lower_marker in lower_text:
            idx = lower_text.rfind(lower_marker)
            text = text[idx + len(marker):].strip()
            break

    # Take first non-empty line
    for line in text.split('\n'):
        line = line.strip()
        if line:
            text = line
            break

    # Clean up markdown artifacts and quotes
    text = re.sub(r'^[\*\#\d\.\:\-]+\s*', '', text)
    text = text.strip('"\'')

    return text.strip()


def build_prompt(source):
    """Same prompt format used during training."""
    return f"Translate from English to Xhosa:\n{source}\nTranslation: "


def translate_batch(model, tokenizer, sources, max_new_tokens, batch_size):
    """Batched translation with left-padding for decoder-only models."""
    tokenizer.padding_side = 'left'
    translations = []

    for i in tqdm(range(0, len(sources), batch_size), desc="Translating"):
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

        # Decode only the generated tokens
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for row, prompt_len in zip(outputs, input_lengths):
            generated = row[int(prompt_len):]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            translations.append(extract_translation(decoded))

    return translations


def score_translations(hypotheses, references, sources=None, use_comet=False):
    """Compute BLEU, chrF++, and optionally COMET."""
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

    result = {
        "bleu": round(bleu_score, 2),
        "chrf": round(chrf_score, 2),
    }

    if use_comet and sources is not None:
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        comet_data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        comet_output = comet_model.predict(comet_data, batch_size=8)
        result["comet"] = round(comet_output.system_score, 4)

    return result


def evaluate_model(model, tokenizer, sources, references, args, label):
    """Run translation and scoring for one model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    hypotheses = translate_batch(
        model, tokenizer, sources, args.max_new_tokens, args.batch_size
    )

    scores = score_translations(
        hypotheses, references, sources, use_comet=args.use_comet
    )

    print(f"  BLEU:  {scores['bleu']}")
    print(f"  chrF++: {scores['chrf']}")
    if "comet" in scores:
        print(f"  COMET: {scores['comet']}")

    # Show a few examples
    print(f"\n  Sample translations:")
    for i in range(min(3, len(hypotheses))):
        print(f"    SRC: {sources[i][:80]}...")
        print(f"    HYP: {hypotheses[i][:80]}...")
        print(f"    REF: {references[i][:80]}...")
        print()

    return {
        "label": label,
        "scores": scores,
        "translations": hypotheses,
    }


def main():
    args = get_args()

    print("Loading test data...")
    sources, references = load_test_data(args.n_eval)
    print(f"Loaded {len(sources)} test sentences (en→xho)")

    all_results = []

    # Evaluate base model if requested
    if args.include_baseline:
        print("\nLoading base model...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        result = evaluate_model(
            model, tokenizer, sources, references, args, "baseline (no FT)"
        )
        all_results.append(result)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Evaluate each fine-tuned checkpoint
    for ckpt_dir in args.checkpoint_dirs:
        if not os.path.exists(ckpt_dir):
            print(f"\nSkipping {ckpt_dir} (not found)")
            continue

        # Infer label from path
        if "ioft" in ckpt_dir:
            label = "IoFT"
        elif "cotft_cot" in ckpt_dir:
            label = "CoTFT (CoT)"
        elif "cotft_maps" in ckpt_dir:
            label = "CoTFT (MAPS)"
        else:
            label = os.path.basename(os.path.dirname(ckpt_dir))

        print(f"\nLoading {label} from {ckpt_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model + LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ckpt_dir)
        model.eval()

        result = evaluate_model(
            model, tokenizer, sources, references, args, label
        )
        all_results.append(result)

        # Free memory
        del model, base_model
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: en→xho Fine-Tuning Results")
    print(f"{'='*60}")
    header = f"{'Model':<20} {'BLEU':>8} {'chrF++':>8}"
    if args.use_comet:
        header += f" {'COMET':>8}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        row = f"{r['label']:<20} {r['scores']['bleu']:>8.2f} {r['scores']['chrf']:>8.2f}"
        if "comet" in r["scores"]:
            row += f" {r['scores']['comet']:>8.4f}"
        print(row)

    # Save results (without full translations for readability)
    save_results = []
    for r in all_results:
        save_results.append({
            "label": r["label"],
            "scores": r["scores"],
            "translations": r["translations"][:10],  # save first 10 as samples
        })

    with open(args.output_file, "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
