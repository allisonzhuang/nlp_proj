"""
Experiment: Qwen3-1.7B Thinking vs Non-Thinking Mode
Evaluates on en→fr and fr→en with BLEU, chrF++, and COMET.

Usage:
  python eval_thinking_1_7b.py --n_eval 50 --use_comet
  python eval_thinking_1_7b.py --model Qwen/Qwen3-0.6B  # to re-run 0.6B with COMET
"""

import argparse
import json
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_comet", action="store_true")
    parser.add_argument("--output_file", type=str, default="results_thinking_1_7b.json")
    return parser.parse_args()


def load_flores(src_lang, tgt_lang, n):
    """Load FLORES devtest data."""
    src_ds = load_dataset("openlanguagedata/flores_plus", src_lang, split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", tgt_lang, split="devtest")
    sources = [ex["text"] for ex in src_ds][:n]
    references = [ex["text"] for ex in tgt_ds][:n]
    return sources, references


def make_prompt(src_text, src_name, tgt_name, thinking):
    """Build translation prompt, optionally forcing non-thinking mode."""
    prompt = (
        f"Please write a high-quality {tgt_name} translation of the following {src_name} sentence.\n"
        f"\"{src_text}\"\n"
        "Please provide only the translation, nothing more.\n"
    )
    # For non-thinking mode, prefill empty think block
    if not thinking:
        prompt += "<think>\n\n</think>\n"
    return prompt


def extract_translation(text):
    """Extract clean translation from model output."""
    if not text:
        return ""

    text = text.strip()

    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Look for translation markers
    for marker in ['Final Translation:', 'Translation:', 'Final:']:
        if marker.lower() in text.lower():
            idx = text.lower().rfind(marker.lower())
            text = text[idx + len(marker):]
            break

    # Clean up
    text = text.strip()
    text = re.sub(r'^[\*\#\d\.\:\-]+\s*', '', text)  # Remove markdown artifacts
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = text.strip('"\'')

    # Take first line if multiple
    if '\n' in text:
        text = text.split('\n')[0].strip()

    return text


def translate_batch(model, tokenizer, sources, src_name, tgt_name, thinking, max_new_tokens, batch_size):
    """Batched translation with thinking mode control."""
    tokenizer.padding_side = 'left'
    translations = []
    raw_outputs = []

    for i in tqdm(range(0, len(sources), batch_size), desc=f"{'Thinking' if thinking else 'Non-Thinking'}"):
        batch = sources[i:i + batch_size]

        # Build prompts
        prompts = []
        for src in batch:
            user_content = make_prompt(src, src_name, tgt_name, thinking)
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
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

        # Decode only generated tokens
        prompt_width = inputs["input_ids"].shape[1]
        for row in outputs:
            generated = row[prompt_width:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            raw_outputs.append(decoded)
            translations.append(extract_translation(decoded))

    return translations, raw_outputs


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
        comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1)
        result["comet"] = round(comet_output.system_score, 4)

    return result


def main():
    args = get_args()

    # Language pairs to evaluate
    lang_pairs = [
        ("eng_Latn", "fra_Latn", "English", "French"),
        ("fra_Latn", "eng_Latn", "French", "English"),
    ]

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    all_results = []

    for src_lang, tgt_lang, src_name, tgt_name in lang_pairs:
        print(f"\n{'='*60}")
        print(f"{src_name} → {tgt_name}")
        print(f"{'='*60}")

        sources, references = load_flores(src_lang, tgt_lang, args.n_eval)
        print(f"Loaded {len(sources)} test sentences")

        for thinking in [True, False]:
            mode = "Thinking" if thinking else "Non-Thinking"
            print(f"\n--- {mode} Mode ---")

            translations, raw_outputs = translate_batch(
                model, tokenizer, sources, src_name, tgt_name,
                thinking, args.max_new_tokens, args.batch_size
            )

            scores = score_translations(
                translations, references, sources, use_comet=args.use_comet
            )

            print(f"  BLEU:   {scores['bleu']}")
            print(f"  chrF++: {scores['chrf']}")
            if "comet" in scores:
                print(f"  COMET:  {scores['comet']}")

            # Count think blocks
            n_think = sum(1 for r in raw_outputs if '<think>' in r.lower() or '</think>' in r.lower())
            print(f"  Outputs with <think> block: {n_think}/{len(raw_outputs)}")

            # Show examples
            print(f"\n  Examples:")
            for i in range(min(2, len(translations))):
                print(f"    SRC: {sources[i][:70]}...")
                print(f"    HYP: {translations[i][:70]}...")
                print(f"    REF: {references[i][:70]}...")
                print()

            all_results.append({
                "model": args.model,
                "src": src_name,
                "tgt": tgt_name,
                "mode": mode,
                "scores": scores,
                "n_outputs": len(translations),
                "n_with_think_block": n_think,
                "examples": [
                    {
                        "source": sources[i],
                        "reference": references[i],
                        "hypothesis": translations[i],
                        "raw_output": raw_outputs[i][:500],
                    }
                    for i in range(min(3, len(translations)))
                ]
            })

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")

    header = f"{'Direction':<15} {'Mode':<15} {'BLEU':>8} {'chrF++':>8}"
    if args.use_comet:
        header += f" {'COMET':>8}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        direction = f"{r['src']}→{r['tgt']}"
        row = f"{direction:<15} {r['mode']:<15} {r['scores']['bleu']:>8.2f} {r['scores']['chrf']:>8.2f}"
        if "comet" in r["scores"]:
            row += f" {r['scores']['comet']:>8.4f}"
        print(row)

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
