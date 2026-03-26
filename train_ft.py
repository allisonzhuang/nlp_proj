"""
Fine-tuning script for IoFT / CoTFT experiments.
Replicates Zebaze et al. (2025) setup: LoRA on gemma-3-4b-pt, en→xho, 5000 steps.

Usage:
  python train_ft.py --mode ioft --output_dir ./checkpoints/ioft
  python train_ft.py --mode cotft_cot --output_dir ./checkpoints/cotft_cot
  python train_ft.py --mode cotft_maps --output_dir ./checkpoints/cotft_maps
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["ioft", "cotft_cot", "cotft_maps"],
                        help="Training mode")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--eval_size", type=int, default=1000,
                        help="Number of examples to hold out for validation")
    return parser.parse_args()


def load_data(mode, eval_size, seed):
    """Load the appropriate dataset and return (train, eval) splits."""
    if mode == "cotft_maps":
        ds = load_dataset("almanach/topxgen-llama-4-scout-MAPS", split="MAPS")
        input_col, output_col = "source", "target"
    elif mode == "cotft_cot":
        ds = load_dataset("almanach/topxgen-llama-4-scout-CoT", split="CoT_T1")
        input_col, output_col = "source", "target"
    else:  # ioft - use the MAPS dataset but only source/translation columns
        ds = load_dataset("almanach/topxgen-llama-4-scout-MAPS", split="MAPS")
        input_col, output_col = "source", "translation"

    # Keep only the columns we need
    ds = ds.map(lambda x: {"input": x[input_col], "output": x[output_col]},
                remove_columns=ds.column_names)

    # Filter out empty outputs
    ds = ds.filter(lambda x: len(x["output"].strip()) > 0)

    # Split into train/eval
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]


def tokenize_example(example, tokenizer, max_length):
    """Format as: 'Translate from English to Xhosa:\n{source}\nTranslation: {output}<eos>'"""
    input_text = example["input"]
    output_text = example["output"]

    # Build full text
    prompt = f"Translate from English to Xhosa:\n{input_text}\nTranslation: "
    full_text = prompt + output_text + tokenizer.eos_token

    # Tokenize
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length)

    # Create labels: mask the prompt tokens with -100 so loss is only on the output
    prompt_tokenized = tokenizer(prompt, truncation=True, max_length=max_length)
    prompt_len = len(prompt_tokenized["input_ids"])

    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels

    return tokenized


def main():
    args = get_args()
    set_seed(args.seed)

    # For CoTFT, we need longer sequences to fit the reasoning traces
    # Paper uses grad_accum=4 for IoFT, 16 for CoTFT
    if args.mode.startswith("cotft"):
        args.max_length = max(args.max_length, 2048)
        args.grad_accum = 16

    print(f"=== Fine-tuning: {args.mode} ===")
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} grad accum")
    print()

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # needed for gradient checkpointing + LoRA
    model.print_trainable_parameters()

    # Load data
    print("Loading dataset...")
    train_ds, eval_ds = load_data(args.mode, args.eval_size, args.seed)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Tokenize
    print("Tokenizing...")
    train_ds = train_ds.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
        remove_columns=["input", "output"],
        num_proc=4,
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
        remove_columns=["input", "output"],
        num_proc=4,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=500,
        weight_decay=0.01,
        bf16=True,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=None,
        seed=args.seed,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
    )

    # Data collator
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # Resume from checkpoint if one exists
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        checkpoints = [
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from {last_checkpoint}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model
    print(f"Saving to {args.output_dir}/final")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Done!")


if __name__ == "__main__":
    main()
