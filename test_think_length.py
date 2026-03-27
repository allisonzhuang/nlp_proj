"""Quick test: how many tokens does Qwen3-1.7B need for thinking on 5 sentences."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

src_ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
sources = [src_ds[i]["text"] for i in [0, 10, 50, 100, 150]]

for max_tokens in [1024, 2048, 4096]:
    print(f"\n=== max_new_tokens={max_tokens} ===")
    for j, src in enumerate(sources):
        prompt = (
            f'Please write a high-quality French translation of the following English sentence.\n'
            f'"{src[:200]}"\n'
            f'Please provide only the translation, nothing more.\n'
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        has_close = "</think>" in gen
        if has_close:
            think_part = gen.split("</think>")[0]
            think_toks = len(tokenizer.encode(think_part))
            trans = gen.split("</think>")[-1].strip().split("\n")[0][:80]
        else:
            think_toks = gen_tokens
            trans = "[TRUNCATED]"
        print(f"  sent={j} tokens={gen_tokens:>5} think={think_toks:>5} closed={has_close} => {trans}")
