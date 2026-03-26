#!/bin/bash
#SBATCH --job-name=eval-ioft-v2
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_ioft_v2_%j.log

set -e

source ~/nlp_proj/.venv/bin/activate
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

cd ~/nlp_proj

python -c "
import json, re, torch, transformers, evaluate
from datasets import load_dataset
from peft import PeftModel
from comet import download_model, load_from_checkpoint

N_EVAL = 200
BASE_MODEL = 'google/gemma-3-4b-pt'

# Load test data
src_ds = load_dataset('openlanguagedata/flores_plus', 'eng_Latn', split='devtest')
tgt_ds = load_dataset('openlanguagedata/flores_plus', 'xho_Latn', split='devtest')
sources = [ex['text'] for ex in src_ds][:N_EVAL]
references = [ex['text'] for ex in tgt_ds][:N_EVAL]
print(f'Loaded {len(sources)} sentences')

# Load COMET
comet_path = download_model('Unbabel/wmt22-comet-da')
comet_model = load_from_checkpoint(comet_path)

# Load IoFT v2 model
tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

base_model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base_model, './checkpoints/ioft/final')
model.eval()

# Translate
translations = []
for i in range(0, len(sources), 8):
    batch = sources[i:i+8]
    prompts = [f'Translate from English to Xhosa:\n{s}\nTranslation: ' for s in batch]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
    for row, pl in zip(outputs, input_lengths):
        decoded = tokenizer.decode(row[int(pl):], skip_special_tokens=True)
        translations.append(decoded.strip().split('\n')[0].strip())

# Score
bleu = evaluate.load('sacrebleu')
chrf = evaluate.load('chrf')
b = bleu.compute(predictions=translations, references=[[r] for r in references])['score']
c = chrf.compute(predictions=translations, references=[[r] for r in references], word_order=2)['score']
comet_data = [{'src': s, 'mt': h, 'ref': r} for s, h, r in zip(sources, translations, references)]
comet_out = comet_model.predict(comet_data, batch_size=8)

print(f'IoFT v2 Results:')
print(f'  BLEU:   {b:.2f}')
print(f'  chrF++: {c:.2f}')
print(f'  COMET:  {comet_out.system_score:.4f}')

for i in range(3):
    print(f'  SRC: {sources[i][:90]}')
    print(f'  HYP: {translations[i][:90]}')
    print(f'  REF: {references[i][:90]}')
"

echo "=== IoFT v2 eval complete ==="
