ARGS="\
    --model_name_or_path $MODEL_NAME_OR_PATH\
    --tokenizer_name_or_path $TOKENIZER_NAME_OR_PATH\
    --src $SRC\
    --tgt $TGT\
    --request_batch_size 64\
    --inference_api vllm\
    --max_samples 10000\
    --num_return_sequences 1\
    --num_beams 1\
    --max_new_tokens 2768\
    --temperature 0.0\
    --top_p 1.0\
    --repetition_penalty 1.0\
    --output_dir .../bm25s/GEMMA-3-1B/NOREVERSE/GEMMA/T=1.0/GREEDY/COMPTRA\
    --k $K\
    --seed $SEED\
    --method_divide $METHOD_DIVIDE\
    --merge_prompt $MERGE_PROMPT\
    --method_translate vanilla\
    --selection_method greedy\
    --steps $STEPS\
    --verbose\
    --number_of_subproblems $NUMBER_OF_SUBPROBLEMS\
    --number_of_refining_steps $NUMBER_OF_REFINING_STEPS\
    --template_key 14\
    --retriever_type bm25s\
    --dataset_name_or_path flores\
    --number_of_merge_demonstrations 0\
    --nllb_name_or_path $MODEL_NAME_OR_PATH\
    "

# --enable_lora\
# --lora_rank 32\
# --base_model_name_or_path .../NOREVERSE/LLAMA/T=1.0/COMPTRALLAMA/checkpoints-gemma-3-4b-full-xho-wiki/checkpoint-5000\