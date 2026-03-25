ARGS="\
    --model_name_or_path google/gemma-3-27b-it\
    --tokenizer_name_or_path google/gemma-3-27b-it\
    --inference_api vllm\
    --request_batch_size 2048\
    --seed 122\
    --max_new_tokens 2000\
    --temperature 0.0\
    --top_p 1.0\
    --repetition_penalty 1.0\
    --num_return_sequences 1\
    --num_beams 1\
    --verbose\
    --languages Xhosa\
    --input_filenames Xhosa.jsonl\
    --input_dir data/Llama-4-Scout-17B-16E-Instruct\
    --number_of_demonstrations 5\
    --source_language English\
    --number_of_generations_per_step 5\
   "
# --cot_template 1\