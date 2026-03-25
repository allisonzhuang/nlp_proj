echo "SLURM_ARRAY_TASK_ID is: $SLURM_ARRAY_TASK_ID"
list_of_methodeees=(sbys maps refine tear paraphrase)
list_of_codes=(SBYS MAPS REFINE TEAR COMPTRA)
DATASET_NAME=${list_of_methodeees[($SLURM_ARRAY_TASK_ID - 1)]}
CODE_DS=${list_of_codes[($SLURM_ARRAY_TASK_ID - 1)]}
echo $DATASET_NAME
echo $CODE_DS

ARGS="\
    --model_name_or_path ${PREFIX}${M6}\
    --tokenizer_name_or_path ${PREFIX}${M6}\
    --dataset_name_or_path ${DATASET_NAME}\
    --input_column_name source\
    --output_column_name target\
    --max_length 2048\
    --max_steps 5000\
    --per_device_train_batch_size 4\
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 16\
    --lora_r 32\
    --lora_alpha 64\
    --lora_dropout 0.05\
    --target_modules q_proj k_proj v_proj o_proj\
    --learning_rate 1e-5\
    --lr_scheduler_type constant_with_warmup\
    --min_learning_rate 5e-6\
    --num_warmup_steps 500\
    --weight_decay 0.01\
    --bf16\
    --seed 122\
    --output_dir .../NOREVERSE/LLAMA/T=1.0/${CODE_DS}GEMMA_27B/checkpoints-${T6}-full-xho-wiki\
    --logging_steps 100\
    --eval_steps 200\
    --save_steps 200\
    --src English\
    --target_languages Xhosa\
    --dataset_size -1\
    --strategy soonest\
    --targets_only\
    --data_dir .../data/Llama-4-Scout-17B-16E-Instruct/T=1.0/gemma-3-27b-it\
    --gradient_checkpointing\
    --test_size_ratio 1000\
    --use_flash_attn\
    "