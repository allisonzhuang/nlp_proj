M6="google/gemma-3-4b-pt"
T6="gemma-3-4b"
PREFIX=""

ARGS="\
    --model_name_or_path .../NOREVERSE/LLAMA/T=1.0/COMPTRALLAMA/checkpoints-${T6}-full-xho-wiki/checkpoint-5000\
    --tokenizer_name_or_path ${PREFIX}${M6}\
    --dataset_name_or_path almanach/topxgen-gemma-3-27b-and-nllb-3.3b\
    --split Xhosa\
    --size_valid_set 100\
    --input_column_name source\
    --output_column_name target\
    --max_length 2048\
    --max_steps 5000\
    --per_device_train_batch_size 4\
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 4\
    --lora_r 32\
    --lora_alpha 64\
    --lora_dropout 0.05\
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj\
    --learning_rate 1e-6\
    --lr_scheduler_type constant_with_warmup\
    --num_warmup_steps 100\
    --weight_decay 0.01\
    --bf16\
    --seed 122\
    --output_dir COMPTRA/checkpoints-${T6}-full-xho-wiki-grpo\
    --logging_steps 10\
    --eval_steps 2000\
    --save_steps 200\
    --src English\
    --target_languages Xhosa\
    --dataset_size -1\
    --strategy soonest\
    --targets_only\
    --gradient_checkpointing\
    --test_size_ratio 1000\
    --use_flash_attn\
    --temperature 1.0\
    --top_p 1.0\
    --repetition_penalty 1.00\
    --max_prompt_length 128\
    --max_completion_length 1536\
    --num_generations 12\
    --generation_batch_size 48\
    --beta 0.02\
    --max_grad_norm 1.0\
    --use_liger_loss\
    --use_grpo\
    --use_format_reward\
    --use_comptra\
    --use_peft\
"

CONFIG_FILE="configs/deepspeed_zero3_multi.yaml"
accelerate launch\
    --main_process_port $MASTER_PORT\
    --config_file $CONFIG_FILE\
    --num_processes=$(($GPUS_PER_NODE - 1))\
    train.py\
    $ARGS