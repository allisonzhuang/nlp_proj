ARGS="
    --model_name_or_path google/metricx-24-hybrid-xxl-v2p6\
    --dataset_name_or_path flores\
    --max_input_length 1024\
    --batch_size 1\
    --data_dir /GENERATIONS/FLORES/TRAINING\
    --number_of_predictions 1012\
    --seed 122\
    --num_workers 8\
    --metric metricx\
    --languages Xhosa\
    --strategies ... \
    --names checkpoint-200 checkpoint-400 checkpoint-600 checkpoint-800 checkpoint-1000 checkpoint-1200 checkpoint-1400 checkpoint-1600 checkpoint-1800 checkpoint-2000 checkpoint-2200 checkpoint-2400 checkpoint-2600 checkpoint-2800 checkpoint-3000 checkpoint-3200 checkpoint-3400 checkpoint-3600 checkpoint-3800 checkpoint-4000 checkpoint-4200 checkpoint-4400 checkpoint-4600 checkpoint-4800 checkpoint-5000\
"
# comptra/evaluate/test.py