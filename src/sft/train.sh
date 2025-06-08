
train_data_file=trendmicro-ailab/Primus-Instruct
model_path=trendmicro-ailab/Llama-Primus-Base
ds_config_file=ds_zero2_no_offload.json
output_path=./hf_train_output

mkdir -p ${output_path}
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_path}/"log_${current_time}.txt"

deepspeed train.py \
    --do_train \
    --model_name_or_path ${model_path} \
    --data_path ${train_data_file} \
    --deepspeed ${ds_config_file} \
    --output_dir ${output_path} \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --num_train_epochs 2 \
    --save_steps 500 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.01 \
    --save_strategy steps \
    --save_safetensors False \
    --use_lora True\
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --model_max_length 4096 \
    --max_seq_length 4096 \
    --bf16 | tee ${log_file}
