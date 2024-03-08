#!/bin/bash
# Below script is for running on SLURM-based systems
# Setup according to your system

# Activate your python virtual environment
source /path/to/python/virtual/env/bin/activate

MODEL_NAME_OR_PATH="/path/to/llama2_hf/llama2-7b-hf"
TOKENIZER_PATH="/path/to/tokenizer/LLM500_tokenizer"
ADPT_STRATEGY="lora"
EMBD_SRATEGY="extend"

cache_dir="/path/to/cache_llama2/huggingface"
output_dir="/path/to/model/$(basename $MODEL_NAME_OR_PATH)_${ADPT_STRATEGY}_${EMBD_SRATEGY}"
logging_dir="/path/to/model/reports/$(basename $MODEL_NAME_OR_PATH)_${ADPT_STRATEGY}_${EMBD_SRATEGY}"
proj_dir="/path/to/mala-500"
data_dir="/path/to/data"
mkdir -p $output_dir
mkdir -p $logging_dir

export TRANSFORMERS_CACHE=/path/to/cache_llama2/huggingface/transformers
export HF_DATASETS_CACHE=/path/to/cache_llama2/huggingface/datasets
export HF_MODULES_CACHE=/path/to/cache_llama2/huggingface/modules
export TORCH_EXTENSIONS_DIR=$proj_dir/torch_extensions/dev

# NCCL variables
NCCL_SOCKET_IFNAME=hsn
# export NCCL_DEBUG=INFO

# Set the number of threads based on --cpus-per-task
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export RDZV_HOST=$(hostname)
export RDZV_PORT=29144
echo "Launching on $RDZV_HOST:$RDZV_PORT"

CMD=" \
    --nnode=$SLURM_JOB_NUM_NODES --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    $proj_dir/lang_adapt/continued_clm.py --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $TOKENIZER_PATH \
    --block_size 4096 \
    --train_file $data_dir/Glot500.txt \
    --cache_dir $cache_dir \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --bf16 True \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --do_train \
    --output_dir $output_dir \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --num_train_epochs 3 \
    --save_steps 500 \
    --logging_steps 500 \
    --save_total_limit 300 \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --tie_word_embeddings 0 \
    --trainable "q_proj,v_proj" \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --modules_to_save "embed_tokens,lm_head" \
    --gradient_checkpointing True \
    --deepspeed $proj_dir/continued_pretained/config/ds3_param_optim_bf16.json
    "

echo $CMD

srun python3 -m torch.distributed.run ${CMD}
