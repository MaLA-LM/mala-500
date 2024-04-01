MODEL=${1:-bigscience/bloom-560m}
GPU=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export TRANSFORMERS_CACHE=/PATH/huggingface/cache/

while IFS=$'\t' read -r lang is_seen; do
    python ../lm-evaluation-harness/main.py \
        --model_api_name 'hf-causal' \
        --model_args use_accelerate=True,pretrained=$MODEL \
        --task_name 'taxi_test' \
        --task_args data_dir="/PATH/huggingface/datasets/taxi/$lang",cache_dir='/PATH/huggingface/cache/',download_mode='load_from_disk' \
        --output_dir '/PATH/LLM500/llama2_eval_output/lm-evaluation-harness/' \
        --template_names 'all_templates' \
        --num_fewshot 3 \
        --batch_size 2 \
        --device cuda:0
done < ../lang_list/taxi_lang_list.txt

