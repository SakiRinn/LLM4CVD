#!/bin/bash
DATASET_NAME=$1
MODEL_NAME=$2
LENGTH=$3
BATCH_SIZE=$4
CUDA=${5:-"0"}

# Check if the first three parameters are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <LENGTH> <BATCH_SIZE> [CUDA]"
    exit 1
fi

if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "Bash version 4.0 or higher is required."
    exit 1
fi

declare -A MODEL_MAP
MODEL_MAP["llama3"]='meta-llama/Meta-Llama-3-8B'
MODEL_MAP["llama3.1"]='meta-llama/Llama-3.1-8B'
MODEL_MAP["llama2"]="meta-llama/Llama-2-7b-hf"
MODEL_MAP["codellama"]="codellama/CodeLlama-7b-hf"

mkdir -p "outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/"

echo "Batch size: ${BATCH_SIZE}"
echo "Length: $(echo $LENGTH | awk -F'-' '{print $2}')"

CUDA_VISIBLE_DEVICES="${CUDA}" python LLM/finetuning.py \
    --quantization \
    --use_peft \
    --peft_method lora \
    --batch_size_training $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --context_length $(echo $LENGTH | awk -F'-' '{print $2}') \
    --num_epochs 5 \
    --model_name ${MODEL_MAP[$MODEL_NAME]} \
    --alpaca_dataset.train_data_path "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --alpaca_dataset.valid_data_path "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --output_dir "outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/" \
    >"outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/finetuning_${MODEL_NAME}_lora_${DATASET_NAME}_${LENGTH}.log"
