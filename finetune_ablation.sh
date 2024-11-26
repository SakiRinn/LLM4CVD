#!/bin/bash
DATASET_NAME=$1
MODEL_NAME=$2
R=$3
ALPHA=$4
CUDA=${5:-"0"}

# Check if the first three parameters are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <R> <ALPHA> [CUDA]"
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

declare -A BATCH_MAP
BATCH_MAP["llama2"]=20
BATCH_MAP["codellama"]=20
BATCH_MAP["llama3"]=16
BATCH_MAP["llama3.1"]=16

mkdir -p "outputs/${MODEL_NAME}_lora_ablation/${DATASET_NAME}_${R}_${ALPHA}/"

echo "Batch size: ${BATCH_MAP[$MODEL_NAME]}"
echo "R: $(echo $R)"
echo "Alpha: $(echo $ALPHA)"

CUDA_VISIBLE_DEVICES="${CUDA}" python CodeLlama/finetuning.py \
    --quantization \
    --use_peft \
    --peft_method lora \
    --batch_size_training ${BATCH_MAP[$MODEL_NAME]} \
    --val_batch_size ${BATCH_MAP[$MODEL_NAME]} \
    --context_length 512 \
    --num_epochs 5 \
    --model_name ${MODEL_MAP[$MODEL_NAME]} \
    --alpaca_dataset.train_data_path "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_train.json" \
    --alpaca_dataset.valid_data_path "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_validate.json" \
    --lora_config.r $R \
    --lora_config.lora_alpha $ALPHA \
    --output_dir "outputs/${MODEL_NAME}_lora_ablation/${DATASET_NAME}_${R}_${ALPHA}/" \
    >"outputs/${MODEL_NAME}_lora_ablation/${DATASET_NAME}_${R}_${ALPHA}/finetuning_${MODEL_NAME}_lora_ablation_${DATASET_NAME}_${R}_${ALPHA}.log"
