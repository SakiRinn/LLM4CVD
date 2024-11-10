#!/bin/bash
DATASET_NAME=$1
MODEL_NAME=$2
LENGTH=$3
CUDA=${4:-"0"}

# Check if the first three parameters are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <LENGTH> [CUDA]"
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
BATCH_MAP["llama2_0-512"]=20
BATCH_MAP["llama2_512-1024"]=10
BATCH_MAP["codellama_0-512"]=20
BATCH_MAP["codellama_512-1024"]=10
BATCH_MAP["llama3_0-512"]=16
BATCH_MAP["llama3_512-1024"]=8
BATCH_MAP["llama3.1_0-512"]=16
BATCH_MAP["llama3.1_512-1024"]=8

mkdir -p "outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/"

echo "Batch size: ${BATCH_MAP["${MODEL_NAME}_${LENGTH}"]}"
echo "Length: $(echo $LENGTH | awk -F'-' '{print $2}')"

if [[ "$MODEL_NAME" == "StarCoder" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/finetune.py \
    --quantization \
    --use_peft \
    --peft_method lora \
    --batch_size_training ${BATCH_MAP["${MODEL_NAME}_${LENGTH}"]} \
    --val_batch_size ${BATCH_MAP["${MODEL_NAME}_${LENGTH}"]} \
    --context_length $(echo $LENGTH | awk -F'-' '{print $2}') \
    --num_epochs 5 \
    --model_name ${MODEL_MAP[$MODEL_NAME]} \
    --train_data_path "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --valid_data_path "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --output_dir "outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/" \
    >"outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/finetuning_${MODEL_NAME}_lora_${DATASET_NAME}_${LENGTH}.log"
else
CUDA_VISIBLE_DEVICES="${CUDA}" python CodeLlama/finetuning.py \
    --quantization \
    --use_peft \
    --peft_method lora \
    --batch_size_training ${BATCH_MAP["${MODEL_NAME}_${LENGTH}"]} \
    --val_batch_size ${BATCH_MAP["${MODEL_NAME}_${LENGTH}"]} \
    --context_length $(echo $LENGTH | awk -F'-' '{print $2}') \
    --num_epochs 5 \
    --model_name ${MODEL_MAP[$MODEL_NAME]} \
    --alpaca_dataset.train_data_path "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --alpaca_dataset.valid_data_path "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --output_dir "outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/" \
    >"outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/finetuning_${MODEL_NAME}_lora_${DATASET_NAME}_${LENGTH}.log"
fi

# CUDA_VISIBLE_DEVICES="${CUDA}" python CodeLlama/inference.py \
#     --base_model ${MODEL_MAP[$MODEL_NAME]} \
#     --data_file "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_test.json" \
#     --csv_path "outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/results.csv" \
#     2>&1 1>"outputs/${MODEL_NAME}_lora/${DATASET_NAME}_${LENGTH}/inference_${MODEL_NAME}_lora_${DATASET_NAME}_${LENGTH}.log"
