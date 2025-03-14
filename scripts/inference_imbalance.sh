#!/bin/bash
DATASET_NAME=$1
MODEL_NAME=$2
POS_RATIO=$3
CUDA=${4:-"0"}

# Check if the first three parameters are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <POS_RATIO> [CUDA]"
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

mkdir -p "outputs/${MODEL_NAME}_lora_imbalance/${DATASET_NAME}_${POS_RATIO}/"

echo "POS_RATIO: $(echo $POS_RATIO)"

CUDA_VISIBLE_DEVICES="${CUDA}" python LLM/inference.py \
    --base_model ${MODEL_MAP[$MODEL_NAME]} \
    --tuned_model "outputs/${MODEL_NAME}_lora_imbalance/${DATASET_NAME}_${POS_RATIO}/epoch-4" \
    --data_file "data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_test.json" \
    --csv_path "outputs/${MODEL_NAME}_lora_imbalance/${DATASET_NAME}_${POS_RATIO}/results.csv" \
    >"outputs/${MODEL_NAME}_lora_imbalance/${DATASET_NAME}_${POS_RATIO}/inference_${MODEL_NAME}_lora_${DATASET_NAME}_${POS_RATIO}.log"
