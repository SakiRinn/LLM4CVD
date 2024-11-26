#!/bin/bash
DATASET_NAME=$1
MODEL_NAME=$2
R=$3
ALPHA=$4
CUDA=${5:-"0"}

# Check if the first three parameters are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <R> [CUDA]"
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

mkdir -p "outputs/${MODEL_NAME}_lora_ablation/${DATASET_NAME}_${R}_${ALPHA}/"

echo "R: $(echo $R)"

CUDA_VISIBLE_DEVICES="${CUDA}" python CodeLlama/inference.py \
    --base_model ${MODEL_MAP[$MODEL_NAME]} \
    --tuned_model "outputs/${MODEL_NAME}_lora_ablation/${DATASET_NAME}_${R}_${ALPHA}/epoch-4" \
    --data_file "data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_test.json" \
    --csv_path "outputs/${MODEL_NAME}_lora_ablation/${DATASET_NAME}_${R}_${ALPHA}/results.csv" \
    >"outputs/${MODEL_NAME}_lora_ablation/${DATASET_NAME}_${R}_${ALPHA}/inference_${MODEL_NAME}_lora_ablation_${DATASET_NAME}_${R}_${ALPHA}.log"
