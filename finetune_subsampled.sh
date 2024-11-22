#!/bin/bash
DATASET_NAME=$1
MODEL_NAME=$2
POS_RATIO=$3
BATCH_SIZE=$4
CUDA=${5:-"0"}

# Check if the first three parameters are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <POS_RATIO> <BATCH_SIZE> [CUDA]"
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

mkdir -p "outputs/${MODEL_NAME}_lora_subsampled/${DATASET_NAME}_${POS_RATIO}/"

echo "Batch size: ${BATCH_MAP[$MODEL_NAME]}"
echo "Pos ratio: $(echo $POS_RATIO)"

if [[ "$MODEL_NAME" == "StarCoder" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/finetune.py \
    --quantization \
    --use_peft \
    --peft_method lora \
    --batch_size_training ${BATCH_MAP[$MODEL_NAME]} \
    --val_batch_size ${BATCH_MAP[$MODEL_NAME]} \
    --context_length 512 \
    --num_epochs 5 \
    --model_name ${MODEL_MAP[$MODEL_NAME]} \
    --train_data_path "data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_train.json" \
    --valid_data_path "data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_validate.json" \
    --output_dir "outputs/${MODEL_NAME}_lora_subsampled/${DATASET_NAME}_${POS_RATIO}/" \
    >"outputs/${MODEL_NAME}_lora_subsampled/${DATASET_NAME}_${POS_RATIO}/finetuning_${MODEL_NAME}_lora_${DATASET_NAME}_${POS_RATIO}.log"
else
CUDA_VISIBLE_DEVICES="${CUDA}" python CodeLlama/finetuning.py \
    --quantization \
    --use_peft \
    --peft_method lora \
    --batch_size_training ${BATCH_MAP[$MODEL_NAME]} \
    --val_batch_size ${BATCH_MAP[$MODEL_NAME]} \
    --context_length 512 \
    --num_epochs 5 \
    --model_name ${MODEL_MAP[$MODEL_NAME]} \
    --alpaca_dataset.train_data_path "data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_train.json" \
    --alpaca_dataset.valid_data_path "data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_validate.json" \
    --output_dir "outputs/${MODEL_NAME}_lora_subsampled/${DATASET_NAME}_${POS_RATIO}/" \
    >"outputs/${MODEL_NAME}_lora_subsampled/${DATASET_NAME}_${POS_RATIO}/finetuning_${MODEL_NAME}_lora_${DATASET_NAME}_${POS_RATIO}.log"
fi

CUDA_VISIBLE_DEVICES="${CUDA}" python CodeLlama/inference.py \
    --base_model ${MODEL_MAP[$MODEL_NAME]} \
    --data_file "data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_test.json" \
    --csv_path "outputs/${MODEL_NAME}_lora_subsampled/${DATASET_NAME}_${POS_RATIO}/results.csv" \
    2>&1 1>"outputs/${MODEL_NAME}_lora_subsampled/${DATASET_NAME}_${POS_RATIO}/inference_${MODEL_NAME}_lora_${DATASET_NAME}_${POS_RATIO}.log"
