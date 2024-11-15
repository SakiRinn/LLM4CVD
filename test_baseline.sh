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

mkdir -p "outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/"

echo "Length: $(echo $LENGTH | awk -F'-' '{print $2}')"

if [[ "$MODEL_NAME" == "Devign" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/main.py \
    --output_dir "outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/" \
    --csv_path="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/results.csv" \
    --do_test \
    --input_dir "data/${DATASET_NAME}/graph/${DATASET_NAME}_${LENGTH}/" \
    --seed 42 \
    >"outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/test_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}.log"
elif [[ "$MODEL_NAME" == "GraphCodeBERT" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/" \
    --csv_path="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/results.csv" \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_test.json" \
    --block_size $(echo $LENGTH | awk -F'-' '{print $2}') \
    --eval_batch_size 64 \
    --seed 42 \
    2>"outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/test_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}.log"
elif [[ "$MODEL_NAME" == "UniXcoder" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/" \
    --csv_path="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/results.csv" \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_test.json" \
    --block_size $(echo $LENGTH | awk -F'-' '{print $2}') \
    --eval_batch_size 64 \
    --seed 42 \
    2>"outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/test_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}.log"
else
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/" \
    --csv_path="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/results.csv" \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_test.json" \
    --block_size $(echo $LENGTH | awk -F'-' '{print $2}') \
    --eval_batch_size 64 \
    --seed 42 \
    2>"outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/test_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}.log"
fi
