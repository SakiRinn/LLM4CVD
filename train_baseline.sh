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

if [[ "$MODEL_NAME" == "GraphCodeBERT" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/" \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_test.json" \
    --epoch 5 \
    --block_size $(echo $LENGTH | awk -F'-' '{print $2}') \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --evaluate_during_training \
    --seed 42 \
    1>&2 2>"outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}.log"
elif [[ "$MODEL_NAME" == "UniXcoder" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/" \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_test.json" \
    --num_train_epochs 5 \
    --block_size $(echo $LENGTH | awk -F'-' '{print $2}') \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --seed 42 \
    1>&2 2>"outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}.log"
elif [[ "$MODEL_NAME" == "Devign" ]]; then
python @scripts/to_graph/main.py \
    "data/${DATASET_NAME}/length/${DATASET_NAME}_${LENGTH}.json" \
    --output-dir "data/${DATASET_NAME}/graph/"
CUDA_VISIBLE_DEVICES="${CUDA}" python Devign/main.py \
    --output_dir "outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/" \
    --input_dir "data/${DATASET_NAME}/graph/${DATASET_NAME}_${LENGTH}/" \
    --feature_size 197 \
    2>&1 >"outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}.log"
else
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/" \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_${LENGTH}_test.json" \
    --epoch 5 \
    --block_size $(echo $LENGTH | awk -F'-' '{print $2}') \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --evaluate_during_training \
    --seed 42 \
    1>&2 2>"outputs/${MODEL_NAME}/${DATASET_NAME}_${LENGTH}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}.log"
fi
