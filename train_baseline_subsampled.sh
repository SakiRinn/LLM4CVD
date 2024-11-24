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

mkdir -p "outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/"

echo "POS_RATIO: $(echo $POS_RATIO)"

if [[ "$MODEL_NAME" == "GraphCodeBERT" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/" \
    --csv_path="outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/results.csv" \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_test.json" \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --evaluate_during_training \
    --seed 42 \
    2>"outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/train_${MODEL_NAME}_${DATASET_NAME}_${POS_RATIO}.log"
elif [[ "$MODEL_NAME" == "UniXcoder" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/" \
    --csv_path="outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/results.csv" \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_test.json" \
    --num_train_epochs 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --seed 42 \
    2>"outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/train_${MODEL_NAME}_${DATASET_NAME}_${POS_RATIO}.log"
else
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/code/run.py \
    --output_dir="outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/" \
    --csv_path="outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/results.csv" \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${POS_RATIO}_train.json" \
    --eval_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_validate.json" \
    --test_data_file="data/${DATASET_NAME}/alpaca/${DATASET_NAME}_0-512_test.json" \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --evaluate_during_training \
    --seed 42 \
    2>"outputs/${MODEL_NAME}_subsampled/${DATASET_NAME}_${POS_RATIO}/train_${MODEL_NAME}_${DATASET_NAME}_${POS_RATIO}.log"
fi
