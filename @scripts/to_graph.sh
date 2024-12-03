#!/bin/bash
DATASET_NAME=$1
LENGTH=$2

# Check if the first three parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <DATASET_NAME> <LENGTH>"
    exit 1
fi

if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "Bash version 4.0 or higher is required."
    exit 1
fi

python @scripts/to_graph/main.py \
    "data/${DATASET_NAME}/length/${DATASET_NAME}_${LENGTH}.json" \
    --output-dir "data/${DATASET_NAME}/graph/"
