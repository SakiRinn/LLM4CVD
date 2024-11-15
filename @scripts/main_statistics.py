import json

import numpy as np
from transformers import AutoTokenizer


def len_count(data, tokenizer_name: str, min_length=0, max_length=99999):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    count = 0
    for entry in data:
        if 'code' not in entry.keys():
            continue
        tokens = tokenizer(entry['code'], return_tensors="pt")
        length = len(tokens['input_ids'].squeeze()) if tokens['input_ids'].squeeze().dim() > 0 else 0
        if min_length <= length < max_length:
            count += 1
    return count


def load_and_count_len(json_paths: list['str'], tokenizer_name: str, min_length=512, max_length=1024):
    total_count = 0
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
        total_count += len_count(data, tokenizer_name, min_length, max_length)
    return total_count


def pos_count(data, binary_tag='label'):
    count = 0
    for entry in data:
        if binary_tag in entry.keys() and entry[binary_tag] == 1:
            count += 1
    return count


def load_and_count_pos_neg(json_paths: list['str'], binary_tag='label'):
    total_pos, total_neg = 0, 0
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
        total_pos += pos_count(data, binary_tag)
        total_neg += len(data) - pos_count(data, binary_tag)
    return total_pos, total_neg


def fpr_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.int32), np.array(y_pred, dtype=np.int32)

    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FN = sum((y_true == 1) & (y_pred == 0))

    if (FP + TN) == 0:
        return 0.
    else:
        FPR = FP / (FP + TN)
        return FPR


if __name__ == '__main__':
    dataset_name = 'reveal'
    tokenizer_name = 'meta-llama/Meta-Llama-3-8B'

    json_paths = [
        f'data/{dataset_name}/length/{dataset_name}_0-512.json',
        # f'data/{dataset_name}/length/{dataset_name}_512-1024.json',
        # f'data/{dataset_name}/length/{dataset_name}_1024-*.json'
    ]

    # print(f'-----> {dataset_name} <-----')
    # print('[Length]')
    # print(f'0~512: {load_and_count_len(json_paths, tokenizer_name, 0, 512)}')
    # print(f'512~1024: {load_and_count_len(json_paths, tokenizer_name, 512, 1024)}')
    # print(f'1024~: {load_and_count_len(json_paths, tokenizer_name, 1024, 99999)}')

    num_pos, num_neg = load_and_count_pos_neg(json_paths)
    print('[Pos/Neg]')
    print(f'Positive: {num_pos}')
    print(f'Negative: {num_neg}')
    print(f'Total: {num_pos + num_neg}')
