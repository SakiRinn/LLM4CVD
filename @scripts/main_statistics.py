import json

import numpy as np
from transformers import AutoTokenizer

from utils.process import split_by_length


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


def load_and_count_len(json_paths: list['str'], tokenizer_name: str, lengths=[512, 1024]):
    data = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data.extend(json.load(f))

    dataset_dict = split_by_length(data, tokenizer_name, lengths)
    for k, v in dataset_dict.items():
        print(f"{k}, len: {len(v)}")

    return dataset_dict


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
    dataset_name = 'diversevul'
    tokenizer_name = 'meta-llama/Meta-Llama-3-8B'

    json_paths = [
        # f'data/{dataset_name}/length/{dataset_name}_512-1024.json',
        # f'data/{dataset_name}/length/{dataset_name}_1024-*.json'
        f'data/{dataset_name}_subsampled/{dataset_name}_0·3.json'
    ]

    print(f'-----> {dataset_name} <-----')
    print('[Length]')
    dataset_dict = load_and_count_len(json_paths, tokenizer_name,
                                      [128, 256, 384, 512, 640, 768, 896, 1024])

    num_pos, num_neg = load_and_count_pos_neg(json_paths)
    print('[Pos/Neg]')
    print(f'Positive: {num_pos}')
    print(f'Negative: {num_neg}')
    print(f'Total: {num_pos + num_neg}')
