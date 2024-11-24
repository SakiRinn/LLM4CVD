import random

import pandas as pd
from transformers import AutoTokenizer


def truncate(data, max_samples=25000, shuffle=True):
    if len(data) <= max_samples:
        print(f'Stay data as the same {len(data)}.')
        if shuffle:
            random.shuffle(data)
        return data

    len_ratio = (max_samples + 0.1) / len(data)

    pos_samples = [e for e in data if 'label' in e.keys() and e['label'] == 1]
    neg_samples = [e for e in data if 'label' in e.keys() and e['label'] == 0]

    if shuffle:
        random.shuffle(pos_samples)
    if shuffle:
        random.shuffle(neg_samples)

    pos_samples = pos_samples[:round(len(pos_samples) * len_ratio)]
    neg_samples = neg_samples[:round(len(neg_samples) * len_ratio)]
    data = pos_samples + neg_samples

    if shuffle:
        random.shuffle(data)
    else:
        df = pd.DataFrame(data)
        df_sorted = df.sort_values(by='index')
        data = df_sorted.to_dict(orient='records')

    return data


def truncate_by_ratio(dataset_dict, max_samples=25000, shuffle=True):
    max_len = 1
    for key, data in dataset_dict.items():
        if len(data) > max_len:
            max_len = len(data)
    len_ratio = (max_samples + 0.1) / max_len

    for key, data in dataset_dict.items():

        if max_len > max_samples:
            pos_samples = [e for e in data if 'label' in e.keys() and e['label'] == 1]
            neg_samples = [e for e in data if 'label' in e.keys() and e['label'] == 0]

            if shuffle:
                random.shuffle(pos_samples)
            if shuffle:
                random.shuffle(neg_samples)
            pos_samples = pos_samples[:round(len(pos_samples) * len_ratio)]
            neg_samples = neg_samples[:round(len(neg_samples) * len_ratio)]

            data = pos_samples + neg_samples
            print(f'Dataset {key} is truncated!')
        else:
            print(f'Stay dataset {key} as the same.')

        if shuffle:
            random.shuffle(data)
        else:
            df = pd.DataFrame(data)
            df_sorted = df.sort_values(by='index')
            data = df_sorted.to_dict(orient='records')
        dataset_dict[key] = data

    return dataset_dict


def train_test_split(data, train_ratio=0.8, val_ratio=0.1):
    train_split_index = int(len(data) * train_ratio)
    val_split_index = int(len(data) * (train_ratio + val_ratio))

    train_dataset = data[:train_split_index]
    val_dataset = data[train_split_index:val_split_index]
    test_dataset = data[val_split_index:]

    return {'train': train_dataset,
            'validate': val_dataset,
            'test': test_dataset}


def split_by_length(data, tokenizer_name: str, lengths=[512, 1024]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if 0 not in lengths:
        lengths.insert(0, 0)
    if 99999 not in lengths:
        lengths.append(99999)

    new_data = []
    for entry in data:
        tokens = tokenizer(entry['code'], return_tensors="pt")
        length = len(tokens['input_ids'].squeeze()) if tokens['input_ids'].squeeze().dim() > 0 else 0
        entry['length'] = length
        new_data.append(entry)

    dataset_dict = {}
    for i in range(len(lengths) - 1):
        min_length = lengths[i]
        max_length = lengths[i + 1]

        dataset = [entry for entry in new_data
                   if min_length <= entry['length'] < max_length]
        dataset_dict[f'{min_length}-{max_length if max_length != 99999 else "*"}'] = dataset

    return dataset_dict


def sampling_by_pos_ratio(data, pos_ratio=0.5, shuffle=True):
    pos_samples = [e for e in data if 'label' in e.keys() and e['label'] == 1]
    neg_samples = [e for e in data if 'label' in e.keys() and e['label'] == 0]
    if shuffle:
        random.shuffle(pos_samples)
    if shuffle:
        random.shuffle(neg_samples)

    num_neg = round(len(pos_samples) / pos_ratio - len(pos_samples))
    neg_samples = neg_samples[:num_neg]
    new_data = pos_samples + neg_samples

    if shuffle:
        random.shuffle(new_data)
    else:
        df = pd.DataFrame(new_data)
        df_sorted = df.sort_values(by='index')
        new_data = df_sorted.to_dict(orient='records')
    return new_data
