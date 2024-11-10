import random
from typing import Any

import pandas as pd


def sampling_by_pos_ratio(data, pos_ratio=0.5, shuffle=True):
    pos_samples = [e for e in data if 'label' in e.keys() and e['label'] == 1]
    neg_samples = [e for e in data if 'label' in e.keys() and e['label'] == 0]
    if shuffle:
        random.shuffle(pos_samples)
    if shuffle:
        random.shuffle(neg_samples)

    num_neg = len(data) - round(len(pos_samples) / pos_ratio)
    neg_samples = neg_samples[:num_neg]
    new_data = pos_samples + neg_samples

    if shuffle:
        random.shuffle(new_data)
    else:
        df = pd.DataFrame(new_data)
        df_sorted = df.sort_values(by='index')
        new_data = df_sorted.to_dict(orient='records')
    return new_data


def sampling_by_type_ratio(data, type_tag: str, type_ratio: dict[str, float], shuffle=True):
    total_ratio = sum(type_ratio.values())
    for k in type_ratio.keys():
        type_ratio[k] /= total_ratio    # normalize

    type_counts = {}
    for entry in data:
        typee = entry[type_tag]
        if typee not in type_counts.keys():
            type_counts[typee] = 1
        else:
            type_counts[typee] += 1

    for k in type_counts.keys():
        if k not in type_ratio.keys():
            raise ValueError
