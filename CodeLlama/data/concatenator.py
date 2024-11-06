# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class PadDataset(Dataset):

    IGNORE_INDEX = -100

    def __init__(self, dataset, pad_length=4096, pad_value=0):
        self.dataset = dataset
        self.pad_length = pad_length
        self.pad_value = pad_value if pad_value is not None else 0

        self.samples = []

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            length = len(sample["input_ids"])

            if length < self.pad_length:
                padded_sample = {
                    "input_ids": sample["input_ids"] + [self.pad_value] * (self.pad_length - length),
                    "attention_mask": sample["attention_mask"] + [False] * (self.pad_length - length),
                    "labels": sample["labels"] + [self.IGNORE_INDEX] * (self.pad_length - length)
                }
                self.samples.append(padded_sample)
            elif length > self.pad_length:
                truncated_sample = {
                    "input_ids": sample["input_ids"][:self.pad_length],
                    "attention_mask": sample["attention_mask"][:self.pad_length],
                    "labels": sample["labels"][:self.pad_length]
                }
                self.samples.append(truncated_sample)
            else:
                self.samples.append(sample)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
