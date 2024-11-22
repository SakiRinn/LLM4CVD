import os
import os.path as osp
import json

import sys
sys.path.append(osp.join(osp.dirname(__file__), '..'))

from utils.process import train_test_split, truncate, split_by_length
from utils import save_dataset_dict
from utils.loader import *


os.chdir(osp.join(osp.dirname(__file__), '..', '..'))


def init_mix_data(do_split=False):
    data = []

    data_devign = load_devign('data/devign/function.json')
    data += data_devign

    data_reveal = load_reveal('data/reveal/')
    data += data_reveal

    data_bigvul = load_bigvul('data/bigvul/MSR_data_cleaned.json')
    data += data_bigvul

    data_diversevul = load_diversevul('data/diversevul/diversevul_20230702.jsonl')
    data += data_diversevul

    data_draper = []
    for v in load_draper('data/draper').values():
        data_draper += v
    short, long, long_2, long_3, _ = split_by_length(data_draper, 'meta-llama/Meta-Llama-3-8B',
                                                     [512, 768, 896, 1024]).values()
    data_draper = long[:int(len(long) * 0.5)] + long_2[:int(len(long_2) * 0.7)] + long_3[:int(len(long_3) * 1.)] + \
        truncate(short, 400000 - int(len(long) * 0.5) - int(len(long_2) * 0.7) - int(len(long_3) * 1.))
    data += data_draper

    # shuffle
    data = truncate(data, 9999999)

    if not osp.exists('data/mix'):
        os.makedirs('data/mix')
    with open('data/mix/mix.json', 'w') as f:
        json.dump(data, f, indent=4)

    if do_split:
        datasets = train_test_split(data)
        save_dataset_dict(datasets, 'data/mix', prefix='mix')


if __name__ == '__main__':
    init_mix_data()
