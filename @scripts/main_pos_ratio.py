from glob import glob
import json
import os
import os.path as osp

from utils.loader import load_diversevul, load_draper, load_splitted_json, load_json, save_dataset_dict
from utils.process import *
from utils import to_alpaca

os.chdir(osp.join(osp.dirname(__file__), '..'))


def split_jsons(input_dir, output_dir):
    for file in os.listdir(input_dir):
        json_path = osp.join(input_dir, file)
        if not os.path.isfile(json_path) or not json_path.endswith(".json"):
            continue
        with open(json_path, "r") as f:
            data = json.load(f)

        datasets = train_test_split(data)
        save_dataset_dict(datasets, output_dir, prefix=file.split(".")[0])


def jsons_to_alpaca(input_dir, output_dir, dataset_name, pos_ratios=[0.25, 0.5]):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    pos_ratios = [str(i).replace('.', '·') for i in pos_ratios]

    for i in pos_ratios:
        pattern = osp.join(input_dir, f'{dataset_name}_{i}_*.json')
        paths = glob(pattern)

        for path in paths:
            # if 'train' in path:
            #     to_alpaca(path, osp.join(output_dir, osp.basename(path)))
            #     paths.remove(path)
            #     break
            # combine_and_to_alpaca(paths, osp.join(output_dir, f'{dataset_name}_{min_length}-{max_length}_test.json'))
            to_alpaca(path, osp.join(output_dir, osp.basename(path)))


def main_diversevul():
    pos_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    data = load_diversevul('data/diversevul/diversevul_20230702.jsonl')
    with open('data/diversevul/split/diversevul_0-512_test.json', "r") as f:
        test_data = json.load(f)
    test_indices = [item['index'] for item in test_data]
    data = [item for item in data if item['index'] not in test_indices]

    pos_ratio_datasets = {}
    for pr in pos_ratios:
        sampled_data = sampling_by_pos_ratio(data, pr)
        new_data = split_by_length(sampled_data, 'meta-llama/Meta-Llama-3-8B')['0-512']
        pos_ratio_datasets[str(pr).replace('.', '·')] = truncate(new_data, 25000)

    save_dataset_dict(pos_ratio_datasets, 'data/diversevul_subsampled/', prefix='diversevul')

    split_jsons('data/diversevul_subsampled/', 'data/diversevul_subsampled/split')
    jsons_to_alpaca('data/diversevul_subsampled/split', 'data/diversevul_subsampled/alpaca',
                    'diversevul', pos_ratios=pos_ratios)


def main_draper():
    pos_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    dataset_dict = load_draper('data/draper')
    data = []
    for v in dataset_dict.values():
        data += v

    with open('data/draper/split/draper_0-512_test.json', "r") as f:
        test_data = json.load(f)
    test_indices = [item['index'] for item in test_data]
    data = [item for item in data if item['index'] not in test_indices]

    pos_ratio_datasets = {}
    for pr in pos_ratios:
        sampled_data = sampling_by_pos_ratio(data, pr)
        new_data = split_by_length(sampled_data, 'meta-llama/Meta-Llama-3-8B')['0-512']
        pos_ratio_datasets[str(pr).replace('.', '·')] = truncate(new_data, 25000)

    save_dataset_dict(pos_ratio_datasets, 'data/draper_subsampled/', prefix='draper')

    split_jsons('data/draper_subsampled/', 'data/draper_subsampled/split')
    jsons_to_alpaca('data/draper_subsampled/split', 'data/draper_subsampled/alpaca',
                    'draper', pos_ratios=pos_ratios)


if __name__ == "__main__":
    main_diversevul()
    main_draper()
