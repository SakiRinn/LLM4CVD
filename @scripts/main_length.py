from glob import glob
import json
import os
import os.path as osp

from utils.loader import load_splitted_json, load_json, save_dataset_dict
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

def jsons_to_alpaca(input_dir, output_dir, dataset_name, lengths=[512, 1024]):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    lengths = [str(i) for i in lengths]
    if '0' not in lengths:
        lengths.insert(0, '0')
    if '*' not in lengths:
        lengths.append('*')

    for i in range(len(lengths) - 1):
        min_length = lengths[i]
        max_length = lengths[i + 1]

        pattern = osp.join(input_dir, f'{dataset_name}_{min_length}-{max_length}_*.json')
        paths = glob(pattern)

        for path in paths:
            # if 'train' in path:
            #     to_alpaca(path, osp.join(output_dir, osp.basename(path)))
            #     paths.remove(path)
            #     break
            # combine_and_to_alpaca(paths, osp.join(output_dir, f'{dataset_name}_{min_length}-{max_length}_test.json'))
            to_alpaca(path, osp.join(output_dir, osp.basename(path)))


def main():
    data = load_json('data/mix/mix.json')
    dataset_dict = split_by_length(data, 'meta-llama/Meta-Llama-3-8B',
                                   [128, 256, 384, 512, 640, 768, 896, 1024])
    del dataset_dict['1024-*']
    for key, dataset in dataset_dict.items():
        sampled_data = sampling_by_pos_ratio(dataset, 0.2)
        dataset_dict[key] = truncate(sampled_data, 10000)
    save_dataset_dict(dataset_dict, 'data/mix/length', prefix='mix')

    split_jsons('data/mix/length', 'data/mix/split')
    jsons_to_alpaca('data/mix/split', 'data/mix/alpaca',
                    'mix', lengths=[128, 256, 384, 512, 640, 768, 896, 1024])


if __name__ == "__main__":
    main()
