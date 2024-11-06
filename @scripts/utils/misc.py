import os
import os.path as osp
import json


def save_dataset_dict(dataset_dict, output_dir='.', prefix=''):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    for k, v in dataset_dict.items():
        with open(osp.join(output_dir, f'{prefix}_{k}.json'), "w") as f:
            json.dump(v, f, indent=4)


def to_alpaca(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    new_data = [{'instruction': 'Detect whether the following code contains vulnerabilities.',
                 'input': entry['code'],
                 'output': str(entry['label']),
                 'index': entry['index']}
                for entry in data]
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=4)


def to_alpaca_and_combine(input_paths, output_path):
    data = []
    for input_path in input_paths:
        with open(input_path, "r") as f:
            data.extend(json.load(f))
    new_data = [{'instruction': 'Detect whether the following code contains vulnerabilities.',
                 'input': entry['code'],
                 'output': str(entry['label']),
                 'index': entry['index']}
                for entry in data]
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=4)
