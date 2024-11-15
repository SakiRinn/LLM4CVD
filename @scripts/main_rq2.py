import json
import os.path as osp


def mix_datasets_by_length(data_paths):
    data = []
    for path in data_paths:
        if osp.isfile(path):
            with open(path, 'r') as f:
                data.extend(json.load(f))
        else:
            print(f"[Warning] {path} is not a valid file or directory.")
    return data



if __name__ == "__main__":
    short_data_paths = [
        'data/devign/length/devign_0-512.json',
        'data/reveal/length/devign_0-512.json',
        'data/bigvul/length/devign_0-512.json',
        'data/diversevul/length/devign_0-512.json',
        'data/draper/length/devign_0-512.json',
    ]
    long_data_paths = [
        'data/devign/length/devign_512-1024.json',
        'data/reveal/length/devign_512-1024.json',
        'data/bigvul/length/devign_512-1024.json',
        'data/diversevul/length/devign_512-1024.json',
        'data/draper/length/devign_512-1024.json',
    ]