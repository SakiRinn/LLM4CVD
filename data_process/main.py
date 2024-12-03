import os
import os.path as osp
from glob import glob

from utils.process import *
from utils.loader import *
from utils.misc import to_alpaca, to_alpaca_and_combine

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


def main_devign():
    data = load_devign('data/devign/function.json')
    datasets = split_by_length(data, 'meta-llama/Meta-Llama-3-8B')
    datasets = truncate_by_ratio(datasets)
    save_dataset_dict(datasets, 'data/devign/length', prefix='devign')

    dataset_name = 'devign'
    split_jsons(f'data/{dataset_name}/length', f'data/{dataset_name}/split')
    jsons_to_alpaca(f'data/{dataset_name}/split', f'data/{dataset_name}/alpaca',
                    dataset_name, lengths=[512, 1024])

    print('Devign done.')


def main_reveal():
    data = load_reveal('data/reveal/')
    datasets = split_by_length(data, 'meta-llama/Meta-Llama-3-8B')
    datasets = truncate_by_ratio(datasets)
    save_dataset_dict(datasets, 'data/reveal/length', prefix='reveal')

    dataset_name = 'reveal'
    split_jsons(f'data/{dataset_name}/length', f'data/{dataset_name}/split')
    jsons_to_alpaca(f'data/{dataset_name}/split', f'data/{dataset_name}/alpaca',
                    dataset_name, lengths=[512, 1024])

    print('Reveal done.')


def main_bigvul():
    data = load_bigvul('data/bigvul/MSR_data_cleaned.json')
    datasets = split_by_length(data, 'meta-llama/Meta-Llama-3-8B')
    datasets = truncate_by_ratio(datasets)
    save_dataset_dict(datasets, 'data/bigvul/length', prefix='bigvul')

    dataset_name = 'bigvul'
    split_jsons(f'data/{dataset_name}/length', f'data/{dataset_name}/split')
    jsons_to_alpaca(f'data/{dataset_name}/split', f'data/{dataset_name}/alpaca',
                    dataset_name, lengths=[512, 1024])

    print('BigVul done.')


def main_diversevul():
    data = load_diversevul('data/diversevul/diversevul_20230702.jsonl')
    datasets = split_by_length(data, 'meta-llama/Meta-Llama-3-8B')
    datasets = truncate_by_ratio(datasets)
    save_dataset_dict(datasets, 'data/diversevul/length', prefix='diversevul')

    dataset_name = 'diversevul'
    split_jsons(f'data/{dataset_name}/length', f'data/{dataset_name}/split')
    jsons_to_alpaca(f'data/{dataset_name}/split', f'data/{dataset_name}/alpaca',
                    dataset_name, lengths=[512, 1024])

    print('DiverseVul done.')


def main_draper():
    dataset_dict = load_draper('data/draper')
    data = []
    for v in dataset_dict.values():
        data += v

    datasets = split_by_length(data, 'meta-llama/Meta-Llama-3-8B')
    datasets = truncate_by_ratio(datasets)
    save_dataset_dict(datasets, 'data/draper/length', prefix='draper')

    dataset_name = 'draper'
    split_jsons(f'data/{dataset_name}/length', f'data/{dataset_name}/split')
    jsons_to_alpaca(f'data/{dataset_name}/split', f'data/{dataset_name}/alpaca',
                    dataset_name, lengths=[512, 1024])

    print('Draper done.')


def main_d2a():
    data = load_d2a('data/d2a')
    datasets = split_by_length(data, 'meta-llama/Meta-Llama-3-8B')
    datasets = truncate_by_ratio(datasets)
    save_dataset_dict(datasets, 'data/d2a/length', prefix='d2a')

    dataset_name = 'd2a'
    split_jsons(f'data/{dataset_name}/length', f'data/{dataset_name}/split')
    jsons_to_alpaca(f'data/{dataset_name}/split', f'data/{dataset_name}/alpaca',
                    dataset_name, lengths=[512, 1024])


if __name__ == "__main__":
    main_devign()
    main_reveal()
    main_bigvul()
    main_diversevul()
    main_draper()
    main_d2a()
