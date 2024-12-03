import json
import os
import os.path as osp
import numpy as np
import h5py


def dataset_to_dict(dataset):
        data = dataset[()]
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return data


def hdf5_to_dict(group):
    result = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            result[k] = dataset_to_dict(v)
        elif isinstance(v, h5py.Group):
            result[k] = hdf5_to_dict(v)
    for k, v in group.attrs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        result[k] = v
    return result


def hdf5_to_json(hdf5_dir, json_dir):
    for file in os.listdir(hdf5_dir):
        hdf5_path = osp.join(hdf5_dir, file)
        if not os.path.isfile(hdf5_path) or not hdf5_path.endswith(".hdf5"):
            continue

        with h5py.File(hdf5_path, 'r') as f:
            data = hdf5_to_dict(f)
        functionSource = [func.decode('utf-8') for func in data['functionSource']]
        data['functionSource'] = functionSource

        if not osp.exists(json_dir):
            os.makedirs(json_dir)
        json_path = osp.join(json_dir, file.replace(".hdf5", ".json"))
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    hdf5_to_json('data/draper/', 'data/draper/json')
