import os
import os.path as osp
import gzip
import pickle
import json

ALL_PROJECTS = ['ffmpeg', 'httpd', 'libav', 'libtiff', 'nginx', 'openssl']


def read_pickle_gz(file_path):
    with gzip.open(file_path, mode='rb') as fp:
        data = []
        while True:
            try:
                data.append(pickle.load(fp))
            except EOFError:
                break
    return data


def pickle_gz_to_json(pickle_gz_dir, json_dir):
    for project in ALL_PROJECTS:
        project_pklgz_dir = osp.join(pickle_gz_dir, project)
        project_json_dir = osp.join(json_dir, project)
        if not osp.exists(project_json_dir):
            os.makedirs(project_json_dir)

        for root, dirs, files in os.walk(project_pklgz_dir):
            for f in files:
                file_path = os.path.join(root, f)
                if not file_path.endswith(".pickle.gz"):
                    continue
                data = read_pickle_gz(file_path)
                filename = os.path.basename(file_path).replace(".pickle.gz", ".json")

                output_path = os.path.join(project_json_dir, filename)
                with open(output_path, "w") as fp:
                    json.dump(data, fp, indent=4)
            break


if __name__ == "__main__":
    pickle_gz_to_json('data/d2a/', 'data/d2a/json')
