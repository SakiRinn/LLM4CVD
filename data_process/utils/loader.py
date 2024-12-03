import os
import os.path as osp
import re
import json

from preprocess.d2a import ALL_PROJECTS
from utils import save_dataset_dict


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f'Success to load {json_path}.')
    return data


def load_splitted_json(json_dir, prefix=''):
    dataset_dict = {}

    for filename in os.listdir(json_dir):
        match = re.match(r'^(.*)_(.*)\.json$', filename)
        if not match:
            continue

        key = match.group(2)
        if not prefix:
            prefix = match.group(1)
            print(f'Default prefix: {prefix}')
        if prefix != match.group(1):
            continue

        json_path = os.path.join(json_dir, filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset_dict[key] = data
        print(f'Success to load {json_path}.')

    return dataset_dict


def load_devign(json_path):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    data = []
    for idx, raw_entry in enumerate(raw_data):
        code, label = raw_entry['func'], raw_entry['target']
        del raw_entry['func'], raw_entry['target']
        entry = {
            'index': idx,
            'code': code,
            'label': label,
            **raw_entry
        }
        data.append(entry)

    return data


def load_reveal(json_dir):
    data = []
    with open(osp.join(json_dir, 'non-vulnerables.json'), 'r') as f:
        data += [{**raw_entry, 'label': 0, 'index': idx}
                 for idx, raw_entry in enumerate(json.load(f))]
    with open(osp.join(json_dir, 'vulnerables.json'), 'r') as f:
        data += [{**raw_entry, 'label': 1, 'index': idx}
                 for idx, raw_entry in enumerate(json.load(f))]
    return data


def load_bigvul(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    data = []
    for idx in range(len(raw_data)):
        raw_entry = raw_data[str(idx)]
        entry = {
            "index": idx,
            "code": raw_entry['func_before'],
            "line": None if len(raw_entry['lines_before']) == 0 else raw_entry['lines_before'],
            "label": int(raw_entry['vul']),
            "cwe": None if len(raw_entry['CWE ID']) == 0 else raw_entry['CWE ID'],
            "cve": None if len(raw_entry['CVE ID']) == 0 else raw_entry['CVE ID']
        }
        data.append(entry)

    return data


def load_d2a(json_dir):
    idx = 0
    data = {}

    for project in ALL_PROJECTS:
        project_dir = osp.join(json_dir, project)
        file_paths = [osp.join(project_dir, f'{project}_labeler_0.json'),
                      osp.join(project_dir, f'{project}_labeler_1.json')]

        raw_data = []
        for path in file_paths:
            with open(path, 'r') as f:
                raw_data += json.load(f)

        for raw_entry in raw_data:
            if raw_entry['label_source'] == "after_fix_extractor":
                continue

            file = raw_entry['bug_info']['file']
            func_name = raw_entry['bug_info']['procedure']

            # extract func_key
            skip_flag = True
            for trace in raw_entry['trace']:
                if trace['file'] == file and trace['func_name'] == func_name:
                    func_key = trace['func_key']
                    skip_flag = False
                    break
            if skip_flag:
                print('[Wrong func_key] An erroneous data has been removed.')
                continue

            # extract code
            skip_flag = True
            for v in raw_entry['functions'].values():
                if v['file'] == file and v['name'] == func_name:
                    code = v['code']
                    skip_flag = False
                    break
            if skip_flag:
                print('[Wrong code] An erroneous data has been removed.')
                continue

            # extract bug_line
            bug_line_no = raw_entry["bug_info"]["line"]
            start_line_no = int(re.search(r'\b(\d+):\d+-\d+:\d+\b', func_key).group(1))
            bug_line_idx = bug_line_no - start_line_no
            try:
                bug_line = code.split('\n')[bug_line_idx]
            except IndexError:
                print('[Wrong index] An erroneous data has been removed.')
                continue

            entry = data.get(func_key, None)
            if entry is None:
                data[func_key] = {
                    'index': idx,
                    'code': code,
                    'label': raw_entry['label'],
                    'type': raw_entry['bug_type'],
                    'project': raw_entry['project'],
                    'line': bug_line
                }
            else:
                if isinstance(entry['line'], str) and bug_line != entry['line']:
                    entry['line'] = [entry['line'], bug_line]
                if isinstance(entry['line'], list) and bug_line not in entry['line']:
                    entry['line'].append(bug_line)
                data[func_key] = entry

            idx += 1

    return list(data.values())


def load_diversevul(jsonl_path):
    with open(jsonl_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]

    data = []
    for idx, raw_entry in enumerate(raw_data):
        code, label, cwe = raw_entry['func'], raw_entry['target'], raw_entry['cwe']
        del raw_entry['func'], raw_entry['target'], raw_entry['cwe']
        del raw_entry['message']

        if len(cwe) == 0:
            cwe = None
        elif len(cwe) == 1:
            cwe = cwe[0]

        entry = {
            'index': idx,
            'code': code,
            'label': label,
            'cwe': cwe,
            **raw_entry
        }
        data.append(entry)

    return data


def load_draper(json_dir):
    file_paths = [osp.join(json_dir, "VDISC_train.json"),
                  osp.join(json_dir, "VDISC_validate.json"),
                  osp.join(json_dir, "VDISC_test.json")]

    dataset_dict = {}
    idx = 0

    for path in file_paths:
        with open(path, 'r') as f:
            raw_data = json.load(f)

        data = []
        for i in range(len(raw_data['functionSource'])):
            code = raw_data['functionSource'][i]
            cwe = [k for k, v in raw_data.items() if k != 'functionSource' and v[i]]

            if len(cwe) == 0:
                cwe = None
            elif len(cwe) == 1:
                cwe = cwe[0]

            entry = {
                'index': i,
                'code': code,
                'label': 1 if cwe else 0,
                'cwe': cwe
            }
            data.append(entry)
            idx += 1

            key = re.search(r'^\w+_(\w+)\.json$', osp.split(path)[-1]).group(1)
            dataset_dict[key] = data

    return dataset_dict
