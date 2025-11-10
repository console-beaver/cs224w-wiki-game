# util function with some util stuff

import pandas as pd
from build_dict_parallel import build_dict_parallel

def build_dict(filepath):
    out = dict()
    with open(filepath, 'r') as f:
        total_lines = sum(1 for _ in f)  # for progress bar

    with open(filepath, 'r') as f:
        for line in tqdm(f, total=total_lines):
            line = line.strip()
            if not line or line[0] == '#': continue
            u, v = line.split()
            u = int(u)
            v = int(v)
            if not u in out: out[u] = {v}
            else: out[u].add(v)
    return out

def load_full_dataset_from_folder():
    import os
    # first verify the dataset has been downloaded
    fail = False
    if os.path.isdir('dataset'):
        if not os.path.isfile(os.path.join('dataset', 'enwiki-2013.txt')):
            print('dataset/enwiki-2013.txt is missing, please run download_dataset.sh')
            fail = True
        if not os.path.isfile(os.path.join('dataset', 'enwiki-2013-names.csv')):
            print('dataset/enwiki-2013-names.csv is missing, please run download_dataset.sh')
            fail = True
    else:
        print('dataset directory is missing, please run download_dataset.sh')
        fail = True
    if fail: return

    print('reading edge names')
    names = pd.read_csv('dataset/enwiki-2013-names.csv', comment='#', header=0, on_bad_lines='skip')
    id_to_name = pd.Series(names['name'].values, index=names['node_id']).to_dict()
    name_to_id = pd.Series(names['node_id'].values, index=names['name']).to_dict()
    print('loading graph')
    # edge_dict = build_dict('dataset/enwiki-2013.txt')
    edge_dict = build_dict_parallel('dataset/enwiki-2013.txt')
    return name_to_id, id_to_name, edge_dict
