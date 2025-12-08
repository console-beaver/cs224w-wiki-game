#!/usr/bin/env python3

import pandas as pd
from build_dict_parallel import build_dict_parallel
from collections import deque
from tqdm import tqdm
import os
import pickle
import random


def bfs(edge_dict, start, target):
    """Find shortest path using BFS. Returns path as list of node IDs."""
    if start == target:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        for neighbor in edge_dict.get(current, set()):
            if neighbor == target:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # No path found

def fetch_dataset(n):
    res = None
    if not n is None:
        print(f'loading subsample of graph data (top {n} nodes)')
        if not (os.path.exists(f'pickles/edge_dict_top_{n}.pkl') and
                os.path.exists(f'pickles/id_to_name_top_{n}.pkl') and
                os.path.exists(f'pickles/name_to_id_top_{n}.pkl')):
            print('you have not generated this subsample of the data yet')
            print('generating pickle files (may take a few minutes)...')
            res = subsample_graph_to_pickles(n)
            if not res: return None
        with open(f'pickles/edge_dict_top_{n}.pkl', 'rb') as f:
            edge_dict = pickle.load(f)
        with open(f'pickles/id_to_name_top_{n}.pkl', 'rb') as f:
            id_to_name = pickle.load(f)
        with open(f'pickles/name_to_id_top_{n}.pkl', 'rb') as f:
            name_to_id = pickle.load(f)
        res = (name_to_id, id_to_name, edge_dict)
    else:  # load the whole dataset
        print('loading the whole dataset')
        res = load_full_dataset_from_folder()
    return res

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

def load_full_dataset_from_folder(tab=0):
    import os
    # first verify the dataset has been downloaded
    fail = False
    if os.path.isdir('dataset'):
        if not os.path.isfile(os.path.join('dataset', 'enwiki-2013.txt')):
            print('\t' * tab + 'dataset/enwiki-2013.txt is missing, please run download_dataset.sh')
            fail = True
        if not os.path.isfile(os.path.join('dataset', 'enwiki-2013-names.csv')):
            print('\t' * tab + 'dataset/enwiki-2013-names.csv is missing, please run download_dataset.sh')
            fail = True
    else:
        print('\t' * tab + 'dataset directory is missing, please run download_dataset.sh')
        fail = True
    if fail: return

    print('\t' * tab + 'reading edge names')
    names = pd.read_csv('dataset/enwiki-2013-names.csv', comment='#', header=0, on_bad_lines='skip')
    id_to_name = pd.Series(names['name'].values, index=names['node_id']).to_dict()
    name_to_id = pd.Series(names['node_id'].values, index=names['name']).to_dict()
    print('\t' * tab + 'loading graph')
    # edge_dict = build_dict('dataset/enwiki-2013.txt')
    edge_dict = build_dict_parallel('dataset/enwiki-2013.txt', tab=tab)
    return name_to_id, id_to_name, edge_dict

def subsample_graph_to_pickles(subsample_count):
    print('first, loading the full dataset:')
    res = load_full_dataset_from_folder(tab=1)
    if not res: return False  # if loading failed, return False
    name_to_id, id_to_name, edge_dict = res

    print('\tcounting undirected edge counts')
    undirected_edge_count = dict()
    for node_id in tqdm(edge_dict):
        for neighbor in edge_dict[node_id]:
            undirected_edge_count[node_id] = undirected_edge_count.get(node_id, 0) + 1
            undirected_edge_count[neighbor] = undirected_edge_count.get(neighbor, 0) + 1

    node_ids_we_care_about = set(sorted(undirected_edge_count.keys(),
                                        reverse=True,
                                        key=lambda x : undirected_edge_count[x]
                                       )[:subsample_count])

    print('\tpruning unneeded nodes')
    keys = list(edge_dict.keys())
    for node_id in tqdm(keys):
        if node_id not in node_ids_we_care_about: del edge_dict[node_id]

    print('\tpruning unneeded edges')
    for node_id in tqdm(edge_dict):
        edge_dict[node_id].intersection_update(node_ids_we_care_about)

    print('\tupdating node names')
    keys = list(id_to_name.keys())
    for node_id in tqdm(keys):
        if not node_id in node_ids_we_care_about: del id_to_name[node_id]
    node_names_we_care_about = set(id_to_name.values())
    keys = list(name_to_id.keys())
    for node_name in tqdm(keys):
        if not node_name in node_names_we_care_about: del name_to_id[node_name]

    print('\tpickling files')
    with open(f'pickles/edge_dict_top_{subsample_count}.pkl', 'wb') as f:
        pickle.dump(edge_dict, f)
    with open(f'pickles/name_to_id_top_{subsample_count}.pkl', 'wb') as f:
        pickle.dump(name_to_id, f)
    with open(f'pickles/id_to_name_top_{subsample_count}.pkl', 'wb') as f:
        pickle.dump(id_to_name, f)
    return True

def sample_src_dst(id_to_name, edge_dict):
    while True:  # repeat until we find a valid src -> dst
        src, dst = random.sample(list(id_to_name.keys()), 2)
        path = bfs(edge_dict, src, dst)
        if path is not None: return src, dst, len(path)

if __name__ == '__main__':
    import sys
    argv = sys.argv
    subsample_count = 1000
    if len(argv) > 1:
        subsample_count = int(argv[1])
    print(f'subsampling {subsample_count} highest-degree nodes from full dataset...')
    subsample_graph_to_pickles(subsample_count)
