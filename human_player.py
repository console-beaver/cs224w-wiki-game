#!/usr/bin/env python3

import pandas as pd
from tqdm import tqdm
from build_dict_parallel import build_dict_parallel

def play_game_human(page_names, page_edges):
    name_to_id, id_to_name = page_names
    pos, end = get_start_end(name_to_id, id_to_name)
    # ASSUMPTION: start and end were entered correctly and are reasonable
    path = [pos]
    while pos != end:
        print('\n\ncurrent article:', id_to_name[pos])
        print('target article:', id_to_name[end])
        print('clicks made so far:', len(path) - 1)
        print('type a node_id to jump to the next article from the list below:')
        options = []
        for i, neighbor in enumerate(page_edges[pos]):
            print('\t(', i+1, ')\t:', id_to_name[neighbor])
            options.append(neighbor)
        choice = int(input('next link? : '))
        while choice <= 0 or choice > len(options):
            choice = int(input('invalid selection, next link? : '))
        pos = options[choice-1]
        path.append(pos)
    print(f'\n\nyou finished the game in {len(path)-1} clicks!')
    print('path:')
    # print(path)
    for i, link in enumerate(path):
        print(f'\t{i+1}.', id_to_name[link])

def get_start_end(name_to_id, id_to_name):
    start = int(input('what is your starting node id? : '))
    print('got starting node id =', start, '(', id_to_name[start], ')')
    target = int(input('what is your target node id? : '))
    print('got target node id =', target, '(', id_to_name[target], ')')
    return start, target

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

if __name__ == '__main__':
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

    if not fail:
        print('reading edge names')
        names = pd.read_csv('dataset/enwiki-2013-names.csv', comment='#', header=0, on_bad_lines='skip')
        id_to_name = pd.Series(names['name'].values, index=names['node_id']).to_dict()
        name_to_id = pd.Series(names['node_id'].values, index=names['name']).to_dict()
        print('loading graph')
        # edge_dict = build_dict('dataset/enwiki-2013.txt')
        edge_dict = build_dict_parallel('dataset/enwiki-2013.txt')
        print('running demo')
        play_game_human((name_to_id, id_to_name), edge_dict)
