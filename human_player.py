#!/usr/bin/env python3

from util import fetch_dataset, sample_src_dst
import numpy as np
import sys

def human_baseline(num_trials, page_names, page_edges):
    name_to_id, id_to_name = page_names
    data = np.zeros((num_trials, 2), dtype=int)  # human length vs min length
    for t in range(num_trials):
        print(f'STARTING TRIAL {t+1}/{num_trials}')
        pos, end, length = sample_src_dst(id_to_name, page_edges)
        data[t, 0] = length
        path = play_game_human(page_names, page_edges, posend=(pos, end), say_results=True)
        print(f'\n\nyou finished the game in {len(path)-1} clicks!')
        print(f'shortest path had length {data[t, 0]}\n\n')
        data[t, 1] = len(path)
    return data

def play_game_human(page_names, page_edges, posend=None, say_results=True):
    name_to_id, id_to_name = page_names
    if posend is None: pos, end = get_start_end_name(name_to_id)
    else: pos, end = posend
    # ASSUMPTION: start and end were entered correctly and are reasonable
    path = [pos]
    while pos != end:
        print('\n\ncurrent article:', id_to_name[pos])
        print('target article:', id_to_name[end])
        print('clicks made so far:', len(path) - 1)
        print('type a node_id to jump to the next article from the list below:')
        options = []
        for i, neighbor in enumerate(sorted(page_edges[pos], key = lambda x : id_to_name[x])):
            print('\t(', i+1, ')\t:', id_to_name[neighbor])
            options.append(neighbor)
        choice = int(input('next link? : '))
        while choice <= 0 or choice > len(options):
            choice = int(input('invalid selection, next link? : '))
        pos = options[choice-1]
        path.append(pos)
    if say_results:
        print(f'\n\nyou finished the game in {len(path)-1} clicks!')
        print('path:')
        for i, link in enumerate(path):
            print(f'\t{i+1}.', id_to_name[link])
    return path

def get_start_end_id(id_to_name):
    start = int(input('what is your starting node id? : '))
    print('got starting node id =', start, '(', id_to_name[start], ')')
    target = int(input('what is your target node id? : '))
    print('got target node id =', target, '(', id_to_name[target], ')')
    return start, target

def get_start_end_name(name_to_id):
    start = input('what is your starting article title? : ')
    print('got starting at node id =', name_to_id[start], f'({start})')
    target = input('what is your target article title? : ')
    print('got target at node id =', name_to_id[target], f'({target})')
    return name_to_id[start], name_to_id[target]

if __name__ == '__main__':
    n = None
    baseline_mode = False
    if len(sys.argv) > 2 and sys.argv[2] == 'b': baseline_mode = True
    if len(sys.argv) > 1: n = int(sys.argv[1])
    res = fetch_dataset(n)
    if not res: exit()  # exit if dataset fetching fails
    name_to_id, id_to_name, edge_dict = res
    if not baseline_mode: play_game_human((name_to_id, id_to_name), edge_dict)
    else:  # run some number of trials to test human!
        import os.path
        if os.path.exists('mytestresults.npy'):
            print('mytestresults.npy already exists in this directory! please delete or rename it')
            exit()
        num_trials = 20
        res = human_baseline(num_trials, (name_to_id, id_to_name), edge_dict)
        print('good work human!')
        print('saving human test results to mytestresults.npy')
        np.save('mytestresults.npy', res)
