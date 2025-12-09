#!/usr/bin/env python3

from util import fetch_dataset, sample_src_dst
import numpy as np
import sys
import random

def human_baseline(num_trials, page_names, page_edges):
    name_to_id, id_to_name = page_names
    data = []
    for t in range(num_trials):
        print(f'STARTING TRIAL {t+1}/{num_trials}')
        random.seed(12736712376 + t)  # magic number for consistency
        pos, end, length = sample_src_dst(id_to_name, page_edges)
        path = play_game_human(page_names, page_edges, posend=(pos, end), say_results=True)
        print(f'\n\nyou finished the game in {len(path)-1} clicks!')
        print(f'shortest path had {length - 1} clicks\n\n')
        data.append((length, path))
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
        choice = input('next link? : ')
        try: choice = int(choice)
        except: choice = -1  # guaranteed fail
        while choice <= 0 or choice > len(options):
            choice = input('invalid selection, next link? : ')
            try: choice = int(choice)
            except: choice = -1  # guaranteed fail
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
        if os.path.exists('mytestresults.pkl'):
            print('mytestresults.pkl already exists in this directory! please delete or rename it')
            exit()
        num_trials = 20
        res = human_baseline(num_trials, (name_to_id, id_to_name), edge_dict)
        print('good work human!')
        print('saving human test results to mytestresults.pkl')
        import pickle
        with open('mytestresults.pkl', 'wb') as f:
            pickle.dump(res, f)
