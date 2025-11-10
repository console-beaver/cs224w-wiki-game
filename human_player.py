#!/usr/bin/env python3

from util import fetch_dataset
import sys

def play_game_human(page_names, page_edges):
    name_to_id, id_to_name = page_names
    # pos, end = get_start_end_id(id_to_name)
    pos, end = get_start_end_name(name_to_id)
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
    print(f'\n\nyou finished the game in {len(path)-1} clicks!')
    print('path:')
    # print(path)
    for i, link in enumerate(path):
        print(f'\t{i+1}.', id_to_name[link])

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
    if len(sys.argv) > 1:
        n = sys.argv[1]
    name_to_id, id_to_name, edge_dict = fetch_dataset(n)
    play_game_human((name_to_id, id_to_name), edge_dict)
