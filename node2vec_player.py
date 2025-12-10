#!/usr/bin/env python3

"""
This code is lightly modified from the human_player.py python script.
It currently only works with the 1000-node subsample. To simulate the node2vec-based agent,
simply run `python3 node2vec_player.py 1000` and on each trial, repeatedly press (1) to select
the topmost choice.
"""


from util import fetch_dataset, sample_src_dst
import numpy as np
import sys
import random
import os.path
from gensim.models import KeyedVectors



def node2vec_baseline(num_trials, page_names, page_edges):
    _, id_to_name = page_names
    embeddings = KeyedVectors.load_word2vec_format('embeddings/node_embeddings_1000.kv')
    data = []
    for t in range(num_trials):
        print(f'STARTING TRIAL {t+1}/{num_trials}')
        pos, end, length = sample_src_dst(id_to_name, page_edges, seed=12736712376 + t)
        path = play_game_node2vec(page_names, page_edges, embeddings, posend=(pos, end), say_results=True)
        print(f'\n\nyou finished the game in {len(path)-1} clicks!')
        print(f'shortest path had {length - 1} clicks\n\n')
        data.append((length, path))
    return data

def play_game_node2vec(page_names, page_edges, embeddings, posend=None, say_results=True):
    name_to_id, id_to_name = page_names
    if posend is None: pos, end = get_start_end_name(name_to_id)
    else: pos, end = posend
    # ASSUMPTION: start and end were entered correctly and are reasonable
    path = [pos]
    traversed_edges = set()
    while pos != end:
        print('\n\ncurrent article:', id_to_name[pos])
        print('target article:', id_to_name[end])
        print('clicks made so far:', len(path) - 1)
        print('type a node_id to jump to the next article from the list below:')
        options = []
        end_embedding = embeddings[str(end)]
        # Sort neighbors by cosine similarity to target
        neighbors_with_sim = []
        for neighbor in page_edges[pos]:
            neighbor_embedding = embeddings[str(neighbor)]
            similarity = np.dot(neighbor_embedding, end_embedding) / (np.linalg.norm(neighbor_embedding) * np.linalg.norm(end_embedding))
            edge = (pos, neighbor)
            print(edge)
            traversed = edge in traversed_edges
            neighbors_with_sim.append((neighbor, similarity, traversed))
        # Sort by traversed status (untraversed first), then by similarity
        sorted_neighbors = sorted(neighbors_with_sim, key=lambda x: (x[2], -x[1]))
        for i, (neighbor, sim, traversed) in enumerate(sorted_neighbors):
            print('\t(', i+1, ')\t:', id_to_name[neighbor], f'(sim: {sim:.3f})', "traversed:", traversed)
            options.append(neighbor)
        choice = input('next link? : ')
        try: choice = int(choice)
        except: choice = -1
        while choice <= 0 or choice > len(options):
            choice = input('invalid selection, next link? : ')
            try: choice = int(choice)
            except: choice = -1
        pos = options[choice-1]
        traversed_edges.add((path[-1], pos))
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
    baseline_mode = True
    if len(sys.argv) > 1: n = int(sys.argv[1])
    res = fetch_dataset(n)
    if not res: exit()  # exit if dataset fetching fails
    name_to_id, id_to_name, edge_dict = res
    if os.path.exists('node2vecresults.pkl'):
        print('node2vecresults.pkl already exists in this directory! please delete or rename it')
        exit()
    num_trials = 20
    res = node2vec_baseline(num_trials, (name_to_id, id_to_name), edge_dict)
    print('saving human test results to mytestresults.pkl')
    import pickle
    with open('mytestresults.pkl', 'wb') as f:
        pickle.dump(res, f)
