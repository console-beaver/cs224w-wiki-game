#!/usr/bin/env python3

from util import fetch_dataset, sample_src_dst
import numpy as np
import sys
from gensim.models import KeyedVectors


def gan_baseline(num_trials, page_names, page_edges, gnn_model, embeddings):
    name_to_id, id_to_name = page_names
    data = []
    for t in range(num_trials):
        print(f'STARTING TRIAL {t+1}/{num_trials}')
        pos, end, length = sample_src_dst(id_to_name, page_edges, seed=12736712376 + t)
        path = play_game_gnn(page_names, page_edges, gnn_model, embeddings, posend=(pos, end), say_results=True)
        print(f'\n\nyou finished the game in {len(path)-1} clicks!')
        print(f'shortest path had {length - 1} clicks\n\n')
        data.append((length, path))
    return data

def choose_neighbor_id(src, neighbors, dst, traversed_edges, gnn_model, embeddings):
    # from the set of neighbor node ids, select edge to traverse
    # which has highest score, avoid repeat traversals of edges (cycle)
    choice = None
    traversed_choice = None
    highest_score = -1e9
    traversed_score = -1e9
    dst_embed = embeddings[str(dst)]
    for neighbor_id in neighbors:
        neighbor_embed = embeddings[str(neighbor_id)]
        score = ... # TODO: something like gnn_model(neighbor_embed, dst_embed)
        if (src, neighbor_id) in traversed_edges and score > traversed_score:
            choice = neighbor_id
            traversed_score = score
        elif score > highest_score:
            choice = neighbor_id
            highest_score = score
    if choice is None: return traversed_choice
    return choice

def play_game_gnn(page_names, page_edges, gnn_model, embeddings, posend=None, say_results=True):
    name_to_id, id_to_name = page_names
    assert posend is not None, 'gnn needs a start and end node'
    pos, end = posend
    path = [pos]
    traversed_edges = set()
    while pos != end:
        pos = choose_neighbor_id(list(page_edges[pos]), end, traversed_edges, gnn_model, embeddings)
        assert pos is not None, 'gnn agent entered a dead end'
        traversed_edges.add((path[-1], pos))
        path.append(pos)
    if say_results:
        print(f'\n\nyou finished the game in {len(path)-1} clicks!')
        print('path:')
        for i, link in enumerate(path):
            print(f'\t{i+1}.', id_to_name[link])
    return path

if __name__ == '__main__':
    n = None
    baseline_mode = False
    if len(sys.argv) > 2 and sys.argv[2] == 'b': baseline_mode = True
    if len(sys.argv) > 1: n = int(sys.argv[1])
    res = fetch_dataset(n)
    if not res: exit()  # exit if dataset fetching fails
    name_to_id, id_to_name, edge_dict = res

    gnn_model = None  # TODO: load the gnn_model or import it from another python file
    embeddings = KeyedVectors.load_word2vec_format(f'embeddings/node_embeddings_{n}.kv')

    if not baseline_mode: play_game_gnn((name_to_id, id_to_name), edge_dict, gnn_model, embeddings)
    else:  # run some number of trials to test robot!
        import os.path
        if os.path.exists('mytestresults.pkl'):
            print('mytestresults.pkl already exists in this directory! please delete or rename it')
            exit()
        num_trials = 20
        res = gan_baseline(num_trials, (name_to_id, id_to_name), edge_dict, gnn_model, embeddings)
        print('good work robot!')
        print('saving robot test results to mytestresults.pkl')
        import pickle
        with open('mytestresults.pkl', 'wb') as f:
            pickle.dump(res, f)
