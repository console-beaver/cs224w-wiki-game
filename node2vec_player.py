#!/usr/bin/env python3

"""
This code is lightly modified from the human_player.py python script.
It currently only works with the 1000-node subsample. To simulate the node2vec-based agent,
simply run `python3 node2vec_player.py 1000 <num_trials>`, and the results will be placed in
node2vecresults.pkl.
"""

from util import fetch_dataset, sample_src_dst
import numpy as np
import sys
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
    assert posend is not None, "node2vec needs start and end"
    pos, end = posend
    path = [pos]
    traversed_edges = set()
    while pos != end:
        # print('type a node_id to jump to the next article from the list below:')
        options = []
        end_embedding = embeddings[str(end)]
        # Sort neighbors by cosine similarity to target
        neighbors_with_sim = []
        for neighbor in page_edges[pos]:
            neighbor_embedding = embeddings[str(neighbor)]
            similarity = np.dot(neighbor_embedding, end_embedding) / (np.linalg.norm(neighbor_embedding) * np.linalg.norm(end_embedding))
            edge = (pos, neighbor)
            traversed = edge in traversed_edges
            neighbors_with_sim.append((neighbor, similarity, traversed))
        # Sort by traversed status (untraversed first), then by similarity
        sorted_neighbors = sorted(neighbors_with_sim, key=lambda x: (x[2], -x[1]))
        for i, (neighbor, sim, traversed) in enumerate(sorted_neighbors):
            # print('\t(', i+1, ')\t:', id_to_name[neighbor], f'(sim: {sim:.3f})', "traversed:", traversed)
            options.append(neighbor)
        # select topmost choice
        pos = options[0]
        traversed_edges.add((path[-1], pos))
        path.append(pos)
    if say_results:
        print(f'\n\nyou finished the game in {len(path)-1} clicks!')
        print('path:')
        for i, link in enumerate(path):
            print(f'\t{i+1}.', id_to_name[link])
    return path

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python3 node2vec_player.py [num_nodes] [num_trials]")
        print("  num_nodes: size of the node subsample (default: None for full dataset)")
        print("  num_trials: number of trials to run (default: 20)")
        exit()

    n = None
    num_trials = 20
    try:
        if len(sys.argv) > 1: n = int(sys.argv[1])
        if len(sys.argv) > 2: num_trials = int(sys.argv[2])
    except ValueError:
        print("Error: num_nodes and num_trials must be integers")
        print("Usage: python3 node2vec_player.py [num_nodes] [num_trials]")
        print("  num_nodes: size of the node subsample (default: None for full dataset)")
        print("  num_trials: number of trials to run (default: 20)")
        exit()

    res = fetch_dataset(n)
    if not res: exit()  # exit if dataset fetching fails
    name_to_id, id_to_name, edge_dict = res
    if os.path.exists('node2vecresults.pkl'):
        print('node2vecresults.pkl already exists in this directory! please delete or rename it')
        exit()
    res = node2vec_baseline(num_trials, (name_to_id, id_to_name), edge_dict)
    print('saving human test results to node2vecresults.pkl')
    import pickle
    with open('node2vecresults.pkl', 'wb') as f:
        pickle.dump(res, f)
