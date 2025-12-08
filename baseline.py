#!/usr/bin/env python3
"""
BFS baseline for Wikipedia Game.
Randomly picks two articles and finds shortest path.

Usage: python baseline.py
"""

import random
import pickle
import os
from util import load_full_dataset_from_folder, bfs


def test_random_path():
    """Pick random articles and find shortest path."""
    # Load full Wikipedia graph (from cache if available)
    print("Loading graph...")

    # Check if full dataset is cached
    if (os.path.exists('pickles/edge_dict_full.pkl') and
        os.path.exists('pickles/id_to_name_full.pkl') and
        os.path.exists('pickles/name_to_id_full.pkl')):
        print("Loading from cache (fast)...")
        with open('pickles/edge_dict_full.pkl', 'rb') as f:
            edge_dict = pickle.load(f)
        with open('pickles/id_to_name_full.pkl', 'rb') as f:
            id_to_name = pickle.load(f)
        with open('pickles/name_to_id_full.pkl', 'rb') as f:
            name_to_id = pickle.load(f)
    else:
        print("No cache found. Run 'python cache_full_dataset.py' first to cache the dataset.")
        print("Loading from raw files (this will take ~5-10 minutes)...")
        name_to_id, id_to_name, edge_dict = load_full_dataset_from_folder()

    print(f"Loaded {len(id_to_name):,} articles\n")

    # Pick random start and target
    start, target = random.sample(list(id_to_name.keys()), 2)

    print(f"Start:  {id_to_name[start]}")
    print(f"Target: {id_to_name[target]}\n")

    # Find path
    path = bfs(edge_dict, start, target)

    # Print result
    if path:
        print(f"Path length: {len(path) - 1} clicks\n")
        print("Path:")
        for i, node_id in enumerate(path):
            print(f"  {i}. {id_to_name[node_id]}")
    else:
        print("No path found (disconnected)")


if __name__ == '__main__':
    test_random_path()
