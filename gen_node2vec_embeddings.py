#!/usr/bin/env python3

"""
Script that takes the Wikipedia sub-graph specified by the first argument, and runs Node2Vec on it
to generate embeddings in the 'embeddings' folder. These embeddings are then used by node2vec_player.py
to simulte Wikiracing games.

Usage: python3 ./gen_node2vec_embeddings.py <top K nodes from wikipedia graph>
"""

import sys
from node2vec import Node2Vec
from util import fetch_dataset
import networkx as nx
import os

# Converts from the pickle output to a networkx graph
def pickleToGraph(id_to_name, edge_dict):
    g = nx.DiGraph()
    # Add nodes with 'name' attribute
    for node_id, node_name in id_to_name.items():
        g.add_node(node_id, name=node_name)
    # Add edges
    for src_id, dst_ids in edge_dict.items():
        for dst_id in dst_ids:
            g.add_edge(src_id, dst_id)
    return g


def main():
    n = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "-h" or sys.argv[1] == "--help":
            print("Usage: ./gen_node2vec_embeddings.py [top K nodes from Wikipedia graph]")
            return
        n = int(sys.argv[1])
    res = fetch_dataset(n)
    if not res:  # fetching dataset fails
        print("Dataset Fetching Failed.")
        return
    if not os.path.isdir('embeddings'):
        print("'embeddings' directory missing. Please either (re)-run download_dataset.sh or create the embeddings directory.")
        return
    _, id_to_name, edge_dict = res
    g = pickleToGraph(id_to_name, edge_dict)
    # Run node2Vec and save results
    node2vec = Node2Vec(g, dimensions=64, walk_length=30, num_walks=200, workers=8)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format(f"embeddings/node_embeddings_{n}.kv")

if __name__ == "__main__":
    main()
