#!/usr/bin/env python3

from util import fetch_dataset
import sys
import networkx as nx

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

if __name__ == '__main__':
    n = None
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    res = fetch_dataset(n)
    if res:  # fetching dataset did not fail
        name_to_id, id_to_name, edge_dict = res
        g = pickleToGraph(id_to_name, edge_dict)
        print(f"Number of nodes: {g.number_of_nodes()}")
        print(f"Number of edges: {g.number_of_edges()}")
        print(f"Is weakly connected: {nx.is_weakly_connected(g)}")
        print(f"Is strongly connected: {nx.is_strongly_connected(g)}")
        print(f"Average degree: {sum(dict(g.degree()).values()) / g.number_of_nodes()}")
        print(f"Average clustering coefficient: {nx.average_clustering(g.to_undirected())}")
        print(f"Diameter of largest strongly connected component: {nx.diameter(nx.subgraph(g, max(nx.strongly_connected_components(g), key=len)))}")
        print(f"Number of nodes in largest SCC: {len(max(nx.strongly_connected_components(g), key=len))}")
