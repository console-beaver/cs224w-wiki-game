#!/usr/bin/env python3

import pickle
from util import fetch_dataset
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys

FILENAMES = (
                'testresults_michael.pkl',
                'testresults_cary.pkl'
            )
NUM_PROMPTS = 20

# WARNING: ASSUMES THAT ALL FILES ABOVE HAD THE SAME NUMBER OF NODES AND THE
# SAME NUMBER OF TRIALS, if it doesn't then something will break

if __name__ == '__main__':
    print('fetching dataset...')
    n = 1000
    if len(sys.argv) > 1: n = int(sys.argv[1])
    res = fetch_dataset(n)
    if not res: exit()  # exit if dataset fetching fails
    name_to_id, id_to_name, edge_dict = res
    print('fetched dataset')
    print('loading human baseline pickles...')
    pkls = []
    for name in FILENAMES:
        with open('players_to_visualize/' + name, 'rb') as f:
            pkls.append(pickle.load(f))
    print('pickles loaded')
    print('visualizing paths...')
    player_colors = ['salmon', 'turquoise', 'green', 'olive', 'gold', 'orange', 'blue', 'indigo'][:len(FILENAMES)]
    for prompt in range(NUM_PROMPTS):
        # create the subgraph
        all_node_ids = set()
        for prompts in pkls: all_node_ids.update(prompts[prompt][1])  # that's the path
        subgraph = nx.DiGraph()
        for node_id in all_node_ids:
            for neighbor in edge_dict[node_id]:
                if neighbor in all_node_ids: subgraph.add_edge(node_id, neighbor)

        # pick edge colors
        edge_colors = []
        for u, v in subgraph.edges():
            color = 'black'
            for player, prompts in enumerate(pkls):
                path = prompts[prompt][1]
                if u in path and v in path and path.index(u) + 1 == path.index(v):
                    if color == 'black': color = player_colors[player]  # one occurance
                    else: color = 'magenta'  # multiple occurances
            edge_colors.append(color)

        # pick node colors
        node_id_to_color = dict()
        for node_id in subgraph.nodes():
            for player, prompts in enumerate(pkls):
                if node_id in prompts[prompt][1]:
                    if node_id in node_id_to_color: node_id_to_color[node_id] = 'magenta'
                    else: node_id_to_color[node_id] = player_colors[player]
        node_colors = []
        for node_id in subgraph.nodes(): node_colors.append(node_id_to_color[node_id])

        # pick line styles
        edge_widths = []
        edge_styles = []
        for color in edge_colors:
            if color == 'black':
                edge_widths.append(1.0)
                edge_styles.append('dashed')
            else:
                edge_widths.append(3.0)
                edge_styles.append('solid')

        labels = { node_id : id_to_name[node_id] for node_id in subgraph.nodes() }

        # draw colored edges first, then gray after
        pos = nx.spring_layout(subgraph)
        nx.draw_networkx_edges(subgraph,
                               pos,
                               edgelist=[e for i, e in enumerate(subgraph.edges()) if edge_colors[i] != 'black'],
                               edge_color=[edge_colors[i] for i, e in enumerate(subgraph.edges()) if edge_colors[i] != 'black'],
                               style=[edge_styles[i] for i, e in enumerate(subgraph.edges()) if edge_colors[i] != 'black'],
                               width=[edge_widths[i] for i, e in enumerate(subgraph.edges()) if edge_colors[i] != 'black']
                              )

        nx.draw_networkx_edges(subgraph,
                               pos,
                               edgelist=[e for i, e in enumerate(subgraph.edges()) if edge_colors[i] == 'black'],
                               edge_color=[edge_colors[i] for i, e in enumerate(subgraph.edges()) if edge_colors[i] == 'black'],
                               style=[edge_styles[i] for i, e in enumerate(subgraph.edges()) if edge_colors[i] == 'black'],
                               width=[edge_widths[i] for i, e in enumerate(subgraph.edges()) if edge_colors[i] == 'black']
                              )

        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors)

        # draw colored node labels
        for node_id in subgraph.nodes():
            color = 'magenta' if node_id in [pkls[0][prompt][1][0], pkls[0][prompt][1][-1]] else 'black'
            bbox = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.4)
            plt.text(pos[node_id][0], pos[node_id][1], labels[node_id], bbox=bbox, color='black', ha='center', va='center')

        # draw colored legend
        legend_elements = []
        for player, (filename, prompts) in enumerate(zip(FILENAMES, pkls)):
            name = filename.split('_')[1].split('.')[0]
            path = prompts[prompt][1]
            color = player_colors[player]
            length = len(path)
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=f'{name}: {length}'))

        plt.legend(handles=legend_elements, loc='upper left')

        plt.margins(0.2)
        plt.title(f'[{prompt+1}/{NUM_PROMPTS}] {id_to_name[pkls[0][prompt][1][0]]} -> {id_to_name[pkls[0][prompt][1][-1]]} [min dist {pkls[0][prompt][0]}]')
        plt.show()
