# cs224w-wiki-game
Project for cs224w, GNN for wiki game.

Authors: Michael Rybalkin, Cary Xiao, Noah Islam

## Dataset
This project uses the dataset [TODO] found here: https://snap.stanford.edu/data/enwiki-2013.html. Download the dataset using `download_dataset.sh`

## Setup
- `pip install -r requirements.txt`
- `./download_dataset.sh`

## Scripts
- `download_dataset.sh`: Downloads the dataset, sets up the repo directory for the other scripts.
- `util.py`: Contains utils used by other scripts. Is also runnable, and takes optional argument `n` (usage: `./util.py 1000`). When ran, the script subsamples the full dataset to only inculde the `n` highest-degree articles, and saves the objects to `pickles/...`. Edges are treated as bidirectional when degree is counted (both incoming and outgoing links are counted). Defaults to `n=1000` when unspecified.
- `human_player.py`: CLI demo of the Wiki Game, prompts for a start and target article title, then lists all neighboring article titles and prompts the user to make a selection. After the target is reached, displays the path of articles taken from start to target. Takes optional argument `n` (usage: `./human_player.py 1000`) which subsamples the dataset to only include the `n` highest-degree articles. If `n` is not specified, uses the full dataset. To play 20 trials and write results to a file, add an attional argument like so: `./human_player.py 1000 b`.
- `node2vec_player.py`: Simulates the simple node2vec agent playing the wiki game. Takes two args. First is `n` for the number of ndoes. Second is optional, is the number of trials to run in baseline mode (defaults to 20). Example usage: `./node2vec_player.py 1000 20`. If in baseline mode, write results of running baseline trials to filesystem.
- `gnn_player.py`: Simulates the GNN agent playing the wiki game. Takes four args. First is `n` for the number of nodes, defaults to `n=1000`. Second is `b` for baseline mode, where the GNN is evaluated on some number of trials selected randomly with seed. Third is the filepath to the checkpoint of the trained GNN. Fourth is the number of  trials to use in baseline mode (default is 20). Example usage: `./gnn_player 1000 b checkpoints/best_model.pt 200`. If in baseline mode, write results of running baseline trials to filesystem.
- `visualize_test_results.py`: Visualization script used for comparing paths of different approaches. Visualizes all path files in the dir `players_to_visualize`. Black edges showing untraversed edges between visited nodes can be disabled with the boolean flag `SHOW_GRAY_EDGES`. This script assumes that all path files were generated in baseline mode using the same src,dst node pairs and the same number of trials. Trials if the path files must be consistent with `NUM_PROMPTS`. Example usage: `./visualize_test_results.npy`.
- `node_names.py`: Takes an argument `n` (usage: `./node_names.py 1000`), and prints the names of all nodes in the subsampled dataset of top `n` nodes of highest degree. Requires that the dataset has already been generated.
- `gen_node2vec_embeddings.py`: Uses Node2vec to create node an embedding for each node in the subgraph. Saves embeddings in `embeddings` folder. Takes parameter `n` for number of nodes in the graph. Usage: `./gen_node2vec_embeddings.py 1000`.
- `baseline.py`: Plays 20 trials of the wiki game using BFS, then reports the shortest path length and number of visited nodes using BFS.
- `comprehensive_eval.py`: TODO! whoever wrote this, explain what it is
- `graph_stats.py`: TODO! whoever wrote this, explain what it is
- `graph_train_loss.py`: TODO! Whoever wrote this, explain what it is
