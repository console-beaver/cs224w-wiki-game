# cs224w-wiki-game
Project for cs224w, GNN for wiki game.

Authors: Michael Rybalkin, Cary Xiao, Noah Islam

## Dataset
This project uses the dataset [TODO] found here: https://snap.stanford.edu/data/enwiki-2013.html. Download the dataset using `download_dataset.sh`

## Setup
- `pip install -r requirements.txt`
- `./download_dataset.sh`

## Scripts
- `download_dataset.sh`: downloads the dataset, sets up the repo directory for the other scripts.
- `util.py`: contains utils used by other scripts. Is also runnable, and takes optional argument `n` (usage: `./util.py 10000`). When ran, the script subsamples the full dataset to only inculde the `n` highest-degree articles, and saves the objects to `pickles/...`. Edges are treated as bidirectional when degree is counted (both incoming and outgoing links are counted). Defaults to `n=1000` when unspecified.
- `human_player.py`: CLI demo of the Wiki Game, prompts for a start and target article title, then lists all neighboring article titles and prompts the user to make a selection. After the target is reached, displays the path of articles taken from start to target. Takes optional argument `n` (usage: `./human_player.py 10000`) which subsamples the dataset to only include the `n` highest-degree articles. If `n` is not specified, uses the full dataset.
- `node_names.py`: takes an argument `n` (usage: `./node_names.py 10000`), and prints the names of all nodes in the subsampled dataset of top `n` nodes of highest degree. Requires that the dataset has already been generated.
