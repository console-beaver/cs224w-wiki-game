#!/usr/bin/env python3

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import sys
import torch

from util import fetch_dataset, sample_src_dst
from gnn.model import get_model


def load_gnn_model(checkpoint_path, device):
    """Load trained GNN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']

    # Use load_datasets to get proper metadata with node_to_idx
    from gnn.torch_dataset import load_datasets
    _, _, _, metadata = load_datasets(
        data_dir=args['data_dir'],
        max_neighbors=args.get('max_neighbors', 100)
    )

    # Create model with same architecture
    model = get_model(
        args['model'],
        num_nodes=metadata['num_nodes'],
        embed_dim=args['embed_dim'],
        hidden_dim=args['hidden_dim'],
        dropout=args['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # node_to_idx maps node_id -> embedding index
    id_to_idx = metadata['node_to_idx']

    return model, metadata, id_to_idx


def gnn_baseline(num_trials, page_names, page_edges, gnn_model, id_to_idx, device, max_neighbors=100):
    name_to_id, id_to_name = page_names
    data = []
    for t in range(num_trials):
        print(f'STARTING TRIAL {t+1}/{num_trials}')
        pos, end, length = sample_src_dst(id_to_name, page_edges, seed=12736712376 + t)
        path = play_game_gnn(page_names, page_edges, gnn_model, id_to_idx, device,
                            posend=(pos, end), say_results=True, max_neighbors=max_neighbors)
        print(f'shortest path had {length - 1} clicks\n\n')
        data.append((length, path))
    return data


def choose_neighbor_gnn(current_id, neighbors, target_id, traversed_edges,
                        gnn_model, id_to_idx, device, max_neighbors=100):
    """
    Use GNN to select best neighbor to traverse to.
    Avoids cycles by preferring unvisited edges.
    """
    if not neighbors:
        return None

    neighbors = list(neighbors)

    # Convert to indices
    current_idx = id_to_idx.get(current_id, 0)
    target_idx = id_to_idx.get(target_id, 0)
    neighbor_indices = [id_to_idx.get(n, 0) for n in neighbors]

    # Pad/truncate neighbors
    num_neighbors = len(neighbor_indices)
    if num_neighbors > max_neighbors:
        neighbor_indices = neighbor_indices[:max_neighbors]
        neighbors = neighbors[:max_neighbors]
        num_neighbors = max_neighbors

    # Create tensors
    current_tensor = torch.tensor([current_idx], device=device)
    target_tensor = torch.tensor([target_idx], device=device)

    # Pad neighbor indices
    padded_neighbors = neighbor_indices + [0] * (max_neighbors - num_neighbors)
    neighbor_tensor = torch.tensor([padded_neighbors], device=device)

    # Create mask
    mask = torch.zeros(1, max_neighbors, dtype=torch.bool, device=device)
    mask[0, :num_neighbors] = True

    # Get scores from model
    with torch.no_grad():
        scores = gnn_model(current_tensor, target_tensor, neighbor_tensor, mask)

    scores = scores[0, :num_neighbors].cpu().numpy()

    # Find best unvisited neighbor
    best_choice = None
    best_score = float('-inf')
    best_visited_choice = None
    best_visited_score = float('-inf')

    for i, (neighbor_id, score) in enumerate(zip(neighbors, scores)):
        if (current_id, neighbor_id) in traversed_edges:
            if score > best_visited_score:
                best_visited_score = score
                best_visited_choice = neighbor_id
        else:
            if score > best_score:
                best_score = score
                best_choice = neighbor_id

    # Prefer unvisited, fall back to visited if needed
    return best_choice if best_choice is not None else best_visited_choice


def play_game_gnn(page_names, page_edges, gnn_model, id_to_idx, device,
                  posend=None, say_results=True, max_neighbors=100, max_steps=100):
    name_to_id, id_to_name = page_names
    assert posend is not None, 'gnn needs a start and end node'
    pos, end = posend
    path = [pos]
    traversed_edges = set()

    while pos != end and len(path) < max_steps:
        neighbors = page_edges.get(pos, set())
        next_pos = choose_neighbor_gnn(
            pos, neighbors, end, traversed_edges,
            gnn_model, id_to_idx, device, max_neighbors
        )

        if next_pos is None:
            print(f'GNN agent stuck at {id_to_name[pos]} with no valid moves!')
            break

        traversed_edges.add((pos, next_pos))
        path.append(next_pos)
        pos = next_pos

    if say_results:
        if pos == end:
            print(f'\n\nGNN finished the game in {len(path)-1} clicks!')
        else:
            print(f'\n\nGNN failed to reach target in {len(path)-1} clicks')
        print('path:')
        for i, link in enumerate(path):
            print(f'\t{i+1}.', id_to_name[link])

    return path


if __name__ == '__main__':
    n = 1000  # default to 1000 node subsample
    baseline_mode = False
    checkpoint_path = 'checkpoints/best_model.pt'

    # Parse arguments: python gnn_player.py [n] [b] [checkpoint_path]
    if len(sys.argv) > 2 and sys.argv[2] == 'b':
        baseline_mode = True
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 3:
        checkpoint_path = sys.argv[3]

    # Load dataset
    res = fetch_dataset(n)
    if not res:
        exit()
    name_to_id, id_to_name, edge_dict = res

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load GNN model
    print(f'Loading GNN model from {checkpoint_path}...')
    if not os.path.exists(checkpoint_path):
        print(f'Error: checkpoint not found at {checkpoint_path}')
        print('Please train the model first: python -m gnn.train')
        exit()

    gnn_model, metadata, id_to_idx = load_gnn_model(checkpoint_path, device)
    print(f'Loaded {metadata.get("num_nodes", "?")} node model')

    if not baseline_mode:
        # Interactive single game mode
        print('Enter start and target node names to play')
        start_name = input('Start node: ')
        target_name = input('Target node: ')
        if start_name not in name_to_id or target_name not in name_to_id:
            print('Invalid node names')
            exit()
        play_game_gnn((name_to_id, id_to_name), edge_dict, gnn_model, id_to_idx, device,
                     posend=(name_to_id[start_name], name_to_id[target_name]))
    else:
        # Run benchmark trials
        import os.path
        if os.path.exists('mytestresults.pkl'):
            print('mytestresults.pkl already exists in this directory! please delete or rename it')
            exit()
        num_trials = 20
        res = gnn_baseline(num_trials, (name_to_id, id_to_name), edge_dict,
                          gnn_model, id_to_idx, device)
        print('good work robot!')
        print('saving robot test results to mytestresults.pkl')
        with open('mytestresults.pkl', 'wb') as f:
            pickle.dump(res, f)
