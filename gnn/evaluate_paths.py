#!/usr/bin/env python3
"""
Path-level evaluation for Wiki Game GNN models.

Tests whether the model can navigate complete paths from source to target.

Usage:
    python -m gnn.evaluate_paths --checkpoint checkpoints/best_model.pt
"""

import os
import sys

# Fix OpenMP duplicate library issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import pickle
import random
from collections import deque

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.model import get_model
from gnn.torch_dataset import load_datasets
from util import fetch_dataset


def bfs_shortest_path(edge_dict, source, target, max_steps=20):
    """Find shortest path length using BFS. Returns path length or -1 if not found."""
    if source == target:
        return 0

    queue = deque([(source, 0)])
    visited = {source}

    while queue:
        current, dist = queue.popleft()

        if dist >= max_steps:
            continue

        for neighbor in edge_dict.get(current, set()):
            if neighbor == target:
                return dist + 1

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return -1  # No path found


def model_navigate(model, edge_dict, source, target, node_to_idx, idx_to_node,
                   device, max_steps=20):
    """
    Have the model navigate from source to target.

    Returns:
        success: bool - did it reach target?
        path_length: int - number of steps taken (-1 if failed)
        path: list - nodes visited
    """
    model.eval()
    current = source
    path = [current]
    visited = {current}

    with torch.no_grad():
        for step in range(max_steps):
            if current == target:
                return True, len(path) - 1, path

            neighbors = list(edge_dict.get(current, set()))
            if not neighbors:
                return False, -1, path  # Dead end

            # Prepare input tensors
            current_idx = torch.tensor([node_to_idx[current]], dtype=torch.long, device=device)
            target_idx = torch.tensor([node_to_idx[target]], dtype=torch.long, device=device)

            # Map neighbors to indices
            neighbor_indices = []
            valid_neighbors = []
            for n in neighbors:
                if n in node_to_idx:
                    neighbor_indices.append(node_to_idx[n])
                    valid_neighbors.append(n)

            if not valid_neighbors:
                return False, -1, path  # No valid neighbors

            # Pad to fixed size
            max_neighbors = 100
            num_valid = len(neighbor_indices)
            padded_indices = neighbor_indices + [0] * (max_neighbors - num_valid)
            neighbor_mask = [True] * num_valid + [False] * (max_neighbors - num_valid)

            neighbor_indices_t = torch.tensor([padded_indices], dtype=torch.long, device=device)
            neighbor_mask_t = torch.tensor([neighbor_mask], dtype=torch.bool, device=device)

            # Get model prediction
            scores = model(current_idx, target_idx, neighbor_indices_t, neighbor_mask_t)
            scores = scores[0, :num_valid]  # Only valid neighbors

            # Pick highest scoring neighbor (that we haven't visited if possible)
            sorted_indices = scores.argsort(descending=True)

            next_node = None
            for idx in sorted_indices:
                candidate = valid_neighbors[idx.item()]
                if candidate not in visited:
                    next_node = candidate
                    break

            # If all neighbors visited, pick the highest scoring one anyway
            if next_node is None:
                next_node = valid_neighbors[sorted_indices[0].item()]

            visited.add(next_node)
            path.append(next_node)
            current = next_node

    # Didn't reach target in max_steps
    return False, -1, path


def random_navigate(edge_dict, source, target, max_steps=20):
    """Random baseline: pick random neighbor at each step."""
    current = source
    path = [current]
    visited = {current}

    for step in range(max_steps):
        if current == target:
            return True, len(path) - 1, path

        neighbors = list(edge_dict.get(current, set()))
        if not neighbors:
            return False, -1, path

        # Prefer unvisited neighbors
        unvisited = [n for n in neighbors if n not in visited]
        if unvisited:
            next_node = random.choice(unvisited)
        else:
            next_node = random.choice(neighbors)

        visited.add(next_node)
        path.append(next_node)
        current = next_node

    return False, -1, path


def evaluate_paths(model, edge_dict, node_to_idx, idx_to_node, device,
                   num_trials=200, max_steps=20, seed=42):
    """
    Evaluate model on random source-target pairs.

    Returns dict with metrics.
    """
    random.seed(seed)

    # Get nodes that can be sources (have outgoing edges)
    source_candidates = list(edge_dict.keys())
    all_nodes = set(edge_dict.keys())
    for neighbors in edge_dict.values():
        all_nodes.update(neighbors)

    # Filter to nodes in our index
    source_candidates = [n for n in source_candidates if n in node_to_idx]
    target_candidates = [n for n in all_nodes if n in node_to_idx]

    results = {
        'model': {'success': 0, 'optimal': 0, 'total_ratio': 0, 'paths': []},
        'random': {'success': 0, 'optimal': 0, 'total_ratio': 0, 'paths': []},
    }

    valid_trials = 0
    pbar = tqdm(total=num_trials, desc='Evaluating paths')

    while valid_trials < num_trials:
        source = random.choice(source_candidates)
        target = random.choice(target_candidates)

        if source == target:
            continue

        # Get optimal path length
        optimal_len = bfs_shortest_path(edge_dict, source, target, max_steps)
        if optimal_len < 0:
            continue  # No path exists

        valid_trials += 1
        pbar.update(1)

        # Model navigation
        model_success, model_len, model_path = model_navigate(
            model, edge_dict, source, target, node_to_idx, idx_to_node, device, max_steps
        )

        if model_success:
            results['model']['success'] += 1
            if model_len == optimal_len:
                results['model']['optimal'] += 1
            results['model']['total_ratio'] += model_len / optimal_len
        results['model']['paths'].append({
            'source': source, 'target': target,
            'success': model_success, 'length': model_len, 'optimal': optimal_len
        })

        # Random baseline
        random_success, random_len, random_path = random_navigate(
            edge_dict, source, target, max_steps
        )

        if random_success:
            results['random']['success'] += 1
            if random_len == optimal_len:
                results['random']['optimal'] += 1
            results['random']['total_ratio'] += random_len / optimal_len
        results['random']['paths'].append({
            'source': source, 'target': target,
            'success': random_success, 'length': random_len, 'optimal': optimal_len
        })

    pbar.close()

    # Compute final metrics
    for method in ['model', 'random']:
        r = results[method]
        r['success_rate'] = r['success'] / num_trials
        r['optimal_rate'] = r['optimal'] / num_trials
        if r['success'] > 0:
            r['avg_ratio'] = r['total_ratio'] / r['success']
        else:
            r['avg_ratio'] = float('inf')

    return results


def main():
    parser = argparse.ArgumentParser(description='Path-level evaluation for Wiki Game GNN')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--subsample', type=int, default=1000,
                        help='Graph subsample size')
    parser.add_argument('--num-trials', type=int, default=200,
                        help='Number of source-target pairs to test')
    parser.add_argument('--max-steps', type=int, default=20,
                        help='Maximum steps allowed per navigation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load graph
    print(f"Loading graph (top {args.subsample} nodes)...")
    result = fetch_dataset(args.subsample)
    if result is None:
        print("Failed to load dataset")
        return
    name_to_id, id_to_name, edge_dict = result

    # Load training data metadata for node mapping
    print("Loading model...")
    _, _, _, metadata = load_datasets()
    node_to_idx = metadata['node_to_idx']
    idx_to_node = metadata['idx_to_node']

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['args']

    model = get_model(
        model_args['model'],
        num_nodes=metadata['num_nodes'],
        embed_dim=model_args['embed_dim'],
        hidden_dim=model_args['hidden_dim'],
        dropout=model_args['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_args['model']} model from epoch {checkpoint['epoch']}")

    # Evaluate
    print(f"\nEvaluating on {args.num_trials} random paths (max {args.max_steps} steps)...")
    results = evaluate_paths(
        model, edge_dict, node_to_idx, idx_to_node, device,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        seed=args.seed
    )

    # Print results
    print("\n" + "=" * 50)
    print("PATH-LEVEL EVALUATION RESULTS")
    print("=" * 50)

    print(f"\n{'Metric':<25} {'Model':>12} {'Random':>12}")
    print("-" * 50)
    print(f"{'Success Rate':<25} {results['model']['success_rate']:>11.1%} {results['random']['success_rate']:>11.1%}")
    print(f"{'Optimal Path Rate':<25} {results['model']['optimal_rate']:>11.1%} {results['random']['optimal_rate']:>11.1%}")
    print(f"{'Avg Path Ratio (if success)':<25} {results['model']['avg_ratio']:>11.2f}x {results['random']['avg_ratio']:>11.2f}x")

    # Show some example paths
    print("\n" + "-" * 50)
    print("EXAMPLE PATHS (first 5):")
    print("-" * 50)

    for i, (m_path, r_path) in enumerate(zip(results['model']['paths'][:5],
                                              results['random']['paths'][:5])):
        source_name = id_to_name.get(m_path['source'], str(m_path['source']))[:20]
        target_name = id_to_name.get(m_path['target'], str(m_path['target']))[:20]

        print(f"\n{i+1}. {source_name} -> {target_name}")
        print(f"   Optimal: {m_path['optimal']} steps")
        print(f"   Model:   {m_path['length'] if m_path['success'] else 'FAIL'} steps {'✓' if m_path['success'] else '✗'}")
        print(f"   Random:  {r_path['length'] if r_path['success'] else 'FAIL'} steps {'✓' if r_path['success'] else '✗'}")


if __name__ == '__main__':
    main()
