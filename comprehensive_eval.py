#!/usr/bin/env python3
"""
Comprehensive evaluation comparing GNN models vs baselines on the Wikipedia Game.

Evaluates:
- MLP model
- GraphSAGE model
- Node2Vec baseline
- Random baseline

All methods are tested on the same random source-target pairs for fair comparison.

Usage:
    python comprehensive_eval.py --num-trials 200 --seed 42
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import pickle
import random
from collections import deque
import numpy as np

import torch
from tqdm import tqdm
from gensim.models import KeyedVectors

from util import fetch_dataset, bfs
from gnn.model import get_model
from gnn.torch_dataset import load_datasets


def load_gnn_model(checkpoint_path, device):
    """Load a trained GNN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']

    _, _, _, metadata = load_datasets(
        data_dir=args['data_dir'],
        max_neighbors=args.get('max_neighbors', 100)
    )

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

    return model, metadata, args['model']


def gnn_navigate(model, edge_dict, source, target, node_to_idx, device, max_steps=20):
    """Navigate using GNN model."""
    current = source
    path = [current]
    visited_edges = set()

    with torch.no_grad():
        for _ in range(max_steps):
            if current == target:
                return True, len(path) - 1, path

            neighbors = list(edge_dict.get(current, set()))
            if not neighbors:
                return False, -1, path

            # Prepare tensors
            current_idx = node_to_idx.get(current, 0)
            target_idx = node_to_idx.get(target, 0)
            neighbor_indices = [node_to_idx.get(n, 0) for n in neighbors]

            num_neighbors = len(neighbor_indices)
            max_neighbors = 100

            if num_neighbors > max_neighbors:
                neighbor_indices = neighbor_indices[:max_neighbors]
                neighbors = neighbors[:max_neighbors]
                num_neighbors = max_neighbors

            padded = neighbor_indices + [0] * (max_neighbors - num_neighbors)
            mask = [True] * num_neighbors + [False] * (max_neighbors - num_neighbors)

            current_t = torch.tensor([current_idx], device=device)
            target_t = torch.tensor([target_idx], device=device)
            neighbor_t = torch.tensor([padded], device=device)
            mask_t = torch.tensor([mask], dtype=torch.bool, device=device)

            scores = model(current_t, target_t, neighbor_t, mask_t)
            scores = scores[0, :num_neighbors].cpu().numpy()

            # Pick best unvisited neighbor
            sorted_indices = np.argsort(-scores)
            next_node = None

            for idx in sorted_indices:
                candidate = neighbors[idx]
                if (current, candidate) not in visited_edges:
                    next_node = candidate
                    break

            if next_node is None:
                next_node = neighbors[sorted_indices[0]]

            visited_edges.add((current, next_node))
            path.append(next_node)
            current = next_node

    return False, -1, path


def node2vec_navigate(embeddings, edge_dict, source, target, max_steps=20):
    """Navigate using Node2Vec embeddings (cosine similarity to target)."""
    current = source
    path = [current]
    visited_edges = set()

    target_emb = embeddings[str(target)]

    for _ in range(max_steps):
        if current == target:
            return True, len(path) - 1, path

        neighbors = list(edge_dict.get(current, set()))
        if not neighbors:
            return False, -1, path

        # Score by cosine similarity to target
        scores = []
        for n in neighbors:
            n_emb = embeddings[str(n)]
            sim = np.dot(n_emb, target_emb) / (np.linalg.norm(n_emb) * np.linalg.norm(target_emb))
            traversed = (current, n) in visited_edges
            scores.append((n, sim, traversed))

        # Sort: untraversed first, then by similarity
        sorted_neighbors = sorted(scores, key=lambda x: (x[2], -x[1]))
        next_node = sorted_neighbors[0][0]

        visited_edges.add((current, next_node))
        path.append(next_node)
        current = next_node

    return False, -1, path


def random_navigate(edge_dict, source, target, max_steps=20):
    """Random baseline: pick random unvisited neighbor."""
    current = source
    path = [current]
    visited = {current}

    for _ in range(max_steps):
        if current == target:
            return True, len(path) - 1, path

        neighbors = list(edge_dict.get(current, set()))
        if not neighbors:
            return False, -1, path

        unvisited = [n for n in neighbors if n not in visited]
        next_node = random.choice(unvisited) if unvisited else random.choice(neighbors)

        visited.add(next_node)
        path.append(next_node)
        current = next_node

    return False, -1, path


def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of Wiki Game methods')
    parser.add_argument('--num-trials', type=int, default=200, help='Number of trials')
    parser.add_argument('--max-steps', type=int, default=20, help='Max steps per navigation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--subsample', type=int, default=1000, help='Node subsample size')
    parser.add_argument('--save-results', type=str, default='comprehensive_eval_results.pkl',
                        help='Output file for detailed results')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load graph
    print(f"\nLoading graph (top {args.subsample} nodes)...")
    result = fetch_dataset(args.subsample)
    if result is None:
        print("Failed to load dataset")
        return
    name_to_id, id_to_name, edge_dict = result

    # Load models
    print("\nLoading models...")

    # MLP
    mlp_model, mlp_metadata, _ = load_gnn_model('checkpoints/best_model.pt', device)
    mlp_node_to_idx = mlp_metadata['node_to_idx']
    print("  Loaded MLP model")

    # GraphSAGE
    graphsage_model, gs_metadata, _ = load_gnn_model('checkpoints/graphsage/best_model.pt', device)
    gs_node_to_idx = gs_metadata['node_to_idx']
    print("  Loaded GraphSAGE model")

    # Node2Vec embeddings (optional)
    n2v_embeddings = None
    try:
        n2v_embeddings = KeyedVectors.load_word2vec_format('embeddings/node_embeddings_1000.kv')
        print("  Loaded Node2Vec embeddings")
    except FileNotFoundError:
        print("  Node2Vec embeddings not found - skipping Node2Vec baseline")

    # Prepare source-target pairs
    print(f"\nGenerating {args.num_trials} valid source-target pairs...")
    source_candidates = list(edge_dict.keys())
    all_nodes = set(edge_dict.keys())
    for neighbors in edge_dict.values():
        all_nodes.update(neighbors)
    target_candidates = list(all_nodes)

    pairs = []
    attempts = 0
    pbar = tqdm(total=args.num_trials, desc="Finding valid pairs")

    while len(pairs) < args.num_trials and attempts < args.num_trials * 100:
        source = random.choice(source_candidates)
        target = random.choice(target_candidates)

        if source == target:
            attempts += 1
            continue

        # Check path exists
        optimal_path = bfs(edge_dict, source, target)
        if optimal_path is None:
            attempts += 1
            continue

        optimal_len = len(optimal_path) - 1
        pairs.append((source, target, optimal_len))
        pbar.update(1)
        attempts += 1

    pbar.close()
    print(f"  Found {len(pairs)} valid pairs")

    # Evaluate all methods
    methods = {
        'MLP': lambda s, t: gnn_navigate(mlp_model, edge_dict, s, t, mlp_node_to_idx, device, args.max_steps),
        'GraphSAGE': lambda s, t: gnn_navigate(graphsage_model, edge_dict, s, t, gs_node_to_idx, device, args.max_steps),
        'Random': lambda s, t: random_navigate(edge_dict, s, t, args.max_steps),
    }

    if n2v_embeddings is not None:
        methods['Node2Vec'] = lambda s, t: node2vec_navigate(n2v_embeddings, edge_dict, s, t, args.max_steps)

    results = {name: {'successes': [], 'optimal_matches': [], 'path_ratios': [], 'path_lengths': []}
               for name in methods}

    print(f"\nEvaluating {len(pairs)} pairs across all methods...")

    for source, target, optimal_len in tqdm(pairs, desc="Evaluating"):
        for method_name, navigate_fn in methods.items():
            success, path_len, path = navigate_fn(source, target)

            results[method_name]['successes'].append(success)

            if success:
                results[method_name]['optimal_matches'].append(path_len == optimal_len)
                results[method_name]['path_ratios'].append(path_len / optimal_len)
                results[method_name]['path_lengths'].append(path_len)
            else:
                results[method_name]['optimal_matches'].append(False)
                results[method_name]['path_ratios'].append(None)
                results[method_name]['path_lengths'].append(None)

    # Compute statistics
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 70)
    print(f"Trials: {len(pairs)}, Max Steps: {args.max_steps}, Seed: {args.seed}")
    print("=" * 70)

    print(f"\n{'Method':<15} {'Success%':>10} {'Optimal%':>10} {'Avg Ratio':>12} {'Avg Length':>12}")
    print("-" * 70)

    summary = {}
    for method_name in methods:
        r = results[method_name]

        success_rate = sum(r['successes']) / len(r['successes'])
        optimal_rate = sum(r['optimal_matches']) / len(r['optimal_matches'])

        valid_ratios = [x for x in r['path_ratios'] if x is not None]
        avg_ratio = np.mean(valid_ratios) if valid_ratios else float('inf')

        valid_lengths = [x for x in r['path_lengths'] if x is not None]
        avg_length = np.mean(valid_lengths) if valid_lengths else float('inf')

        summary[method_name] = {
            'success_rate': success_rate,
            'optimal_rate': optimal_rate,
            'avg_ratio': avg_ratio,
            'avg_length': avg_length,
            'std_ratio': np.std(valid_ratios) if valid_ratios else 0,
            'std_length': np.std(valid_lengths) if valid_lengths else 0,
        }

        print(f"{method_name:<15} {success_rate:>9.1%} {optimal_rate:>9.1%} {avg_ratio:>11.3f}x {avg_length:>11.2f}")

    # Additional statistics
    print("\n" + "-" * 70)
    print("ADDITIONAL STATISTICS")
    print("-" * 70)

    # Distribution of optimal path lengths
    optimal_lens = [p[2] for p in pairs]
    print(f"\nOptimal path length distribution:")
    print(f"  Mean: {np.mean(optimal_lens):.2f}, Std: {np.std(optimal_lens):.2f}")
    print(f"  Min: {min(optimal_lens)}, Max: {max(optimal_lens)}")

    # Path ratio statistics per method
    print(f"\nPath ratio statistics (path_length / optimal_length):")
    method_names_to_report = ['MLP', 'GraphSAGE']
    if 'Node2Vec' in methods:
        method_names_to_report.append('Node2Vec')
    for method_name in method_names_to_report:
        valid_ratios = [x for x in results[method_name]['path_ratios'] if x is not None]
        if valid_ratios:
            print(f"  {method_name}: mean={np.mean(valid_ratios):.3f}, std={np.std(valid_ratios):.3f}, "
                  f"median={np.median(valid_ratios):.3f}, max={max(valid_ratios):.3f}")

    # Save detailed results
    all_results = {
        'args': vars(args),
        'pairs': pairs,
        'results': results,
        'summary': summary,
        'optimal_lengths': optimal_lens,
    }

    with open(args.save_results, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nDetailed results saved to {args.save_results}")


if __name__ == '__main__':
    main()
