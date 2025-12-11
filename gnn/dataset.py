#!/usr/bin/env python3
"""
Generate training data for Wiki Game GNN from BFS optimal paths.

Usage: python -m gnn.dataset [--num-pairs 1000] [--subsample 1000]

Output files saved to training_data/:
  - train_samples.pkl
  - val_samples.pkl
  - test_samples.pkl
  - metadata.pkl
"""

import os
import sys
import pickle
import random
import argparse
from collections import deque
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import fetch_dataset


def build_reverse_graph(edge_dict):
    """Build reverse graph for backward BFS from target."""
    reverse = {}
    for node, neighbors in edge_dict.items():
        for neighbor in neighbors:
            if neighbor not in reverse:
                reverse[neighbor] = set()
            reverse[neighbor].add(node)
    return reverse


def bfs_distances_from_target(reverse_graph, target, max_dist=10):
    """
    Run BFS backward from target to compute distances.
    Returns dict: {node_id: distance_to_target}
    """
    distances = {target: 0}
    queue = deque([target])

    while queue:
        current = queue.popleft()
        current_dist = distances[current]

        if current_dist >= max_dist:
            continue

        # Traverse reverse edges (nodes that link TO current)
        for prev_node in reverse_graph.get(current, set()):
            if prev_node not in distances:
                distances[prev_node] = current_dist + 1
                queue.append(prev_node)

    return distances


def generate_samples_from_pair(edge_dict, reverse_graph, source, target, max_path_len=10):
    """
    Generate training samples from one (source, target) pair.

    Returns list of samples, where each sample represents one step along
    the optimal path.
    """
    # Get distances from all reachable nodes to target
    distances = bfs_distances_from_target(reverse_graph, target, max_dist=max_path_len)

    # Check if source can reach target
    if source not in distances:
        return []

    samples = []
    current = source
    visited = {current}

    # Walk the optimal path, generating a sample at each step
    while current != target:
        neighbors = list(edge_dict.get(current, set()))

        if not neighbors:
            break

        # Get distances for each neighbor
        neighbor_distances = []
        for n in neighbors:
            if n in distances:
                neighbor_distances.append(distances[n])
            else:
                neighbor_distances.append(-1)  # Unreachable

        # Find optimal neighbors (those with minimum distance)
        reachable_dists = [d for d in neighbor_distances if d >= 0]
        if not reachable_dists:
            break

        min_dist = min(reachable_dists)
        optimal_mask = [d == min_dist for d in neighbor_distances]

        # Create sample
        sample = {
            'current_id': current,
            'target_id': target,
            'neighbor_ids': neighbors,
            'neighbor_distances': neighbor_distances,
            'optimal_mask': optimal_mask,
        }
        samples.append(sample)

        # Move to an optimal neighbor (pick first one for determinism)
        next_node = None
        for i, is_optimal in enumerate(optimal_mask):
            if is_optimal and neighbors[i] not in visited:
                next_node = neighbors[i]
                break

        # If all optimal neighbors visited, pick any unvisited reachable one
        if next_node is None:
            for i, dist in enumerate(neighbor_distances):
                if dist >= 0 and neighbors[i] not in visited:
                    next_node = neighbors[i]
                    break

        if next_node is None:
            break  # Stuck (shouldn't happen if path exists)

        visited.add(next_node)
        current = next_node

        # Safety limit
        if len(samples) >= max_path_len:
            break

    return samples


def generate_training_data(edge_dict, num_pairs=1000, max_path_len=10, seed=42,
                           min_samples_per_target=3):
    """
    Generate training data ensuring all nodes appear as targets.

    Strategy:
    1. First, ensure every node appears as a target at least min_samples_per_target times
    2. Then, fill remaining pairs randomly

    Returns:
        all_samples: list of all training samples
        pair_indices: list of (start_idx, end_idx) marking samples per pair
    """
    random.seed(seed)

    print("Building reverse graph...")
    reverse_graph = build_reverse_graph(edge_dict)

    # Get all nodes that have outgoing edges (can be sources)
    source_candidates = list(edge_dict.keys())
    # Get all nodes (can be targets)
    all_nodes = set(edge_dict.keys())
    for neighbors in edge_dict.values():
        all_nodes.update(neighbors)
    target_candidates = list(all_nodes)

    print(f"Graph has {len(all_nodes)} nodes, {len(source_candidates)} with outgoing edges")

    all_samples = []
    pair_indices = []  # Track which samples belong to which pair
    target_counts = {t: 0 for t in target_candidates}  # Track samples per target

    # Phase 1: Ensure all nodes appear as targets at least min_samples_per_target times
    print(f"Phase 1: Ensuring all {len(target_candidates)} nodes appear as targets (min {min_samples_per_target}x each)...")

    uncovered_targets = set(target_candidates)
    pairs_tried = 0
    pairs_successful = 0

    pbar = tqdm(total=len(target_candidates), desc="Covering targets")
    while uncovered_targets and pairs_tried < len(target_candidates) * 50:
        # Pick an uncovered target
        target = random.choice(list(uncovered_targets))
        source = random.choice(source_candidates)

        if source == target:
            pairs_tried += 1
            continue

        samples = generate_samples_from_pair(
            edge_dict, reverse_graph, source, target, max_path_len
        )

        pairs_tried += 1

        if samples:
            start_idx = len(all_samples)
            all_samples.extend(samples)
            end_idx = len(all_samples)
            pair_indices.append((start_idx, end_idx))
            pairs_successful += 1

            target_counts[target] += 1
            if target_counts[target] >= min_samples_per_target:
                uncovered_targets.discard(target)
                pbar.update(1)

    pbar.close()
    print(f"  Phase 1: {pairs_successful} pairs, {len(uncovered_targets)} targets still uncovered")

    if uncovered_targets:
        print(f"  Warning: Could not reach {len(uncovered_targets)} targets from any source")

    # Phase 2: Generate additional random pairs up to num_pairs
    remaining_pairs = num_pairs - pairs_successful
    if remaining_pairs > 0:
        print(f"Phase 2: Generating {remaining_pairs} additional random pairs...")

        pbar = tqdm(total=remaining_pairs, desc="Random pairs")
        phase2_tried = 0
        phase2_successful = 0

        while phase2_successful < remaining_pairs and phase2_tried < remaining_pairs * 10:
            source = random.choice(source_candidates)
            target = random.choice(target_candidates)

            if source == target:
                phase2_tried += 1
                continue

            samples = generate_samples_from_pair(
                edge_dict, reverse_graph, source, target, max_path_len
            )

            phase2_tried += 1

            if samples:
                start_idx = len(all_samples)
                all_samples.extend(samples)
                end_idx = len(all_samples)
                pair_indices.append((start_idx, end_idx))
                phase2_successful += 1
                target_counts[target] += 1
                pbar.update(1)

        pbar.close()
        pairs_successful += phase2_successful

    # Print coverage stats
    covered = sum(1 for c in target_counts.values() if c > 0)
    print(f"\nGenerated {len(all_samples)} samples from {pairs_successful} pairs")
    print(f"Target coverage: {covered}/{len(target_candidates)} nodes ({100*covered/len(target_candidates):.1f}%)")
    if covered > 0:
        print(f"Min samples per covered target: {min(c for c in target_counts.values() if c > 0)}")
        print(f"Max samples per target: {max(target_counts.values())}")

    return all_samples, pair_indices


def split_by_pairs(all_samples, pair_indices, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split samples by source-target pairs (not individual samples).
    This ensures no data leakage between splits.
    """
    random.seed(seed)

    num_pairs = len(pair_indices)
    indices = list(range(num_pairs))
    random.shuffle(indices)

    train_end = int(num_pairs * train_ratio)
    val_end = int(num_pairs * (train_ratio + val_ratio))

    train_pair_indices = indices[:train_end]
    val_pair_indices = indices[train_end:val_end]
    test_pair_indices = indices[val_end:]

    def gather_samples(pair_idx_list):
        samples = []
        for pi in pair_idx_list:
            start, end = pair_indices[pi]
            samples.extend(all_samples[start:end])
        return samples

    train_samples = gather_samples(train_pair_indices)
    val_samples = gather_samples(val_pair_indices)
    test_samples = gather_samples(test_pair_indices)

    return train_samples, val_samples, test_samples


def main():
    parser = argparse.ArgumentParser(description='Generate training data for Wiki Game GNN')
    parser.add_argument('--num-pairs', type=int, default=1000,
                        help='Number of source-target pairs to sample')
    parser.add_argument('--subsample', type=int, default=1000,
                        help='Node subsample size (must have pickle files)')
    parser.add_argument('--max-path-len', type=int, default=10,
                        help='Maximum path length to consider')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='training_data',
                        help='Output directory for pickle files')
    parser.add_argument('--min-per-target', type=int, default=3,
                        help='Minimum samples per target node for coverage')
    args = parser.parse_args()

    # Load graph
    print(f"Loading top {args.subsample} node subsample...")
    result = fetch_dataset(args.subsample)
    if result is None:
        print("Failed to load dataset")
        return
    name_to_id, id_to_name, edge_dict = result

    # Generate training data
    all_samples, pair_indices = generate_training_data(
        edge_dict,
        num_pairs=args.num_pairs,
        max_path_len=args.max_path_len,
        seed=args.seed,
        min_samples_per_target=args.min_per_target
    )

    if not all_samples:
        print("No samples generated!")
        return

    # Split into train/val/test
    print("\nSplitting into train/val/test...")
    train_samples, val_samples, test_samples = split_by_pairs(
        all_samples, pair_indices, seed=args.seed
    )

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save samples
    print(f"\nSaving to {args.output_dir}/...")

    with open(os.path.join(args.output_dir, 'train_samples.pkl'), 'wb') as f:
        pickle.dump(train_samples, f)

    with open(os.path.join(args.output_dir, 'val_samples.pkl'), 'wb') as f:
        pickle.dump(val_samples, f)

    with open(os.path.join(args.output_dir, 'test_samples.pkl'), 'wb') as f:
        pickle.dump(test_samples, f)

    # Save metadata
    metadata = {
        'subsample_size': args.subsample,
        'num_pairs': len(pair_indices),
        'num_samples': len(all_samples),
        'max_path_len': args.max_path_len,
        'seed': args.seed,
        'train_size': len(train_samples),
        'val_size': len(val_samples),
        'test_size': len(test_samples),
        'id_to_name': id_to_name,
        'name_to_id': name_to_id,
    }

    with open(os.path.join(args.output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print("Done!")

    # Print sample statistics
    print("\n--- Sample Statistics ---")
    avg_neighbors = sum(len(s['neighbor_ids']) for s in all_samples) / len(all_samples)
    avg_optimal = sum(sum(s['optimal_mask']) for s in all_samples) / len(all_samples)
    print(f"Avg neighbors per sample: {avg_neighbors:.1f}")
    print(f"Avg optimal neighbors: {avg_optimal:.1f}")


if __name__ == '__main__':
    main()
