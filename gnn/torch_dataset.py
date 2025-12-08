#!/usr/bin/env python3
"""
PyTorch Dataset for Wiki Game GNN training.
"""

import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


class WikiGameDataset(Dataset):
    """
    Dataset for Wiki Game navigation task.

    Each sample contains:
        - current_idx: Index of current node
        - target_idx: Index of target node
        - neighbor_indices: Padded tensor of neighbor indices
        - neighbor_mask: Boolean mask for valid neighbors (not padding)
        - labels: Multi-hot tensor indicating optimal neighbors
    """

    def __init__(self, samples, node_to_idx, max_neighbors=None):
        """
        Args:
            samples: List of sample dicts from dataset.py
            node_to_idx: Dict mapping node IDs to contiguous indices
            max_neighbors: Max neighbors to consider (None = use max in data)
        """
        self.samples = samples
        self.node_to_idx = node_to_idx
        self.num_nodes = len(node_to_idx)

        # Find max neighbors if not specified
        if max_neighbors is None:
            max_neighbors = max(len(s['neighbor_ids']) for s in samples)
        self.max_neighbors = max_neighbors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Map node IDs to indices
        current_idx = self.node_to_idx[sample['current_id']]
        target_idx = self.node_to_idx[sample['target_id']]

        # Handle neighbors (with padding)
        neighbor_ids = sample['neighbor_ids']
        optimal_mask = sample['optimal_mask']

        num_neighbors = min(len(neighbor_ids), self.max_neighbors)

        # Initialize padded arrays
        neighbor_indices = torch.zeros(self.max_neighbors, dtype=torch.long)
        neighbor_mask = torch.zeros(self.max_neighbors, dtype=torch.bool)
        labels = torch.zeros(self.max_neighbors, dtype=torch.float)

        # Fill in actual neighbors
        for i in range(num_neighbors):
            nid = neighbor_ids[i]
            if nid in self.node_to_idx:
                neighbor_indices[i] = self.node_to_idx[nid]
                neighbor_mask[i] = True
                labels[i] = float(optimal_mask[i])

        return {
            'current_idx': torch.tensor(current_idx, dtype=torch.long),
            'target_idx': torch.tensor(target_idx, dtype=torch.long),
            'neighbor_indices': neighbor_indices,
            'neighbor_mask': neighbor_mask,
            'labels': labels,
            'num_neighbors': torch.tensor(num_neighbors, dtype=torch.long),
        }


def load_datasets(data_dir='training_data', max_neighbors=100):
    """
    Load train/val/test datasets from pickle files.

    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    # Load samples
    with open(os.path.join(data_dir, 'train_samples.pkl'), 'rb') as f:
        train_samples = pickle.load(f)
    with open(os.path.join(data_dir, 'val_samples.pkl'), 'rb') as f:
        val_samples = pickle.load(f)
    with open(os.path.join(data_dir, 'test_samples.pkl'), 'rb') as f:
        test_samples = pickle.load(f)
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # Build node_to_idx mapping (contiguous indices for embeddings)
    all_node_ids = set()
    for samples in [train_samples, val_samples, test_samples]:
        for s in samples:
            all_node_ids.add(s['current_id'])
            all_node_ids.add(s['target_id'])
            all_node_ids.update(s['neighbor_ids'])

    node_to_idx = {nid: idx for idx, nid in enumerate(sorted(all_node_ids))}

    # Create datasets
    train_dataset = WikiGameDataset(train_samples, node_to_idx, max_neighbors)
    val_dataset = WikiGameDataset(val_samples, node_to_idx, max_neighbors)
    test_dataset = WikiGameDataset(test_samples, node_to_idx, max_neighbors)

    # Add mapping info to metadata
    metadata['node_to_idx'] = node_to_idx
    metadata['idx_to_node'] = {v: k for k, v in node_to_idx.items()}
    metadata['num_nodes'] = len(node_to_idx)

    return train_dataset, val_dataset, test_dataset, metadata


if __name__ == '__main__':
    # Test loading
    train_ds, val_ds, test_ds, meta = load_datasets()
    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples")
    print(f"Num nodes: {meta['num_nodes']}")

    # Test a sample
    sample = train_ds[0]
    print(f"\nSample 0:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
