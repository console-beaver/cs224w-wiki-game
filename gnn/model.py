#!/usr/bin/env python3
"""
GNN models for Wiki Game navigation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNavigator(nn.Module):
    """
    MLP baseline for Wiki Game navigation.

    For each neighbor, computes a score based on:
        - Current node embedding
        - Target node embedding
        - Neighbor node embedding

    Architecture:
        concat(current, target, neighbor) -> MLP -> score
    """

    def __init__(self, num_nodes, embed_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # Node embeddings (learned)
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)

        # MLP for scoring neighbors
        # Input: concat(current, target, neighbor) = 3 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings
        nn.init.xavier_uniform_(self.node_embedding.weight)

        # Initialize MLP
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, current_idx, target_idx, neighbor_indices, neighbor_mask):
        """
        Args:
            current_idx: [batch_size] Current node indices
            target_idx: [batch_size] Target node indices
            neighbor_indices: [batch_size, max_neighbors] Neighbor indices (padded)
            neighbor_mask: [batch_size, max_neighbors] Valid neighbor mask

        Returns:
            scores: [batch_size, max_neighbors] Score for each neighbor
        """
        batch_size, max_neighbors = neighbor_indices.shape

        # Get embeddings
        current_emb = self.node_embedding(current_idx)  # [batch, embed_dim]
        target_emb = self.node_embedding(target_idx)    # [batch, embed_dim]
        neighbor_emb = self.node_embedding(neighbor_indices)  # [batch, max_neighbors, embed_dim]

        # Expand current and target for each neighbor
        current_expanded = current_emb.unsqueeze(1).expand(-1, max_neighbors, -1)
        target_expanded = target_emb.unsqueeze(1).expand(-1, max_neighbors, -1)

        # Concatenate: [batch, max_neighbors, 3 * embed_dim]
        combined = torch.cat([current_expanded, target_expanded, neighbor_emb], dim=-1)

        # Score each neighbor
        scores = self.mlp(combined).squeeze(-1)  # [batch, max_neighbors]

        # Mask out invalid neighbors with large negative value
        scores = scores.masked_fill(~neighbor_mask, float('-inf'))

        return scores

    def predict(self, current_idx, target_idx, neighbor_indices, neighbor_mask):
        """Get predicted neighbor (highest score)."""
        scores = self.forward(current_idx, target_idx, neighbor_indices, neighbor_mask)
        return scores.argmax(dim=-1)


class GraphSAGENavigator(nn.Module):
    """
    GraphSAGE-style model for Wiki Game navigation.

    Uses mean aggregation over neighbors, conditioned on target.
    """

    def __init__(self, num_nodes, embed_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)

        # Aggregation MLP (combines current with neighbor aggregate)
        self.aggregate_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Scoring MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        for module in list(self.aggregate_mlp) + list(self.score_mlp):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, current_idx, target_idx, neighbor_indices, neighbor_mask):
        batch_size, max_neighbors = neighbor_indices.shape

        # Get embeddings
        current_emb = self.node_embedding(current_idx)
        target_emb = self.node_embedding(target_idx)
        neighbor_emb = self.node_embedding(neighbor_indices)

        # Mean aggregation over valid neighbors
        mask_expanded = neighbor_mask.unsqueeze(-1).float()
        neighbor_sum = (neighbor_emb * mask_expanded).sum(dim=1)
        neighbor_count = mask_expanded.sum(dim=1).clamp(min=1)
        neighbor_agg = neighbor_sum / neighbor_count  # [batch, embed_dim]

        # Combine current with aggregated neighbors
        combined_context = torch.cat([current_emb, neighbor_agg], dim=-1)
        context_emb = self.aggregate_mlp(combined_context)  # [batch, embed_dim]

        # Score each neighbor using context + target + neighbor
        context_expanded = context_emb.unsqueeze(1).expand(-1, max_neighbors, -1)
        target_expanded = target_emb.unsqueeze(1).expand(-1, max_neighbors, -1)

        score_input = torch.cat([context_expanded, target_expanded, neighbor_emb], dim=-1)
        scores = self.score_mlp(score_input).squeeze(-1)

        scores = scores.masked_fill(~neighbor_mask, float('-inf'))

        return scores

    def predict(self, current_idx, target_idx, neighbor_indices, neighbor_mask):
        scores = self.forward(current_idx, target_idx, neighbor_indices, neighbor_mask)
        return scores.argmax(dim=-1)


def get_model(model_name, num_nodes, **kwargs):
    """Factory function to get model by name."""
    models = {
        'mlp': MLPNavigator,
        'graphsage': GraphSAGENavigator,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(models.keys())}")
    return models[model_name](num_nodes, **kwargs)
