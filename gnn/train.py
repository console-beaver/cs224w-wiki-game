#!/usr/bin/env python3
"""
Training script for Wiki Game GNN models.

Usage:
    python -m gnn.train --model mlp --epochs 50
    python -m gnn.train --model graphsage --embed-dim 128
"""

import os
import sys

# Fix OpenMP duplicate library issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.torch_dataset import load_datasets
from gnn.model import get_model


def compute_metrics(scores, labels, neighbor_mask):
    """
    Compute evaluation metrics.

    Args:
        scores: [batch, max_neighbors] Model scores
        labels: [batch, max_neighbors] Ground truth (multi-hot)
        neighbor_mask: [batch, max_neighbors] Valid neighbor mask

    Returns:
        dict with metrics
    """
    batch_size = scores.shape[0]

    # Top-1 accuracy: Does the highest-scored neighbor have label=1?
    predictions = scores.argmax(dim=-1)  # [batch]
    top1_correct = labels.gather(1, predictions.unsqueeze(1)).squeeze(1)  # [batch]
    top1_acc = top1_correct.mean().item()

    # Top-3 accuracy: Is any of top-3 predictions optimal?
    _, top3_indices = scores.topk(3, dim=-1)  # [batch, 3]
    top3_labels = labels.gather(1, top3_indices)  # [batch, 3]
    top3_acc = (top3_labels.sum(dim=-1) > 0).float().mean().item()

    # Mean reciprocal rank (MRR)
    # Rank of first correct answer
    sorted_indices = scores.argsort(dim=-1, descending=True)
    sorted_labels = labels.gather(1, sorted_indices)
    # Find first 1 in each row
    ranks = (sorted_labels.cumsum(dim=-1) == 1).float().argmax(dim=-1) + 1
    mrr = (1.0 / ranks.float()).mean().item()

    return {
        'top1_acc': top1_acc,
        'top3_acc': top3_acc,
        'mrr': mrr,
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_metrics = {'top1_acc': 0, 'top3_acc': 0, 'mrr': 0}
    num_batches = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        # Move to device
        current_idx = batch['current_idx'].to(device)
        target_idx = batch['target_idx'].to(device)
        neighbor_indices = batch['neighbor_indices'].to(device)
        neighbor_mask = batch['neighbor_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        scores = model(current_idx, target_idx, neighbor_indices, neighbor_mask)

        # Compute loss (cross-entropy over valid neighbors)
        # Mask invalid neighbors with large negative value (not -inf to avoid NaN)
        scores_for_loss = scores.clone()
        scores_for_loss[~neighbor_mask] = -1e9
        log_probs = F.log_softmax(scores_for_loss, dim=-1)

        # Normalize labels to sum to 1 for each sample
        label_sums = labels.sum(dim=-1, keepdim=True).clamp(min=1)
        soft_labels = labels / label_sums

        # Only compute loss over valid neighbors
        log_probs_masked = log_probs * neighbor_mask.float()
        loss = -(soft_labels * log_probs_masked).sum(dim=-1).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        with torch.no_grad():
            metrics = compute_metrics(scores, labels, neighbor_mask)
            for k, v in metrics.items():
                all_metrics[k] += v
        num_batches += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    for k in all_metrics:
        all_metrics[k] /= num_batches

    return avg_loss, all_metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_metrics = {'top1_acc': 0, 'top3_acc': 0, 'mrr': 0}
    num_batches = 0

    for batch in tqdm(dataloader, desc='Evaluating', leave=False):
        current_idx = batch['current_idx'].to(device)
        target_idx = batch['target_idx'].to(device)
        neighbor_indices = batch['neighbor_indices'].to(device)
        neighbor_mask = batch['neighbor_mask'].to(device)
        labels = batch['labels'].to(device)

        scores = model(current_idx, target_idx, neighbor_indices, neighbor_mask)

        # Loss
        scores_for_loss = scores.clone()
        scores_for_loss[~neighbor_mask] = -1e9
        log_probs = F.log_softmax(scores_for_loss, dim=-1)
        label_sums = labels.sum(dim=-1, keepdim=True).clamp(min=1)
        soft_labels = labels / label_sums
        log_probs_masked = log_probs * neighbor_mask.float()
        loss = -(soft_labels * log_probs_masked).sum(dim=-1).mean()

        total_loss += loss.item()
        metrics = compute_metrics(scores, labels, neighbor_mask)
        for k, v in metrics.items():
            all_metrics[k] += v
        num_batches += 1

    avg_loss = total_loss / num_batches
    for k in all_metrics:
        all_metrics[k] /= num_batches

    return avg_loss, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Wiki Game GNN')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'graphsage'])
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--max-neighbors', type=int, default=100)
    parser.add_argument('--data-dir', type=str, default='training_data')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("Loading datasets...")
    train_ds, val_ds, test_ds, metadata = load_datasets(
        data_dir=args.data_dir,
        max_neighbors=args.max_neighbors
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Num nodes: {metadata['num_nodes']}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model
    print(f"Creating {args.model} model...")
    model = get_model(
        args.model,
        num_nodes=metadata['num_nodes'],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    best_val_acc = 0
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_metrics['top1_acc'])

        # Log
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['top1_acc']:.3f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['top1_acc']:.3f}, MRR: {val_metrics['mrr']:.3f}")

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_metrics['top1_acc'],
            'val_loss': val_loss,
            'val_acc': val_metrics['top1_acc'],
            'val_mrr': val_metrics['mrr'],
        })

        # Save best model
        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'args': vars(args),
            }, os.path.join(args.save_dir, 'best_model.pt'))

    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Test Evaluation:")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Top-1 Accuracy: {test_metrics['top1_acc']:.3f}")
    print(f"  Top-3 Accuracy: {test_metrics['top3_acc']:.3f}")
    print(f"  MRR: {test_metrics['mrr']:.3f}")

    # Load best model and evaluate
    print("\nBest Model Test Evaluation:")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Top-1 Accuracy: {test_metrics['top1_acc']:.3f}")
    print(f"  Top-3 Accuracy: {test_metrics['top3_acc']:.3f}")
    print(f"  MRR: {test_metrics['mrr']:.3f}")

    # Save history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nModel saved to {args.save_dir}/best_model.pt")


if __name__ == '__main__':
    main()
