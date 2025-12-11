"""
This script takes the .json coming from training the GNN as a single argument, and creates / shows
a plot of the loss over each epoch.
"""
import json
import sys
import matplotlib.pyplot as plt


# Given a filepath for the training of a .json file, creates a graph of the loss for each epoch.
def graph_train_loss(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    epochs = []
    accuracies = []
    
    for i, obj in enumerate(data):
        # print(obj)
        if 'train_loss' in obj:
            accuracies.append(obj['train_loss'])
            epochs.append(i)
    # print(accuracies)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Accuracy Over Epochs: GraphSAGE')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    print("Usage: python3 ./graph_train_loss.py <json file from GNN training>")
    if len(sys.argv) > 1:
        graph_train_loss(sys.argv[1])