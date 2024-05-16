import os
import numpy as np
import pandas as pd
import requests
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from utils.graph import GraphHelper

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 16)  # Adjust input feature size to match node feature dimension
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, 2)  # Adjust output size based on number of classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0, keepdim=True)  # Global mean pooling
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":

    # Main script
    BASE_DIR = './assets/graphs'
    files = os.listdir(BASE_DIR)

    graph_helper = GraphHelper()
    graphs = []
    labels = []

    for filepath in files:
        if filepath.endswith('.gexf'):
            graph = graph_helper.load_transaction_graph_from_gexf(f'{BASE_DIR}/{filepath}')
            print(f'{BASE_DIR}/{filepath}')
            print(graph)
            graph_pyg = from_networkx(graph)
            # graph_helper.show(graph)
            # Add dummy node features for demonstration
            num_nodes = graph_pyg.num_nodes
            graph_pyg.x = torch.tensor(np.random.rand(num_nodes, 2), dtype=torch.float)

            # Add dummy label for demonstration
            graph_pyg.y = torch.tensor([0], dtype=torch.long)

            graphs.append(graph_pyg)

    # Create DataLoader for batch processing
    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    # Training function
    def train(loader):
        model.train()
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)


    # Training loop
    for epoch in range(50):
        loss = train(loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    # Test the model
    model.eval()
    correct = 0
    for data in loader:
        _, pred = model(data).max(dim=1)
        correct += int((pred == data.y).sum())
    accuracy = correct / len(loader)
    print(f'Accuracy: {accuracy:.4f}')
