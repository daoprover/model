import torch
from torch_geometric.nn import GATConv, global_max_pool
import torch.nn as nn
import torch.nn.functional as F


class GraphGATConv(torch.nn.Module):
    def __init__(self, in_channels, edge_in_channels, num_classes=2, heads=4):
        super(GraphGATConv, self).__init__()

        # First GAT layer: Multi-head with 'heads' attention heads
        self.conv1 = GATConv(in_channels, 16, heads=heads, edge_dim=edge_in_channels, concat=True)

        # Second GAT layer: Multi-head with 'heads' attention heads
        self.conv2 = GATConv(16 * heads, 16, heads=heads, edge_dim=edge_in_channels, concat=True)

        # Fully connected layers after pooling
        self.fc1 = nn.Linear(16 * heads, 32)
        self.fc2 = nn.Linear(32, num_classes)

        # Dropout and LayerNorm
        self.dropout = nn.Dropout(p=0.4)
        self.norm1 = nn.LayerNorm(16 * heads)
        self.norm2 = nn.LayerNorm(32)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # First GAT convolution + ReLU + normalization
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.norm1(x)

        # Second GAT convolution + ReLU
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Global max pooling
        x = global_max_pool(x, batch)

        # Fully connected layers
        x = self.fc1(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)