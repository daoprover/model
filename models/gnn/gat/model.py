import torch
from torch_geometric.nn import GINEConv, global_max_pool
import torch.nn as nn
import torch.nn.functional as F


class GraphGINConv(torch.nn.Module):
    def __init__(self, in_channels, edge_in_channels, num_classes=2):
        super(GraphGINConv, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.conv1 = GINEConv(nn1, edge_dim=edge_in_channels)

        nn2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.conv2 = GINEConv(nn2, edge_dim=edge_in_channels)

        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.norm1 = nn.LayerNorm(16)
        self.norm2 = nn.LayerNorm(32)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_max_pool(x, batch)  # Global max pooling
        x = self.fc1(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)