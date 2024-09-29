import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, LayerNorm, Dropout
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, JumpingKnowledge
from torch_geometric.nn import MessagePassing


class GraphGATConv(torch.nn.Module):
    def __init__(self, in_channels, edge_in_channels, num_classes=2, heads=4, hidden_channels=16, num_layers=3):
        super(GraphGATConv, self).__init__()

        # Initial node feature transformation
        self.node_proj = Linear(in_channels, hidden_channels)

        # Initial edge feature transformation
        self.edge_proj = Linear(edge_in_channels, hidden_channels)

        # GAT layers with enhanced edge conditioning
        self.gat_layers = ModuleList()
        self.edge_update_networks = ModuleList()
        self.norms = ModuleList()

        # Build multiple GAT layers with edge-conditioned transformations
        for i in range(num_layers):
            input_dim = hidden_channels * heads if i > 0 else hidden_channels
            self.gat_layers.append(
                GATConv(input_dim, hidden_channels, heads=heads, edge_dim=hidden_channels, concat=True))

            # Edge update network for transforming edge features after each GAT layer
            self.edge_update_networks.append(Linear(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels * heads))

        # Jumping Knowledge module to aggregate features from different GAT layers
        self.jump = JumpingKnowledge(mode='cat')

        # Fully connected layers for graph-level classification
        self.fc1 = Linear(hidden_channels * heads * num_layers, 64)
        self.fc2 = Linear(64, num_classes)

        # Dropout and LayerNorm for MLP layers
        self.dropout = Dropout(p=0.5)
        self.norm_fc1 = LayerNorm(64)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Project node features to higher dimension
        x = self.node_proj(x)
        x = F.relu(x)

        # Project edge features to hidden dimension
        edge_attr = self.edge_proj(edge_attr)
        edge_attr = F.relu(edge_attr)

        # Store outputs from each GAT layer for Jumping Knowledge
        xs = []

        # GAT layers with edge-conditioned transformations
        for i, conv in enumerate(self.gat_layers):
            # Perform graph convolution with edge features
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.norms[i](x)

            # Update edge features using a simple MLP
            edge_attr = self.edge_update_networks[i](edge_attr)
            edge_attr = F.relu(edge_attr)

            # Store the output of each GAT layer
            xs.append(x)

        # Jumping Knowledge to aggregate layer outputs
        x = self.jump(xs)

        # Global mean pooling + global max pooling combination
        x = global_mean_pool(x, batch) + global_max_pool(x, batch)

        # Fully connected layers with Dropout and LayerNorm
        x = self.fc1(x)
        x = F.relu(self.norm_fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)