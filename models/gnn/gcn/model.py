import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 16)  # Adjust input feature size to match node feature dimension
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, num_classes)  # Adjust output size based on number of classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0, keepdim=True)  # Global mean pooling
        x = self.fc(x)
        return F.log_softmax(x, dim=1)