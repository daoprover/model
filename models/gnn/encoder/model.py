import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),  # Additional hidden layer
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, encoded_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoded_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class GraphSAGEWithAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim, num_classes):
        super(GraphSAGEWithAutoencoder, self).__init__()
        self.autoencoder = Autoencoder(input_dim, hidden_dim, encoded_dim)
        self.conv1 = SAGEConv(encoded_dim, 64)  # Increase output dimension
        self.conv2 = SAGEConv(64, 64)  # Increase input and output dimensions
        self.conv3 = SAGEConv(64, 32)  # Additional convolutional layer
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        encoded, decoded = self.autoencoder(x)
        x = self.conv1(encoded, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # Apply the third convolutional layer
        x = F.relu(x)
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.fc(x)
        return torch.sigmoid(x)
