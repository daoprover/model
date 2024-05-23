# import os
# import numpy as np
# import pandas as pd
# import requests
# import networkx as nx
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from torch_geometric.data import Data
# from torch_geometric.utils import from_networkx
# from torch_geometric.loader import DataLoader  # Updated import
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
#
# from utils.graph import GraphHelper
# from sklearn import preprocessing
#
#
# class GCN(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(2, 16)  # Adjust input feature size to match node feature dimension
#         self.conv2 = GCNConv(16, 16)
#         self.fc = torch.nn.Linear(16, num_classes)  # Adjust output size based on number of classes
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = torch.mean(x, dim=0, keepdim=True)  # Global mean pooling
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)
#
# if __name__ == "__main__":
#
#     # Main script
#     BASE_DIR = './assets/graphs'
#     files = os.listdir(BASE_DIR)
#
#     graph_helper = GraphHelper()
#     graphs = []
#     labels = []
#
#     for filepath in files:
#         if filepath.endswith('.gexf'):
#             graph, label = graph_helper.load_transaction_graph_from_gexf(f'{BASE_DIR}/{filepath}')
#             print(f"Loaded label: {label}")  # Verify the label is loaded correctly
#             graph_pyg = from_networkx(graph)
#
#             # Add dummy node features for demonstration
#             num_nodes = graph_pyg.num_nodes
#             graph_pyg.x = torch.tensor(np.random.rand(num_nodes, 2), dtype=torch.float)
#
#             # Use the label loaded from the GEXF file or add a default label if not present
#             if label is not None:
#                 labels.append(label)
#             else:
#                 labels.append("default_label")
#
#             graphs.append(graph_pyg)
#
#     # Encode string labels to integers
#     label_encoder = LabelEncoder()
#     encoded_labels = label_encoder.fit_transform(labels)
#
#     # Print unique labels and their encoded values for verification
#     print("Unique labels and their encoded values:")
#     for label, encoded_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
#         print(f"{label}: {encoded_label}")
#
#     # Assign encoded labels to the graphs
#     for i, graph_pyg in enumerate(graphs):
#         graph_pyg.y = encoded_labels[i]
#     print("graphs: ", graphs)
#
#     loader = DataLoader(graphs, batch_size=1, shuffle=True)
#
#     print("loader: ", loader)
#
#     num_classes = len(label_encoder.classes_)
#     print("num_classes: ", num_classes)
#
#     model = GCN(num_classes)
#     print("model: ", model)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#     print("optimizer: ", optimizer)
#
#     # Training function
#     def train(loader):
#         model.train()
#         total_loss = 0
#         for data in loader:
#             print("Data batch:", data)
#             print("Data batch x shape:", data.x.shape)
#             print("Data batch edge_index shape:", data.edge_index.shape)
#             print("Data batch y:", data.y)
#
#             # optimizer.zero_grad()
#             out = model(data)
#             print("Model output:", out)
#
#             loss = F.nll_loss(out, data.y)
#             print("Loss:", loss.item())
#
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         return total_loss / len(loader)
#
#
#     # Training loop
#     for epoch in range(50):
#         loss = train(loader)
#         # if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss:.4f}')
#
#     # Test the model
#     model.eval()
#     correct = 0
#     for data in loader:
#         _, pred = model(data).max(dim=1)
#         correct += int((pred == data.y).sum())
#     accuracy = correct / len(loader)
#     print(f'Accuracy: {accuracy:.4f}')


import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder

from utils.graph import GraphHelper

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

# Main script
BASE_DIR = './assets/graphs'
files = os.listdir(BASE_DIR)

graph_helper = GraphHelper()
graphs = []
labels = []

# Ensure some sample graphs are created and saved with labels

for filepath in files:
    if filepath.endswith('.gexf'):
        graph, label = graph_helper.load_transaction_graph_from_gexf(f'{BASE_DIR}/{filepath}')
        print(f"Loaded label: {label}")  # Verify the label is loaded correctly
        graph_pyg = from_networkx(graph)
        graph_helper.show(graph)
        # Add dummy node features for demonstration
        num_nodes = graph_pyg.num_nodes
        graph_pyg.x = torch.tensor(np.random.rand(num_nodes, 2), dtype=torch.float)

        # Use the label loaded from the GEXF file or add a default label if not present
        if label is not None:
            labels.append(label)
        else:
            labels.append("default_label")

        graphs.append(graph_pyg)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

print("Unique labels and their encoded values:")
for label, encoded_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{label}: {encoded_label}")

print("Encoded labels:", encoded_labels)
assert max(encoded_labels) < len(label_encoder.classes_), "Encoded label exceeds number of classes"

for i, graph_pyg in enumerate(graphs):
    graph_pyg.y = torch.tensor([encoded_labels[i]], dtype=torch.long)

loader = DataLoader(graphs, batch_size=1, shuffle=True)

print("DataLoader created. Verifying batches...")
for i, batch in enumerate(loader):
    print(f"Batch {i}: {batch}")

num_classes = len(label_encoder.classes_)
print("Number of classes:", num_classes)
model = GCN(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(loader):
    model.train()
    total_loss = 0
    for data in loader:

        optimizer.zero_grad()
        out = model(data)
        assert torch.max(data.y) < out.shape[1], "Target label out of bounds for the number of classes"

        loss = F.nll_loss(out, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(100):
    print(f"Epoch {epoch}")
    loss = train(loader)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')


model.eval()

torch.save(model.state_dict(), "model.h5")

correct = 0
for data in loader:
    print("Data batch:", data)
    _, pred = model(data).max(dim=1)
    print("Prediction:", pred)
    correct += int((pred == data.y).sum())
accuracy = correct / len(loader)
print(f'Accuracy: {accuracy:.4f}')
