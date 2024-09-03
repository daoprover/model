import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, label_binarize
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGINConv
from utils.graph import GraphHelper


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
        graph_pyg = from_networkx(graph)

        # Extract node features
        node_features = []
        for node in graph.nodes(data=True):
            if 'feature' in node[1]:
                node_features.append(node[1]['feature'])
            else:
                # Provide a random feature vector if the feature attribute is missing
                random_feature = np.random.rand(2)  # Assuming the feature size is 2
                node_features.append(random_feature)

        # Convert node features to tensor
        graph_pyg.x = torch.tensor(node_features, dtype=torch.float)

        # Extract edge features
        edge_features = []
        for edge in graph.edges(data=True):
            attvalues = edge[2].get('attvalues', {})
            feature = []
            for attvalue in attvalues:
                feature.append(float(attvalue['value']))
            edge_features.append(feature)

        # Convert edge features to tensor
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        graph_pyg.edge_attr = edge_features_tensor

        if label is not None:
            if label != "white":
                label = "anomaly"
            labels.append(label)
        else:
            labels.append("white")
        print("label: ", label)
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

loader = DataLoader(graphs, batch_size=32, shuffle=True)

print("DataLoader created. Verifying batches...")
for i, batch in enumerate(loader):
    print(f"Batch {i}: {batch}")

num_classes = len(label_encoder.classes_)
print("Number of classes:", num_classes)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your model and optimizer
model = GraphGINConv(in_channels=2, edge_in_channels=len(edge_features[0]), num_classes=num_classes).to(
    device)  # Adjust in_channels based on your features
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)


def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)  # Move data to GPU
        optimizer.zero_grad()
        out = model(data)
        assert torch.max(data.y) < out.shape[1], "Target label out of bounds for the number of classes"
        loss = F.nll_loss(out, data.y)
        # print("loss ", loss.item())
        # print("out ", out)
        # print("data.y ", data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Training loop
for epoch in range(25):
    print(f"Epoch {epoch}")
    loss = train(loader)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

model.eval()

torch.save(model.state_dict(), "gin_model_new.h5")
