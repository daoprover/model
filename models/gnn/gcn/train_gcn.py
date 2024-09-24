import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from models.gnn.gcn.model import GCN
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
