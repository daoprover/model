import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch
from torch_geometric.nn import SAGEConv, global_max_pool
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from utils.graph import GraphHelper


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, 16)
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.norm1 = nn.LayerNorm(16)
        self.norm2 = nn.LayerNorm(32)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)  # Global max pooling
        x = self.fc1(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Main script
BASE_DIR = './assets/graphs'
files = os.listdir(BASE_DIR)

graph_helper = GraphHelper()
graphs = []
labels = []

# Ensure some sample graphs are created and saved with labels
for filepath in files[:100]:
    if filepath.endswith('.gexf'):
        graph, label = graph_helper.load_transaction_graph_from_gexf(f'{BASE_DIR}/{filepath}')
        graph_pyg = from_networkx(graph)

        # Extract node features
        node_features = []
        for node in graph.nodes(data=True):
            if 'feature' in node[1]:
                node_features.append(node[1]['feature'])
            else:
                # Provide a default feature vector, e.g., a zero vector
                default_feature = np.zeros(2)  # Assuming the feature size is 2
                node_features.append(default_feature)

        # Convert node features to tensor
        graph_pyg.x = torch.tensor(node_features, dtype=torch.float)

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
model = GraphSAGE(in_channels=2, num_classes=num_classes).to(device)  # Adjust in_channels based on your features
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
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Training loop
for epoch in range(100):
    print(f"Epoch {epoch}")
    loss = train(loader)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

model.eval()

torch.save(model.state_dict(), "sage_model_new_v2.h5")

# Testing
correct = 0
all_preds = []
all_labels = []
all_probs = []
with torch.no_grad():
    for data in loader:
        data = data.to(device)  # Move data to GPU
        out = model(data)
        prob = torch.exp(out)  # Convert log probabilities to probabilities
        _, preds = out.max(dim=1)
        correct += int((preds == data.y).sum())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_probs.extend(prob.cpu().numpy())  # Store all probabilities

accuracy = correct / len(loader.dataset)
print(f'Accuracy: {accuracy:.4f}')

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))

# Binarize the labels for ROC curve
all_labels_bin = label_binarize(all_labels, classes=range(2))
all_probs = np.array(all_probs)

all_probs_np = np.vstack(all_probs)
all_labels_np = np.array(all_labels)

fpr, tpr, _ = roc_curve(all_labels_bin, all_probs_np[:, 0])
roc_auc = auc(tpr, fpr)

# Plot ROC curve
plt.figure()
plt.plot(tpr, fpr, label=f'ROC Curve (AUC = {roc_auc:0.2f})')
plt.show()


