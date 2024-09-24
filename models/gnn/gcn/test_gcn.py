import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

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
        # graph_helper.show(graph)
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

# Ensure that the model is defined with the correct number of classes before loading state
model = GCN(num_classes=num_classes)

# Load model state with the correct number of classes
model.load_state_dict(torch.load("model.h5"))
model.eval()

correct = 0
all_preds = []
all_labels = []
all_probs = []
for data in loader:
    print("Data batch:", data)
    out = model(data)
    prob = torch.exp(out)  # Convert log probabilities to probabilities
    _, pred = out.max(dim=1)
    print("Prediction:", pred)
    correct += int((pred == data.y).sum())
    all_preds.append(pred.item())
    all_labels.append(data.y.item())
    all_probs.append(prob[0][1].item())  # Assuming binary classification

accuracy = correct / len(loader)
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

# Classification report
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
