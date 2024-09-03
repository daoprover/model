import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from utils.graph import GraphHelper



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = './assets/test'
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

assert max(encoded_labels) < len(label_encoder.classes_), "Encoded label exceeds number of classes"

for i, graph_pyg in enumerate(graphs):
    graph_pyg.y = torch.tensor([encoded_labels[i]], dtype=torch.long)

loader = DataLoader(graphs, batch_size=1, shuffle=True)


# Define your model and optimizer
model = GraphSAGE(in_channels=2, num_classes=2).to(device)  # Adjust in_channels based on your features
model.load_state_dict(torch.load("sage_model_new.h5"))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)


val_loader = DataLoader(graphs, batch_size=1, shuffle=False)


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
tick_marks = np.arange(2)
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
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
