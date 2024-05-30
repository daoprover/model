import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from utils.graph import GraphHelper

# Define the Autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, encoded_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoded_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the GraphSAGE model with Autoencoder
class GraphSAGEWithAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim, num_classes):
        super(GraphSAGEWithAutoencoder, self).__init__()
        self.autoencoder = Autoencoder(input_dim, hidden_dim, encoded_dim)
        self.conv1 = SAGEConv(encoded_dim, 16)  # Adjust input feature size to match encoded dimension
        self.conv2 = SAGEConv(16, 16)
        self.fc = torch.nn.Linear(16, num_classes)  # Adjust output size based on number of classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        encoded, _ = self.autoencoder(x)
        x = self.conv1(encoded, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0, keepdim=True)  # Global mean pooling
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


BASE_DIR = './assets/test'
files = os.listdir(BASE_DIR)

graph_helper = GraphHelper()
graphs = []
labels = []

# Ensure some sample graphs are created and saved with labels
for filepath in files[:100]:
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


# Load the model
num_classes = 18
input_dim = 2
hidden_dim = 16
encoded_dim = 8
model = GraphSAGEWithAutoencoder(input_dim, hidden_dim, encoded_dim, num_classes)
model.load_state_dict(torch.load("sage_autoencoder_model.h5"))
model.eval()

# Now you can use the loaded model for evaluation
correct = 0
all_preds = []
all_labels = []
all_probs = []
for data in loader:
    out = model(data)
    prob = torch.exp(out)  # Convert log probabilities to probabilities
    _, pred = out.max(dim=1)
    correct += int((pred == data.y).sum())
    all_preds.append(pred.item())
    all_labels.append(data.y.item())
    all_probs.append(prob.detach().numpy())  # Store all probabilities

accuracy = correct / len(loader)
print(f'Accuracy: {accuracy:.4f}')

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
# plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
# plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
# print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))

# Binarize the labels for ROC curve
all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
all_probs = np.array(all_probs)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
optimal_thresholds = dict()
for i in range(num_classes):
    fpr[i], tpr[i], thresholds = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    # Find the optimal threshold (Youden's J statistic)
    j_scores = tpr[i] - fpr[i]
    optimal_idx = np.argmax(j_scores)
    optimal_thresholds[i] = thresholds[optimal_idx]

print("Optimal Thresholds:", optimal_thresholds)

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'yellow', 'black', 'purple', 'brown']  # Add more colors if necessary
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
