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
from sklearn.preprocessing import LabelBinarizer

from utils.graph import GraphHelper

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

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


def validate(loader, model):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item()
            prob = torch.exp(out)
            _, pred = out.max(dim=1)
            correct += int((pred == data.y).sum())
            all_preds.extend(pred.tolist())
            all_labels.extend(data.y.tolist())
            all_probs.extend(prob.tolist())
    accuracy = correct / len(loader)
    return total_loss / len(loader), accuracy, all_preds, all_labels, all_probs

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


## Initialize the model
input_dim = 2
hidden_dim = 32
encoded_dim = 32
num_classes = len(label_encoder.classes_)
model = GraphSAGEWithAutoencoder(input_dim, hidden_dim, encoded_dim, num_classes)
model.load_state_dict(torch.load("sage_autoencoder_model.pth"))
model.eval()

val_loader = DataLoader(graphs, batch_size=1, shuffle=False)

val_loss, val_accuracy, all_preds, all_labels, all_probs = validate(val_loader, model)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

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


all_probs_np = np.vstack(all_probs)
all_labels_np = np.array(all_labels)
print("Shape of prediction probabilities:", all_probs_np.shape)

# Classification report
# print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))

# Compute ROC curve and ROC area for each class
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    # Binarize the labels for the current class
    y_true = (encoded_labels == i).astype(int)
    fpr[i], tpr[i], _ = roc_curve(y_true, all_probs_np[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (AUC = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()