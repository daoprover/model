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
            torch.nn.Dropout(0.3),  # Add dropout for regularization
            torch.nn.Linear(hidden_dim, encoded_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoded_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),  # Add dropout for regularization
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
input_dim = 2
hidden_dim = 16
encoded_dim = 8
model = GraphSAGEWithAutoencoder(input_dim, hidden_dim, encoded_dim, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Pre-train the autoencoder
autoencoder = Autoencoder(input_dim, hidden_dim, encoded_dim)
autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01, weight_decay=5e-4)

def train_autoencoder(loader, model, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        encoded, decoded = model(data.x)
        loss = F.mse_loss(decoded, data.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Train the autoencoder
for epoch in range(50):
    loss = train_autoencoder(loader, autoencoder, autoencoder_optimizer)
    if epoch % 10 == 0:
        print(f'Autoencoder Epoch {epoch}, Loss: {loss:.4f}')

# Save the autoencoder state_dict
torch.save(autoencoder.state_dict(), "autoencoder.pth")

# Load the pre-trained autoencoder weights
model.autoencoder.load_state_dict(torch.load("autoencoder.pth"))

# Train the combined model
def train(loader, model, optimizer):
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

# Validation
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

# Split the data into train and validation sets
train_graphs = graphs[:int(0.8 * len(graphs))]
val_graphs = graphs[int(0.8 * len(graphs)):]

train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)

# Training loop with validation
for epoch in range(100):
    train_loss = train(train_loader, model, optimizer)
    val_loss, val_accuracy, all_preds, all_labels, all_probs = validate(val_loader, model)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Save the combined model state_dict
torch.save(model.state_dict(), "sage_autoencoder_model.pth")

# Testing
val_loss, val_accuracy, all_preds, all_labels, all_probs = validate(val_loader, model)
print(f'Final Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

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
unique_labels = np.unique(all_labels)
print("Unique labels in the data:", unique_labels)
print("Label Encoder classes:", label_encoder.classes_)
# Ensure the target names and labels match
target_names = [label_encoder.classes_[i] for i in unique_labels]
print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

# Binarize the labels for ROC curve
all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
all_probs = np.array(all_probs)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
optimal_thresholds = dict()
for i in range(num_classes):
    if i in unique_labels:  # Ensure that the class is in the data
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
    if i in unique_labels:  # Ensure that the class is in the data
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
