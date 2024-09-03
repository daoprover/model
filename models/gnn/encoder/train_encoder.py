import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from models.gnn.encoder.model import Autoencoder, GraphSAGEWithAutoencoder
from utils.graph import GraphHelper

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.001)

def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - grad mean: {param.grad.mean()} grad std: {param.grad.std()}")
        else:
            print(f"{name} has no grad")

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

# Main script
BASE_DIR = '../../../assets/graphs'
files = os.listdir(BASE_DIR)

graph_helper = GraphHelper()
graphs = []
labels = []

for filepath in files:
    if filepath.endswith('.gexf'):
        graph, label = graph_helper.load_transaction_graph_from_gexf(f'{BASE_DIR}/{filepath}')
        graph_pyg = from_networkx(graph)
        num_nodes = graph_pyg.num_nodes
        graph_pyg.x = torch.tensor(np.random.rand(num_nodes, 2), dtype=torch.float)

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

for i, graph_pyg in enumerate(graphs):
    graph_pyg.y = torch.tensor([encoded_labels[i]], dtype=torch.long)

loader = DataLoader(graphs, batch_size=1, shuffle=True)

input_dim = 2
hidden_dim = 32
encoded_dim = 32
num_classes = len(label_encoder.classes_)

autoencoder = Autoencoder(input_dim, hidden_dim, encoded_dim)
initialize_weights(autoencoder)
autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-5, weight_decay=5e-4)

# Ensure all parameters require gradients
for name, param in autoencoder.named_parameters():
    print(f'{name} requires grad: {param.requires_grad}')

# Train the autoencoder
for epoch in range(50):
    loss = train_autoencoder(loader, autoencoder, autoencoder_optimizer)
    if epoch % 10 == 0:
        print(f'Autoencoder Epoch {epoch}, Loss: {loss:.4f}')

# Save and load the autoencoder state_dict
torch.save(autoencoder.state_dict(), "autoencoder.pth")
model = GraphSAGEWithAutoencoder(input_dim, hidden_dim, encoded_dim, num_classes)
model.autoencoder.load_state_dict(torch.load("autoencoder.pth"))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)

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
        print(f"Batch loss: {loss.item()}")
        print_gradients(model)
    return total_loss / len(loader)

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

torch.save(model.state_dict(), "sage_autoencoder_model.pth")

val_loss, val_accuracy, all_preds, all_labels, all_probs = validate(val_loader, model)
print(f'Final Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

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

unique_labels = np.unique(all_labels)
target_names = [label_encoder.classes_[i] for i in unique_labels]
print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
all_probs = np.array(all_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()
optimal_thresholds = dict()
for i in range(num_classes):
    if i in unique_labels:
        fpr[i], tpr[i], thresholds = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        j_scores = tpr[i] - fpr[i]
        optimal_idx = np.argmax(j_scores)
        optimal_thresholds[i] = thresholds[optimal_idx]

print("Optimal Thresholds:", optimal_thresholds)

plt.figure(figsize=(10, 8))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'yellow', 'black', 'purple', 'brown']
for i, color in zip(range(num_classes), colors):
    if i in unique_labels:
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
