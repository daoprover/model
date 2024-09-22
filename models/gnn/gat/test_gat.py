import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import torch
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from dataset.data_loader import GraphDataset

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGINConv
from utils.graph import GraphHelper

# Main script
BASE_DIR = '/home/sempai/Desktop/Projects/validation-model/assets/test'


labels = ["anomaly", "white"]
label_encoder = LabelEncoder()
label_encoder.fit(labels)

dataset = GraphDataset(base_dir=BASE_DIR, label_encoder=label_encoder)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = GraphGINConv(in_channels=2, edge_in_channels=2, num_classes=len(labels)).to(device)
# Define your model and optimizer

model.load_state_dict((torch.load("/home/sempai/Desktop/Projects/validation-model/gin_model_best_v1_new_dataset.h5", weights_only=True )))
model.eval()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)


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
