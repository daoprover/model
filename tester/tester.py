import logging

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import numpy as np
import matplotlib.pyplot as plt


class Tester:
    def __init__(self, device: torch.device, model: torch.nn.Module, logger: logging.Logger):
        self.device = device
        self.model = model
        self.logger = logger

    def test(self, loader: DataLoader, label_encoder: LabelEncoder):
        correct = 0
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)  # Move data to GPU
                out = self.model(data)
                prob = torch.exp(out)  # Convert log probabilities to probabilities
                _, preds = out.max(dim=1)
                correct += int((preds == data.y).sum())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())  # Store all probabilities

        accuracy = correct / len(loader.dataset)
        self.logger.debug(f'Accuracy: {accuracy:.4f}')

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
        self.logger.debug(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))

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
