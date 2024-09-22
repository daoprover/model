import os
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import numpy as np
import random  # Для случайного перемешивания

from sklearn.preprocessing import LabelEncoder
import sys
import gc  # Для явного вызова сборщика мусора

from dataset.data_loader import GraphDataset

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGINConv
from utils.graph import GraphHelper

model_version = "v1_new_dataset_adam"


labels = ["anomaly", "white"]
label_encoder = LabelEncoder()
label_encoder.fit(labels)

BASE_DIR = './assets/train'

dataset = GraphDataset(base_dir=BASE_DIR, label_encoder=label_encoder)

loader = DataLoader(dataset, batch_size=512, shuffle=True)


print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Определение модели и оптимизатора
num_classes = len(label_encoder.classes_)
model = GraphGINConv(in_channels=2, edge_in_channels=2, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-8, weight_decay=5e-7)

# Функция тренировки
def train(loader):
    i = 0
    model.train()
    total_loss = 0
    dataset.shuffle()
    for data in loader:
        print(f"iteration {i} data len {len(data)}")

        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        del data
        torch.cuda.empty_cache()
        gc.collect()

        i += 1
    return total_loss / len(loader)
best_loss = 1
# Цикл обучения
model.load_state_dict(
    (torch.load("/home/sempai/Desktop/Projects/validation-model/assets/models/gin_model_best_v1_new_dataset.h5", weights_only=True)))

for epoch in range(200):

    print(f"start {epoch} epoch")
    loss = train(loader)
    if loss < best_loss:
        torch.save(model.state_dict(), f"gin_model_best_{model_version}.h5")

    print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Сохранение модели
torch.save(model.state_dict(), f"gin_model_{model_version}.h5")
