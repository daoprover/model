import logging
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

from dataset.data_loader import GraphDatasetLoader
from models.gnn.gat.hyperparams import GatHyperParams

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGATConv
from utils.graph import GraphHelper


class GatTrainer():
    def __init__(self,hyperparams: GatHyperParams, logger: logging.Logger):
        self.hyperparams = hyperparams
        self.logger = logger

    def train_gat(self):
        labels = ["anomaly", "white"]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        dataset = GraphDatasetLoader(base_dir=self.hyperparams.dataset.train, label_encoder=label_encoder, logger=self.logger)

        loader = DataLoader(dataset, batch_size=self.hyperparams.training.batch_size, shuffle=True)


        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_classes = len(label_encoder.classes_)
        model = GraphGATConv(in_channels=2, edge_in_channels=2, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-8, weight_decay=5e-7)

        def train(loader):
            i = 0
            model.train()
            total_loss = 0
            dataset.shuffle()
            for data in loader:

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

        # model.load_state_dict(
        #     (torch.load("/home/sempai/Desktop/Projects/validation-model/assets/models/gin_model_best_v1_new_dataset.h5", weights_only=True)))

        for epoch in range(self.hyperparams.training.epochs_number):

            print(f"start {epoch} epoch")
            loss = train(loader)
            if loss < best_loss:
                torch.save(model.state_dict(), f"gat_model_best_{self.hyperparams.meta.version}.h5")

            print(f'Epoch {epoch}, Loss: {loss:.4f}')

        torch.save(model.state_dict(), f"gat_model_{self.hyperparams.meta.version}.h5")