import os
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader
import torch
import logging
from sklearn.preprocessing import LabelEncoder

from dataset.data_loader import GraphDatasetLoader
from models.gnn.gat.hyperparams import GatHyperParams
from tester.tester import Tester

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGATConv


class TestGAT(Tester):
    def __init__(self, hyperparams: GatHyperParams, logger: logging.Logger, labels=None):
        if labels is None:
            labels = ["anomaly", "white"]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.hyperparams = hyperparams
        dataset = GraphDatasetLoader(base_dir=hyperparams.dataset.test, label_encoder=self.label_encoder, logger=logger)
        self.loader = DataLoader(dataset, batch_size=self.hyperparams.testing.batch_size, shuffle=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = GraphGATConv(in_channels=2, edge_in_channels=2, num_classes=len(labels)).to(self.device)
        self.model.load_state_dict((torch.load(self.hyperparams.testing.model_path, weights_only=True)))
        self.model.eval()
        self.logger = logger

        super().__init__(device=self.device, logger=self.logger, model=self.model)

    def test(self):
        super().test(self.loader, self.label_encoder)
