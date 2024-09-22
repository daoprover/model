import os
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader
import torch
import logging
from sklearn.preprocessing import LabelEncoder, label_binarize

from dataset.data_loader import GraphDatasetLoader
from tester.tester import Tester

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGINConv


def test_gat(dataset_path: Path, model_path: Path, logger: logging.Logger, labels=None):
    if labels is None:
        labels = ["anomaly", "white"]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    dataset = GraphDatasetLoader(base_dir=dataset_path, label_encoder=label_encoder, logger=logger)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GraphGINConv(in_channels=2, edge_in_channels=2, num_classes=len(labels)).to(device)
    model.load_state_dict((torch.load(model_path, weights_only=True)))
    model.eval()

    tester = Tester(device, model, logger)
    tester.test(loader, label_encoder)

