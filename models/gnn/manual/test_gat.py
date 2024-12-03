import os
import sys
from torch_geometric.loader import DataLoader
import torch
import logging
from sklearn.preprocessing import LabelEncoder

from models.gnn.manual.data_loader import GraphDatasetLoader
from models.gnn.manual.model import GraphGNNWithEmbeddings
from tester.tester import Tester

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))



class TestGraphGNNWithEmbeddings(Tester):



    def __init__(self,  logger: logging.Logger, labels=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device: ", device)

        if labels is None:
            labels = ["anomaly", "white"]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        dataset = GraphDatasetLoader(base_dir='assets/test', label_encoder=self.label_encoder,
                                     logger=logging.getLogger(), dataset_size=600)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        num_time_labels = 4
        embedding_dim = 8
        hidden_dim = 16
        self.model = GraphGNNWithEmbeddings(
            node_input_dim=5,  # Adjusted to match the number of node features
            edge_input_dim=4,  # Adjusted to match the number of edge features
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_time_labels=num_time_labels
        )

        self.model.load_state_dict((torch.load("manual_1.h5", weights_only=True)))
        self.model.eval()
        self.logger = logger

        super().__init__(device=self.device, logger=self.logger, model=self.model)

    def test(self):
        super().test(self.loader, self.label_encoder)
