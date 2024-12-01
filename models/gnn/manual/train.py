import logging
import os

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder

import sys



sys.path.insert(1, os.path.join(sys.path[0], "../../.."))
from models.gnn.gat.hyperparams import GatHyperParams
from models.gnn.manual.data_loader import GraphDatasetLoader
from models.gnn.manual.model import GraphGNNWithEmbeddings


class ManualGNNTrainer():
    def __init__(self, logger: logging.Logger):
        # self.hyperparams = hyperparams
        self.logger = logger

    def train(self):

        # Initialize label encoder
        labels = ["anomaly", "white"]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        # Load dataset
        dataset = GraphDatasetLoader(base_dir='data/train', label_encoder=label_encoder,
                                     logger=logging.getLogger())
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Initialize model
        num_time_labels = 4
        embedding_dim = 8
        hidden_dim = 16
        model = GraphGNNWithEmbeddings(
            node_input_dim=5,  # Adjusted to match the number of node features
            edge_input_dim=4,  # Adjusted to match the number of edge features
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_time_labels=num_time_labels
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss for graph classification

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print("device: ", device)

        # Training Loop
        epochs = 50
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            model.train()
            total_loss = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                try:
                    output = model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        node_time_label=batch.x[:, 2].long(),
                        edge_time_label=batch.edge_attr[:, 2].long(),
                        batch=batch.batch
                    )
                except RuntimeError as e:
                    print(f"RuntimeError: {e}")
                    continue
                loss = criterion(output, batch.y.float().view(-1, 1))  # Reshape batch.y to match output
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

        # Testing
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                test_output = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    node_time_label=batch.x[:, 2].long(),
                    edge_time_label=batch.edge_attr[:, 2].long(),
                    batch=batch.batch
                )
                print("Predicted graph label:", test_output)



