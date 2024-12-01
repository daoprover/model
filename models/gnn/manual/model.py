
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax


class GraphGNNWithEmbeddings(MessagePassing):
    def __init__(self, node_input_dim, edge_input_dim, embedding_dim, hidden_dim, num_time_labels):
        super(GraphGNNWithEmbeddings, self).__init__(aggr='add')  # Add aggregation
        self.node_time_embedding = nn.Embedding(num_time_labels, embedding_dim)
        self.edge_time_embedding = nn.Embedding(num_time_labels, embedding_dim)

        print("node_input_dim:", node_input_dim)  # Expected size of node features
        print("embedding_dim:", embedding_dim)  # Expected size of embeddings


        self.node_transform = nn.Linear(node_input_dim + embedding_dim, hidden_dim)
        self.edge_transform = nn.Linear(edge_input_dim + embedding_dim, hidden_dim)
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability for binary classification
        )

    def forward(self, x, edge_index, edge_attr, node_time_label, edge_time_label, batch):

        embedded_node_time = self.node_time_embedding(node_time_label)
        embedded_edge_time = self.edge_time_embedding(edge_time_label)

        print("Shape of x (node features):", x.shape)  # Should be [num_nodes, node_input_dim]
        print("Shape of embedded_node_time:", embedded_node_time.shape)  # Should be [num_nodes, embedding_dim]
        concatenated_features = torch.cat([x, embedded_node_time], dim=-1)
        print("Shape of concatenated features:", concatenated_features.shape)



        x = torch.cat([x, embedded_node_time], dim=-1)



        edge_attr = torch.cat([edge_attr, embedded_edge_time], dim=-1)
        x = self.node_transform(x)
        edge_attr = self.edge_transform(edge_attr)

        # Validate edge_index
        if edge_index.max().item() >= x.size(0):
            raise ValueError(f"Invalid edge_index: max index {edge_index.max().item()} >= number of nodes {x.size(0)}")

        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        graph_embedding = global_mean_pool(x, batch)  # Aggregate node features for each graph
        return self.graph_classifier(graph_embedding)  # Graph-level output

    def message(self, x_i, x_j, edge_attr, edge_index):
        num_nodes = x_i.shape[0]  # Total number of nodes
        print("Number of nodes:", num_nodes)

        # Check the range of edge_index[1]
        print("Max index in edge_index[1]:", edge_index[1].max().item())

        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        attention_logits = self.attention_mlp(combined)
        attention = softmax(attention_logits, edge_index[1], num_nodes=x_i.shape[0])
        return attention * x_j

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(combined)