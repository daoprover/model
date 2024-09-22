import os
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import numpy as np
import random


from utils.graph import GraphHelper


class GraphDataset(Dataset):
    def __init__(self, base_dir, label_encoder):
        self.dataset_size = 120
        self.base_dir = base_dir
        self.all_files = [f for f in os.listdir(base_dir) if f.endswith('.gexf')]
        self.files =  random.sample(self.all_files, self.dataset_size)
        self.graph_helper = GraphHelper()
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= len(self.files):
            raise IndexError("Индекс за пределами доступных файлов.")

        filepath = os.path.join(self.base_dir, self.files[idx])
        graph, label = self.graph_helper.load_transaction_graph_from_gexf(filepath)
        graph_pyg = from_networkx(graph)

        # Преобразование списка numpy-массивов в один массив перед созданием тензора
        node_features = np.array([node[1].get('feature', np.random.rand(2)) for node in graph.nodes(data=True)])
        graph_pyg.x = torch.tensor(node_features, dtype=torch.float)

        # Признаки ребер
        edge_features = []
        for edge in graph.edges(data=True):
            attvalues = edge[2].get('attvalues', {})
            feature = [float(attvalue['value']) for attvalue in attvalues] if attvalues else [0.0, 0.0]
            edge_features.append(feature)

        # Проверка наличия признаков у всех ребер
        if edge_features:
            edge_features = np.array(edge_features)
            graph_pyg.edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            print(f"edge_attr is empty. {filepath}")
            # Если признаки отсутствуют, создаем тензор с нулями нужного размера
            graph_pyg.edge_attr = torch.zeros((graph.number_of_edges(), 2), dtype=torch.float)

        # Если метка отсутствует, удаляем файл и переходим к следующему
        if label is None or len(label) == 0:
            os.remove(filepath)
            print(f"Label is empty. Removed file: {filepath}")
            return self.__getitem__(idx + 1)

        # Обработка меток
        label = "anomaly" if label and label != "white" else "white"
        encoded_label = self.label_encoder.transform([label])
        if len(encoded_label) == 0:
            os.remove(filepath)
            print(f"Encoded label is empty. Removed file: {filepath}")
            return self.__getitem__(idx + 1)

        graph_pyg.y = torch.tensor([encoded_label[0]], dtype=torch.long)

        return graph_pyg

    def shuffle(self):
        random.shuffle(self.files)
        self.files =  random.sample(self.all_files, self.dataset_size)
        random.shuffle(self.files)
