import os
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
import gc  # Для явного вызова сборщика мусора
import random  # Для случайного перемешивания



sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGINConv
from utils.graph import GraphHelper

class GraphDataset(Dataset):
    def __init__(self, base_dir, label_encoder):
        self.base_dir = base_dir
        self.files = [f for f in os.listdir(base_dir) if f.endswith('.gexf')]
        self.graph_helper = GraphHelper()
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= len(self.files):  # Защита от выхода за пределы индексов
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
            feature = [float(attvalue['value']) for attvalue in attvalues] if attvalues else [0.0, 0.0]  # По умолчанию [0.0, 0.0]
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

# Создание и кодирование меток
labels = ["anomaly", "white"]
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Путь к директории с графами
BASE_DIR = './assets/graphs'

# Создание кастомного датасета
dataset = GraphDataset(base_dir=BASE_DIR, label_encoder=label_encoder)

# Использование DataLoader из torch_geometricss
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Определение модели и оптимизатора
num_classes = len(label_encoder.classes_)
model = GraphGINConv(in_channels=2, edge_in_channels=2, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

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
        # Очистка данных после обработки
        del data
        torch.cuda.empty_cache()
        gc.collect()  # Явный вызов сборщика мусора

        i += 1
    return total_loss / len(loader)

# Цикл обучения
for epoch in range(25):
    print(f"start {epoch} epoch")
    loss = train(loader)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Сохранение модели
torch.save(model.state_dict(), "gin_model_new.h5")
