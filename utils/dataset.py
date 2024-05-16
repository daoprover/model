from pathlib import Path

import pandas as pd
from networkx import DiGraph


class Dataset:
    def __init__(self, filepath: Path):
        self.data = list((DiGraph, str))
        self.filepath = filepath

    def prepare_dateset(self):
        data = pd.read_csv(self.filepath).values
        my_list = [(item[0], item[9]) for item in data]

        return my_list

    def set_data(self, data: (DiGraph, str)):
        self.data = data

    def append_data(self, data: (DiGraph, str)):
        self.data.append(data)

    def split_data(self, train_size=0.8):
        self.data = self.data[:int(len(self.data) * train_size)]
