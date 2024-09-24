import time as t
from pathlib import Path
import sys
import os
import logging

import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils.graph import GraphHelper


class Indexer:
    def __init__(self, logger: logging.Logger, sleep_time=4):
        self.sleep_time = sleep_time
        self.logger = logger

    def index_white(self, save_path, block_numbers: list[int], tx_per_block: int = 50):
        graph_helper = GraphHelper(self.logger)
        i = 0

        for block in block_numbers:
            addresses = graph_helper.get_white_addresses(block)[:tx_per_block]
            for address in addresses:
                self.__process_address_info(save_path, address, graph_helper)

                self.logger.info(f"i: {i}", )
                i += 1
                t.sleep(self.sleep_time)

    def index_black_addresses(self, save_path: Path, csv_path: Path):
        labeled_addresses = [(item[0], item[9]) for item in pd.read_csv(csv_path).values]

        print(len(labeled_addresses))

        graphHelper = GraphHelper()

        i = 0

        for address in labeled_addresses:
            self.__process_address_info(save_path, address[0], graphHelper)

            self.logger.debug("i:", i)
            i += 1
            t.sleep(self.sleep_time)

    def __process_address_info(self, save_path: Path, address: str, graph_helper: GraphHelper):
        address_file = f"{save_path}/{address}.gexf"

        if os.path.exists(address_file):
            self.logger.debug(f"File for address {address} already exists. Skipping...")
            return

        self.logger.debug("address:", address)
        transactions = graph_helper.get_transactions(address)

        if transactions:
            self.logger.debug("Transactions for address:", address)
            graph = graph_helper.build_transaction_graph(transactions)
            graph_helper.save_transaction_graph_to_gexf(graph, address_file, "white")
        else:
            self.logger.error("No transactions found.")
