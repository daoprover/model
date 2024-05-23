import os
import time as t
from pathlib import Path

from utils.dataset import Dataset
from utils.graph import GraphHelper

if __name__ == "__main__":
    labeled_addresses = Dataset(Path('assets/BitcoinHeistData.csv')).prepare_dateset()
    print(len(labeled_addresses))
    graphHelper = GraphHelper()
    i = 0

    for address in labeled_addresses:
        address_file = f"assets/graphs/{address[0]}.gexf"

        if os.path.exists(address_file):
            print(f"File for address {address[0]} already exists. Skipping...")
            continue

        print("address:", address)
        transactions = graphHelper.get_transactions(address[0])

        if transactions:
            print("Transactions for address:", address)
            graph = graphHelper.build_transaction_graph(transactions)
            graphHelper.save_transaction_graph_to_gexf(graph, address_file, address[1])
        else:
            print("No transactions found.")

        print("i:", i)
        i += 1
        t.sleep(4)


