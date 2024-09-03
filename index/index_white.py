import os
import time as t
from pathlib import Path
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils.dataset import Dataset
from utils.graph import GraphHelper

if __name__ == "__main__":
    # labeled_addresses = Dataset(Path('assets/BitcoinHeistData.csv')).prepare_dateset()
    # print(len(labeled_addresses))
    graphHelper = GraphHelper()
    i = 0
    blocks = [123152, 312731, 389893,463234,456139, 643044, 34534, 121898, 704000, 704564, 750555, 230234, 541244, 1214, 335633, 340034, 294516, 3156,85324, 165948]
    for block in blocks:
        addresses = graphHelper.get_white_addresses(block)[:50]
        print("addresses: ", addresses)
        for address in addresses:
            address_file = f"assets/graphs/{address}.gexf"

            if os.path.exists(address_file):
                print(f"File for address {address} already exists. Skipping...")
                continue

            print("address:", address)
            transactions = graphHelper.get_transactions(address)

            if transactions:
                print("Transactions for address:", address)
                graph = graphHelper.build_transaction_graph(transactions)
                graphHelper.save_transaction_graph_to_gexf(graph, address_file, "white")
            else:
                print("No transactions found.")

            print("i:", i)
            i += 1
            t.sleep(4)

    labeled_addresses = Dataset(Path('../assets/BitcoinHeistData.csv')).prepare_dateset()
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