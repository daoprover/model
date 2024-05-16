import time as t
from pathlib import Path

from utils.dataset import Dataset
from utils.graph import GraphHelper

if __name__ == "__main__":
    labeled_addresses = Dataset(Path('assets/BitcoinHeistData.csv')).prepare_dateset()
    print(len(labeled_addresses))
    graphHelper = GraphHelper()
    for address in labeled_addresses:
        t.sleep(4)

        transactions = graphHelper.get_transactions(address[0])

        if transactions:
            print("Transactions for address:", address)
            graph = graphHelper.build_transaction_graph(transactions)
            graphHelper.save_transaction_graph_to_gexf(graph, f"assets/{address[0]}")
        else:
            print("No transactions found.")


