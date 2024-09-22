import requests
import networkx as nx
import matplotlib.pyplot as plt
import logging


class GraphHelper:
    def __init__(self, logger: logging.Logger ):
        self.logger = logger

    def build_transaction_graph(self, transactions):
        G = nx.DiGraph()
        for tx in transactions:
            timestamp = tx['time']
            fee = tx['fee']
            size = tx['size']

            input_addresses = [input_tx['prev_out']['addr']
                               for input_tx in tx['inputs']
                               if 'prev_out' in input_tx
                               and 'addr' in input_tx['prev_out']]
            output_addresses = [output_tx['addr'] for output_tx in tx['out'] if 'addr' in output_tx]

            for input_address in input_addresses:
                for output_address in output_addresses:
                    value = sum(output_tx['value']
                                for output_tx in tx['out']
                                if 'addr' in output_tx
                                and output_tx['addr'] == output_address)
                    edge_attrs = {
                        'amount': value,
                        'fee': fee,
                        'size': size,
                        'timestamp': timestamp
                    }
                    G.add_node(input_address)
                    G.add_node(output_address)
                    G.add_edge(input_address, output_address, **edge_attrs)
        return G

    def save_transaction_graph_to_gexf(self, G, filepath, label=None):
        self.logger.debug("label: ", label)
        if label is not None:
            self.logger.debug("Label is added")
            G.graph['name'] = label
        else:
            self.logger.debug("Default_label is added")
            G.graph['name'] = "default_label"
        nx.write_gexf(G, filepath)

    def load_transaction_graph_from_gexf(self, filepath):
        G = nx.read_gexf(filepath)
        label = G.graph.get('name')
        return G, label

    def show(self, graph):
        layout = nx.spring_layout(graph)

        node_colors = ['skyblue' for _ in graph.nodes()]
        node_sizes = [10 for _ in graph.nodes()]

        nx.draw(graph, layout, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=10)
        edge_labels = nx.get_edge_attributes(graph, 'amount')
        nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, font_size=8)
        plt.title("Bitcoin Transaction Graph")
        plt.show()

    def get_transactions(self, address):
        url = f"https://blockchain.info/rawaddr/{address}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            transactions = data['txs']
            return transactions
        else:
            self.logger.debug("Error fetching data:", response.status_code)
            return None

    def get_white_addresses(self, block_id):
        url = f"https://blockchain.info/rawblock/{block_id}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            transactions = data['tx']
            addresses = []
            for tx in transactions:
                inputs = tx['inputs']
                for input in inputs:
                    try:
                        address = input["addr"]
                        addresses.append(address)
                    except KeyError:
                        continue
                outs = tx['out']
                for out in outs:
                    try:
                        address = out["addr"]
                        addresses.append(address)
                    except KeyError:
                        continue
            return addresses
        else:
            self.logger.error("Error fetching data:", response.status_code)
            return None
