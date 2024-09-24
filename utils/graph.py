import requests
import networkx as nx
import matplotlib.pyplot as plt
import logging


class GraphHelper:
    def __init__(self, logger: logging.Logger ):
        self.logger = logger

    import networkx as nx

    def build_transaction_graph(self, transactions):
        G = nx.DiGraph()

        # Standard node attributes to ensure consistency across all nodes
        def initialize_node_attributes():
            return {
                'total_sent': 0,
                'total_received': 0,
                'num_transactions': 0,
                'total_fees': 0,
                'avg_transaction_value': 0,
                'last_transaction_time': None
            }

        for tx in transactions:
            timestamp = tx['time']
            fee = tx['fee']
            size = tx['size']

            # Extract input and output addresses
            input_addresses = [input_tx['prev_out']['addr']
                               for input_tx in tx['inputs']
                               if 'prev_out' in input_tx
                               and 'addr' in input_tx['prev_out']]
            output_addresses = [output_tx['addr'] for output_tx in tx['out'] if 'addr' in output_tx]

            for input_address in input_addresses:
                for output_address in output_addresses:
                    # Calculate total value transferred to the output address
                    value = sum(output_tx['value']
                                for output_tx in tx['out']
                                if 'addr' in output_tx
                                and output_tx['addr'] == output_address)

                    # Edge attributes for the transaction
                    edge_attrs = {
                        'amount': value,
                        'fee': fee,
                        'size': size,
                        'timestamp': timestamp
                    }

                    # Initialize both input and output nodes if they don't exist
                    if not G.has_node(input_address):
                        G.add_node(input_address, **initialize_node_attributes())
                    if not G.has_node(output_address):
                        G.add_node(output_address, **initialize_node_attributes())

                    # Update input node (sender) attributes
                    G.nodes[input_address]['total_sent'] += value
                    G.nodes[input_address]['total_fees'] += fee
                    G.nodes[input_address]['num_transactions'] += 1
                    G.nodes[input_address]['avg_transaction_value'] = (
                            (G.nodes[input_address]['total_sent'] + G.nodes[input_address]['total_received'])
                            / G.nodes[input_address]['num_transactions']
                    )
                    G.nodes[input_address]['last_transaction_time'] = timestamp

                    # Update output node (receiver) attributes
                    G.nodes[output_address]['total_received'] += value
                    G.nodes[output_address]['total_fees'] += fee
                    G.nodes[output_address]['num_transactions'] += 1
                    G.nodes[output_address]['avg_transaction_value'] = (
                            (G.nodes[output_address]['total_sent'] + G.nodes[output_address]['total_received'])
                            / G.nodes[output_address]['num_transactions']
                    )
                    G.nodes[output_address]['last_transaction_time'] = timestamp

                    # Add the edge between input and output addresses with transaction details
                    G.add_edge(input_address, output_address, **edge_attrs)

        return G

    def save_transaction_graph_to_gexf(self, G, filepath, label=None):
        self.logger.debug("label: ", label)
        if label is not None:
            self.logger.debug("Label is added")
            G.graph['name'] = label
        else:
            self.logger.debug("Default_label is added")
            G.graph['name'] = "white"
        nx.write_gexf(G, filepath)


    def rebuild_transaction_graph(self, filepath, new_path):
        G, label = self.load_transaction_graph_from_gexf(filepath)
        for node in G.nodes():
            G.nodes[node].update({
                'total_sent': 0,
                'total_received': 0,
                'num_transactions': 0,
                'total_fees': 0,
                'avg_transaction_value': 0,
                'last_transaction_time': None
            })

        for u, v, data in G.edges(data=True):
            value = data.get('amount', 0)
            fee = data.get('fee', 0)
            timestamp = data.get('timestamp', None)

            # Update the sender (u) node attributes
            G.nodes[u]['total_sent'] += value
            G.nodes[u]['total_fees'] += fee
            G.nodes[u]['num_transactions'] += 1
            G.nodes[u]['last_transaction_time'] = max(G.nodes[u]['last_transaction_time'], timestamp) if G.nodes[u][
                'last_transaction_time'] else timestamp

            # Update the receiver (v) node attributes
            G.nodes[v]['total_received'] += value
            G.nodes[v]['total_fees'] += fee
            G.nodes[v]['num_transactions'] += 1
            G.nodes[v]['last_transaction_time'] = max(G.nodes[v]['last_transaction_time'], timestamp) if G.nodes[v][
                'last_transaction_time'] else timestamp

        # Calculate the average transaction value for each node
        for node in G.nodes():
            total_value = G.nodes[node]['total_sent'] + G.nodes[node]['total_received']
            if G.nodes[node]['num_transactions'] > 0:
                G.nodes[node]['avg_transaction_value'] = total_value / G.nodes[node]['num_transactions']

        self.save_transaction_graph_to_gexf(G, new_path)

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
