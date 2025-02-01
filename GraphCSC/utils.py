import numpy as np
import random
import json
import os
from networkx.readwrite import json_graph
import networkx as nx
import tensorflow as tf

WALK_LEN = 5
N_WALKS = 50


def load_data(prefix,directory, normalize=True, load_walks=False):
    """Load graph, features, and class map from files."""

    G_data = json.load(open(os.path.join(directory, prefix + "-G.json")))

    G = json_graph.node_link_graph(G_data)

    # Handle node ID conversion
    if isinstance(list(G.nodes())[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    # Load features
    feats_path = os.path.join(directory, prefix + "-feats.npy")
    if os.path.exists(feats_path):
        feats = np.load(feats_path)
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    # Load ID map
    id_map = json.load(open(os.path.join(directory, prefix + "-id_map.json")))

    id_map = {conversion(k): int(v) for k, v in id_map.items()}

    # Load class map
    class_map = json.load(open(os.path.join(directory, prefix + "-class_map.json")))

    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)
    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    # Remove nodes without validation/test annotations
    broken_count = 0
    for node in list(G.nodes()):
        if 'val' not in G.node[node] or 'test' not in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print(f"Removed {broken_count} nodes that lacked proper annotations.")

    # Annotate edges for training
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    # Normalize features
    if normalize and feats is not None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    # Load Centrality Based random walks
    walks = []
    if load_walks:
        with open(os.path.join(directory, prefix + "-walks.txt")) as fp:
            for line in fp:
                walks.append(list(map(conversion, line.split())))
    return G, feats, id_map, walks, class_map

# Function to load the pre-computed centrality measures
def load_centrality_measures(directory, file_name):
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as f:
        centrality_measures = json.load(f)
    return centrality_measures


# Function to normalize centrality scores
def normalize_centrality(centrality_scores):
    min_centrality = min(centrality_scores.values())
    max_centrality = max(centrality_scores.values())
    normalized_scores = {
        node: (score - min_centrality) / (max_centrality - min_centrality + 1e-9)
        for node, score in centrality_scores.items()
    }
    return normalized_scores

def calculate_centrality_measures(graph):
    centrality_measures = {
        "degree": nx.degree_centrality(graph),

        # In my research I focus soley on degree, therefore the other centralities are not necessary to run.
        #"closeness": nx.closeness_centrality(graph),
        #"betweenness": nx.betweenness_centrality(graph),
        #"eigenvector": nx.eigenvector_centrality(graph),
        #"pagerank": nx.pagerank(graph),
    }
    return centrality_measures

def save_centrality_as_json(centrality_measures, output_file):
    with open(output_file, 'w') as f:
        json.dump(centrality_measures, f, indent=4)
    print(f"Centrality measures saved to {output_file}")


def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=name)


def run_random_walks_with_centrality(G, nodes, centrality_scores, num_walks=N_WALKS):
    pairs = []
    normalized_scores = normalize_centrality(centrality_scores)

    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue

        # Get the normalized centrality score for the node
        normalized_score = normalized_scores.get(str(node), 0)

        # Nodes which have
        num_walks_for_node = max(int(num_walks * normalized_score), 1)

        # Perform the scaled number of random walks
        for _ in range(num_walks_for_node):
            curr_node = node
            for _ in range(WALK_LEN):
                neighbors = list(G.neighbors(curr_node))
                if not neighbors:
                    break
                next_node = random.choice(neighbors)
                if curr_node != node:  # Skip self-co-occurrences
                    pairs.append((node, curr_node))
                curr_node = next_node

        if count % 1000 == 0:
            print(f"Done walks for {count} nodes")

    return pairs


if __name__ == "__main__":

    prefix = "ppi"
    directory = r"C:\Users\User\PycharmProjects\ML4Graph\PPI_Data"

    # Load the data
    G, feats, id_map, walks, class_map = load_data(prefix, directory, load_walks=False)

    # Get the nodes to run random walks
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)

    '''
    # Calculate the Centrality, this needs to be done only oncce, thus as I cun it already it is not necessary to run it again.
    centrality_measures = calculate_centrality_measures(G)
    save_centrality_as_json(centrality_measures, "../PPI_Data/normalized_degree_centrality.json")
    centrality_measures = load_centrality_measures("../PPI_Data/normalized_degree_centrality.json")
    '''

    '''
    As I calulate the only the degree centrality, I dont need to normalize them, as the networkx package does this automatically when calculating the degree
    normalize_centralized_degree = normalize_centrality(centrality_measures["degree"])
    '''

    '''
    Calculate once the bridge strength once
    bridge_strength_file = os.path.join(directory, "ppi-bridge_strength_normalized.json")
    
    '''

    with open(bridge_strength_file, "r") as f:
        normalize_centralized_degree = json.load(f)

    directory = r"C:\Users\User\PycharmProjects\ML4Graphs\GraphSAGE\PPI"

    # Run random walks and save the output
    pairs = run_random_walks_with_centrality(G, G.nodes(), normalize_centralized_degree)
    with open("ppi-walks_full_dataset_bridge_centrality.txt", "w") as fp:
            fp.write("\n".join([f"{p[0]}\t{p[1]}" for p in pairs]))
