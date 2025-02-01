# negative_sampling.py
import numpy as np
import random
import tensorflow as tf
from utils import load_centrality_measures

def generate_negative_samples_from_labels(labels, num_samples, graph, alpha=0.75,centrality=True):
    """
    Produce exactly `num_samples` negative samples as a 1D tensor [num_samples],
    excluding any node that:
      - is itself a label in `labels`,
      - or is a neighbor of any label in `labels`.
    """

    if centrality:
        directory = r"C:\Users\User\PycharmProjects\ML4Graph\PPI_Data"
        centrality_data = load_centrality_measures(directory, "normalized_degree_centrality.json")
        centrality_degree = centrality_data["degree"]
    else:
        directory = r"C:\Users\User\PycharmProjects\ML4Graph\PPI_Data"
        centrality_degree = load_centrality_measures(directory, "ppi-bridge_strength_normalized.json")

    # Use only the actual nodes in the graph
    nodes = list(graph.nodes())

    centrality_list = [centrality_degree.get(str(node), 0.0) for node in nodes]
    centrality_list = np.array(centrality_list, dtype=np.float32)

    centrality_list = centrality_list ** alpha
    total_scores = np.sum(centrality_list)
    base_probabilities = centrality_list / total_scores  # shape: [len(nodes)]

    def py_negative_sampler(labels_np):
        """
        Python function called by tf.py_func at runtime.
        Excludes label nodes and their neighbors from sampling.
        Returns a NumPy array of shape [num_samples].
        """
        labels_np = labels_np.reshape(-1).astype(np.int64)
        label_set = set(labels_np)

        # Collect neighbors for each label node
        neighbor_set = set()
        for lbl in label_set:
            # Only do this if lbl is indeed a node in the graph
            if graph.has_node(lbl):
                for nbr in graph.neighbors(lbl):
                    neighbor_set.add(nbr)

        # The union of label nodes + their neighbors
        exclude_set = label_set.union(neighbor_set)

        # Build a local distribution that excludes these nodes
        local_nodes = []
        local_probs = []
        for node_id, prob in zip(nodes, base_probabilities):
            if node_id not in exclude_set:
                local_nodes.append(node_id)
                local_probs.append(prob)

        local_probs = np.array(local_probs, dtype=np.float32)

        # If everything got excluded (edge case), revert to full distribution
        if np.sum(local_probs) <= 0.0:
            local_nodes = nodes
            local_probs = base_probabilities.copy()

        # Renormalize local_probs
        local_probs /= np.sum(local_probs)

        # Sample exactly `num_samples` node IDs
        sampled_neg = random.choices(local_nodes, weights=local_probs, k=num_samples)

        return np.array(sampled_neg, dtype=np.int64)

    neg_samples_tf = tf.py_func(
        func=py_negative_sampler,
        inp=[labels],
        Tout=tf.int64
    )

    neg_samples_tf.set_shape([num_samples])

    return neg_samples_tf