from layers import Layer
from utils import load_centrality_measures
import numpy
import tensorflow as tf
import random


class NeighborSampler(Layer):
    """
    Samples neighbors based on centrality scores.
    Assumes that adj lists are padded with random re-sampling.
    """

    def __init__(self, adj_info, centrality_dict, **kwargs):
        super(NeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

        # Convert centrality dictionary to tensor for direct lookup
        max_node_id = max(map(int, centrality_dict.keys()))
        centrality_list = [centrality_dict.get(str(i), 0.0) for i in range(max_node_id)]
        self.centrality_tensor = tf.convert_to_tensor(centrality_list, dtype=tf.float32)

    def _call(self, inputs):
        ids, num_samples = inputs

        # Look up the adjacency list for the given IDs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        max_node_id = tf.shape(self.centrality_tensor)[0] - 1
        adj_lists = tf.clip_by_value(adj_lists, 0, max_node_id)

        # Shape: [batch_size, max_degree]

        # Gather centrality scores for neighbors
        centrality_scores = tf.gather(self.centrality_tensor, adj_lists)  # Shape: [batch_size, max_degree]

        # Sort neighbors based on centrality scores
        _, sorted_indices = tf.nn.top_k(centrality_scores, k=tf.shape(centrality_scores)[1], sorted=True)

        # Reorder adjacency list based on sorted indices
        batch_size = tf.shape(adj_lists)[0]
        max_degree = tf.shape(adj_lists)[1]
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, max_degree])
        gather_indices = tf.stack([batch_indices, sorted_indices], axis=-1)
        sorted_adj_lists = tf.gather_nd(adj_lists, gather_indices)  # Shape: [batch_size, max_degree]

        # Slice to get the top-k neighbors
        top_neighbors = tf.slice(sorted_adj_lists, [0, 0], [-1, num_samples])  # Shape: [batch_size, num_samples]

        return top_neighbors

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