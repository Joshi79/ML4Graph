from layers import Layer
import tensorflow as tf


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

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        max_node_id = tf.shape(self.centrality_tensor)[0] - 1
        adj_lists = tf.clip_by_value(adj_lists, 0, max_node_id)


        centrality_scores = tf.gather(self.centrality_tensor, adj_lists)

        _, sorted_indices = tf.nn.top_k(centrality_scores, k=tf.shape(centrality_scores)[1], sorted=True)

        batch_size = tf.shape(adj_lists)[0]
        max_degree = tf.shape(adj_lists)[1]
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, max_degree])
        gather_indices = tf.stack([batch_indices, sorted_indices], axis=-1)
        sorted_adj_lists = tf.gather_nd(adj_lists, gather_indices)

        top_neighbors = tf.slice(sorted_adj_lists, [0, 0], [-1, num_samples])

        return top_neighbors
