from collections import namedtuple
import numpy as np
import tensorflow as tf
import math
import random
import layers as layers
import metrics as metrics

from prediction import BipartiteEdgePredLayer
from aggregators import  CSCAggregator
from utils1 import load_centrality_measures

# Disclaimer
# The Model object
# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    """ A standard multi-layer perceptron """
    def __init__(self, placeholders, dims, categorical=True, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.categorical = categorical

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']

        self.optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if self.categorical:
            self.loss += metrics.masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))

    def _accuracy(self):
        if self.categorical:
            self.accuracy = metrics.masked_accuracy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(layers.Dense(input_dim=self.input_dim,
                                 output_dim=self.dims[1],
                                 act=tf.nn.relu,
                                 dropout=self.placeholders['dropout'],
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(layers.Dense(input_dim=self.dims[1],
                                 output_dim=self.output_dim,
                                 act=lambda x: x,
                                 dropout=self.placeholders['dropout'],
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

# SAGEInfo is a namedtuple that specifies the parameters 
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])

class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
            layer_infos, G, concat=False, aggregator_type="mean",
            model_size="small", identity_dim=0,learning_rate=0.7, neg_sample_size=5, weight_decay=0.0,
            **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features. 
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)
        #if aggregator_type == "mean":
        self.aggregator_cls = CSCAggregator

        '''
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)
        '''

        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.model_size = model_size
        self.adj_info = adj
        self.learning_rate = learning_rate
        self.neg_sample_size = neg_sample_size
        self.weight_decay = weight_decay
        self.G = G

        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat

        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        self.build()

    def generate_negative_samples_from_labels(self, labels, num_samples, graph, alpha=0.75):
        """
        Produce exactly `num_samples` negative samples as a 1D tensor [num_samples],
        excluding any node that:
          - is itself a label in `labels`,
          - or is a neighbor of any label in `labels`.

        It samples nodes from a "centrality^alpha" distribution,
        using random.choices(...) in Python.

        Args:
            labels:      tf.Tensor of shape [batch_size, 1] or [batch_size].
            num_samples: Scalar int, how many negative samples to draw.
            graph:       NetworkX graph (assumed to have integer node IDs).
            alpha:       Exponent for distortion (like in unigram sampling).

        Returns:
            neg_samples_tf: A tf.Tensor of shape [num_samples], dtype=int64.
        """

        # 1) Load your centrality measures (like "degrees") from JSON.
        centrality_data = load_centrality_measures("centrality_measures_final_complete.json")
        centrality_degree = centrality_data["degree"]  # dict: str(node_id) -> float

        # Use only the actual nodes in the graph
        nodes = list(graph.nodes())

        # Build centrality probabilities only for nodes in the graph
        centrality_list = [centrality_degree.get(str(node), 0.0) for node in nodes]
        centrality_list = np.array(centrality_list, dtype=np.float32)

        # Apply exponent distortion
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

        # 2) Wrap the sampler in tf.py_func so we can call it inside the TF graph
        neg_samples_tf = tf.py_func(
            func=py_negative_sampler,
            inp=[labels],
            Tout=tf.int64
        )

        # 3) Set shape to [num_samples]
        neg_samples_tf.set_shape([num_samples])

        return neg_samples_tf

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
                  aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []

        for layer in range(len(num_samples)):
            if new_agg:
                # Check if using CSCAggregator to adjust dim_mult
                dim_mult = 1  # Disable concat-based dimension doubling


                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult * dims[layer], dims[layer + 1],
                                                     act=lambda x: x, dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult * dims[layer], dims[layer + 1],
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              dim_mult * dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])

        '''
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled= self.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))
            
        '''

        self.neg_samples = self.generate_negative_samples_from_labels(labels, self.neg_sample_size, self.G)

        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        self.outputs2, _ = self.aggregate(samples2, [self.features], self.dims, num_samples,
                support_sizes2, aggregators=self.aggregators, concat=self.concat,
                model_size=self.model_size)

        neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos,
            self.neg_sample_size)
        self.neg_outputs, _ = self.aggregate(neg_samples, [self.features], self.dims, num_samples,
                neg_support_sizes, batch_size=self.neg_sample_size, aggregators=self.aggregators,
                concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
                dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid, 
                bilinear_weights=False,
                name='edge_predict')

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

    def build(self):
        self._build()

        # TF graph management
        self._loss()
        self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        # Add aggregator L2 regularization, etc.
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.total_lost(self.outputs1,
                                                     self.outputs2,
                                                     self.neg_outputs)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, self.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)