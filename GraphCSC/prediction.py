from utils import zeros
from layers import Layer
import tensorflow as tf

class BipartiteEdgePredLayer(Layer):
    def __init__(self, input_dim1, input_dim2, placeholders, dropout=False, act=tf.nn.sigmoid,
                 loss_fn='xent', neg_sample_weights=1.0, bias=False, **kwargs):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+similar node and node and negative samples)
        Args:
            input_dim1: dimension of the first input (e.g., node embeddings)
            input_dim2: dimension of the second input (e.g., context embeddings)
            placeholders: dictionary of TensorFlow placeholders
            dropout: whether to apply dropout
            act: activation function
            loss_fn: loss function to use ('xent' for cross-entropy)
            neg_sample_weights: weight for negative samples
            bias: whether to use bias in the output layer
        """
        super(BipartiteEdgePredLayer, self).__init__(**kwargs)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.act = act
        self.bias = bias

        # Margin for hinge loss
        self.neg_sample_weights = neg_sample_weights

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.output_dim = 1
        with tf.variable_scope(self.name + '_vars'):
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss

        if self.logging:
            self._log_vars()

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
            inputs2: tensor of shape [batch_size x feature_size].
        """
        # Dot product between inputs1 and inputs2
        result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        return result

    def neg_cost(self, inputs1, neg_samples):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
        return neg_aff

    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def centrality_score(self, emb, reuse=False):
        """
        That is the mapping function f(v) as in the paper, which maps the
        embedding to a centrality score
        emb: tensor of shape [batch_size, embedding_dim].
        Returns: tensor of shape [batch_size], a scalar per node.
        """
        with tf.variable_scope("centrality_nn", reuse=reuse):
            # Example: one hidden layer + one output
            hidden = tf.layers.dense(emb, 64, activation=tf.nn.relu,
                                     name='hidden_layer')
            score = tf.layers.dense(hidden, 1, activation=None,
                                    name='output_layer')
            score = tf.squeeze(score, axis=1)
        return score

    def _xent_loss(self, inputs1, inputs2, neg_samples):
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss

    def ranking_loss(self, inputs1, inputs2):
        score_i = self.centrality_score(inputs1, reuse=False)
        score_j = self.centrality_score(inputs2, reuse=True)

        # sigma( f(vi) - f(vj) )
        diff = score_i - score_j
        ranking_prob = tf.nn.sigmoid(diff)

        # Added +1e-10 for numerical stability
        loss_c = -tf.reduce_sum(tf.log(ranking_prob + 1e-10))

        return loss_c

    def total_lost(self, inputs1, inputs2, neg_samples):
        Lg_loss = self._xent_loss(inputs1, inputs2, neg_samples)
        Lc_loss = self.ranking_loss(inputs1, inputs2)

        lambda_c = 0.5
        total_loss = Lg_loss + lambda_c * Lc_loss
        return total_loss