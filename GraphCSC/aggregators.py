import tensorflow as tf
from layers import Layer
from utils import glorot, zeros
import tensorflow as tf
import numpy as np

# Simple glorot init (same as in your code)
'''
def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=name)
'''
# Base Layer class

class CSCAggregator(Layer):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0.0, bias=False, act=tf.nn.relu, name=None, **kwargs):
        super(CSCAggregator, self).__init__(**kwargs)


        self.dropout = dropout
        self.bias = bias
        self.act = act

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''
        print(name)

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim], name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim + output_dim, output_dim], name='self_weights')

            if bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        # Transform neighbors and aggregate
        neigh_reshaped = tf.reshape(neigh_vecs, [-1, self.neigh_input_dim])
        neigh_transformed = tf.matmul(neigh_reshaped, self.vars['neigh_weights'])
        neigh_transformed = tf.reshape(neigh_transformed, [tf.shape(neigh_vecs)[0], -1, self.output_dim])
        neigh_agg = tf.reduce_sum(neigh_transformed, axis=1)

        # Concatenate and apply weights
        concat_vec = tf.concat([self_vecs, neigh_agg], axis=1)
        output = tf.matmul(concat_vec, self.vars['self_weights'])

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

