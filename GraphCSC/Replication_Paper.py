import tensorflow as tf
import numpy as np
from models import SampleAndAggregate, SAGEInfo
from minibatch import EdgeMinibatchIterator
from neigh_samplers import NeighborSampler
from utils import load_data, load_centrality_measures
import os
import json
import time

# Parameters
batch_size = 1024
max_degree = 30
neg_sample_size = 20
samples_1 = 25
samples_2 = 10
dim_1 = 128
dim_2 = 128
model_size = "small"
identity_dim = 0
dropout = 0.0
validate_iter = 5000
validate_batch_size = 1024
print_every = 20
max_total_steps = 100000
log_device_placement = False
save_embeddings = True
train_prefix = "ppi"
random_context = True
epochs = 5


def construct_placeholders():
    return {
        'batch1': tf.placeholder(tf.int32, shape=(None,), name='batch1'),
        'batch2': tf.placeholder(tf.int32, shape=(None,), name='batch2'),
        'dropout': tf.placeholder_with_default(0.0, shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }


def save_embeddings(sess, model, minibatch_iter):
    """Save node embeddings to a file."""
    embeddings = []
    nodes = []
    finished = False
    iter_num = 0

    while not finished:
        feed_dict, finished, batch_nodes = minibatch_iter.incremental_embed_feed_dict(batch_size, iter_num)
        print(f"Batch {iter_num}, Finished: {finished}, Batch Nodes: {len(batch_nodes)}")
        iter_num += 1
        outputs = sess.run(model.outputs1, feed_dict=feed_dict)
        embeddings.append(outputs)
        nodes.extend(batch_nodes)

    embeddings = np.vstack(embeddings)
    return embeddings, nodes


def evaluate(sess, model, minibatch_iter, size=None):
    """
    Evaluate the model on a single minibatch of validation data.
    Returns the loss for the validation set.
    """
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    val_loss = sess.run(model.loss, feed_dict=feed_dict_val)
    return val_loss


def incremental_evaluate(sess, model, minibatch_iter, size):
    """
    Evaluate the model on the entire validation set in smaller minibatches.
    Returns the average loss over the entire validation set.
    """
    finished = False
    val_losses = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        val_loss = sess.run(model.loss, feed_dict=feed_dict_val)
        val_losses.append(val_loss)
    return np.mean(val_losses)


def train(train_data, centrality_measures, context=True):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    if not features is None:
        features = np.vstack([features, np.zeros((features.shape[1],))])

    if context:
        context_pairs_walk = train_data[3]
        print(f"Number of random walk pairs: {len(context_pairs_walk)}")
        print(f"Sample pairs: {context_pairs_walk[:10]}")


    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(
        G, id_map, placeholders, centrality_measures, context_pairs = context_pairs_walk, batch_size=batch_size, max_degree=max_degree
    )

    # Define adjacency placeholder
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    # Define neighbor sampling
    sampler = NeighborSampler(adj_info, centrality_measures["degree"])
    layer_infos = [
        SAGEInfo("node", sampler, samples_1, dim_1),
        SAGEInfo("node", sampler, samples_2, dim_2)
    ]

    # Create GraphSAGE model
    model = SampleAndAggregate(
        placeholders,
        features,
        adj_info,
        minibatch.deg,
        layer_infos=layer_infos,
        G=G,
        model_size="small",
        identity_dim=0,
        logging=True
    )

    # Set up TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    # Train model
    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    for epoch in range(epochs):
        minibatch.shuffle()
        print(f"Epoch: {epoch + 1}")
        iter = 0

        while not minibatch.end():
            # Training step
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: dropout})
            _, train_loss = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

            # Print training progress
            if iter % print_every == 0:
                print(f"Iter: {iter}, Train Loss: {train_loss:.5f}")

            # Perform validation at regular intervals
            if iter % validate_iter == 0:
                # Switch to validation adjacency
                sess.run(val_adj_info.op)
                val_loss = incremental_evaluate(sess, model, minibatch, validate_batch_size)
                print(f"Validation Loss after {iter} iterations: {val_loss:.5f}")
                # Switch back to training adjacency
                sess.run(train_adj_info.op)

            iter += 1

    # Save all embeddings
    sess.run(train_adj_info.op)  # Ensure training adjacency is active
    all_embeddings, all_nodes = save_embeddings(sess, model, minibatch)
    all_nodes = [node[0] for node in all_nodes]  # All nodes as a flat list

    print(f"Total Nodes Processed in Embeddings: {len(all_nodes)}")

    train_nodes = [n for n, data in G.nodes(data=True) if not data.get('val', False) and not data.get('test', False)]
    test_nodes = [n for n, data in G.nodes(data=True) if data.get('val', False) or data.get('test', False)]


    # Create a mapping from node ID to its index
    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
    train_indices = [node_to_index[n] for n in train_nodes]
    test_indices = [node_to_index[n] for n in test_nodes]

    # Extract train and test embeddings using the indices
    train_embeddings = all_embeddings[train_indices]
    test_embeddings = all_embeddings[test_indices]

    # Save embeddings and corresponding nodes separately
    np.save("train_embeddings.npy", train_embeddings)
    np.save("test_embeddings.npy", test_embeddings)

    with open("train_nodes.txt", "w") as f:
        for node in train_nodes:
            f.write(f"{node}\n")

    with open("test_nodes.txt", "w") as f:
        for node in test_nodes:
            f.write(f"{node}\n")


def main():
    print("Loading data...")
    directory = r"C:\Users\User\PycharmProjects\ML4Graph\PPI_Data"
    centrality_dict = load_centrality_measures(directory, "normalized_degree_centrality.json")
    train_data = load_data(train_prefix, directory,  load_walks=True)
    print("Data loaded.")
    train(train_data, centrality_dict)


if __name__ == "__main__":
    main()
