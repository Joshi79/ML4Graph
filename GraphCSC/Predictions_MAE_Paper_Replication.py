from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
from utils import load_centrality_measures

# Load the train and test embeddings
train_embeddings = np.load("train_embeddings.npy")
test_embeddings = np.load("test_embeddings.npy")

print(f"Train Embeddings Shape: {train_embeddings.shape}")
print(f"Test Embeddings Shape: {test_embeddings.shape}")

# Load the node lists for train and test
with open("train_nodes.txt", "r") as f:
    train_nodes = [line.strip() for line in f]

with open("test_nodes.txt", "r") as f:
    test_nodes = [line.strip() for line in f]

directory = r"C:\Users\User\PycharmProjects\ML4Graph\PPI_Data"
centrality_measure = load_centrality_measures(directory,"normalized_degree_centrality.json")
centrality = centrality_measure["degree"]

mean_centrality = np.mean(list(centrality.values()))
print(mean_centrality)

# Align centrality values with the embeddings
train_centrality = np.array([centrality.get(node, 0) for node in train_nodes]).reshape(-1, 1)
test_centrality = np.array([centrality.get(node, 0) for node in test_nodes]).reshape(-1, 1)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_dim=train_embeddings.shape[1]),  # Hidden layer
    Dense(1, activation='sigmoid')                                     # Output layer
])

# Compile the model
model.compile(optimizer=SGD(lr=0.001), loss='mean_absolute_error', metrics=['mae'])
# Train the model
model.fit(train_embeddings, train_centrality, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, mae = model.evaluate(test_embeddings, test_centrality)
print("Test MAE:", mae)
