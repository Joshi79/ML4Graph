from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import numpy as np
from utils import load_centrality_measures

train_embeddings = np.load("train_embeddings.npy")
test_embeddings = np.load("test_embeddings.npy")

print(f"Train Embeddings Shape: {train_embeddings.shape}")
print(f"Test Embeddings Shape: {test_embeddings.shape}")

with open("train_nodes.txt", "r") as f:
    train_nodes = [line.strip() for line in f]

with open("test_nodes.txt", "r") as f:
    test_nodes = [line.strip() for line in f]

directory = r"C:\Users\User\PycharmProjects\ML4Graph\PPI_Data"
centrality_measure = load_centrality_measures(directory, "normalized_degree_centrality.json")
centrality = centrality_measure["degree"]

mean_centrality = np.mean(list(centrality.values()))
print("Mean Centrality:", mean_centrality)
std_centrality = np.std(list(centrality.values()))
print(std_centrality)

train_centrality = np.array([centrality.get(node, 0) for node in train_nodes]).reshape(-1, 1)
test_centrality = np.array([centrality.get(node, 0) for node in test_nodes]).reshape(-1, 1)

log_train_centrality = np.log1p(train_centrality)
log_test_centrality = np.log1p(test_centrality)

model = Sequential([
    Dense(64, activation='relu', input_dim=train_embeddings.shape[1]),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=SGD(lr=0.07), loss='mean_absolute_error', metrics=['mae'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)

model.fit(
    train_embeddings, train_centrality,
    epochs=100   ,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)

loss, mae = model.evaluate(test_embeddings, test_centrality)
print("Test MAE:", mae)

y_max = np.max(test_centrality)
y_min = np.min(test_centrality)
y_mean = np.mean(test_centrality)

nmae = mae / (y_max - y_min)
cv_mae = mae / y_mean

print("\nEvaluation Metrics:")
print(f"Test MAE: {mae}")
print(f"Test NMAE: {nmae}")
print(f"Test CV(MAE): {cv_mae}")
