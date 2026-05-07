import numpy as np


def format_input(data):
    return np.array(data, dtype=np.float32)


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    return np.mean(predictions == y_true)


def shuffle_data(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
