import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        # Better initialization (break symmetry)
        self.W = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        self.b = np.random.randn(1, output_size) * 0.01

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ, lr):
        m = self.X.shape[0]

        dW = np.dot(self.X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W.T)

        self.W -= lr * dW
        self.b -= lr * db

        return dX