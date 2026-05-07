import numpy as np

class NeuralNet:
    def __init__(self, input_size, num_classes):
        self.W1 = np.random.randn(input_size, 64) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, 64))

        self.W2 = np.random.randn(64, 32) * np.sqrt(2 / 64)
        self.b2 = np.zeros((1, 32))

        self.W3 = np.random.randn(32, num_classes) * np.sqrt(2 / 32)
        self.b3 = np.zeros((1, num_classes))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.softmax(self.Z3)

        return self.A3

    def backward(self, X, y, lr):
        m = X.shape[0]

        dZ3 = self.A3 - y
        dW3 = (self.A2.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (self.Z2 > 0)
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1