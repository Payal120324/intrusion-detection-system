class NeuralNetwork:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations

    def forward(self, X):
        A = X
        for layer, activation in zip(self.layers, self.activations):
            Z = layer.forward(A)
            A = activation.forward(Z)
        return A

    def backward(self, dA, lr):
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
            # last layer (NO activation backward)
               dZ = dA
            else:
               dZ = self.activations[i].backward(dA)

            dA = self.layers[i].backward(dZ, lr)