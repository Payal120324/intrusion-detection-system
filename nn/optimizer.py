class GradientDescent:
    def __init__(self, lr):
        self.lr = lr
    def update(self, weights, gradients):
        return weights - self.lr * gradients
