from training.metrics import accuracy

class Trainer:
    def __init__(self, model, loss_fn, lr):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.model.forward(X)

            loss = self.loss_fn.forward(y_pred, y)

            dA = self.loss_fn.backward()
            self.model.backward(dA, self.lr)

            if epoch % 10 == 0:
                acc = accuracy(y_pred, y)
                print(f"Epoch {epoch} | Loss {loss:.4f} | Acc {acc:.4f}")