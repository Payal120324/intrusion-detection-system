from training.metrics import accuracy

def evaluate(model, X, y):
    y_pred = model.forward(X)

    acc = accuracy(y_pred, y)
    print("Accuracy:", acc)

    return acc