import numpy as np

def accuracy(y_pred, y_true):
    preds = np.argmax(y_pred, axis=1)
    true = np.argmax(y_true, axis=1)
    return np.mean(preds == true)