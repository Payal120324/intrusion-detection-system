import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
from preprocessing.pipeline import process

# Load data
X_train, X_test, y_train, y_test, scaler, columns = process("data/raw/nsl_kdd.csv")

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

# Predict
y_pred = model.predict(X_test)
# Labels
class_names = ["Normal", "DoS", "Probe", "U2R", "R2L"]

# ----------------------------
# 1. ACCURACY
# ----------------------------
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")

# ----------------------------
# 2. CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

# Visual confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

print("\n📈 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names
))

# ----------------------------
# 4. PRECISION / RECALL / F1
# ----------------------------
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-Score')

plt.xticks(x, class_names)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Precision / Recall / F1-Score")
plt.legend()
plt.show()

# ----------------------------
# 5. PREDICTION DISTRIBUTION
# ----------------------------
unique, counts = np.unique(y_pred, return_counts=True)

plt.figure(figsize=(8, 5))
plt.bar(class_names, counts)
plt.title("Prediction Distribution")
plt.ylabel("Count")
plt.show()