import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from preprocessing.pipeline import process

# Load data
X_train, X_test, y_train, y_test, scaler, columns = process("data/raw/nsl_kdd.csv")

# Train model (multi-class)
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    random_state=42
)

model.fit(X_train, y_train)

# Save everything
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
json.dump(columns, open("models/columns.json", "w"))

print("✅ Multi-class model trained successfully")