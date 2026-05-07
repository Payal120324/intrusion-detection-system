import pickle
import json
import pandas as pd

class Predictor:
    def __init__(self):
        self.model = pickle.load(open("models/model.pkl", "rb"))
        self.scaler = pickle.load(open("models/scaler.pkl", "rb"))
        self.columns = json.load(open("models/columns.json"))

    def predict_with_proba(self, data_dict):
        df = pd.DataFrame([data_dict])

        df = pd.get_dummies(df)
        df = df.reindex(columns=self.columns, fill_value=0)

        X = self.scaler.transform(df.values)

        pred = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]

        return int(pred), probs.tolist()