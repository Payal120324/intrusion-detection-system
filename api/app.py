from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from inference.predictor import Predictor
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()

# 5-class labels
LABELS = ["Normal", "DoS", "Probe", "U2R", "R2L"]

@app.get("/")
def home():
    return {"message": "IDS Multi-Class API Running"}

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, header=None)

    results = []

    for i, row in df.iterrows():
        data = row.to_dict()

        pred, probs = predictor.predict_with_proba(data)

        results.append({
            "row": i+1,
            "label": LABELS[pred],
            "confidence": float(max(probs))
        })

    return {"results": results}