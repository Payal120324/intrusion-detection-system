# AI-Based Intrusion Detection System (IDS)

A full-stack Machine Learning based Intrusion Detection System capable of detecting and classifying multiple types of network attacks using the NSL-KDD dataset.

This project combines:
- Custom Neural Network implementation from scratch using NumPy
- Multi-class Machine Learning classification
- FastAPI backend
- Interactive dashboard frontend
- Data preprocessing pipeline
- Model evaluation and visualization

---

#  Features

##  Multi-Class Intrusion Detection
Detects and classifies:
- Normal Traffic
- DoS Attacks
- Probe Attacks
- U2R Attacks
- R2L Attacks

---

##  Custom Neural Network From Scratch
Implemented manually using NumPy:
- Forward propagation
- Backpropagation
- ReLU activation
- Softmax output
- Gradient descent optimization

---

##  Random Forest Production Model
Used for final deployment due to better stability and performance on the NSL-KDD dataset.

---

##  Interactive Dashboard
Includes:
- CSV upload
- Real-time predictions
- Attack analytics
- Prediction charts
- Metrics cards
- Attack distribution visualization

---

##  Model Evaluation
Supports:
- Accuracy score
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Tech Stack

## Backend
- Python
- FastAPI

## Machine Learning
- Scikit-learn
- NumPy
- Pandas

## Visualization
- Matplotlib
- Chart.js

## Frontend
- HTML
- CSS
- JavaScript

---

#  Project Structure

```text
ids-neural-network/
│
├── api/
│   └── app.py
│
├── config/
│   └── config.py
│
├── dashboard/
│   └── frontend/
│       └── index.html
│
├── data/
│   └── raw/
│       └── nsl_kdd.csv
│
├── inference/
│   └── predictor.py
│
├── models/
│   ├── columns.json
│   ├── model.pkl
│   └── scaler.pkl
│
├── nn/
│   ├── activations.py
│   ├── layers.py
│   ├── loss.py
│   ├── model.py
│   ├── network.py
│   ├── optimizer.py
│   └── utils.py
│
├── preprocessing/
│   ├── encoder.py
│   ├── loader.py
│   ├── pipeline.py
│   ├── scaler.py
│   └── splitter.py
│
├── training/
│   ├── evaluator.py
│   ├── logger.py
│   ├── metrics.py
│   └── trainer.py
│
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

#  Dataset

Dataset used:
- NSL-KDD Dataset

The dataset contains labeled network traffic records used for intrusion detection research.

---

#  Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/ai-intrusion-detection-system.git
```

---

## 2. Enter Project Folder

```bash
cd ai-intrusion-detection-system
```

---

## 3. Create Virtual Environment

```bash
python -m venv venv
```

---

## 4. Activate Virtual Environment

### Windows
```bash
venv\Scripts\activate
```


## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Training the Model

```bash
python train.py
```

This generates:
- model.pkl
- scaler.pkl
- columns.json

inside the `models/` directory.

---

#  Evaluate Model

```bash
python evaluate.py
```

Metrics generated:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

#  Run Backend API

```bash
uvicorn api.app:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

#  Run Frontend Dashboard

```bash
cd dashboard/frontend
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

---

#  Using the Dashboard

1. Upload CSV file
2. Click **Upload & Predict**
3. View:
   - attack predictions
   - attack distribution
   - analytics graphs
   - metrics

---

#  Model Performance

Example evaluation results:

| Metric | Score |
|---|---|
| Accuracy | ~98% |
| Multi-Class Classification | 
| Real-Time Prediction | 

---

#  Key Learning Outcomes

This project demonstrates:
- Neural Networks from scratch
- Feature engineering
- Multi-class classification
- ML pipeline design
- Model evaluation
- API development
- Frontend integration
- Real-world debugging and deployment workflow

---

#  Future Improvements

Possible future upgrades:
- Live packet sniffing
- Real-time network monitoring
- Docker deployment
- XGBoost integration
- SHAP feature importance
- User authentication
- Database logging

---

