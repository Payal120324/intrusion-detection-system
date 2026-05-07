import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Map attack labels → 5 classes
def map_label(label):
    if label == "normal":
        return 0
    elif label in ["neptune","smurf","back","teardrop","pod","land"]:
        return 1   # DoS
    elif label in ["satan","ipsweep","nmap","portsweep","mscan","saint"]:
        return 2   # Probe
    elif label in ["buffer_overflow","rootkit","perl","loadmodule"]:
        return 3   # U2R
    else:
        return 4   # R2L


def process(path):
    df = pd.read_csv(path, header=None)

    # All features except last 2 columns
    X = df.iloc[:, :-2]
    y = df.iloc[:, -2].apply(map_label)

    # One-hot encode categorical features
    X = pd.get_dummies(X)
    columns = X.columns.tolist()

    X = X.values.astype(float)
    y = y.values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, columns


def transform_input(df, columns, scaler):
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    X = df.values.astype(float)
    X = scaler.transform(X)

    return X