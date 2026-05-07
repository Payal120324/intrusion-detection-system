import pandas as pd

def encode(df):
    return pd.get_dummies(df)