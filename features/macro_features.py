"""
features/macro_features.py

Macro Feature Engineering

Reference:
Shen & Xiu (2025)
Can Machines Learn Weak Signals?

Input:
    FRED-MD transformed variables

Output:
    Weak Signal Features
"""

import numpy as np
import pandas as pd

from config import RANDOM_SEED

np.random.seed(RANDOM_SEED)

# ==========================================================
# Parameters
# ==========================================================

#MAX_LAG = 12

#ROLLING_WINDOWS = [3,6,12]

MAX_LAG = 3

ROLLING_WINDOWS = [3]

TARGET_VARIABLE = "INDPRO"

# ==========================================================
# Lag Features
# ==========================================================

def create_lags(df,max_lag=MAX_LAG):

    cols={}

    for col in df.columns:

        if col=="sasdate":
            continue

        for lag in range(1,max_lag+1):

            cols[f"{col}_lag{lag}"]=df[col].shift(lag)

    return pd.DataFrame(cols,index=df.index)


# ==========================================================
# Rolling Statistics
# ==========================================================

def create_rolling(df,windows=ROLLING_WINDOWS):

    cols={}

    for col in df.columns:

        if col=="sasdate":
            continue

        for w in windows:

            cols[f"{col}_mean{w}"]=df[col].rolling(w).mean()

            cols[f"{col}_std{w}"]=df[col].rolling(w).std()

    return pd.DataFrame(cols,index=df.index)


# ==========================================================
# Momentum Features
# ==========================================================

def create_momentum(df):

    cols={}

    for col in df.columns:

        if col=="sasdate":
            continue

        cols[f"{col}_mom3"]=df[col]-df[col].shift(3)
        cols[f"{col}_mom6"]=df[col]-df[col].shift(6)
        cols[f"{col}_mom12"]=df[col]-df[col].shift(12)

    return pd.DataFrame(cols,index=df.index)
# ==========================================================
# Build Final Dataset
# ==========================================================

def build_features(df,target=TARGET_VARIABLE,horizon=1):

    data=df.copy()

    y=data[target].shift(-horizon)

    lag_features=create_lags(data)

    rolling_features=create_rolling(data)

    momentum_features=create_momentum(data)

    X=pd.concat(
        [lag_features,rolling_features,momentum_features],
        axis=1
    )

    dataset=pd.concat(
        [X,y.rename("target")],
        axis=1
    )

    dataset=dataset.dropna().reset_index(drop=True)

    X=dataset.drop(columns="target")

    y=dataset["target"]

    return X,y


# ==========================================================
# Summary
# ==========================================================

def summarize_features(X):

    print("="*70)

    print("Feature Matrix Shape:")

    print(X.shape)

    print("="*70)

    print("First 10 Features:")

    print(X.columns[:10].tolist())

    print("="*70)

