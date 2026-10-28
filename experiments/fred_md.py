"""
experiments/fred_md.py

Real Macro Forecasting Experiment
(FRED-MD)

Reference:
Shen & Xiu (2025)
Can Machines Learn Weak Signals?
"""

import os
import time
import numpy as np
import pandas as pd

from config import RANDOM_SEED

from models import get_model

from evaluation.rolling import (
    run_rolling,
    EXPANDING
)

np.random.seed(RANDOM_SEED)

# ==========================================================
# Paths
# ==========================================================

DATA_PATH = "data/FRED_MD"

FRED_FILE = os.path.join(
    DATA_PATH,
    "current.csv"
)

APPENDIX_FILE = os.path.join(
    DATA_PATH,
    "FRED-MD_updated_appendix.csv"
)

RESULT_PATH = "results"

# ==========================================================
# Models
# ==========================================================

MODELS = [

    "ridge",

    "lasso",

    "elasticnet"

]

# ==========================================================
# Target Variable (Shen & Xiu, Macro 1)
# ==========================================================

TARGET = "INDPRO"

# ==========================================================
# Read Appendix
# ==========================================================

# ==========================================================
# Load Appendix
# ==========================================================

def load_appendix():

    encodings = [
        "utf-8",
        "cp1252",
        "latin1"
    ]

    appendix = None

    for enc in encodings:

        try:

            appendix = pd.read_csv(
                APPENDIX_FILE,
                encoding=enc
            )

            print("Appendix encoding:", enc)

            break

        except UnicodeDecodeError:

            continue

    if appendix is None:
        raise Exception("Cannot read appendix file.")

    appendix.columns = appendix.columns.str.strip()

    return appendix


# ==========================================================
# Read FRED-MD
# ==========================================================
def load_fred():

    df = pd.read_csv(FRED_FILE)

    # ---------- Transformation Codes ----------
    tcodes = df.iloc[0].copy()

    # ---------- Remove Transform row ----------
    df = df.iloc[1:].reset_index(drop=True)

    # ---------- Date ----------
    df["sasdate"] = pd.to_datetime(
        df["sasdate"],
        format="%m/%d/%Y"
    )

    # ---------- Numeric ----------
    for c in df.columns:

        if c != "sasdate":

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    df = df.sort_values("sasdate").reset_index(drop=True)

    return df, tcodes

# ==========================================================
# Merge Appendix with Dataset
# ==========================================================

# ==========================================================
# Prepare Dataset
# ==========================================================

def prepare_dataset():

    df, tcodes_from_data = load_fred()

    appendix = load_appendix()

    available = []

    tcodes = {}

    for col in df.columns:

        if col == "sasdate":
            continue

        if col in appendix["fred"].values:

            available.append(col)

            tcodes[col] = int(tcodes_from_data[col])

    df = df[
        ["sasdate"] + available
    ]

    print("=" * 70)
    print("Variables :", len(available))
    print("=" * 70)

    return df, available, tcodes
