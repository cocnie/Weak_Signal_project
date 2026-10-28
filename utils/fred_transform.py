"""
utils/fred_transform.py

FRED-MD Transformation Codes

Reference:
McCracken & Ng (2016)
Shen & Xiu (2025)
"""

import numpy as np
import pandas as pd


# ==========================================================
# Transformation 1
# Level
# ==========================================================

def transform_1(x):

    return x.copy()


# ==========================================================
# Transformation 2
# First Difference
# ==========================================================

def transform_2(x):

    return x.diff()


# ==========================================================
# Transformation 3
# Second Difference
# ==========================================================

def transform_3(x):

    return x.diff().diff()


# ==========================================================
# Transformation 4
# Natural Log
# ==========================================================

def transform_4(x):

    x = x.where(x > 0)

    return np.log(x)

# ==========================================================
# Transformation 5
# First Difference of Log
# ==========================================================

def transform_5(x):

    x = x.where(x > 0)

    return np.log(x).diff()


# ==========================================================
# Transformation 6
# Second Difference of Log
# ==========================================================

def transform_6(x):

    x = x.where(x > 0)

    return np.log(x).diff().diff()


# ==========================================================
# Transformation 7
# Difference of Percent Change
#
# Δ((Xt/Xt-1)-1)
#
# مطابق مقاله FRED-MD
# ==========================================================

def transform_7(x):

    return x.pct_change().diff()


# ==========================================================
# Dictionary
# ==========================================================

TRANSFORMERS = {

    1: transform_1,

    2: transform_2,

    3: transform_3,

    4: transform_4,

    5: transform_5,

    6: transform_6,

    7: transform_7

}

# ==========================================================
# Apply Transformations
# ==========================================================

def apply_transformations(df, tcodes):

    cols = {"sasdate": df["sasdate"]}

    for column in df.columns:

        if column == "sasdate":
            continue

        if column not in tcodes:
            continue

        code = int(tcodes[column])

        if code not in TRANSFORMERS:
            raise ValueError(f"Unknown transformation code {code} for {column}")

        cols[column] = TRANSFORMERS[code](df[column])

    transformed = pd.DataFrame(cols)

    transformed = transformed.dropna().reset_index(drop=True)

    return transformed


