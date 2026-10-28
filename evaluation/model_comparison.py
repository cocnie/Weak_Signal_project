"""
evaluation/model_comparison.py

Compare Multiple Models

Reference:
Shen & Xiu (2025)
"""

import os
import time
import pandas as pd

RESULT_PATH = "results"


# ==========================================================
# Compare Models
# ==========================================================

def compare_models(results_dict):

    rows = []

    for model_name, result in results_dict.items():

        row = {
            "Model": model_name,
            "R2": result["summary"]["R2"],
            "R2_OOS": result["summary"]["R2_OOS"],
            "MSPE": result["summary"]["MSPE"],
            "RMSE": result["summary"]["RMSE"]
        }

        if "parameters" in result:

            for key, value in result["parameters"].items():
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by="R2_OOS",
        ascending=False
    ).reset_index(drop=True)

    return df


# ==========================================================
# Add Execution Time
# ==========================================================

def add_execution_time(df, execution_times):

    df = df.copy()

    df["Execution_Time"] = df["Model"].map(execution_times)

    return df


# ==========================================================
# Add Number of Selected Variables
# ==========================================================

def add_nonzero_variables(df, nonzero_dict):

    df = df.copy()

    df["NonZero"] = df["Model"].map(nonzero_dict)

    return df


# ==========================================================
# Rank Models
# ==========================================================

def rank_models(df):

    ranking = df.copy()

    ranking["Rank"] = range(1, len(df) + 1)

    cols = ["Rank"] + [c for c in ranking.columns if c != "Rank"]

    return ranking[cols]


# ==========================================================
# Save Results
# ==========================================================

def save_results(df, filename="model_comparison.csv"):

    os.makedirs(
        os.path.join(RESULT_PATH, "tables"),
        exist_ok=True
    )

    path = os.path.join(
        RESULT_PATH,
        "tables",
        filename
    )

    df.to_csv(path, index=False)

    return path


# ==========================================================
# Pretty Print
# ==========================================================

def print_results(df):

    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(df)
    print("=" * 70)