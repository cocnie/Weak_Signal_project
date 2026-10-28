"""
experiments/fred_forecast.py

Macro Forecasting Experiment
(FRED-MD)

Reference
---------
Shen & Xiu (2025)
Can Machines Learn Weak Signals?
"""

import os
import time
import numpy as np
import pandas as pd

from config import RANDOM_SEED
from models import get_model
from evaluation.rolling import run_rolling, EXPANDING

from experiments.fred_transform import (
    load_transformed_dataset
)

np.random.seed(RANDOM_SEED)

RESULT_PATH = "results"

MODELS = [
    "ridge",
    "lasso",
    "elasticnet"
]

# ==========================================================
# Forecast Targets
# ==========================================================

FORECAST_TARGETS = [

    "INDPRO",

    "PAYEMS",

    "UNRATE",

    "CPIAUCSL",

    "PCEPI",

    "HOUST",

    "PERMIT",

    "S&P 500",

    "GS10",

    "FEDFUNDS",

    "BUSLOANS",

    "VIXCLSx"

]

# ==========================================================
# Build Forecast Dataset
# ==========================================================

def build_dataset(
    df,
    target,
    horizon=1
):
    """
    Build forecasting dataset.

    Parameters
    ----------
    df : transformed FRED dataframe

    target : forecasting target

    horizon : forecast horizon

    Returns
    -------
    X
    y
    """

    if target not in df.columns:
        raise ValueError(f"{target} not found.")

    y = df[target].shift(-horizon)

    X = df.drop(
        columns=[
            "sasdate",
            target
        ]
    ).copy()

    data = pd.concat(
        [
            X,
            y.rename("target")
        ],
        axis=1
    )

    data = data.dropna().reset_index(drop=True)

    X = data.drop(columns="target")

    y = data["target"]

    return X, y


# ==========================================================
# Run One Forecast
# ==========================================================

def run_one_forecast(
    df,
    target,
    model_name,
    horizon=1
):

    X, y = build_dataset(
        df,
        target,
        horizon
    )

    model = get_model(model_name)

    results = run_rolling(
        model_module=model,
        X=X,
        y=y,
        train_size=200,
        step_size=1,
        mode=EXPANDING
    )

    summary = dict(results["summary"])

    summary["Target"] = target
    summary["Model"] = model_name
    summary["Horizon"] = horizon
    summary["Windows"] = results["windows"]

    for k, v in results["parameters"].items():
        summary[k] = v

    prediction = results["results"].copy()

    prediction["Target"] = target
    prediction["Model"] = model_name
    prediction["Horizon"] = horizon

    return summary, prediction

# ==========================================================
# Run All Forecasts
# ==========================================================

def run_all_forecasts(
    df,
    horizon=1
):

    summaries = []

    predictions = []

    for target in FORECAST_TARGETS:

        print("=" * 70)
        print(f"Target : {target}")
        print("=" * 70)

        if target not in df.columns:

            print(f"{target} not found.")
            continue

        for model_name in MODELS:

            print(f"Running {model_name}")

            summary, prediction = run_one_forecast(
                df=df,
                target=target,
                model_name=model_name,
                horizon=horizon
            )

            summaries.append(summary)

            predictions.append(prediction)

    summary_df = pd.DataFrame(summaries)

    prediction_df = pd.concat(
        predictions,
        ignore_index=True
    )

    return summary_df, prediction_df


# ==========================================================
# Save Results
# ==========================================================

def save_results(
    summary_df,
    prediction_df,
    horizon
):

    os.makedirs(
        os.path.join(
            RESULT_PATH,
            "tables"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            RESULT_PATH,
            "raw"
        ),
        exist_ok=True
    )

    summary_df.to_csv(

        os.path.join(
            RESULT_PATH,
            "tables",
            f"fred_forecast_summary_h{horizon}.csv"
        ),

        index=False

    )

    prediction_df.to_csv(

        os.path.join(
            RESULT_PATH,
            "raw",
            f"fred_forecast_predictions_h{horizon}.csv"
        ),

        index=False

    )
# ==========================================================
# Main Experiment
# ==========================================================

def run_experiment(
    horizon=1
):

    start = time.time()

    print("=" * 70)
    print("Loading FRED-MD")
    print("=" * 70)

    df, variables, tcodes = load_transformed_dataset()

    print(f"Observations : {len(df)}")
    print(f"Variables    : {len(variables)}")
    print(f"Horizon      : {horizon}")

    print("=" * 70)
    print("Running Forecast Experiment")
    print("=" * 70)

    summary_df, prediction_df = run_all_forecasts(
        df=df,
        horizon=horizon
    )

    save_results(
        summary_df,
        prediction_df,
        horizon
    )

    elapsed = time.time() - start

    print("=" * 70)
    print("Finished Successfully")
    print(f"Execution Time : {elapsed:.2f} sec")
    print("=" * 70)

    return summary_df, prediction_df


# ==========================================================
# Script Entry
# ==========================================================

if __name__ == "__main__":

    run_experiment(
        horizon=1
    )

