"""
experiments/fred_experiment.py

Real Macro Forecasting Experiment
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

from experiments.fred_md import (
    prepare_dataset,
    TARGET,
    MODELS,
    RESULT_PATH
)

from utils.fred_transform import (
    apply_transformations
)

from evaluation.rolling import (
    run_rolling,
    EXPANDING
)

np.random.seed(RANDOM_SEED)


# ==========================================================
# Load and Transform Dataset
# ==========================================================

def load_transformed_dataset():
    """
    Load FRED-MD dataset and apply
    McCracken-Ng transformation codes.

    Returns
    -------
    transformed_df : DataFrame

    variables : list

    tcodes : dict
    """

    df, variables, tcodes = prepare_dataset()

    transformed = apply_transformations(
        df=df,
        tcodes=tcodes
    )

    transformed = (
        transformed
        .dropna()
        .reset_index(drop=True)
    )

    return transformed, variables, tcodes


# ==========================================================
# Select Target Variable
# ==========================================================

def prepare_target(
    transformed_df
):
    """
    Separate target variable.

    Returns
    -------
    y : Series
    """

    if TARGET not in transformed_df.columns:

        raise ValueError(
            f"{TARGET} not found."
        )

    y = transformed_df[TARGET].copy()

    return y


# ==========================================================
# Select Predictor Variables
# ==========================================================

def prepare_predictors(
    transformed_df
):
    """
    Remove date column and target.

    Returns
    -------
    X : DataFrame
    """

    X = transformed_df.drop(
        columns=[
            "sasdate",
            TARGET
        ]
    )

    return X

# ==========================================================
# Build Feature Matrix
# ==========================================================

def build_feature_matrix(df):

    X = df.drop(columns=["sasdate", TARGET]).copy()
    y = df[TARGET].copy()

    data = pd.concat([X, y.rename("target")], axis=1)
    data = data.dropna().reset_index(drop=True)

    X = data.drop(columns="target")
    y = data["target"]

    return X, y


# ==========================================================
# Run One Model
# ==========================================================

def run_one_model(model_name, X, y):

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
    summary["Model"] = model_name
    summary["Windows"] = results["windows"]

    for k, v in results["parameters"].items():
        summary[k] = v

    prediction = results["results"].copy()
    prediction["Model"] = model_name

    return summary, prediction


# ==========================================================
# Run All Models
# ==========================================================

def run_all_models(X, y):

    summaries = []
    predictions = []

    for model_name in MODELS:

        print(f"Running {model_name}")

        summary, prediction = run_one_model(
            model_name,
            X,
            y
        )

        summaries.append(summary)
        predictions.append(prediction)

    summary_df = pd.DataFrame(summaries)
    prediction_df = pd.concat(predictions, ignore_index=True)

    return summary_df, prediction_df

# ==========================================================
# Save Results
# ==========================================================

def save_results(summary_df, prediction_df):

    os.makedirs(os.path.join(RESULT_PATH, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_PATH, "raw"), exist_ok=True)

    summary_df.to_csv(
        os.path.join(
            RESULT_PATH,
            "tables",
            "fred_md_summary.csv"
        ),
        index=False
    )

    prediction_df.to_csv(
        os.path.join(
            RESULT_PATH,
            "raw",
            "fred_md_predictions.csv"
        ),
        index=False
    )


# ==========================================================
# Main Experiment
# ==========================================================

def run_fred_experiment():

    start = time.time()

    print("=" * 70)
    print("Loading FRED-MD Dataset")
    print("=" * 70)

    transformed_df, variables, tcodes = load_transformed_dataset()

    print(f"Observations : {len(transformed_df)}")
    print(f"Variables    : {len(variables)}")

    X, y = build_feature_matrix(transformed_df)

    print("=" * 70)
    print("Running Models")
    print("=" * 70)

    summary_df, prediction_df = run_all_models(X, y)

    save_results(
        summary_df,
        prediction_df
    )

    elapsed = time.time() - start

    print("=" * 70)
    print("Finished")
    print(f"Execution Time : {elapsed:.2f} sec")
    print("=" * 70)

    return summary_df, prediction_df


# ==========================================================
# Script Entry
# ==========================================================

if __name__ == "__main__":

    run_fred_experiment()