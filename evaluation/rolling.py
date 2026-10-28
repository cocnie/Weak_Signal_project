"""
evaluation/rolling.py

Rolling / Expanding Window Evaluation
Reference:
Shen & Xiu (2025)
"""

import numpy as np
import pandas as pd
from utils.preprocessing import preprocess
from evaluation.metrics import evaluate
from tqdm import tqdm
from config import ROLLING, EXPANDING

def run_rolling(model_module,
                X,
                y,
                train_size=500,
                step_size=10,
                mode=ROLLING,
                progress=True):

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    n = len(X)

    y_true = []
    y_pred = []

    parameter_history = []

    start = 0
    n_windows = max(0, len(X) - train_size)

    pbar = tqdm(total=n_windows, desc="Rolling windows") if progress else None
    while True:

        if mode == ROLLING:
            train_start = start
            train_end = start + train_size
        else:
            train_start = 0
            train_end = train_size + start

        test_start = train_end

        if test_start >= n:
            break

        test_end = min(test_start + step_size, n)

        if test_end <= test_start:
            break

        if test_end > n:
            break

        X_train = X.iloc[train_start:train_end].copy()
        y_train = y.iloc[train_start:train_end].copy()

        X_test = X.iloc[test_start:test_end].copy()
        y_test = y.iloc[test_start:test_end].copy()

        if len(X_test) == 0:
            break

        X_train, X_test, scaler = preprocess(
            X_train,
            X_test
        )

        model = model_module.train(
            X_train,
            y_train
        )

        pred = model.predict(X_test)

        y_true.extend(y_test.tolist())
        y_pred.extend(pred.tolist())

        if hasattr(model_module, "get_parameters"):
            parameter_history.append(
                model_module.get_parameters(model)
            )
        else:
            parameter_history.append({})

        start += step_size
        if pbar is not None:
            pbar.update(step_size)
    if pbar is not None:
        pbar.close()
    
    test_results = evaluate(
        np.array(y_true),
        np.array(y_pred)
    )

    summary = {
        "R2": test_results["R2"],
        "R2_OOS": test_results["R2_OOS"],
        "MSPE": test_results["MSPE"],
        "RMSE": test_results["RMSE"]
    }

    mean_parameters = {}

    if len(parameter_history) > 0:

        keys = parameter_history[0].keys()

        for key in keys:

            values = [
                p[key]
                for p in parameter_history
                if key in p and isinstance(p[key], (int, float, np.number))
            ]

            if len(values) > 0:
                mean_parameters[key] = float(np.mean(values))

    prediction_df = pd.DataFrame({
        "Target": y_true,
        "Prediction": y_pred
    })

    return {
        "summary": summary,
        "results": prediction_df,
        "parameters": mean_parameters,
        "windows": len(parameter_history)
    }