"""
ridge.py

Ridge Regression Model
Shen & Xiu (2025)

Author:
Ali Heidarpoor

Project:
Weak Signals Detection
"""

import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

import config


# ============================================================
# Train
# ============================================================

def train(X_train, y_train):
    """
    Train Ridge Regression model.

    Parameters
    ----------
    X_train : ndarray

    y_train : ndarray

    Returns
    -------
    model
    """

    lambda_grid = np.logspace(-4, 4, 50)
    #lambda_grid = np.logspace(-8,8,200)
    #alpha_grid = (config.N_FEATURES/config.N_OBSERVATIONS) * lambda_grid
    alpha_grid = config.N_FEATURES * lambda_grid
    model = RidgeCV(

        alphas=alpha_grid,
        fit_intercept=False,

        cv=config.RIDGE_CV

    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# Evaluate
# ============================================================

def evaluate(model,
             X_test,
             y_test):
    """
    Evaluate trained Ridge model.

    Returns
    -------
    dict
    """

    y_pred = model.predict(X_test)

    mspe = mean_squared_error(
        y_test,
        y_pred
    )

    mspe_zero = mean_squared_error(

        y_test,

        np.zeros_like(y_test)

    )

    r2_oos = 1 - mspe / mspe_zero

    rmse = np.sqrt(mspe)

    ordinary_r2 = r2_score(
        y_test,
        y_pred
    )

    return {

        "model": model,

        "y_pred": y_pred,

        "R2_OOS": r2_oos,

        "R2": ordinary_r2,

        "MSPE": mspe,

        "RMSE": rmse,

        "alpha": model.alpha_

    }