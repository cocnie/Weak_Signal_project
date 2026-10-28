"""
models/elasticnet.py

ElasticNet Regression Model

Reference:
Shen & Xiu (2025)
"""

import numpy as np

from sklearn.linear_model import ElasticNetCV

from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

import config


# ==========================================================
# Train
# ==========================================================

def train(
    X_train,
    y_train
):
    """
    Train ElasticNet using Cross Validation
    """

    alphas = (
        config.LAMBDA_GRID /
        np.sqrt(config.N_OBSERVATIONS)
    )

    model = ElasticNetCV(

        alphas=alphas,

        l1_ratio=[0.2, 0.5, 0.8],

        cv=config.CV_LASSO,

        fit_intercept=False,

        max_iter=200_000,
        tol = 1e-4, 
        selection = "random",

        random_state=config.RANDOM_SEED

    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ==========================================================
# Evaluation
# ==========================================================

def evaluate(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    mspe = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mspe)

    r2 = r2_score(
        y_test,
        y_pred
    )

    mspe_zero = np.mean(
        y_test ** 2
    )

    r2_oos = 1 - mspe / mspe_zero

    results = {

        "alpha":
            model.alpha_,

        "l1_ratio":
            model.l1_ratio_,

        "R2":
            r2,

        "R2_OOS":
            r2_oos,

        "MSPE":
            mspe,

        "RMSE":
            rmse,

        "NonZero":
            np.sum(model.coef_ != 0),

        "Coefficients":
            model.coef_,

        "y_pred":
            y_pred

    }

    return results