"""
evaluation/metrics.py

Evaluation Metrics

Reference:
Shen & Xiu (2025)
"""

import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

# ==========================================================
# Mean Squared Prediction Error
# ==========================================================

def mspe(
    y_true,
    y_pred
):
    """
    Mean Squared Prediction Error
    """

    return mean_squared_error(
        y_true,
        y_pred
    )


# ==========================================================
# Root Mean Squared Prediction Error
# ==========================================================

def rmse(
    y_true,
    y_pred
):
    """
    Root Mean Squared Prediction Error
    """

    return np.sqrt(

        mspe(

            y_true,

            y_pred

        )

    )


# ==========================================================
# Ordinary R²
# ==========================================================

def r2(
    y_true,
    y_pred
):
    """
    Ordinary coefficient of determination
    """

    return r2_score(

        y_true,

        y_pred

    )


# ==========================================================
# Out-of-Sample R²
# ==========================================================

def r2_oos(
    y_true,
    y_pred
):
    """
    Out-of-Sample R²

    R²_OOS = 1 - MSPE(model) / MSPE(zero predictor)

    Zero predictor = predicting zero return.
    """

    model_mspe = mspe(

        y_true,

        y_pred

    )

    zero_mspe = np.mean(

        y_true ** 2

    )

    return 1 - model_mspe / zero_mspe


# ==========================================================
# Collect All Metrics
# ==========================================================

def evaluate(
    y_true,
    y_pred
):
    """
    Return all metrics
    """

    return {

        "R2":

            r2(

                y_true,

                y_pred

            ),

        "R2_OOS":

            r2_oos(

                y_true,

                y_pred

            ),

        "MSPE":

            mspe(

                y_true,

                y_pred

            ),

        "RMSE":

            rmse(

                y_true,

                y_pred

            )

    }