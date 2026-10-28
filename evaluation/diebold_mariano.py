"""
evaluation/diebold_mariano.py

Diebold-Mariano Test

Reference
---------
Diebold & Mariano (1995)

Shen & Xiu (2025)
"""

import numpy as np

from scipy.stats import norm


# ==========================================================
# Forecast Errors
# ==========================================================

def forecast_errors(
    y_true,
    y_pred
):
    """
    Forecast errors

    e = y - yhat
    """

    y_true = np.asarray(y_true)

    y_pred = np.asarray(y_pred)

    return y_true - y_pred


# ==========================================================
# Squared Loss
# ==========================================================

def squared_loss(errors):

    return errors ** 2


# ==========================================================
# Absolute Loss
# ==========================================================

def absolute_loss(errors):

    return np.abs(errors)


# ==========================================================
# Loss Differential
# ==========================================================

def loss_differential(
    loss_1,
    loss_2
):
    """
    d_t

    Positive :
        model 2 better

    Negative :
        model 1 better
    """

    return loss_1 - loss_2

# ==========================================================
# Long-run Variance (Newey-West)
# ==========================================================

def long_run_variance(
    d,
    horizon=1
):
    """
    HAC variance estimator.

    Parameters
    ----------
    d : loss differential

    horizon : forecast horizon

    Returns
    -------
    Long-run variance
    """

    d = np.asarray(d)

    T = len(d)

    gamma0 = np.var(
        d,
        ddof=1
    )

    lrv = gamma0

    for lag in range(1, horizon):

        cov = np.cov(
            d[:-lag],
            d[lag:],
            ddof=1
        )[0, 1]

        weight = 1 - lag / horizon

        lrv += 2 * weight * cov

    return lrv


# ==========================================================
# Diebold-Mariano Statistic
# ==========================================================

def dm_test(
    y_true,
    y_pred_1,
    y_pred_2,
    loss="mse",
    horizon=1
):
    """
    Diebold-Mariano Test

    Parameters
    ----------
    y_true

    y_pred_1

    y_pred_2

    loss : "mse" or "mae"

    horizon : forecast horizon

    Returns
    -------
    dict
    """

    e1 = forecast_errors(
        y_true,
        y_pred_1
    )

    e2 = forecast_errors(
        y_true,
        y_pred_2
    )

    if loss.lower() == "mse":

        l1 = squared_loss(e1)
        l2 = squared_loss(e2)

    elif loss.lower() == "mae":

        l1 = absolute_loss(e1)
        l2 = absolute_loss(e2)

    else:

        raise ValueError(
            "loss must be mse or mae."
        )

    d = loss_differential(
        l1,
        l2
    )

    T = len(d)

    d_bar = np.mean(d)

    lrv = long_run_variance(
        d,
        horizon
    )

    dm = d_bar / np.sqrt(
        lrv / T
    )

    p_value = 2 * (
        1 - norm.cdf(
            np.abs(dm)
        )
    )

    return {

        "DM": dm,

        "p_value": p_value,

        "mean_loss_diff": d_bar,

        "long_run_variance": lrv,

        "n": T

    }
# ==========================================================
# Diebold-Mariano Test
# ==========================================================

def diebold_mariano(
    y_true,
    pred_1,
    pred_2,
    loss="mse"
):

    y_true = np.asarray(y_true)

    pred_1 = np.asarray(pred_1)

    pred_2 = np.asarray(pred_2)

    if loss == "mse":

        e1 = (y_true - pred_1) ** 2

        e2 = (y_true - pred_2) ** 2

    elif loss == "mae":

        e1 = np.abs(y_true - pred_1)

        e2 = np.abs(y_true - pred_2)

    else:

        raise ValueError("Unknown loss.")

    d = e1 - e2

    dm = dm_statistic(d)

    p = dm_pvalue(dm)

    return {

        "DM": dm,

        "PValue": p,

        "MeanLoss1": np.mean(e1),

        "MeanLoss2": np.mean(e2),

        "MeanDifference": np.mean(d)

    }
