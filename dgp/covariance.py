"""
covariance.py

Generate covariance matrices according to Assumption 1
of Shen & Xiu (2025).

Author:
Ali Heidarpoor

Project:
Weak Signals Detection
"""

import numpy as np
import config


def generate_covariance(rng):
    """
    Generate covariance matrices Σ1 and Σ2.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    Sigma1 : ndarray
        Row covariance matrix.

    Sigma1_half : ndarray
        Square root of Sigma1.

    Sigma2 : ndarray
        Column covariance matrix.

    Sigma2_half : ndarray
        Square root of Sigma2.
    """

    

    n = config.N_OBSERVATIONS
    p = config.N_FEATURES

    # ======================================================
    # Generate Sigma1
    # ======================================================

    eigvals_1 = rng.uniform(
        config.SIGMA1_MIN,
        config.SIGMA1_MAX,
        n
    )

    Q1, _ = np.linalg.qr(
        rng.standard_normal((n, n))
    )

    Sigma1 = Q1 @ np.diag(eigvals_1) @ Q1.T

    Sigma1_half = (
        Q1
        @ np.diag(np.sqrt(eigvals_1))
        @ Q1.T
    )

    # ======================================================
    # Generate Sigma2
    # ======================================================

    eigvals_2 = rng.uniform(
        config.SIGMA2_MIN,
        config.SIGMA2_MAX,
        p
    )

    Q2, _ = np.linalg.qr(
        rng.standard_normal((p, p))
    )

    Sigma2 = Q2 @ np.diag(eigvals_2) @ Q2.T

    Sigma2_half = (
        Q2
        @ np.diag(np.sqrt(eigvals_2))
        @ Q2.T
    )

    return (
        Sigma1,
        Sigma1_half,
        Sigma2,
        Sigma2_half
    )