"""
noise_generator.py

Generate noise vector ε according to
Assumption 2 of Shen & Xiu (2025).

Author:
Ali Heidarpoor

Project:
Weak Signals Detection
"""

import numpy as np
import config


def generate_noise(rng):
    """
    Generate noise vector ε.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    epsilon : ndarray
        Noise vector.

    Sigma_eps : ndarray
        Noise covariance matrix.

    Sigma_eps_half : ndarray
        Square root of Sigma_eps.
    """

    

    n = config.N_OBSERVATIONS

    # ======================================================
    # Generate noise covariance matrix
    # ======================================================

    eigvals_eps = rng.uniform(
        config.NOISE_MIN,
        config.NOISE_MAX,
        n
    )

    Qe, _ = np.linalg.qr(
        rng.standard_normal((n, n))
    )

    Sigma_eps = (
        Qe
        @ np.diag(eigvals_eps)
        @ Qe.T
    )

    Sigma_eps_half = (
        Qe
        @ np.diag(np.sqrt(eigvals_eps))
        @ Qe.T
    )

    # ======================================================
    # Generate iid Gaussian vector
    # ======================================================

    z = rng.standard_normal(n)

    # ======================================================
    # Generate epsilon
    # ======================================================

    epsilon = Sigma_eps_half @ z

    return (
        epsilon,
        Sigma_eps,
        Sigma_eps_half
    )