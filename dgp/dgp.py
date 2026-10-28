"""
dgp.py

Generate synthetic dataset according to
Shen & Xiu (2025).

Author:
Ali Heidarpoor

Project:
Weak Signals Detection
"""

import numpy as np

import config

from dgp.covariance import generate_covariance
from dgp.beta_generator import generate_beta
from dgp.noise_generator import generate_noise


def generate_data(seed=None):
    
    """
    Generate one synthetic dataset.

    Parameters
    ----------
    seed : int or None

    Returns
    -------
    X

    y

    beta

    epsilon

    Sigma1

    Sigma2

    active_index
    """

    rng = np.random.default_rng(seed)

    n = config.N_OBSERVATIONS
    p = config.N_FEATURES

    # =====================================================
    # Covariance Matrices
    # =====================================================

    Sigma1, Sigma1_half, Sigma2, Sigma2_half = \
        generate_covariance(rng)

    # =====================================================
    # Beta
    # =====================================================

    beta, b0, active_index = generate_beta(rng)

    # =====================================================
    # Noise
    # =====================================================

    epsilon, Sigma_eps, Sigma_eps_half = \
        generate_noise(rng)

    # =====================================================
    # Generate Z
    # =====================================================

    Z = rng.standard_normal((n, p))

    # =====================================================
    # Generate X
    # =====================================================

    X = Sigma1_half @ Z @ Sigma2_half

    # =====================================================
    # Generate y
    # =====================================================

    y = X @ beta + epsilon

    return {

        "X": X,

        "y": y,

        "beta": beta,

        "b0": b0,

        "epsilon": epsilon,

        "Sigma1": Sigma1,

        "Sigma2": Sigma2,

        "Sigma_eps": Sigma_eps,

        "active_index": active_index

    }