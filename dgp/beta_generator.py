"""
beta_generator.py

Generate sparse coefficient vector β according to
Assumption 3 of Shen & Xiu (2025).

Author:
Ali Heidarpoor

Project:
Weak Signals Detection
"""

import numpy as np
import config


def generate_beta(rng):
    """
    Generate sparse coefficient vector β.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    beta : ndarray
        Final coefficient vector.

    b0 : ndarray
        Sparse normalized coefficient vector.

    active_index : ndarray
        Indices of active (non-zero) signals.
    """

    

    p = config.N_FEATURES
    q = config.Q
    tau = config.TAU

    # -----------------------------------------------------
    # Bernoulli indicator
    # -----------------------------------------------------

    b1 = rng.binomial(
        n=1,
        p=q,
        size=p
    )

    # -----------------------------------------------------
    # Sub-exponential prior
    # (Laplace distribution)
    # -----------------------------------------------------

    b2 = rng.laplace(
        loc=0,
        scale=np.sqrt(config.BETA_VARIANCE),
        size=p
    )

    # -----------------------------------------------------
    # Sparse normalized vector
    # -----------------------------------------------------

    b0 = (b1 * b2) / np.sqrt(q)

    # -----------------------------------------------------
    # Final coefficient vector
    # -----------------------------------------------------

    beta = np.sqrt(tau / p) * b0

    active_index = np.where(b1 == 1)[0]

    return beta, b0, active_index