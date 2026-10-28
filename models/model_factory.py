"""
models/model_factory.py

Central Model Factory

This file provides a unified interface for all models
used in the Weak Signals project.

Reference:
Shen & Xiu (2025)
"""

from models import ridge
from models import lasso
from models import elasticnet
from models import random_forest
from models import gbrt
from models import neural_net


# ==========================================================
# Dictionary of all available models
# ==========================================================

MODELS = {

    "ridge": ridge,

    "lasso": lasso,

    "elasticnet": elasticnet,

    "random_forest": random_forest,

    "gbrt": gbrt,

    "neural_net": neural_net

}


# ==========================================================
# Return module
# ==========================================================

def get_model(model_name):
    """
    Return the selected model module.

    Example
    -------
    model = get_model("ridge")
    """

    if model_name not in MODELS:

        raise ValueError(

            f"Unknown model: {model_name}"

        )

    return MODELS[model_name]


# ==========================================================
# Return list of available models
# ==========================================================

def available_models():

    return list(MODELS.keys())