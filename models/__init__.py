"""
models/__init__.py

Central Model Loader
"""

from importlib import import_module

AVAILABLE_MODELS = {
    "ridge": "models.ridge",
    "lasso": "models.lasso",
    "elasticnet": "models.elasticnet",
    "random_forest": "models.random_forest",
    "gbrt": "models.gbrt",
    "neural_net": "models.neural_net"
}

def get_model(name):
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return import_module(AVAILABLE_MODELS[name])

def list_models():
    return list(AVAILABLE_MODELS.keys())