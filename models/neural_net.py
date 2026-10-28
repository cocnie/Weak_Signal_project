"""
models/neural_net.py

Feed Forward Neural Network (MLP)

Reference:
Shen & Xiu (2025)
"""

import numpy as np

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV

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
    Train Feed Forward Neural Network
    """

    base_model = MLPRegressor(

        random_state=config.RANDOM_SEED,

        max_iter=2000

    )

    param_grid = {

        "hidden_layer_sizes": [

            (20,),
            (50,),
            (100,),

            (50,50)

        ],

        "alpha": [

            1e-4,
            1e-3,
            1e-2

        ],

        "learning_rate_init": [

            0.001,
            0.01

        ]

    }

    grid = GridSearchCV(

        estimator=base_model,

        param_grid=param_grid,

        cv=5,

        scoring="r2",

        n_jobs=-1

    )

    grid.fit(

        X_train,

        y_train

    )

    return grid.best_estimator_


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

        "hidden_layer_sizes":

            model.hidden_layer_sizes,

        "alpha":

            model.alpha,

        "learning_rate_init":

            model.learning_rate_init,

        "iterations":

            model.n_iter_,

        "loss":

            model.loss_,

        "R2":

            r2,

        "R2_OOS":

            r2_oos,

        "MSPE":

            mspe,

        "RMSE":

            rmse,

        "y_pred":

            y_pred

    }

    return results