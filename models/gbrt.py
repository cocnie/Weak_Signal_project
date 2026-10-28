"""
models/gbrt.py

Gradient Boosting Regression

Reference:
Shen & Xiu (2025)
"""

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

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
    Train Gradient Boosting Regression
    """

    base_model = GradientBoostingRegressor(

        random_state=config.RANDOM_SEED

    )

    param_grid = {

        "n_estimators": [100, 200],

        "learning_rate": [0.01, 0.05, 0.10],

        "max_depth": [2, 3, 4],

        "min_samples_leaf": [1, 3, 5]

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

        "n_estimators":

            model.n_estimators,

        "learning_rate":

            model.learning_rate,

        "max_depth":

            model.max_depth,

        "min_samples_leaf":

            model.min_samples_leaf,

        "R2":

            r2,

        "R2_OOS":

            r2_oos,

        "MSPE":

            mspe,

        "RMSE":

            rmse,

        "FeatureImportance":

            model.feature_importances_,

        "y_pred":

            y_pred

    }

    return results