"""
preprocessing.py

Data preprocessing utilities.

Author:
Ali Heidarpoor

Project:
Weak Signals Detection
"""

from sklearn.preprocessing import StandardScaler


# ============================================================
# Fit scaler on training data
# ============================================================

def fit_scaler(X_train):
    """
    Fit StandardScaler on training data.

    Parameters
    ----------
    X_train : ndarray

    Returns
    -------
    scaler : StandardScaler
    """

    scaler = StandardScaler()

    scaler.fit(X_train)

    return scaler


# ============================================================
# Transform training data
# ============================================================

def transform_train(scaler, X_train):
    """
    Standardize training data.
    """

    return scaler.transform(X_train)


# ============================================================
# Transform test data
# ============================================================

def transform_test(scaler, X_test):
    """
    Standardize test data.
    """

    return scaler.transform(X_test)


# ============================================================
# One-step preprocessing
# ============================================================

def preprocess(X_train, X_test):
    """
    Fit scaler on train and transform both train/test.

    Returns
    -------
    X_train_std

    X_test_std

    scaler
    """

    scaler = fit_scaler(X_train)

    X_train_std = transform_train(
        scaler,
        X_train
    )

    X_test_std = transform_test(
        scaler,
        X_test
    )

    return X_train_std, X_test_std, scaler