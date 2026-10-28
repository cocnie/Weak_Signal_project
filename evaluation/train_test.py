"""
evaluation/train_test.py

Unified Train/Test Pipeline
"""

from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocess
from evaluation.metrics import evaluate

def run_train_test(model_module, X, y, test_size, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train, X_test, scaler = preprocess(X_train, X_test)

    model = model_module.train(X_train, y_train)

    results = model_module.evaluate(model, X_test, y_test)

    metrics = evaluate(y_test, results["y_pred"])

    results.update(metrics)

    return {
        "model": model,
        "scaler": scaler,
        "results": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }