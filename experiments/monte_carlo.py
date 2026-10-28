"""
experiments/monte_carlo.py

Monte Carlo Simulation Framework

Reference:
Shen & Xiu (2025)
"""

import numpy as np
import pandas as pd

import config

from dgp.dgp import generate_data
from models.model_factory import get_model
from evaluation.train_test import run_train_test
from evaluation.rolling import run_rolling


# ==========================================================
# Available Evaluation Modes
# ==========================================================

TRAIN_TEST = "train_test"
ROLLING = "rolling"


# ==========================================================
# Monte Carlo Simulator
# ==========================================================

class MonteCarloSimulator:

    def __init__(
        self,
        models=None,
        evaluation_mode=TRAIN_TEST,
        repeats=config.MC_REPEATS,
        train_size=200,
        step_size=10,
        random_seed=config.RANDOM_SEED
    ):

        if models is None:
            models = [
                "ridge",
                "lasso",
                "elasticnet",
                "random_forest",
                "gbrt",
                "neural_net"
            ]

        self.models = models
        self.evaluation_mode = evaluation_mode
        self.repeats = repeats
        self.train_size = train_size
        self.step_size = step_size
        self.random_seed = random_seed

        self.results = []


# ==========================================================
# Reset Results
# ==========================================================

    def reset(self):
        self.results = []


# ==========================================================
# Generate Dataset
# ==========================================================

    def generate_dataset(self, seed):

        return generate_data(seed=seed)


# ==========================================================
# Get Model
# ==========================================================

    def load_model(self, model_name):

        return get_model(model_name)

# ==========================================================
# Run One Simulation
# ==========================================================

    def run_single(self, seed):

        data = self.generate_dataset(seed)

        X = data["X"]
        y = data["y"]

        simulation_results = {}

        for model_name in self.models:

            model = self.load_model(model_name)

            if self.evaluation_mode == TRAIN_TEST:

                output = run_train_test(
                    model_module=model,
                    X=X,
                    y=y,
                    test_size=config.TEST_SIZE,
                    random_state=seed
                )

            elif self.evaluation_mode == ROLLING:

                output = run_rolling(
                    model_module=model,
                    X=X,
                    y=y,
                    train_size=self.train_size,
                    step_size=self.step_size
                )

            else:

                raise ValueError("Unknown evaluation mode.")

            simulation_results[model_name] = output

        return simulation_results


# ==========================================================
# Run Monte Carlo
# ==========================================================

    def run(self):

        self.reset()

        np.random.seed(self.random_seed)

        seeds = np.random.randint(
            0,
            1000000,
            self.repeats
        )

        for i, seed in enumerate(seeds):

            print(
                f"Simulation {i+1}/{self.repeats}",
                end="\r"
            )

            result = self.run_single(int(seed))

            self.results.append(result)

        print()

        return self.results


# ==========================================================
# Extract Metric
# ==========================================================

    def extract_metric(self, metric_name):

        table = []

        for sim in self.results:

            row = {}

            for model_name in self.models:

                r = sim[model_name]["results"]

                if metric_name in r:
                    row[model_name] = r[metric_name]

                elif metric_name + "_mean" in r:
                    row[model_name] = r[metric_name + "_mean"]

                else:
                    row[model_name] = np.nan

            table.append(row)

        return pd.DataFrame(table)


# ==========================================================
# Summary Statistics
# ==========================================================

    def summary(self, metric_name="R2_OOS"):

        df = self.extract_metric(metric_name)

        summary = pd.DataFrame(index=self.models)

        summary = dict(results["summary"])
        summary["Std"] = df.std()
        summary["Median"] = df.median()
        summary["Min"] = df.min()
        summary["Max"] = df.max()

        return summary


# ==========================================================
# Save Results
# ==========================================================

    def save_results(
        self,
        metric_name="R2_OOS",
        filename=None
    ):

        df = self.extract_metric(metric_name)

        if filename is None:

            filename = f"results/raw/{metric_name}_monte_carlo.csv"

        df.to_csv(filename, index=False)

        return filename


# ==========================================================
# Save Summary
# ==========================================================

    def save_summary(
        self,
        metric_name="R2_OOS",
        filename=None
    ):

        table = self.summary(metric_name)

        if filename is None:

            filename = f"results/tables/{metric_name}_summary.csv"

        table.to_csv(filename)

        return filename


# ==========================================================
# Print Summary
# ==========================================================

    def print_summary(
        self,
        metric_name="R2_OOS"
    ):

        print("="*70)
        print(f"Monte Carlo Summary ({metric_name})")
        print("="*70)
        print(self.summary(metric_name))
        print("="*70)

