"""
experiments/tau_experiment.py

Experiment:
Effect of Signal Strength (Tau)

Reference:
Shen & Xiu (2025)
"""

import pandas as pd
import config
from experiments.monte_carlo import MonteCarloSimulator, TRAIN_TEST


class TauExperiment:

    def __init__(self,
                 tau_grid=None,
                 repeats=config.MC_REPEATS,
                 models=None):

        if tau_grid is None:
            tau_grid = config.TAU_GRID

        if models is None:
            models = [
                "ridge",
                "lasso",
                "elasticnet",
                "random_forest",
                "gbrt",
                "neural_net"
            ]

        self.tau_grid = tau_grid
        self.repeats = repeats
        self.models = models
        self.results = []

    def run(self):

        original_tau = config.TAU

        self.results = []

        for tau in self.tau_grid:

            print(f"\nRunning Tau = {tau}")

            config.TAU = tau

            mc = MonteCarloSimulator(
                models=self.models,
                evaluation_mode=TRAIN_TEST,
                repeats=self.repeats
            )

            mc.run()

            summary = mc.summary("R2_OOS")

            summary["Tau"] = tau
            summary["Model"] = summary.index

            self.results.append(summary.reset_index(drop=True))

        config.TAU = original_tau

        return pd.concat(self.results, ignore_index=True)

    def save(self,
             df,
             filename="results/tables/tau_experiment.csv"):

        df.to_csv(filename, index=False)

        return filename