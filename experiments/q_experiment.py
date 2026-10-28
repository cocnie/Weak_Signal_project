"""
experiments/q_experiment.py

Experiment:
Effect of Sparsity Level (Q)

Reference:
Shen & Xiu (2025)
"""

import pandas as pd
import config
from experiments.monte_carlo import MonteCarloSimulator, TRAIN_TEST

class QExperiment:

    def __init__(self,q_grid=None,repeats=config.MC_REPEATS,models=None):

        if q_grid is None:
            q_grid=config.Q_GRID

        if models is None:
            models=[
                "ridge",
                "lasso",
                "elasticnet",
                "random_forest",
                "gbrt",
                "neural_net"
            ]

        self.q_grid=q_grid
        self.repeats=repeats
        self.models=models
        self.results=[]

    def run(self):

        original_q=config.Q
        self.results=[]

        for q in self.q_grid:

            print(f"\nRunning Q = {q}")

            config.Q=q

            mc=MonteCarloSimulator(
                models=self.models,
                evaluation_mode=TRAIN_TEST,
                repeats=self.repeats
            )

            mc.run()

            summary=mc.summary("R2_OOS")

            summary["Q"]=q
            summary["Model"]=summary.index

            self.results.append(summary.reset_index(drop=True))

        config.Q=original_q

        return pd.concat(self.results,ignore_index=True)

    def save(self,df,filename="results/tables/q_experiment.csv"):

        df.to_csv(filename,index=False)

        return filename