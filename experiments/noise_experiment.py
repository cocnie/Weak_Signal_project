"""
experiments/noise_experiment.py

Experiment:
Effect of Noise Level

Reference:
Shen & Xiu (2025)
"""

import pandas as pd
import config
from experiments.monte_carlo import MonteCarloSimulator, TRAIN_TEST

class NoiseExperiment:

    def __init__(self,noise_grid=None,repeats=config.MC_REPEATS,models=None):

        if noise_grid is None:
            noise_grid=[(0.5,1.0),(0.7,1.5),(1.0,2.0)]

        if models is None:
            models=[
                "ridge",
                "lasso",
                "elasticnet",
                "random_forest",
                "gbrt",
                "neural_net"
            ]

        self.noise_grid=noise_grid
        self.repeats=repeats
        self.models=models
        self.results=[]

    def run(self):

        original_min=config.NOISE_MIN
        original_max=config.NOISE_MAX

        self.results=[]

        for noise_min,noise_max in self.noise_grid:

            print(f"\nRunning Noise = ({noise_min},{noise_max})")

            config.NOISE_MIN=noise_min
            config.NOISE_MAX=noise_max

            mc=MonteCarloSimulator(
                models=self.models,
                evaluation_mode=TRAIN_TEST,
                repeats=self.repeats
            )

            mc.run()

            summary=mc.summary("R2_OOS")

            summary["Noise_Min"]=noise_min
            summary["Noise_Max"]=noise_max
            summary["Model"]=summary.index

            self.results.append(summary.reset_index(drop=True))

        config.NOISE_MIN=original_min
        config.NOISE_MAX=original_max

        return pd.concat(self.results,ignore_index=True)

    def save(self,df,filename="results/tables/noise_experiment.csv"):

        df.to_csv(filename,index=False)

        return filename