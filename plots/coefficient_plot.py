"""
plots/coefficient_plot.py

Coefficient Visualization

Reference:
Shen & Xiu (2025)

Plots:
    • Coefficient Magnitudes
    • Top Features
    • Non-zero Coefficients
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from config import FIGURE_DPI
from config import FIGURE_WIDTH
from config import FIGURE_HEIGHT
from config import FIGURE_PATH


# ==========================================================
# Default Figure Style
# ==========================================================

plt.rcParams["figure.figsize"] = (
    FIGURE_WIDTH,
    FIGURE_HEIGHT
)

plt.rcParams["figure.dpi"] = FIGURE_DPI

plt.rcParams["axes.grid"] = True

plt.rcParams["grid.alpha"] = 0.30

plt.rcParams["font.size"] = 11


# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs(
    FIGURE_PATH,
    exist_ok=True
)

# ==========================================================
# Plot Coefficients
# ==========================================================

def plot_coefficients(coefficients,feature_names,title="Coefficients",filename="coefficients.png",sort=True):

    coef=np.asarray(coefficients)

    names=np.asarray(feature_names)

    if sort:

        idx=np.argsort(np.abs(coef))[::-1]

        coef=coef[idx]

        names=names[idx]

    plt.figure()

    plt.bar(range(len(coef)),coef)

    plt.xticks(range(len(names)),names,rotation=90)

    plt.xlabel("Features")

    plt.ylabel("Coefficient")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_PATH,filename))

    plt.close()

# ==========================================================
# Plot Top Features
# ==========================================================

def plot_top_features(coefficients,feature_names,top_n=20,title="Top Features",filename="top_features.png"):

    coef=np.asarray(coefficients)

    names=np.asarray(feature_names)

    idx=np.argsort(np.abs(coef))[::-1][:top_n]

    coef=coef[idx]

    names=names[idx]

    plt.figure()

    plt.barh(range(len(coef)),coef)

    plt.yticks(range(len(names)),names)

    plt.gca().invert_yaxis()

    plt.xlabel("Coefficient")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_PATH,filename))

    plt.close()

# ==========================================================
# Plot Non-Zero Coefficients
# ==========================================================

def plot_nonzero(coefficients,feature_names,title="Non-Zero Coefficients",filename="nonzero.png"):

    coef=np.asarray(coefficients)

    names=np.asarray(feature_names)

    mask=coef!=0

    coef=coef[mask]

    names=names[mask]

    if len(coef)==0:
        print("No non-zero coefficients.")
        return

    idx=np.argsort(np.abs(coef))[::-1]

    coef=coef[idx]

    names=names[idx]

    plt.figure()

    plt.barh(range(len(coef)),coef)

    plt.yticks(range(len(names)),names)

    plt.gca().invert_yaxis()

    plt.xlabel("Coefficient")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_PATH,filename))

    plt.close()

# ==========================================================
# Compare Models
# ==========================================================

def compare_models(models,feature_names,title="Coefficient Comparison",filename="compare_models.png"):

    plt.figure()

    for name,coef in models.items():

        coef=np.asarray(coef)

        idx=np.argsort(np.abs(coef))[::-1][:20]

        plt.plot(
            range(20),
            coef[idx],
            marker="o",
            label=name
        )

    plt.xticks(
        range(20),
        np.asarray(feature_names)[idx],
        rotation=90
    )

    plt.xlabel("Features")

    plt.ylabel("Coefficient")

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_PATH,filename))

    plt.close()

