"""
config.py

Configuration file for the Weak Signals Detection Project

Reference:
Shen, Z. & Xiu, D. (2025)
Can Machines Learn Weak Signals?
"""

import numpy as np

# ==========================================================
# Project Information
# ==========================================================

PROJECT_NAME = "Weak Signals Detection"

REFERENCE = "Shen & Xiu (2025)"

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# Data Dimensions
# ==========================================================

# Number of observations (n)

N_OBSERVATIONS = 500

# Number of predictors (p)

N_FEATURES = 100

# ==========================================================
# Signal Parameters
# ==========================================================

# Sparsity level (Assumption 3)

Q = 0.05

# Signal strength

TAU = 0.10

# Variance of sub-exponential prior

BETA_VARIANCE = 0.30

# ==========================================================
# Row Covariance Matrix Σ1
# ==========================================================

SIGMA1_MIN = 0.5
SIGMA1_MAX = 2.0

# ==========================================================
# Column Covariance Matrix Σ2
# ==========================================================

SIGMA2_MIN = 1.0
SIGMA2_MAX = 3.0

# ==========================================================
# Noise Covariance Matrix
# ==========================================================

NOISE_MIN = 0.7
NOISE_MAX = 1.5

# ==========================================================
# Train/Test Split
# ==========================================================

TEST_SIZE = 0.30

TRAIN_SIZE = 0.70

# ==========================================================
# Cross Validation
# ==========================================================

CV_RIDGE = 6

CV_LASSO = 6

# ==========================================================
# Lambda Grid
# ==========================================================

LAMBDA_GRID = np.logspace(-4, 4, 50)

# ==========================================================
# Monte Carlo Simulation
# ==========================================================

MC_REPEATS = 100

# ==========================================================
# Experimental Grids
# ==========================================================

TAU_GRID = [
    0.01,
    0.02,
    0.05,
    0.10
]

Q_GRID = [
    0.01,
    0.03,
    0.05,
    0.10
]

# ==========================================================
# Future Noise Scenarios
# ==========================================================

NOISE_LEVELS = [
    "Low",
    "Medium",
    "High"
]

# CV params.

RIDGE_CV = 6

# ==========================================================
# Figure Settings
# ==========================================================

FIGURE_PATH = 'results/figures'

FIGURE_DPI = 150

FIGURE_WIDTH = 10

FIGURE_HEIGHT = 6


IRAN_MAX_OBSERVATIONS = 750   # برای استفاده از کل داده: None
IRAN_TRAIN_SIZE = 300
IRAN_STEP_SIZE = 20
IRAN_WINDOW_MODE = "expanding"  # یا "rolling"

ROLLING = "rolling"
EXPANDING = "expanding"

# ==========================================================
# Adjusment setting
# ==========================================================

USE_BACKWARD_ADJUSTMENT = True
ADJUSTMENT_THRESHOLD = 0.20