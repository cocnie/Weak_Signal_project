"""
experiments/main_iran.py

Main Experiment

Iran Stock Market

Reference:
Shen & Xiu (2025)
Can Machines Learn Weak Signals?
"""

import os
import time
import pandas as pd

from experiments.iran_market import (
    run_market
)

from plots.performance_plot import (
    compare_performance
)

RESULT_RAW="results/raw"
RESULT_TABLE="results/tables"

os.makedirs(RESULT_RAW,exist_ok=True)
os.makedirs(RESULT_TABLE,exist_ok=True)

# ==========================================================
# Run Iran Market Experiment
# ==========================================================

print('='*70)
print('IRAN MARKET EXPERIMENT')
print('='*70)

start=time.time()

market_results,market_predictions=run_market()

print('='*70)
print('Experiment completed')
print('='*70)

print('Summary shape    :',market_results.shape)
print('Prediction shape :',market_predictions.shape)

summary_file=os.path.join(
    RESULT_TABLE,
    'iran_market_summary.csv'
)

prediction_file=os.path.join(
    RESULT_RAW,
    'iran_market_predictions.csv'
)

market_results.to_csv(summary_file,index=False)

market_predictions.to_csv(prediction_file,index=False)

print('Saved:')
print(summary_file)
print(prediction_file)

# ==========================================================
# Performance Plots
# ==========================================================

compare_performance(
    market_results,
    title='Iran Market Performance',
    filename='iran_market_performance.png'
)

print('Performance figure saved in results/figures')

# ==========================================================
# Final Report
# ==========================================================

elapsed=time.time()-start

print('='*70)
print('IRAN MARKET EXPERIMENT FINISHED')
print('='*70)

print('Execution time: %.2f sec' % elapsed)

print('='*70)
print(market_results.head())
print('='*70)

best_model=market_results.groupby('Model')['R2_OOS'].mean().sort_values(ascending=False).iloc[0]

print('Best mean R2_OOS:',best_model)

print('='*70)

