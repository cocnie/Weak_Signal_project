"""
main_fred.py

Main Experiment

FRED-MD

Reference:
Shen & Xiu (2025)
Can Machines Learn Weak Signals?
"""

import os
import time
import pandas as pd

from experiments.fred_md import (
    prepare_dataset
)

from utils.fred_transform import (
    apply_transformations
)

from features.macro_features import (
    build_features
)

from models import get_model

from evaluation.rolling import (
    run_rolling,
    EXPANDING
)

RESULT_RAW = "results/raw"
RESULT_TABLE = "results/tables"

os.makedirs(RESULT_RAW,exist_ok=True)
os.makedirs(RESULT_TABLE,exist_ok=True)

# ==========================================================
# Load Data
# ==========================================================

print('='*70)
print('FRED-MD EXPERIMENT')
print('='*70)

start=time.time()

df,variables,tcodes=prepare_dataset()

print('Raw shape:',df.shape)

# ==========================================================
# Apply Transformations
# ==========================================================

df=apply_transformations(df,tcodes)

print('Transformed shape:',df.shape)

# ==========================================================
# Build Features
# ==========================================================

X,y=build_features(df)

print('Feature matrix:',X.shape)
print('Target vector :',y.shape)

# Optional quick run
X=X.iloc[:300].reset_index(drop=True)
y=y.iloc[:300].reset_index(drop=True)

print('Test subset   :',X.shape)

# ==========================================================
# Models
# ==========================================================

MODELS=[
    'ridge',
    'lasso',
    'elasticnet'
]

all_results=[]
all_predictions=[]

# ==========================================================
# Run Models
# ==========================================================

for model_name in MODELS:

    print('='*70)
    print('Running',model_name)
    print('='*70)

    model=get_model(model_name)

    results=run_rolling(
        model_module=model,
        X=X,
        y=y,
        train_size=200,
        step_size=1,
        mode=EXPANDING,
        progress=True
    )

    summary=dict(results['summary'])

    summary['Model']=model_name
    summary['N']=len(X)

    all_results.append(summary)

    pred=results['results'].copy()

    pred['Model']=model_name

    all_predictions.append(pred)

    print(summary)

# ==========================================================
# Save Results
# ==========================================================

result_df=pd.DataFrame(all_results)

prediction_df=pd.concat(
    all_predictions,
    ignore_index=True
)

summary_file=os.path.join(
    RESULT_TABLE,
    'fred_summary.csv'
)

prediction_file=os.path.join(
    RESULT_RAW,
    'fred_predictions.csv'
)

result_df.to_csv(summary_file,index=False)

prediction_df.to_csv(prediction_file,index=False)

print('='*70)
print('Saved:')
print(summary_file)
print(prediction_file)

# ==========================================================
# Plots
# ==========================================================

from plots.forecast_plot import (
    plot_forecast,
    plot_all_models
)

from plots.performance_plot import (
    compare_performance
)

plot_all_models(
    prediction_df,
    title='FRED-MD Forecast Comparison',
    filename='fred_forecast_comparison.png'
)

compare_performance(
    result_df,
    title='FRED-MD Performance',
    filename='fred_performance.png'
)

print('Figures saved in results/figures')

# ==========================================================
# Final Report
# ==========================================================

elapsed=time.time()-start

print('='*70)
print('FRED-MD EXPERIMENT FINISHED')
print('='*70)

print('Execution time: %.2f sec' % elapsed)

print('='*70)
print(result_df)
print('='*70)

best_model=result_df.sort_values(
    'R2_OOS',
    ascending=False
).iloc[0]

print('Best model based on R2_OOS:')
print(best_model)

print('='*70)

