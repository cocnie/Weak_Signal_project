"""
experiments/main_iran_parallel.py

Parallel Iran Market Experiment

Run multiple tickers simultaneously.

Reference:
Shen & Xiu (2025)
Can Machines Learn Weak Signals?
"""

import os,re,time
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from contextlib import contextmanager
import joblib
from plots.performance_plot import compare_performance
from experiments.iran_market import run_symbol

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"



DATA_FOLDER = 'data/CSV'
RESULT_RAW = "results/raw"
RESULT_TABLE = "results/tables"

RESULT_PER_STOCK = "results_per_stock"
RESULT_PER_STOCK_RAW = os.path.join(RESULT_PER_STOCK, "raw")
RESULT_PER_STOCK_TABLE = os.path.join(RESULT_PER_STOCK, "tables")
RESULT_PER_STOCK_FIGURE = os.path.join(RESULT_PER_STOCK, "figures")

os.makedirs(RESULT_PER_STOCK_FIGURE, exist_ok=True)
os.makedirs(RESULT_PER_STOCK_RAW, exist_ok=True)
os.makedirs(RESULT_PER_STOCK_TABLE, exist_ok=True)

os.makedirs(RESULT_RAW, exist_ok=True)
os.makedirs(RESULT_TABLE, exist_ok=True)

def get_max_files(total_files):
    while True:
        user_input = input(
            f"Maximum number of CSV files to process "
            f"(1 to {total_files}; press Enter for all files): "
        ).strip()

        if user_input == "":
            return None

        try:
            max_files = int(user_input)

            if 1 <= max_files <= total_files:
                return max_files

            print(f"Please enter an integer between 1 and {total_files}.")

        except ValueError:
            print("Invalid input. Enter a positive integer or press Enter.")




# Number of CPU workers
N_JOBS = 4   # use all available cores
# ==========================================================
# Safe wrapper
# ==========================================================

def safe_run_symbol(file_path):

    try:
        output = run_symbol(file_path)

        if output is None:
            return None

        result_df, prediction_df = output

        return file_path, result_df, prediction_df

    except Exception as e:
        print(f'SKIPPED {os.path.basename(file_path)} : {e}')
        return None
# ==========================================================
# Collect CSV Files
# ==========================================================
csv_files = sorted([
    os.path.join(DATA_FOLDER, f)
    for f in os.listdir(DATA_FOLDER)
    if f.lower().endswith(".csv")
])

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {DATA_FOLDER}")

MAX_FILES = get_max_files(len(csv_files))

if MAX_FILES is not None:
    csv_files = csv_files[:MAX_FILES]

print('=' * 70)
print('IRAN MARKET PARALLEL EXPERIMENT')
print('=' * 70)
print('Files found :', len(csv_files))
print('Parallel jobs:', N_JOBS)
print('=' * 70)

start = time.time()

# ==========================================================
# Parallel Execution with tqdm
# ==========================================================

@contextmanager
def tqdm_joblib(tqdm_object):

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):

        def __call__(self,*args,**kwargs):

            tqdm_object.update(n=self.batch_size)

            return super().__call__(*args,**kwargs)

    old_callback=joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack=TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack=old_callback
        tqdm_object.close()

with tqdm_joblib(tqdm(total=len(csv_files),desc='Tickers')):

    outputs=Parallel(n_jobs=N_JOBS,backend="threading")(
        delayed(safe_run_symbol)(file_path)
        for file_path in csv_files
    )

print('=' * 70)
print('Parallel execution finished')
print('=' * 70)

# ==========================================================
# Collect Outputs
# ==========================================================
outputs = [o for o in outputs if o is not None]

result_frames = []
prediction_frames = []

for file_path, result_df, prediction_df in outputs:

    # نام CSV اصلی، بدون پسوند؛ برای نام فایل ویندوز ایمن می‌شود
    stock_name = os.path.splitext(os.path.basename(file_path))[0]
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", stock_name)

    # خروجی summary مربوط به همان سهم و هر سه مدل
    result_df.to_csv(
        os.path.join(
            RESULT_PER_STOCK_TABLE,
            f"{safe_name}_summary.csv"
        ),
        index=False
    )

    # همه پیش‌بینی‌های خارج از نمونهٔ همان سهم و هر سه مدل
    prediction_df.to_csv(
        os.path.join(
            RESULT_PER_STOCK_RAW,
            f"{safe_name}_predictions.csv"
        ),
        index=False
    )

    compare_performance(
    result_df,
    title=f"{stock_name} - Model Performance",
    filename=f"{safe_name}_performance.png",
    output_dir=RESULT_PER_STOCK_FIGURE)

    # سپس همان خروجی‌ها برای فایل‌های تجمیعی نگه داشته می‌شوند
    if result_df is not None and len(result_df) > 0:
        result_frames.append(result_df)

    if prediction_df is not None and len(prediction_df) > 0:
        prediction_frames.append(prediction_df)

##===================================
if len(result_frames) == 0:
    raise RuntimeError("No successful ticker results.")

market_results = pd.concat(
    result_frames,
    ignore_index=True
)

market_predictions = pd.concat(
    prediction_frames,
    ignore_index=True
)

print("=" * 70)
print("Successful tickers :", market_results["Ticker"].nunique())
print("Result rows        :", len(market_results))
print("Prediction rows    :", len(market_predictions))
print("=" * 70)
# ==========================================================
# Save Results
# ==========================================================

summary_file=os.path.join(
    RESULT_TABLE,
    'iran_market_parallel_summary.csv'
)

prediction_file=os.path.join(
    RESULT_RAW,
    'iran_market_parallel_predictions.csv'
)

market_results.to_csv(summary_file,index=False)
market_predictions.to_csv(prediction_file,index=False)

print('Saved:')
print(summary_file)
print(prediction_file)

# ==========================================================
# Performance Plot
# ==========================================================

compare_performance(
    market_results,
    title='Iran Market Parallel Performance',
    filename='iran_market_parallel_performance.png'
)

print('Performance figure saved in results/figures')

# ==========================================================
# Final Report
# ==========================================================

elapsed=time.time()-start

print('='*70)
print('IRAN MARKET PARALLEL EXPERIMENT FINISHED')
print('='*70)

print('Execution time: %.2f sec' % elapsed)
print('Execution time: %.2f min' % (elapsed/60))

print('='*70)
print('Mean Performance by Model')
print('='*70)

mean_table=market_results.groupby('Model')[
    ['R2','R2_OOS','MSPE','RMSE']
].mean().sort_values('R2_OOS',ascending=False)

print(mean_table)

print('='*70)

best_model=mean_table.index[0]

best_r2oos=mean_table.iloc[0]['R2_OOS']

print('Best model:',best_model)
print('Best mean R2_OOS: %.6f' % best_r2oos)

print('='*70)

ticker_count=market_results['Ticker'].nunique()

print('Tickers processed:',ticker_count)

print('='*70)