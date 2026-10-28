"""
experiments/iran_market.py

Real Iran Stock Market Experiment

Reference:
Shen & Xiu (2025)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import config
from config import RANDOM_SEED,IRAN_STEP_SIZE,IRAN_TRAIN_SIZE,IRAN_MAX_OBSERVATIONS,EXPANDING ,ROLLING,USE_BACKWARD_ADJUSTMENT,ADJUSTMENT_THRESHOLD

from models import get_model
from evaluation.rolling import run_rolling

from utils.detect_price_breaks import detect_price_breaks
from utils.apply_backward_adjustment import apply_backward_adjustment

np.random.seed(RANDOM_SEED)

DATA_PATH = "data/CSV"
RESULT_PATH = "results"


MODELS = ["ridge","lasso","elasticnet"]
# ==========================================================
# Adjustment functions 
# ==========================================================

def maybe_adjust_prices(df, threshold=0.20):

    events, _ = detect_price_breaks(df,threshold=threshold)
    
    n_events = len(events)

    if n_events == 0:
        return df, n_events

    df_adj = apply_backward_adjustment(df, events)

    replace_map = {
        "ADJ_<OPEN>": "<OPEN>",
        "ADJ_<FIRST>": "<FIRST>",
        "ADJ_<HIGH>": "<HIGH>",
        "ADJ_<LOW>": "<LOW>",
        "ADJ_<CLOSE>": "<CLOSE>",
        "ADJ_<LAST>": "<LAST>",}

    for new_col, old_col in replace_map.items():
        df_adj[old_col] = df_adj[new_col]

    return df_adj,n_events
# ==========================================================
# Read CSV Files
# ==========================================================

def load_market_file(file_path):

    print("="*70)
    print("Reading File:")
    print(file_path)

    df = pd.read_csv(file_path)
    # مرتب سازی زمانی
    date_col = '<DTYYYYMMDD>'

    if date_col in df.columns:

        df[date_col] = pd.to_datetime(df[date_col].astype(str),format='%Y%m%d')

        df = df.sort_values(date_col).reset_index(drop=True)

        print('First date :', df[date_col].iloc[0])
        print('Last date  :', df[date_col].iloc[-1])

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst Row:")
    print(df.iloc[0])

    print("\nDataFrame Shape:")
    print(df.shape)

    if "<DTYYYYMMDD>" not in df.columns:
        raise Exception("Column <DTYYYYMMDD> NOT FOUND")

    df = df.sort_values("<DTYYYYMMDD>").reset_index(drop=True)

    numeric_cols = [
        "<FIRST>",
        "<OPEN>",
        "<HIGH>",
        "<LOW>",
        "<CLOSE>",
        "<LAST>",
        "<VOL>",
        "<VALUE>"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    return df

def load_all_market_files():

    files = sorted(glob.glob(os.path.join(DATA_PATH, "*.csv")))

    if len(files) == 0:
        raise FileNotFoundError("No CSV files found inside data/CSV")

    return files

# ==========================================================
# Feature Engineering
# ==========================================================
def build_features(df):
    price_cols = ["<FIRST>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<LAST>"]
    df = df.sort_values("<DTYYYYMMDD>").reset_index(drop=True)
    df = df.loc[(df["<VOL>"] > 0) &(df["<VALUE>"] > 0)].copy()
    df = df.loc[df["<OPENINT>"] >= 5].copy() # Remove the QUEUE days.
    df = df.loc[(df[price_cols] > 0).all(axis=1)].copy()
    df = df.reset_index(drop=True)

# ==========================================================
# BACKWARD PRICE ADJUSTMENT
# ==========================================================

    n_events = 0

    if config.USE_BACKWARD_ADJUSTMENT:

        events, _ = detect_price_breaks(
            df,
            threshold=config.ADJUSTMENT_THRESHOLD
        )

        n_events = len(events)

        print(
            f"build_features | threshold={config.ADJUSTMENT_THRESHOLD}|n_events{n_events}")

        df, n_events = maybe_adjust_prices(
            df,
            threshold=config.ADJUSTMENT_THRESHOLD
        )

    else:
        print("build_features | adjustment OFF")
    
    first  = df["<FIRST>"]
    high   = df["<HIGH>"]
    low    = df["<LOW>"]
    close  = df["<CLOSE>"]
    last   = df["<LAST>"]
    prev   = df["<OPEN>"]      # yesterday official close
    volume = df["<VOL>"]
    value  = df["<VALUE>"]
    trades = df["<OPENINT>"]

    X = pd.DataFrame(index=df.index)
    #LOCKED days 
    X["is_locked"] = (high == low).astype(int)
    # returns
    X["ret_1"] = np.log(last).diff()
    X["gap"] = np.log(first / prev)
    X["intraday_ret"] = np.log(close / first)

    # candle structure
    X["range"] = np.log(high / low)
    X["last_close"] = np.log(last / close)
    X["close_strength"] = np.where((high-low) > 0,(close-low)/(high-low),0.5)

    # liquidity
    X["log_volume"] = np.log1p(volume)
    X["log_value"] = np.log1p(value)
    X["avg_trade_price"] = np.log1p(value / (volume + 1))

    # market microstructure
    X["trades_change"] = np.log1p(trades).diff()
    X["avg_trade_size"] = np.log1p(volume / (trades + 1))

    # dynamics
    X["volatility_5"] = X["ret_1"].rolling(5).std()
    X["ma5_ratio"] = last / last.rolling(5).mean()
    X["ma10_ratio"] = last / last.rolling(10).mean()
    X["momentum_5"] = np.log(last).diff(5)
    

    # target: official next-day return
    y = np.log(last.shift(-1)) - np.log(last)

    feature_dates = df["<DTYYYYMMDD>"]
    target_dates = df["<DTYYYYMMDD>"].shift(-1)

    data = pd.concat(
        [
            X,
            y.rename("target"),
            feature_dates.rename("Feature_Date"),
            target_dates.rename("Target_Date"),
        ],
        axis=1
    ).dropna()


    #print("FEATURE CHECK")
    #print(X.describe().T)

    #print("TARGET CHECK")
    #print(y.describe())

    #print("MAX ABS FEATURE")
    #print(X.abs().max())

    
    return (
    data.drop(columns=["target", "Feature_Date", "Target_Date"]),
    data["target"],
    data["Feature_Date"],
    data["Target_Date"],
    n_events)
# ==========================================================
# Run One Symbol
# ==========================================================

def run_symbol(file_path):

    df = load_market_file(file_path)

    X, y, feature_dates, target_dates = build_features(df)
    
    X = X.iloc[:IRAN_MAX_OBSERVATIONS].reset_index(drop=True)
    y = y.iloc[:IRAN_MAX_OBSERVATIONS].reset_index(drop=True)
    
    feature_dates = feature_dates.iloc[:IRAN_MAX_OBSERVATIONS].reset_index(drop=True)
    target_dates = target_dates.iloc[:IRAN_MAX_OBSERVATIONS].reset_index(drop=True)

    train_size = IRAN_TRAIN_SIZE
    step_size=IRAN_STEP_SIZE
    if len(X) <= train_size + step_size:
        print(f'Skipped {file_path}: insufficient usable samples ({len(X)})')
        return None

    symbol = df["<TICKER>"].iloc[0]

    all_results = []

    all_predictions = []

    for model_name in MODELS:

        print(f"Running {symbol} -> {model_name}")

        model = get_model(model_name)

        results = run_rolling(
            model_module=model,
            X=X,
            y=y,
            train_size = IRAN_TRAIN_SIZE,
            step_size=IRAN_STEP_SIZE,
            mode=ROLLING
        )

        summary = dict(results["summary"])

        summary["Model"] = model_name
        summary["Ticker"] = symbol
        summary["N"] = len(X)

        all_results.append(summary)

        pred = results["results"].copy()

        n_predictions = len(pred)
        first_test_index = train_size

        pred["Feature_Date"] = feature_dates.iloc[
            first_test_index:first_test_index + n_predictions].to_numpy()

        pred["Target_Date"] = target_dates.iloc[
            first_test_index:first_test_index + n_predictions].to_numpy()



        pred["Model"] = model_name
        pred["Ticker"] = symbol

        all_predictions.append(pred)

    result_df = pd.DataFrame(all_results)

    prediction_df = pd.concat(all_predictions, ignore_index=True)

    return result_df, prediction_df

#==========================================================================

#BOTH MODE : Rolling and Expanding

#==========================================================================
def run_symbol_both_mode(file_path):

    # 1) Read and prepare one stock's data
    df = load_market_file(file_path)

    X, y, feature_dates, target_dates = build_features(df)


    X = X.iloc[:IRAN_MAX_OBSERVATIONS].reset_index(drop=True)
    y = y.iloc[:IRAN_MAX_OBSERVATIONS].reset_index(drop=True)

    feature_dates = feature_dates.iloc[:IRAN_MAX_OBSERVATIONS].reset_index(drop=True)
    target_dates = target_dates.iloc[:IRAN_MAX_OBSERVATIONS].reset_index(drop=True)

    train_size = IRAN_TRAIN_SIZE
    step_size = IRAN_STEP_SIZE

    if len(X) <= train_size + step_size:
        print(
            f"Skipped {file_path}: "
            f"insufficient usable samples ({len(X)})"
        )
        return None

    symbol = df["<TICKER>"].iloc[0]

    all_results = []
    all_predictions = []

    # 2) Run both window modes
    modes = {
        "Expanding": EXPANDING,
        "Rolling": ROLLING
    }

    for mode_name, mode_value in modes.items():

        print("=" * 70)
        print(f"Running {symbol} | Window mode: {mode_name}")
        print("=" * 70)

        for model_name in MODELS:

            print(f"Running {symbol} -> {model_name} -> {mode_name}")

            model = get_model(model_name)

            results = run_rolling(
                model_module=model,
                X=X,
                y=y,
                train_size=train_size,
                step_size=step_size,
                mode=mode_value,
                progress=True
            )

            # Summary result for one model and one mode
            summary = dict(results["summary"])
            summary["Model"] = model_name
            summary["Ticker"] = symbol
            summary["Window_Mode"] = mode_name
            summary["N"] = len(X)

            all_results.append(summary)

            # Predictions for one model and one mode
            pred = results["results"].copy()
            n_predictions = len(pred)
            first_test_index = train_size

            pred["Feature_Date"] = feature_dates.iloc[
                first_test_index:first_test_index + n_predictions].to_numpy()

            pred["Target_Date"] = target_dates.iloc[
                first_test_index:first_test_index + n_predictions].to_numpy()
            
            pred["Model"] = model_name
            pred["Ticker"] = symbol
            pred["Window_Mode"] = mode_name

            all_predictions.append(pred)

            # Print immediate result
            print(
                f"Result | {symbol} | {model_name} | {mode_name} | "
                f"R2_OOS = {summary['R2_OOS']:.6f} | "
                f"RMSE = {summary['RMSE']:.6f}"
            )

    result_df = pd.DataFrame(all_results)
    prediction_df = pd.concat(all_predictions, ignore_index=True)

    print("=" * 70)
    print(f"FINAL COMPARISON: {symbol}")
    print("=" * 70)

    print(
        result_df[
            ["Window_Mode", "Model", "R2", "R2_OOS", "MSPE", "RMSE", "N"]
        ].sort_values(
            ["Window_Mode", "R2_OOS"],
            ascending=[True, False]
        )
    )

    return result_df, prediction_df


# ==========================================================
# Run Entire Market
# ==========================================================

def run_market():

    files = load_all_market_files()

    market_results = []

    market_predictions = []

    start = time.time()

    for file in files:

        print("="*70)
        print(os.path.basename(file))

        result_df, prediction_df = run_symbol(file)

        market_results.append(result_df)

        market_predictions.append(prediction_df)

    market_results = pd.concat(market_results, ignore_index=True)

    market_predictions = pd.concat(market_predictions, ignore_index=True)

    os.makedirs(os.path.join(RESULT_PATH, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_PATH, "raw"), exist_ok=True)

    market_results.to_csv(
        os.path.join(RESULT_PATH, "tables", "iran_market_summary.csv"),
        index=False
    )

    market_predictions.to_csv(
        os.path.join(RESULT_PATH, "raw", "iran_market_predictions.csv"),
        index=False
    )

    print("="*70)
    print("Finished")
    print(f"Files : {len(files)}")
    print(f"Execution Time : {time.time()-start:.2f} sec")
    print("="*70)

    return market_results, market_predictions

