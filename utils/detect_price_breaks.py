import numpy as np
import pandas as pd

def detect_price_breaks(df, threshold=0.20):

    data = df.copy()
    data = data.loc[(data["<FIRST>"] > 0) &(data["<LAST>"] > 0)].copy()
    # chronological order
    data = (data.sort_values("<DTYYYYMMDD>").reset_index(drop=True))

    data = data.loc[data["<VOL>"] > 0].copy()
    data = data.reset_index(drop=True)

    # Previous trading day's LAST
    data["prev_last"] = data["<LAST>"].shift(1)

    # Opening jump
    data["raw_return"] = np.log(data["<FIRST>"] / data["prev_last"])

    # Abnormal opening jump
    data["price_break"] = (data["raw_return"].abs() >np.log(1 + threshold))

    events = data.loc[
        data["price_break"],
        ["<DTYYYYMMDD>",
            "prev_last",
            "<FIRST>",
            "<LAST>",
            "raw_return",
            "<VOL>",
            "<VALUE>"]].copy()

    # Adjustment factor
    events["factor"] = (events["<FIRST>"] /events["prev_last"])

    return events, data