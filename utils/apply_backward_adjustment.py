def apply_backward_adjustment(df, events):

    data = df.copy()

    # chronological order
    data = (data.sort_values("<DTYYYYMMDD>").reset_index(drop=True))

    # Final cumulative adjustment factor
    data["CUM_FACTOR"] = 1.0

    # Newest → oldest order for events
    events = events.sort_values(
        "<DTYYYYMMDD>",
        ascending=False
    )

    for _, event in events.iterrows():

        event_date = event["<DTYYYYMMDD>"]
        factor = event["factor"]

        # All observations before the event
        mask = data["<DTYYYYMMDD>"] < event_date

        data.loc[mask, "CUM_FACTOR"] *= factor

    # Price columns to adjust
    price_cols = ["<OPEN>","<FIRST>","<HIGH>","<LOW>","<CLOSE>","<LAST>"]

    for col in price_cols:

        data[f"ADJ_{col}"] = (data[col] * data["CUM_FACTOR"])
        
    data["CUM_FACTOR"] = data["CUM_FACTOR"].astype(float)    
    return data