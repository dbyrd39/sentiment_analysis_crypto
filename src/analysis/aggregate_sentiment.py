import pandas as pd

def aggregate_daily(df, date_col="date", score_col="finbert_score"):
    df[date_col] = pd.to_datetime(df[date_col])
    daily = df.groupby([df[date_col].dt.date, "coin"])[score_col].mean().reset_index()
    daily.columns = ["date", "coin", "mean_sentiment"]
    return daily
