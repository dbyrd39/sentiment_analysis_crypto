from extract.load_kaggle import load_kaggle_social
from preprocess.clean_text import clean_text
from preprocess.detect_coins import assign_coin_column
from models.finbert_batched import run_finbert_batched
from models.twitter_batched import run_twitter_batched
from analysis.aggregate_sentiment import aggregate_daily

import pandas as pd

def run_pipeline():

    # Load datasets
    tweets = load_kaggle_social("data/sample/sample_twitter.csv")
    reddit = load_kaggle_social("data/sample/sample_reddit.csv")
    news = load_kaggle_social("data/sample/sample_news.csv")

    # Combine
    df = pd.concat([tweets, reddit, news], ignore_index=True)

    # Clean text
    df["text"] = df["text"].fillna("").astype(str)
    df["clean_text"] = df["text"].apply(clean_text)

    # Detect coins
    df = assign_coin_column(df)

    # Run models
    df = run_finbert_batched(df, text_col="clean_text", batch_size=32)
    df = run_twitter_batched(df, text_col="clean_text", batch_size=32)


    # Save long-form scored data
    df.to_parquet("data/processed/social_scored.parquet")

    # Aggregate by day + coin
    daily = aggregate_daily(df)
    daily.to_csv("data/processed/daily_sentiment.csv", index=False)

    print("Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()
