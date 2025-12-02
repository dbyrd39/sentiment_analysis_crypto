import pandas as pd

def merge_sources(tweet_df, reddit_df, news_df):
    frames = [tweet_df, reddit_df, news_df]
    merged = pd.concat(frames, ignore_index=True)
    return merged
