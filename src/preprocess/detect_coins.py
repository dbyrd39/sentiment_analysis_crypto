import re
import pandas as pd

COIN_MAP = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL",
    "xrp": "XRP", "ripple": "XRP",
}

def detect_coin(text: str) -> str:
    for k, v in COIN_MAP.items():
        if re.search(rf"\b{k}\b", text.lower()):
            return v
    return "OTHER"

def assign_coin_column(df: pd.DataFrame, text_col="clean_text"):
    df["coin"] = df[text_col].apply(detect_coin)
    return df
