from datasets import load_dataset
import pandas as pd

def load_financial_phrasebank() -> pd.DataFrame:
    ds = load_dataset("financial_phrasebank", "sentences_allagree")
    return ds["train"].to_pandas()
