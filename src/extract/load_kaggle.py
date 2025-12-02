import pandas as pd

# Global row limit for proof-of-concept runs
ROW_LIMIT = 5000   # adjust to taste

def load_kaggle_social(path: str, row_limit: int = ROW_LIMIT) -> pd.DataFrame:
    """
    Universal dataset loader that:
      - Reads at most row_limit rows
      - Handles encoding issues
      - Auto-detects the correct text column
      - Safely parses date columns
      - Skips bad lines
    """

    df = pd.read_csv(
        path,
        on_bad_lines="skip",
        engine="python",
        encoding="latin1",
        encoding_errors="ignore",
        nrows=row_limit
    )

    # Candidate text columns commonly found in Kaggle/social datasets
    candidate_text_cols = [
        "text", "body", "selftext", "title",
        "comment", "content", "message", "clean_comment",
    ]

    # Find the first column that exists
    text_col = None
    for col in candidate_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"No usable text column found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"[INFO] Using text column '{text_col}' for {path}")

    # Clean the text column
    df = df[df[text_col].notna()]
    df[text_col] = df[text_col].astype(str)
    df = df.rename(columns={text_col: "text"})  # normalize to a single column name

    # Optional: unify date handling
    date_cols = ["date", "created_utc", "timestamp", "created_at"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df[df[col].notna()]
            df = df.rename(columns={col: "date"})
            break

    print(f"[INFO] Loaded {len(df)} rows from {path} (limit={row_limit})")
    return df


