import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Social-media optimized model using safetensors
tokenizer = AutoTokenizer.from_pretrained(
    "finiteautomata/bertweet-base-sentiment-analysis",
    use_fast=False  # required for BERTweet
)

model = AutoModelForSequenceClassification.from_pretrained(
    "finiteautomata/bertweet-base-sentiment-analysis",
    use_safetensors=True
)

model.eval()

LABELS = ["negative", "neutral", "positive"]


def run_twitter_batched(df: pd.DataFrame, text_col="clean_text", batch_size=16, max_length=128):
    texts = df[text_col].tolist()
    sentiments = []
    scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Twitter sentiment"):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)

        sentiments.extend([LABELS[p] for p in preds])
        scores.extend(probs[:, 2] - probs[:, 0])

    df["tweet_sentiment"] = sentiments
    df["tweet_score"] = scores

    return df
