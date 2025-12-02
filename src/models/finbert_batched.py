import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Load FinBERT-tone (safetensors, macOS compatible)
tokenizer = AutoTokenizer.from_pretrained(
    "yiyanghkust/finbert-tone",
    use_safetensors=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone",
    use_safetensors=True,
    trust_remote_code=True
)

model.eval()  # put in inference mode

LABELS = ["negative", "neutral", "positive"]


def run_finbert_batched(df: pd.DataFrame, text_col="clean_text", batch_size=16, max_length=128):
    texts = df[text_col].tolist()
    sentiments = []
    scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT sentiment"):
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

        # sentiment score = positive - negative
        scores.extend(probs[:, 2] - probs[:, 0])

    df["finbert_sentiment"] = sentiments
    df["finbert_score"] = scores

    return df




