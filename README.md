
# 📊 Crypto Sentiment Analysis Pipeline  
### *Multi-Model Sentiment Analysis on Large-Scale Crypto Social Data (Tweets, Reddit, News)*  
**Python • HuggingFace Transformers • FinBERT • BERTweet • NLP • Data Engineering**

---

## 🚀 Overview  
This project builds a **robust, production-style sentiment analysis pipeline** for cryptocurrency-related social media content. It ingests large-scale datasets (Twitter, Reddit), cleans the data, and applies **batched transformer-based sentiment models** to extract market sentiment for assets like Bitcoin, Ethereum, and other major cryptocurrencies.

The pipeline is engineered for:

- **High performance** (batched GPU-ready inference)  
- **Large datasets** (up to millions of rows)  
- **Fault tolerance** (handling noisy social text, malformed CSVs, encoding issues)  
- **Easy deployment** (10-minute CPU proof-of-concept mode)  

---

## ✨ Features

### 🔹 Multi-model Sentiment Analysis  
- **FinBERT-tone (safetensors)** — Financial tone classification  
- **BERTweet Sentiment (safetensors)** — Social media sentiment  

### 🔹 Batched, Vectorized Inference (20–30× faster)  
- Efficient tokenization (`max_length=64`)  
- Large batch sizes (`batch_size=64–128`)  
- GPU-ready for instant throughput  

### 🔹 Robust Dataset Loader  
- Auto-detects text fields across sources (`text`, `body`, `selftext`, etc.)  
- Handles malformed CSVs safely  
- Fixes encoding errors (latin1, ignore)  
- Optional row caps for development (`ROW_LIMIT=5000`)  

### 🔹 Modular Architecture  
- `/extract` → dataset ingestion  
- `/preprocess` → text cleaning  
- `/models` → batching-optimized transformer inference  
- `pipeline.py` → orchestration  

---

## 📁 Project Structure

```
sentiment_analysis_crypto/
├── src/
│   ├── extract/
│   ├── preprocess/
│   ├── models/
│   └── pipeline.py
│
├── data/
│   ├── raw/        # ignored by Git
│   └── sample/     # optional small demo files
│
├── notebooks/
│   └── colab_sentiment_pipeline.ipynb
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE (optional)
```

---

## 🛠 Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd sentiment_analysis_crypto
```

Create a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

Place datasets into:

```
data/raw/
```

Run:

```bash
python src/pipeline.py
```

### 🔎 Fast Mode (~10 minutes)
Set:

```python
ROW_LIMIT = 5000
```

---

## ⚡ GPU Version (Google Colab)

Use the notebook in `/notebooks` for a fully GPU-accelerated run.

---

## 📈 Output

Creates:

```
data/processed/crypto_sentiment_output.csv
```

With:

- Sentiment scores  
- Labels  
- Cleaned text  
- Combined metrics across models  

---

## 🧩 Future Enhancements

- Streamlit dashboard  
- Topic modeling (BERTopic)  
- LLM-based sentiment explanations  
- Live data ingestion  

---

## 📄 License  
MIT License (optional)

---

## 👤 Author  
**Devyn Byrd**  
Feel free to connect on LinkedIn or GitHub!
