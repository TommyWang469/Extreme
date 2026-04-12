"""
sentiment_scoring.py
Loads article texts from data/articles.txt (one article per line),
scores each with VADER (primary) and TextBlob (validation), and saves
a composite sentiment score to data/sentiment_scores.csv.

Expected format of articles.txt:
    <ISO-date>\t<article text>
e.g.:
    2024-02-01\tBitcoin surges as ETF inflows accelerate…
    2024-02-03\tCrypto market faces headwinds from Fed minutes…

If no date column is present the script assigns sequential dates starting
from a configurable START_DATE so you can still test the pipeline.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os

# ── Settings ──────────────────────────────────────────────────────────────────
ARTICLES_PATH = "data/articles.txt"
OUT_PATH      = "data/sentiment_scores.csv"
START_DATE    = "2024-02-01"   # fallback start date if file has no date column

os.makedirs("data", exist_ok=True)

# ── 1. Load articles ──────────────────────────────────────────────────────────
print("Loading articles …")
rows = []
with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Try to parse a leading date (tab-separated)
        parts = line.split("\t", maxsplit=1)
        if len(parts) == 2:
            date_str, text = parts
        else:
            date_str, text = None, parts[0]
        rows.append({"date": date_str, "text": text})

df = pd.DataFrame(rows)

# Assign sequential dates if none provided
if df["date"].isna().all():
    dates = pd.date_range(start=START_DATE, periods=len(df), freq="D")
    df["date"] = dates.strftime("%Y-%m-%d")

df["date"] = pd.to_datetime(df["date"])
print(f"  {len(df)} articles loaded, spanning {df['date'].min().date()} → {df['date'].max().date()}")

# ── 2. VADER scoring ──────────────────────────────────────────────────────────
print("Scoring with VADER …")
vader = SentimentIntensityAnalyzer()

def vader_scores(text):
    s = vader.polarity_scores(text)
    return pd.Series({
        "vader_neg":      s["neg"],
        "vader_neu":      s["neu"],
        "vader_pos":      s["pos"],
        "vader_compound": s["compound"],   # primary metric  [-1, +1]
    })

df = df.join(df["text"].apply(vader_scores))

# ── 3. TextBlob scoring (validation) ─────────────────────────────────────────
print("Scoring with TextBlob …")

def textblob_scores(text):
    blob = TextBlob(text)
    return pd.Series({
        "tb_polarity":     blob.sentiment.polarity,      # [-1, +1]
        "tb_subjectivity": blob.sentiment.subjectivity,  # [0, 1]
    })

df = df.join(df["text"].apply(textblob_scores))

# ── 4. Composite score: average VADER compound across articles per date ────────
# Mentor's reference score for Feb 2024 is -6 on a 0–100-style scale;
# we keep the raw [-1, +1] VADER compound here and note that scaling by 100
# would reproduce that convention.
daily = (
    df.groupby("date")
    .agg(
        n_articles        = ("text",          "count"),
        vader_compound    = ("vader_compound", "mean"),   # composite VADER
        tb_polarity       = ("tb_polarity",    "mean"),   # composite TextBlob
        tb_subjectivity   = ("tb_subjectivity","mean"),
    )
    .reset_index()
)

# ── 5. Save ───────────────────────────────────────────────────────────────────
daily.to_csv(OUT_PATH, index=False)
print(f"Saved {len(daily)} daily rows → {OUT_PATH}")
print(daily)
