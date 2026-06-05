"""
sentiment_scoring.py
Scores crypto news with VADER (primary) and TextBlob (validation), then collapses
to one monthly composite score per month. Saves data/sentiment_scores.csv.

SPRINT 1 CHANGES (improvement.md T8)
─────────────────────────────────────────────────────────────────────────────
  • LINEAR vs EXPONENTIAL weighting (mentor's hint). Within each month we now
    produce two composites:
        vader_linear   – equal-weight mean of every article in the month
        vader_exp_hlN  – exponentially-weighted mean, half-life HALF_LIFE_DAYS,
                         so articles nearer month-end count more (they are the
                         freshest signal for predicting the *next* month).
  • SNIPPET scoring (mentor's hint). Sentiment lives in the headline + lede, not
    the diluted body. If SCORE_SNIPPET is True we score only the first
    SNIPPET_CHARS characters of each article. Set False to score full text.
  • Input auto-detects format: prefers data/articles_scraped.csv (date,source,
    url,headline,snippet) produced by scrape_articles.py; falls back to the
    legacy tab-separated data/articles.txt.

Backward compatibility: the column `vader_compound` is kept as an alias of
`vader_linear` so the existing analysis.py keeps working unchanged.
"""

import os
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ── Settings ──────────────────────────────────────────────────────────────────
SCRAPED_PATH   = "data/articles_scraped.csv"   # preferred input if present
LEGACY_PATH    = "data/articles.txt"           # fallback input
OUT_PATH       = "data/sentiment_scores.csv"

SCORE_SNIPPET    = True    # score only the lede (mentor: "Snippet")
SNIPPET_CHARS    = 240     # ~ headline + first sentence
HALF_LIFE_DAYS   = 7       # exponential-weight half-life within a month

os.makedirs("data", exist_ok=True)


# ── 1. Load articles (scraped CSV preferred, legacy txt fallback) ─────────────
def load_articles() -> pd.DataFrame:
    if os.path.exists(SCRAPED_PATH):
        print(f"Loading scraped corpus → {SCRAPED_PATH}")
        df = pd.read_csv(SCRAPED_PATH)
        # Build text from headline + snippet; tolerate missing columns
        head = df.get("headline", pd.Series([""] * len(df))).fillna("")
        snip = df.get("snippet",  pd.Series([""] * len(df))).fillna("")
        df["text"] = (head + ". " + snip).str.strip()
        df = df[["date", "text"]]
    else:
        print(f"Scraped corpus not found; loading legacy → {LEGACY_PATH}")
        rows = []
        with open(LEGACY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                date_str, text = (parts if len(parts) == 2 else (None, parts[0]))
                rows.append({"date": date_str, "text": text})
        df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(df["date"])
    if SCORE_SNIPPET:
        df["text"] = df["text"].str.slice(0, SNIPPET_CHARS)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} articles, {df['date'].min().date()} → {df['date'].max().date()}"
          f"  (snippet={'on' if SCORE_SNIPPET else 'off'})")
    return df


# ── 2. Sentiment scoring ──────────────────────────────────────────────────────
def score(df: pd.DataFrame) -> pd.DataFrame:
    vader = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["text"].apply(lambda t: vader.polarity_scores(t)["compound"])
    df["tb_polarity"]    = df["text"].apply(lambda t: TextBlob(t).sentiment.polarity)
    return df


# ── 3. Monthly composites: linear AND exponential weighting ───────────────────
def monthly_composites(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    # month-end timestamp per row, and days-before-month-end for the decay weight
    month_end = df["month"].dt.to_timestamp("M")
    days_before_end = (month_end - df["date"]).dt.days.clip(lower=0)
    decay = np.log(2) / HALF_LIFE_DAYS
    df["w_exp"] = np.exp(-decay * days_before_end)

    out = []
    for period, g in df.groupby("month"):
        w = g["w_exp"].to_numpy()
        v = g["vader_compound"].to_numpy()
        out.append({
            "date":          period.to_timestamp("M"),
            "n_articles":    len(g),
            "vader_linear":  v.mean(),
            f"vader_exp_hl{HALF_LIFE_DAYS}": np.average(v, weights=w),
            "tb_polarity":   g["tb_polarity"].mean(),
        })
    monthly = pd.DataFrame(out).set_index("date").sort_index()
    # Backward-compat alias used by analysis.py
    monthly["vader_compound"] = monthly["vader_linear"]
    return monthly


def main():
    df = load_articles()
    df = score(df)
    monthly = monthly_composites(df)
    monthly.to_csv(OUT_PATH)
    print(f"\nSaved {len(monthly)} monthly rows → {OUT_PATH}")
    print(monthly.round(3).to_string())

    # Quick linear-vs-exponential divergence check
    exp_col = f"vader_exp_hl{HALF_LIFE_DAYS}"
    diff = (monthly[exp_col] - monthly["vader_linear"]).abs().mean()
    print(f"\nMean |exp − linear| divergence: {diff:.4f}  "
          f"(0 ⇒ weighting makes no difference; larger ⇒ recency matters)")


if __name__ == "__main__":
    main()
