"""
finbert_scoring.py
Sprint 2, mentor direction A (Jun 2026): score the scraped crypto headlines with
FinBERT (ProsusAI/finbert) — a transformer fine-tuned on financial text — instead
of the lexicon-based VADER. Produces WEEKLY composites (weekly resolution is the
Sprint-2 statistical-power fix). VADER weekly is also computed for the ablation
table in analysis_v2.py.

Per-article score = P(positive) − P(negative)  ∈ [−1, 1]

Output: data/sentiment_weekly.csv
  week, n_articles, finbert_linear, finbert_exp_hl7, vader_linear, vader_exp_hl7
"""

import os
import numpy as np
import pandas as pd

SCRAPED_PATH   = "data/articles_scraped.csv"
OUT_PATH       = "data/sentiment_weekly.csv"
HALF_LIFE_DAYS = 7
SNIPPET_CHARS  = 240
BATCH          = 32
WEEK_RULE      = "W-FRI"   # week ends Friday, aligns with ARKF/QQQ trading week

os.makedirs("data", exist_ok=True)


def load_articles() -> pd.DataFrame:
    df = pd.read_csv(SCRAPED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    head = df.get("headline", pd.Series([""] * len(df))).fillna("")
    snip = df.get("snippet",  pd.Series([""] * len(df))).fillna("")
    text = (head + ". " + snip).str.strip()
    # GDELT rows duplicate headline into snippet — don't score the same words twice
    dup = head.str.strip() == snip.str.strip()
    text[dup] = head[dup].str.strip()
    df["text"] = text.str.slice(0, SNIPPET_CHARS)
    df = df.dropna(subset=["date", "text"]).sort_values("date").reset_index(drop=True)
    print(f"{len(df)} articles, {df['date'].min().date()} → {df['date'].max().date()}")
    return df[["date", "text"]]


def score_finbert(texts: list) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok   = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    pos = next(i for i, l in id2label.items() if l == "positive")
    neg = next(i for i, l in id2label.items() if l == "negative")
    print(f"FinBERT on {device}, labels={id2label}")

    out = np.empty(len(texts))
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            enc = tok(batch, return_tensors="pt", truncation=True,
                      padding=True, max_length=64).to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out[i:i + len(batch)] = probs[:, pos] - probs[:, neg]
            if (i // BATCH) % 10 == 0:
                print(f"  scored {i + len(batch)}/{len(texts)}")
    return out


def score_vader(texts: list) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    v = SentimentIntensityAnalyzer()
    return np.array([v.polarity_scores(t)["compound"] for t in texts])


def weekly_composites(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = df["date"].dt.to_period(WEEK_RULE).dt.end_time.dt.normalize()
    days_before_end = (df["week"] - df["date"]).dt.days.clip(lower=0)
    df["w_exp"] = np.exp(-np.log(2) / HALF_LIFE_DAYS * days_before_end)

    rows = []
    for wk, g in df.groupby("week"):
        w = g["w_exp"].to_numpy()
        rows.append({
            "week":            wk,
            "n_articles":      len(g),
            "finbert_linear":  g["finbert"].mean(),
            f"finbert_exp_hl{HALF_LIFE_DAYS}": np.average(g["finbert"], weights=w),
            "vader_linear":    g["vader"].mean(),
            f"vader_exp_hl{HALF_LIFE_DAYS}":   np.average(g["vader"], weights=w),
        })
    return pd.DataFrame(rows).set_index("week").sort_index()


def main():
    df = load_articles()
    texts = df["text"].tolist()
    df["finbert"] = score_finbert(texts)
    df["vader"]   = score_vader(texts)
    weekly = weekly_composites(df)
    weekly.to_csv(OUT_PATH)

    corr = df["finbert"].corr(df["vader"])
    print(f"Saved {len(weekly)} weekly rows → {OUT_PATH}")
    print(f"  article-level corr(FinBERT, VADER) = {corr:.3f} "
          f"(low corr ⇒ the domain model reads the news differently)")
    print(f"  weeks with ≥1 article: {len(weekly)}, "
          f"median articles/week: {weekly['n_articles'].median():.0f}")


if __name__ == "__main__":
    main()
