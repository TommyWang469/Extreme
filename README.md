# Social Media Sentiment as a Predictor of Extreme Crypto Events

**Research question:** Does social media sentiment predict extreme crypto events?

The pipeline tests whether VADER sentiment scores from curated crypto articles
(a16z Crypto, CoinDesk) can serve as an early-warning signal for extreme
portfolio returns — framed as signal detection, not causal inference.

---

## Folder structure

```
Extreme/
├── data_collection.py      # Step 1 — fetch BTC/ETH VWAP, build portfolio
├── sentiment_scoring.py    # Step 2 — score articles with VADER + TextBlob
├── event_definition.py     # Step 3 — label extreme events (±2.5 SD threshold)
├── analysis.py             # Step 4 — logistic regression + metrics + ROC plot
├── requirements.txt        # Python dependencies
├── CONTEXT.md              # Research context and methodology notes
├── data/
│   ├── articles.txt        # Input: one article per line (date\tarticle text)
│   ├── price_data.csv      # Output of step 1
│   ├── sentiment_scores.csv# Output of step 2
│   └── labeled_events.csv  # Output of step 3
└── outputs/
    └── roc_curve.png       # Output of step 4
```

---

## Setup

```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # one-time TextBlob corpus download
```

---

## Running the pipeline (in order)

### Step 1 — Collect price data
```bash
python data_collection.py
```
Downloads daily BTC-USD and ETH-USD OHLCV data from Yahoo Finance (2021–2024),
approximates VWAP as `(High + Low + Close) / 3`, computes daily log-returns,
and builds a 50/50 equal-weighted portfolio return series.
**Output:** `data/price_data.csv`

---

### Step 2 — Score sentiment
```bash
python sentiment_scoring.py
```
Reads `data/articles.txt`. Each line should be formatted as:
```
YYYY-MM-DD<TAB>article text here
```
Scores every article with VADER (primary) and TextBlob (validation), then
aggregates to a daily composite score.
**Output:** `data/sentiment_scores.csv`

---

### Step 3 — Define extreme events
```bash
python event_definition.py
```
Calculates 63-trading-day (≈ 3-month) forward returns for the portfolio,
computes rolling mean and SD over the same window, and labels any day whose
forward return exceeds ±2.5 SD as an extreme event (binary: 1 = extreme,
0 = normal).
**Output:** `data/labeled_events.csv`

---

### Step 4 — Run analysis
```bash
python analysis.py
```
Merges sentiment scores with event labels on date, fits a logistic regression
(`extreme_binary ~ vader_compound`), and reports:

| Metric | Description |
|---|---|
| **Odds ratio** | exp(β₁) — multiplicative change in odds per 1-SD sentiment shift |
| **AUC-ROC** | Area under the ROC curve (sklearn `roc_auc_score`) |
| **McFadden R²** | 1 − LL_model / LL_null |

**Output:** printed summary + `outputs/roc_curve.png`

---

## Methodology notes

| Choice | Detail |
|---|---|
| Portfolio | 50% BTC + 50% ETH, VWAP-based daily log-returns |
| Sentiment (primary) | VADER compound score, averaged across 12 articles per period |
| Sentiment (validation) | TextBlob polarity |
| Extreme event threshold | ±2.5 rolling SD of 63-day forward return |
| Model | Logistic regression (statsmodels `Logit`) |
| Robustness check | Re-run `event_definition.py` with `WINDOW = 2` for 48-hour window |
| Pilot reference | Feb 2024 composite VADER score ≈ −6 (×100 scale) |

> **Framing:** results should be interpreted as an early-warning signal
> association, not a causal claim.
