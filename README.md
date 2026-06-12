# Social Media Sentiment as a Predictor of Extreme Crypto Events

**Research question:** Does social-media / news sentiment serve as an early-warning
signal for extreme moves in a 50/50 BTC+ETH portfolio?

The pipeline scores crypto news with VADER (primary) and TextBlob (validation),
labels "extreme" months from forward portfolio returns, and tests whether
sentiment precedes those extremes — via both **logistic regression** and an
**event study**. Framed as signal detection, **not** causal inference.

> See `CONTEXT.md` for methodology + results, `improvement.md` for the roadmap and
> open questions, and `sprint2_results.md` for the latest numbers.

---

## Pipeline at a glance

```
Github Extreme/
├── data_collection.py     # Step 1 — fetch BTC/ETH VWAP, build 50/50 portfolio
├── scrape_articles.py     # Step 2 — (optional) GDELT + CoinDesk RSS article corpus
├── sentiment_scoring.py   # Step 3 — VADER + TextBlob, linear & exponential weighting
├── event_definition.py    # Step 4 — label extreme months (quantile / global_z / EVT)
├── analysis.py            # Step 5 — logistic regression (4 specs) + ROC + benchmark
├── event_study.py         # Step 6 — CAR around known shocks (Terra, FTX, ETF, halving)
├── finbert_scoring.py     # Step 7 — FinBERT weekly sentiment (Sprint 2)
├── build_features_v2.py   # Step 8 — RWDV + scar labels + ARKF/QQQ factors (Sprint 2)
├── analysis_v2.py         # Step 9 — multi-factor weekly logit (Sprint 2)
├── requirements.txt
├── CONTEXT.md  HANDOFF.md  improvement.md  sprint2_results.md
├── data/
│   ├── articles.txt           # legacy manual corpus (fallback input)
│   ├── articles_scraped.csv   # produced by scrape_articles.py (preferred input)
│   ├── raw_html/              # scraper cache (git-ignored)
│   ├── price_data.csv         # Step 1 output
│   ├── sentiment_scores.csv   # Step 3 output (vader_linear, vader_exp_hl7, tb_polarity)
│   └── labeled_events.csv     # Step 4 output (~13% extreme rate)
└── outputs/
    ├── roc_curve.png          # Step 5
    └── event_study_car.png    # Step 6
```

---

## Setup (one time)

```bash
# from the project root
pip install -r requirements.txt
python3 -m textblob.download_corpora      # TextBlob corpora (one time)
```

Requires Python 3.10+. Dependencies: yfinance, vaderSentiment, textblob, pandas,
numpy, scikit-learn, statsmodels, matplotlib, scipy, requests, beautifulsoup4, lxml.

> **Note:** commands below use `python3` (verified on this machine). If your
> system maps `python` to Python 3, you can use `python` interchangeably.

---

## Compile / run the whole project (copy-paste)

These commands assume your terminal is **inside the project folder**. Because the
folder name ends with a space, `cd` into it once with quotes:

```bash
cd "/Users/hongqingwang/Documents/GitHub/Github Extreme "
```

### 1. Byte-compile check (catches syntax errors without running anything)

```bash
python3 -m py_compile data_collection.py scrape_articles.py sentiment_scoring.py \
    event_definition.py analysis.py event_study.py && echo "All scripts compile OK"
```

### 2. Run the full pipeline in order (one block)

```bash
python3 data_collection.py      && \
python3 sentiment_scoring.py    && \
python3 event_definition.py     && \
python3 analysis.py             && \
python3 event_study.py
```

> `scrape_articles.py` is **optional** and intentionally left out of the one-shot
> block because it is slow (GDELT is rate-limited) and network-dependent. Run it
> separately to build the dense corpus (see below). If `data/articles_scraped.csv`
> exists, `sentiment_scoring.py` uses it automatically; otherwise it falls back to
> `data/articles.txt`.

### 3. (Optional) Build the dense article corpus

```bash
python3 scrape_articles.py      # may need 2–3 runs to fill GDELT-rate-limited months
# then regenerate sentiment + results on the dense corpus:
python3 sentiment_scoring.py && python3 analysis.py
```

### 4. Sprint 2 — FinBERT + RWDV multi-factor (weekly)

```bash
python3 finbert_scoring.py     # FinBERT weekly sentiment (first run downloads the model)
python3 build_features_v2.py   # RWDV + scar labels + ARKF/QQQ factors
python3 analysis_v2.py         # multi-factor logit → sprint2_results.md
```

### One-liner: compile, then run everything

```bash
python3 -m py_compile *.py && \
python3 data_collection.py && python3 sentiment_scoring.py && \
python3 event_definition.py && python3 analysis.py && python3 event_study.py
```

---

## What each step does

| Step | Script | Input → Output |
|---|---|---|
| 1 | `data_collection.py` | Yahoo Finance → `data/price_data.csv` (BTC/ETH VWAP, 50/50 log-return portfolio) |
| 2 | `scrape_articles.py` *(optional)* | GDELT + CoinDesk RSS → `data/articles_scraped.csv` (date, source, url, headline, snippet) |
| 3 | `sentiment_scoring.py` | articles → `data/sentiment_scores.csv` (VADER linear **&** exponential, TextBlob) |
| 4 | `event_definition.py` | `price_data.csv` → `data/labeled_events.csv` (one obs/month, ~13% extreme) |
| 5 | `analysis.py` | merge → console table + `outputs/roc_curve.png` (4 predictor specs, pilot benchmark) |
| 6 | `event_study.py` | `price_data.csv` + sentiment → `outputs/event_study_car.png` (CAR around shocks) |

---

## Methodology notes (current)

| Choice | Detail |
|---|---|
| Portfolio | 50% BTC + 50% ETH, VWAP-based daily log-returns |
| Sentiment (primary) | VADER compound — **linear** and **exponential** (7-day half-life) monthly composites |
| Sentiment (validation) | TextBlob polarity |
| Snippet scoring | headline + lede, not full body |
| Extreme-event label | one forward-return obs per month, thresholded on the **whole-sample** distribution; methods: `quantile` (default), `global_z`, `evt` (Generalized Pareto / Black Swan). Base rate ≈ **13%** (was 31.8% in v1) |
| Models | Logistic regression (statsmodels `Logit`, 4 specs) **+** event study (mean-adjusted CAR) |
| Robustness check | set `WINDOW = 2` in `event_definition.py` for the 48-hour window |
| Pilot reference | Feb-2024 composite ≈ −0.06 (mentor); ours +0.275 — unresolved |

To switch the extreme-event method, edit `METHOD` at the top of
`event_definition.py` (`"quantile"` / `"global_z"` / `"evt"`).

> **Framing:** results are an early-warning *association*, not a causal claim.
> Current best spec (exp-weighted, lag-1): AUC ≈ 0.56, not statistically significant.
