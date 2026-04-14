# Project Handoff Note
**Research:** Social Media Sentiment as a Predictor of Extreme Crypto Events
**Status:** Pipeline complete, first results obtained, ready for improvement

---

## What we built

Starting from scratch, we built a full end-to-end research pipeline across five Python scripts:

1. **data_collection.py** — Downloads daily BTC and ETH price data from Yahoo Finance (2021–2024), approximates daily VWAP as (High + Low + Close) / 3, computes log-returns, and constructs a 50/50 equal-weighted portfolio return series. Saves to `data/price_data.csv`.

2. **sentiment_scoring.py** — Reads `data/articles.txt` (tab-separated: date + article text), scores every article with VADER (primary, compound score −1 to +1) and TextBlob (validation), then resamples to monthly frequency so each month has one composite sentiment score. Saves to `data/sentiment_scores.csv`.

3. **event_definition.py** — Computes 63-trading-day (≈3 month) forward returns for the portfolio, calculates rolling mean and SD, and labels any period where the forward return exceeds ±2.5 rolling SD as an extreme event (binary: 1 = extreme, 0 = normal). Resamples daily labels to monthly frequency using MAX — if any day in a month was extreme, the whole month is labelled 1. Saves to `data/labeled_events.csv`.

4. **analysis.py** — Merges monthly sentiment scores with monthly event labels, runs a logistic regression (VADER compound → extreme binary), and reports odds ratio, AUC-ROC, McFadden R², and p-value. Saves ROC curve to `outputs/roc_curve.png`.

5. **articles.txt** — We manually created 66 representative articles spanning January 2021 to December 2024, covering all major crypto events (2021 bull run, Terra/LUNA collapse, FTX bankruptcy, 2023 recovery, Bitcoin ETF approval, 2024 halving and ATH). Each line is formatted as `YYYY-MM-DD\tarticle text`.

We also wrote `requirements.txt`, `README.md`, and `CONTEXT.md` documenting the full methodology.

---

## A problem we ran into and fixed

The first version merged sentiment and event data on exact dates. Because articles.txt had only 1–4 articles per month (not one per trading day), the inner join produced only **54 matched rows and 3 extreme events** — far too few for logistic regression to find any signal.

**Fix:** We resampled both datasets to monthly frequency before merging:
- `sentiment_scoring.py` now averages all articles within a month → one row per month
- `event_definition.py` now takes the MAX of extreme_binary per month → one row per month
- `analysis.py` now aligns both series on month-end timestamps before joining

This gave us **41 monthly observations with 14 extreme event months (34.1%)** — a workable dataset.

---

## Results so far

| Metric | Value | What it means |
|---|---|---|
| Odds Ratio | 1.1985 | A 1-SD rise in sentiment multiplies the odds of an extreme event by 1.20× |
| AUC-ROC | 0.5370 | Barely above random (0.50 = guessing, 1.0 = perfect) |
| McFadden R² | 0.0057 | Sentiment explains ~0.6% of extreme event months |
| p-value | 0.5832 | Not statistically significant — cannot rule out chance |

**Plain-language conclusion:** The odds ratio of 1.20 shows a small positive directional effect — higher sentiment slightly raises the odds of an extreme event — but it is weak and not statistically significant. AUC of 0.537 means the model barely outperforms random guessing. McFadden R² of 0.006 means sentiment alone explains almost nothing. The p-value of 0.58 means there is a 58% chance this result appeared by chance.

This is a valid and honest finding: a small number of curated articles scored with VADER does not strongly predict extreme crypto events on its own. That conclusion is itself reportable.

---

## How to improve next

The following are the most impactful next steps, in priority order:

**1. Add more articles per month (highest priority)**
The biggest weakness is data density — averaging 1–4 articles per month is not enough to produce a reliable composite sentiment score. Target at least 10–15 articles per month. Sources to add: CoinDesk daily news feed, a16z Crypto blog, The Block, Decrypt. Even pulling article headlines rather than full text would help.

**2. Test a lagged sentiment predictor**
Currently, sentiment in month T is matched against the extreme event label for month T. But the research question is about prediction — does sentiment *precede* extreme events? Try shifting the sentiment score forward by 1 month (sentiment at T predicts event at T+1) and re-running analysis.py. This is a one-line change: `df["vader_compound"] = df["vader_compound"].shift(1)` before the regression.

**3. Add a control variable**
A single predictor model is fragile. Adding one control variable — such as monthly BTC trading volume (already downloadable via yfinance) or the VIX index (market fear gauge) — would make the model more robust and allow the mentor to assess whether sentiment adds explanatory power *beyond* what market conditions already predict.

**4. Run the 48-hour robustness check**
The methodology specifies a 48-hour window as a robustness check alongside the 3-month primary window. In `event_definition.py`, change `WINDOW = 63` to `WINDOW = 2` and re-run the pipeline. Compare results. If the signal improves at shorter horizons, that tells you something about how quickly sentiment translates into price action.

**5. Check the pilot period benchmark**
The mentor's reference composite VADER score for February 2024 was −6 (on a ×100 scale, i.e. −0.06 in raw VADER). Our pipeline produced a February 2024 composite of +0.326 for that month — a meaningful discrepancy. This could be due to article selection differences. Worth discussing with the mentor to align on which articles to include.

---

## How to continue in a new conversation

1. Share this file and `CONTEXT.md` with your new Claude session as context
2. The full pipeline is in `/Users/hongqingwang/Documents/GitHub/Extreme/`
3. Run order: `data_collection.py` → `sentiment_scoring.py` → `event_definition.py` → `analysis.py`
4. The most impactful next task is increasing article density in `data/articles.txt`
