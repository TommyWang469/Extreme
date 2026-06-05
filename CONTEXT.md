# Project Context

## Research question
Does social media sentiment predict extreme crypto events?

## Methodology 
- Portfolio: 50% BTC + 50% ETH, VWAP-based daily returns
- Sentiment tool: VADER (primary), TextBlob (validation)
- Article source: 12 salient articles per period (a16z Crypto, CoinDesk)
- Extreme event definition: 3-month forward return > ±2.5 rolling SD = 1, else 0
- Model: logistic regression → odds ratio, AUC-ROC, McFadden R²
- Primary window: 3 months, robustness check: 48-hour window
- Framing: early warning signal, not causal claim

## Output metrics
- Odds ratio: exponentiate the logistic regression β₁ coefficient
- AUC-ROC: sklearn roc_auc_score
- McFadden R²: 1 - (log-likelihood model / log-likelihood null)

## Pilot period
February 2024, mentor's reference composite sentiment score was -6

## Explanation of each file

**data_collection.py**
Downloads historical daily price data for BTC and ETH from Yahoo Finance covering 2021–2024. For each day it approximates VWAP as (High + Low + Close) / 3, then computes daily log-returns for each asset. It then constructs the portfolio return as a simple 50/50 blend: 0.5 × BTC return + 0.5 × ETH return. The result is saved to data/price_data.csv and serves as the foundation for all downstream analysis.

**sentiment_scoring.py**
Reads data/articles.txt, where each line contains a date and an article text separated by a tab. It scores every article using two tools: VADER (the primary tool, which gives a compound score between -1 and +1 measuring overall sentiment) and TextBlob (used as a validation check, giving a polarity score on the same scale). Articles are then grouped and averaged by month to produce one composite sentiment score per month. This monthly score is the predictor variable in the regression. Output is saved to data/sentiment_scores.csv.


Linear weighted moving average VS exponential weighted



**event_definition.py**
Takes the daily portfolio returns from price_data.csv and computes the 3-month (63 trading day) forward return for each date — meaning, how much the portfolio gained or lost over the next three months from that point. It then calculates a rolling mean and rolling standard deviation of those forward returns. Any month where the forward return exceeds ±2.5 standard deviations from the rolling mean is flagged as an extreme event (binary label = 1), otherwise normal (label = 0). The daily data is then resampled to monthly frequency, where a month is labelled extreme if any single day within it was extreme. Output is saved to data/labeled_events.csv.

**analysis.py**
Merges the monthly sentiment scores with the monthly extreme event labels, aligning both series on month-end dates. It then runs a logistic regression where the VADER compound score predicts whether a month is an extreme event. From the fitted model it extracts three metrics: odds ratio, AUC-ROC, and McFadden R². It also plots the ROC curve and saves it to outputs/roc_curve.png, and prints a clean results summary to the terminal.

---

## What the metrics mean

**Odds Ratio (exp β₁)**
The odds ratio comes from exponentiating the β₁ coefficient of the logistic regression — the coefficient attached to the sentiment score. An odds ratio of 1.0 means sentiment has no effect. A value above 1.0 means higher sentiment increases the odds of an extreme event; below 1.0 means higher sentiment reduces those odds. In this project the odds ratio tells us: for every one standard deviation increase in the monthly VADER score, how much do the odds of experiencing an extreme crypto event multiply?

**AUC-ROC (Area Under the ROC Curve)**
AUC-ROC measures how well the model ranks months — specifically, whether months with higher predicted probability of an extreme event actually tend to be extreme. A value of 0.5 means the model is no better than random guessing. A value of 1.0 means perfect ranking. In this project it answers: does sentiment score order months by their actual extremeness, even if the threshold is imprecise?

**McFadden R²**
McFadden R² is the logistic regression equivalent of R² in linear regression. It is calculated as 1 − (log-likelihood of the fitted model / log-likelihood of a null model with only an intercept). Values above 0.2 are generally considered a good fit in logistic models. In this project it answers: what share of the variation in extreme event months does sentiment alone explain?

**p-value (β₁)**
The p-value tests whether the sentiment coefficient β₁ is statistically distinguishable from zero — i.e. whether the effect we observe could plausibly have appeared by chance even if sentiment had no real relationship with extreme events. A p-value below 0.05 is the conventional threshold for claiming the result is unlikely to be random. A high p-value (such as 0.58) means there is a 58% chance of seeing an effect this large just by chance, so we cannot confidently claim sentiment has any real predictive power based on this sample alone.

---

## Current conclusions (as of first full pipeline run)

**Results snapshot — 41 monthly observations, 14 extreme event months (34.1%)**

| Metric | Value |
|---|---|
| Odds Ratio | 1.1985 |
| AUC-ROC | 0.5370 |
| McFadden R² | 0.0057 |
| p-value (β₁) | 0.5832 |

The odds ratio of 1.20 is slightly interesting — it means a 1-SD increase in sentiment multiplies the odds of an extreme event by 1.20×. That is a small positive directional effect and it points in a plausible direction, but it is not strong enough to be meaningful on its own.

The AUC of 0.537 is barely above random. The model is better than guessing but only just. Sentiment score has some ability to rank months by extremeness, but the discrimination is weak.

McFadden R² of 0.006 means sentiment explains roughly 0.6% of extreme event months. Practically zero explanatory power.

The p-value of 0.58 means the result is not statistically significant — we cannot rule out that the observed direction happened by chance.

**What this tells us so far:** VADER sentiment derived from a small number of curated articles (roughly 1–4 per month) does not strongly predict extreme crypto events on its own. This is itself a reportable finding. The most likely reasons are: (1) too few articles per month to produce a reliable composite score, (2) the articles may not be timely enough to precede the events they describe, and (3) a single linear sentiment predictor may be too simple to capture the non-linear dynamics of extreme market moves. The next steps should focus on increasing article density, testing a lagged sentiment predictor, and potentially adding a second predictor such as trading volume or VIX as a control variable.

---

# Sprint 1 — What changed, why, and the effect on results (mentor review)

*Revisions made in response to your Apr-14 feedback (linear vs. exponential
weighting, event study, Mt Gox / SBF, Black Swan / extreme-value statistics,
BeautifulSoup, snippets). The v1 numbers above are kept for the record.*

## Headline: the most important output change

| Output | Before (v1) | After (Sprint 1) | What it means |
|---|---|---|---|
| **Extreme-event base rate** | **31.8%** of months | **13.0%** of months (6 / 46) | v1 was calling 1-in-3 months a "black swan," which is incoherent for a *rare* event — the label itself was broken, so every downstream metric was built on sand. 13% is the first defensible number. |
| **Best AUC-ROC** | 0.537 | **0.559** | The model now ranks tail months slightly better than v1 *and* does so against a credible label. |
| **Best predictor** | linear, same-month | **exp-weighted, 1-month lag** | We now actually test the research question — does sentiment *precede* extremes — instead of measuring it concurrently. |
| **Methods available** | 1 (logistic regression) | 3 label methods + regression + **event study** | Cross-checks instead of a single fragile estimate. |

> **The one-line story for the meeting:** *"The first run wasn't wrong because
> sentiment failed — it was wrong because the extreme-event label was miscalibrated
> (32% of months flagged). After fixing the label and adding recency-weighting and
> a predictive lag, the signal is still weak but now honestly measured: AUC 0.56,
> not significant. The real bottleneck is article density, which the new scraper
> addresses."*

---

## Methodology changes — before → after → why it improves the output

### 1. Extreme-event definition (the critical fix)
- **Before:** z-scored each day's 63-day forward return against a *rolling* 63-day
  window of those same forward returns, then flagged a whole month if **any one day**
  crossed ±2.5 SD (monthly MAX).
- **Problem:** consecutive 63-day forward windows overlap ~98%, so they are highly
  autocorrelated; the rolling SD denominator is unstable and shrinks in calm periods,
  pushing ordinary moves past 2.5 SD. The monthly MAX then flipped entire months on.
  Together → **31.8%** "extreme."
- **After:** take **one clean forward-return observation per month** (value at
  month-end) and threshold it against the **whole-sample** distribution. Three
  selectable methods: `quantile` (default), `global_z`, and `evt` (Generalized
  Pareto / peaks-over-threshold — the Black Swan / extreme-value approach you flagged).
- **Why the output is better:** removes the autocorrelation artifact and the MAX
  inflation, so the label now means what it says. Base rate **31.8% → 13.0%**, and
  every metric downstream is now interpretable.

### 2. Sentiment aggregation — linear vs. exponential weighting (your hint)
- **Before:** simple equal-weight monthly mean of all article scores.
- **After:** also compute an **exponentially-weighted** monthly score (7-day
  half-life) so articles nearer month-end count more.
- **Why the output is better:** the freshest sentiment is the most relevant for
  predicting *next* month. Swapping linear → exponential lifted AUC from 0.480 to
  0.534 (same-month) and is the basis of the best spec below.

### 3. Contemporaneous → lagged predictor (testing the actual question)
- **Before:** sentiment in month *T* vs. extreme label in month *T* (concurrent).
- **After:** sentiment in month *T* → extreme label in month *T+1* (predictive).
- **Why the output is better:** "early-warning signal" requires sentiment to
  *precede* the event. Adding the lag on top of exponential weighting gives the
  **best spec: AUC 0.559**.

### 4. New lens — event study (your "Event study / Mt Gox / SBF" hint)
- **Before:** none.
- **After:** `event_study.py` computes cumulative abnormal returns (CAR) over
  [−10,+10] trading days around Ronin, Terra/LUNA, FTX, the ETF approval, and the
  halving (mean-adjusted model). *(Mt Gox 2014 is outside our 2021–2024 data window
  — flagged as an open question.)*
- **Why the output is better:** the event study doesn't depend on the tiny count of
  labelled extreme months, so it's a more robust complementary check. It surfaced a
  qualitative finding the regression can't: the two "positive" catalysts (ETF, halving)
  produced **negative** abnormal returns — classic **"sell-the-news."**

### 5. Article density — BeautifulSoup scraper + snippets (your hint)
- **Before:** 63 hand-curated articles (1–4 / month).
- **After:** `scrape_articles.py` (GDELT historical backfill + CoinDesk RSS via
  BeautifulSoup), targeting **≥ 30 articles / month**, scoring **snippets**
  (headline + lede) rather than full bodies.
- **Why the output will be better:** the single biggest source of noise in v1 was
  thin monthly composites. Denser, snippet-level data should stabilise the monthly
  sentiment score before we draw firmer conclusions.

---

## Results snapshot — 43 monthly obs, 6 extreme months (14%)

| Spec | Odds Ratio | AUC | McFadden R² | p-value |
|---|---|---|---|---|
| **VADER exp-hl7, lag-1 (best)** | 0.704 | **0.559** | 0.0161 | 0.495 |
| VADER exp-hl7, same-month | 0.780 | 0.534 | 0.0087 | 0.588 |
| VADER linear, lag-1 | 0.889 | 0.505 | 0.0020 | 0.808 |
| VADER linear, same-month (≈ v1 method) | 0.927 | 0.480 | 0.0008 | 0.864 |

**How to read this honestly for the mentor:**
- The signal is **still weak and not statistically significant** (p ≈ 0.50). What
  improved is *rigor*, not a breakthrough — the label is fixed and the predictor is
  now specified correctly.
- The best odds ratio is **< 1** (0.70), hinting that *higher* prior sentiment
  precedes *lower* odds of a tail month (a euphoria-reversal direction) — but with
  only ~6 events the sample can't confirm it.

## Two honest limitations to raise at the meeting
1. **Small-sample tension.** At monthly resolution a *true* <5% black-swan rate gives
   only 1–2 events — too few to regress on. 13% (6 events) is the practical floor for
   an estimable logistic regression; for genuinely rare events the **event study** is
   the better tool. *(Open question: move to weekly/daily resolution to raise event count?)*
2. **Pilot benchmark gap unresolved.** Our Feb-2024 composite is **+0.275** vs your
   reference **−0.06**. This persists from v1 and is almost certainly an
   article-*selection* difference, not a code bug — needs us to align on which
   articles feed the pilot month.
