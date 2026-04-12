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
