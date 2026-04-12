# Project Context

## Research question
Does social media sentiment predict extreme crypto events?

## Methodology (agreed with mentor)
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
