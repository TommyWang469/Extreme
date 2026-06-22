
---

## 1. What Sprint 2 set out to do


1. **Switch the sentiment model** 
2. **Create an volatility calculation** 
3. **Build a multi-factor model** 
4. **Firth penalized logistic regression**

---

## 2. What we did

- **FinBERT sentiment** replaces VADER — this is the sentiment *scorer* only; it does
  **not** change the number of extreme events.
- **Invented RWDV — Recovery-Weighted Downside Volatility**, a literature-scan-verified
  *original* metric. It is a downside-only (lower-partial-moment) volatility in which
  RWDV weights every trading day by how stubbornly its losses persisted. Up-days are ignored entirely (it is a downside-only, "lower partial moment" measure), and each down-day is assigned a weight between 0.1 and 1.0 based on its recovery speed: starting from that day's one-day drop, we define a half-recovery target halfway back up (for example, a fall from 100 to 92 sets a target of 96), then look forward up to ten trading days to see how long the price takes to reach it. If it recovers the very next day the weight is just 1/10 = 0.1, if it takes five days the weight is 0.5, and if it never claws back half the loss within ten days the weight is capped at 1.0. 
  flash crashes that bounce back are nearly ignored, sustained declines count fully.
- **Multi-factor weekly logistic regression**: 
  Using four predictors together — FinBERT news sentiment (the variable under test), RWDV (our recovery-weighted downside volatility, to control for volatility clustering), ARKF (the ARK Fintech ETF's trailing return), and QQQ (the Nasdaq-100's trailing return, for broad tech-market risk appetite).
- **Extended the data** from end-2024 to **mid-2026** and **backfilled the article
  corpus** so every month is covered.
- **Applied rare-event-correct statistics** 
  Firth penalized logistic regression. Ordinary logistic regression is biased when events are rare — it tends to push coefficients too far and report p-values that are too optimistic (or it can fail outright). Firth adds a tiny mathematical penalty (a "Jeffreys prior") that corrects this bias and produces stable, more honest estimates. It's the textbook fix for rare-event logistic regression. We implemented it from scratch so every step is auditable.

  Permutation test. Instead of trusting a formula for the p-value, this builds the "no relationship exists" world directly from your data: it randomly shuffles the crash labels thousands of times, refits the model each time, and checks how often a sentiment effect as large as ours shows up purely by chance. If 14% of random shuffles produce an effect that big, your p-value is 0.14. It makes almost no assumptions, so it's a reality check on the Firth result.

  Block-permutation test. A more honest version of the permutation test for time-series data. The catch with our predictors (RWDV is a 63-day rolling window, sentiment is a 4-week average) is that adjacent weeks are nearly identical — heavily autocorrelated — so although there are 269 rows, the number of truly independent observations is far smaller (~40–70). A plain permutation pretends every week is independent and therefore overstates significance. Block permutation shuffles contiguous chunks of weeks instead of individual weeks, preserving that week-to-week stickiness and giving the most honest p-value of the three.
- **Tested the regime hypothesis** (2021–22 vs 2023–26 split + a formal interaction test) 
  The regime hypothesis
  Crypto didn't behave like one consistent market across 2021–2026. The eras were different in character:

  2021–22 — "retail/chaos" regime: speculative, retail-driven, full of blowups (Terra/LUNA, FTX). Sentiment and herd behavior arguably drove prices.
  2023–26 — "institutional" regime: ETFs approved, big institutional players, more "mature" market.
  The hypothesis was that sentiment might predict crashes in one regime but not the other — e.g., news mood mattered in the wild early era and stopped working once the market institutionalized. (We initially even assumed "strong early, faded late.")

---

## 3. New files and what they are for

| File | Type | What it does |
|---|---|---|
| `finbert_scoring.py` | script | Scores every article with **FinBERT** (+ VADER for comparison); outputs weekly sentiment composites. |
| `build_features_v2.py` | script | Builds the **RWDV** metric, the **scar-event** labels, and downloads the **ARKF / QQQ** factors. |
| `analysis_v2.py` | script | The **multi-factor weekly logistic regression** (sentiment + RWDV + ARKF + QQQ), with ablations and out-of-sample test. |
| `analysis_firth.py` | script | **Firth penalized logistic regression** (rare-event-correct) + permutation test — hand-implemented so every step is auditable. |
| `analysis_regime.py` | script | **Regime split** (early vs late) with **block-permutation** and a formal **sentiment×regime interaction** test. |
| `data/sentiment_weekly.csv` | data | Weekly FinBERT + VADER sentiment composites. |
| `data/dataset_weekly.csv` | data | Weekly RWDV, scar labels, and ARKF/QQQ factors. |
| `sprint2_results.md` | output | The multi-factor logit results table. |
| `outputs/run_log_v2.txt` | output | Tidy run-log of the weekly model (tables only). |
| `outputs/firth_results.txt` / `.md` | output | Firth results — raw table (`.txt`) and interpretation (`.md`). |
| `outputs/regime_results.txt` / `.md` | output | Regime-split results — raw table (`.txt`) and interpretation (`.md`). |


---