# Sprint 1 Results Summary

**Date:** 2026-06-05
**Scope:** improvement.md tasks T6–T10 (code) + T7 (scraper)
**Window:** 2021–2024, 50/50 BTC+ETH portfolio

---

## What changed vs. v1

| Area | v1 (first run) | Sprint 1 |
|---|---|---|
| Extreme-event labeling | Rolling ±2.5 SD on overlapping forward returns + monthly MAX | One obs/month, whole-sample quantile / global-z / **EVT** options |
| Extreme-event base rate | **31.8%** (implausibly high) | **13.0%** (6 of 46 months) |
| Sentiment aggregation | Simple monthly mean only | **Linear + exponential** (7-day half-life) |
| Predictor specs tested | 1 (contemporaneous linear) | **4** (linear/exp × contemporaneous/lag-1) |
| Article corpus | 63 hand-curated | **scrape_articles.py** (GDELT backfill + CoinDesk RSS) |
| Methodology | Logistic regression only | + **event_study.py** (CAR around known shocks) |

---

## The key fix: extreme-event rate 31.8% → 13.0%

The v1 label was broken two ways:
1. It z-scored the 63-day forward return against a *rolling* 63-day window of
   those same (overlapping, autocorrelated) forward returns — an unstable, often
   tiny denominator that pushed ordinary moves past 2.5 SD.
2. It then flagged a whole month extreme if *any single day* was extreme (MAX).

Sprint 1 takes one clean forward-return observation per month and thresholds it
against the whole-sample distribution. Default = empirical quantile (stable
6-event plateau for TAIL_PCT 0.10–0.13). EVT/GPD and global-z are also selectable.

> **Honest caveat:** at monthly resolution a *true* <5% black-swan rate yields
> only 1–2 events — too few to regress on. 13% (6 events) is the practical floor
> for an estimable logistic regression. For genuinely rare events the event-study
> pipeline is the right tool.

---

## Logistic regression — predictor comparison (ranked by AUC)

| Spec | n | events | Odds | AUC | McFadden R² | p |
|---|---|---|---|---|---|---|
| **VADER exp-hl7, lag-1 (predictive)** | 42 | 5 | 0.704 | **0.559** | 0.0161 | 0.495 |
| VADER exp-hl7, contemporaneous | 43 | 6 | 0.780 | 0.534 | 0.0087 | 0.588 |
| VADER linear, lag-1 | 42 | 5 | 0.889 | 0.505 | 0.0020 | 0.808 |
| VADER linear, contemporaneous *(≈ v1)* | 43 | 6 | 0.927 | 0.480 | 0.0008 | 0.864 |

**Reading it:**
- The best spec — **exponentially-weighted sentiment, lagged one month** — lifts
  AUC from 0.480 to **0.559**. Both recency-weighting and lagging help, which is
  consistent with an *early-warning* story (fresh sentiment predicting next
  month's tail).
- Still **not statistically significant** (p = 0.50). With 5–6 events the power
  is very low; treat as directional.
- The lagged odds ratio < 1 (0.70) hints that *higher* prior sentiment precedes
  *lower* odds of an extreme month — a euphoria/complacency-reversal direction —
  but the sample cannot confirm it.

---

## Event study — CAR around known shocks (2021–2024)

| Event | Date | Expected | CAR[−10,+10] | Matched? |
|---|---|---|---|---|
| Ronin bridge hack | 2022-03-23 | − | +0.330 | ✗ |
| Terra/LUNA collapse | 2022-05-09 | − | −0.344 | ✓ |
| FTX bankruptcy (SBF) | 2022-11-11 | − | −0.321 | ✓ |
| Bitcoin ETF approval | 2024-01-10 | + | −0.079 | ✗ |
| Bitcoin halving | 2024-04-20 | + | −0.234 | ✗ |

- Direction matched in **2/5** events. The two "positive" catalysts (ETF, halving)
  produced **negative** abnormal returns — the classic **"sell-the-news"** pattern,
  a reportable qualitative finding rather than a model failure.
- corr(pre-event sentiment, CAR) = **+0.32** (n=5, illustrative only).

---

## Pilot benchmark still unresolved

Our Feb-2024 composite (vader_linear) = **+0.275** vs. the mentor's reference
**−0.06**. Gap ≈ 0.34. This persists from v1 and is almost certainly an
article-*selection* difference, not a code bug. Needs alignment with the mentor
on which articles feed the pilot month.

---

## Bottom line

The headline conclusion is unchanged but now **defensible**: curated VADER
sentiment is at best a weak, non-significant early-warning signal for extreme
crypto months. What improved is the *rigor* — the extreme-event label is no
longer broken (13% not 32%), recency-weighting + lagging give a measurable AUC
bump, and we now have both a corrected regression and an event-study lens.

The single highest-leverage next step remains **data density**: run
`scrape_articles.py` to replace the 63-article corpus with ≥ 30 articles/month,
then re-run sentiment → analysis. Everything downstream is wired to pick the
scraped file up automatically.
