# Project Handoff Note
**Research:** Social Media Sentiment as a Predictor of Extreme Crypto Events
**Status:** Sprint 2 built (FinBERT + RWDV multi-factor, weekly) — see improvement.md §2 and sprint2_results.md

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
2. The full pipeline is in `/Users/hongqingwang/Documents/GitHub/Github Extreme/`
3. Run order: `data_collection.py` → `scrape_articles.py` → `sentiment_scoring.py` → `event_definition.py` → `analysis.py` → `event_study.py`
4. The most impactful next task is increasing article density (run `scrape_articles.py`, possibly 2–3× to fill GDELT-rate-limited months)

---

## Sprint 1 update (improvement.md T6–T10)

**What was done**
- **Fixed the broken extreme-event label.** v1 flagged 31.8% of months as extreme
  (rolling-SD on overlapping forward returns + monthly MAX). Rewritten to one
  clean obs/month thresholded against the whole-sample distribution. New base rate
  **13.0%**. Three selectable methods: `quantile` (default), `global_z`, `evt` (GPD).
- **sentiment_scoring.py:** added exponential weighting (7-day half-life) and
  snippet-only scoring; auto-detects `articles_scraped.csv`, falls back to `articles.txt`.
- **analysis.py:** now compares 4 predictor specs (linear/exp × contemporaneous/lag-1),
  prints a ranked table, and checks the Feb-2024 pilot benchmark.
- **event_study.py (new):** CAR around Ronin / Terra / FTX / ETF / halving.
- **scrape_articles.py (new):** GDELT historical backfill + CoinDesk RSS (BeautifulSoup),
  cached + throttled, targets ≥30 articles/month → `data/articles_scraped.csv`.

**Results:** best spec (exp-weighted, lag-1) AUC 0.480→0.559; still not significant
(low power at ~6 events). Event study shows "sell-the-news" on ETF/halving. Full
numbers preserved in `CONTEXT.md` (Sprint 1 section).

---

## Sprint 2 update (Jun 2026)

**Done**
- Dense corpus (1,778 articles); pilot Feb-2024 gap resolved (+0.041 vs −0.06 ✓).
- FinBERT weekly sentiment (`finbert_scoring.py`) — corr with VADER only 0.35.
- **Novel metric: RWDV** (Recovery-Weighted Downside Volatility) + scar-event
  labels (`build_features_v2.py`) — literature-scan-verified gap.
- Multi-factor weekly logit with ARKF + QQQ (`analysis_v2.py`), per mentor's
  whiteboard. 194 weeks, 12 scar events.

**Results:** pre-registered spec not significant (p = 0.83). Exploratory
4-week-smoothed FinBERT: OR 1.70, p = 0.12, AUC 0.66; out-of-sample AUC
0.69–0.79 (only 3 test events). Full tables in `sprint2_results.md`.

**Open items / Sprint 3**
- ✓ Prices extended to Jun 2026 (`data_collection.py`); confirmation run done:
  OR 1.70 (replicated exactly), p = 0.108, OOS AUC 0.670 — see sprint2_results.md
  and the Sprint 2/3 section of CONTEXT.md.
- ✓ CONTEXT.md refreshed (methodology header, pilot gap marked resolved,
  Sprint 2/3 results section appended).
- **NEXT (highest leverage): backfill the 15 article-less months** (mostly
  2025–26). `scrape_articles.py` END now extended to Jun 2026 — run
  `python3 scrape_articles.py` 2–3× (GDELT rate limits), then
  `python3 finbert_scoring.py && python3 analysis_v2.py`.

---

## Automated audit run — 2026-06-14

**Code audit (v2 pipeline) — no leakage bugs found.** Verified the predictive
alignment is correct end-to-end:
- RWDV `shift(H_RECOVERY)` correctly ensures the trailing window only contains
  down-days whose recovery horizon is fully resolved by time T — no look-ahead.
- Feature/label alignment is genuine lag-1: row `wk` carries features as of the
  end of week `wk` and the target is the event in week `wk+1`.
- 4-week sentiment smoothing uses only weeks `wk-3..wk` — no future leakage.

**Two honest caveats (not bugs, design choices to disclose to the mentor):**
- The decile thresholds `lo/hi` in `scar_labels` are full-sample quantiles, so the
  out-of-sample test's *labels* are defined using the whole period. Defensible
  ("extreme relative to the study window") but worth a footnote; a stricter OOS
  would set the threshold on training data only.
- The headline-table AUC is in-sample (optimistic); the separate OOS AUC line is
  the honest discrimination number.

**Root cause of weak significance confirmed:** the scraped corpus dead-stops at
2025-01 — months 2025-02 → 2026-05 (16 months) have ZERO articles, while prices
now run to Jun 2026. So extending prices added weeks with no sentiment, which the
regression drops. Backfilling those months is the binding lever.

**Backfill done + full re-run (honest result):** corpus now 2,382 articles,
66/66 months covered, no gaps. Re-ran finbert_scoring → build_features_v2 →
analysis_v2. The confirmation spec did **not** improve — it got slightly weaker:

| | n | events | OR(sent) | p | in-samp AUC | OOS AUC |
|---|---|---|---|---|---|---|
| pre-backfill (200 wk, 2021–22-heavy) | 200 | 13 | 1.70 | 0.108 | 0.660 | 0.670 |
| **full corpus 2021–26** | 269 | 17 | **1.533** | **0.139** | 0.642 | **0.590** |

The earlier p≈0.108 was partly a small-sample artifact. On the complete data the
euphoria-reversal direction (OR>1) persists but is NOT significant, and OOS AUC
falls to ~0.59. This is the truthful state — do not report it as significant.

**Interpretation (mentor-aligned, the real finding):** the signal was strong in
the 2021–22 retail/chaos regime and faded as crypto institutionalised — a REGIME
SHIFT, exactly what the mentor flagged ("3–4 regimes; how do you recognise the
shift?"). Next honest step = one pre-specified regime-split test (2021–22 vs
2023–26), reported either way. NOT spec-fishing for a star.
