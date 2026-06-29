# Project Context

## Research question
Does social media sentiment predict extreme crypto events?

## Project timeline — what we did, in order
*(Chronological record; doubles as the skeleton of the paper's Methods/Discussion.)*

1. **v1 (first pipeline).** Monthly VADER logistic regression. Found a broken extreme-event
   label (31.8% of months flagged "extreme") — fixed to one obs/month on the whole-sample
   distribution → ~13%.
2. **Sprint 1.** Added exponential weighting (7-day half-life) + lagged (predictive)
   sentiment; 3 label methods (quantile / global-z / EVT); a GDELT+CoinDesk scraper; and an
   event study (CAR around Ronin/Terra/FTX/ETF/halving → "sell-the-news"). Best AUC ~0.56–0.60,
   still not significant.
3. **Sprint 2 — model.** Switched VADER → **FinBERT**; invented **RWDV** (recovery-weighted
   downside volatility) + **scar-event** labels; built a **multi-factor weekly logit**
   (sentiment + RWDV + ARKF + QQQ). Resolution monthly → weekly (6 → 17 events).
4. **Sprint 2 — statistics.** Tested the best spec with **Firth** penalized regression +
   **permutation** + **block-permutation** → honest **near-null** (odds ≈ 1.5, Firth p = 0.028
   but permutation/block ≈ 0.12–0.14; OOS AUC ≈ 0.6).
5. **Regime test.** Split 2021–22 vs 2023–26 + interaction test → **no statistical regime
   shift** (p = 0.217); the earlier "strong early, faded late" story was retracted.
6. **Data backfill.** Extended prices + articles to mid-2026 → **66/66 months covered**
   (2,382 articles); the full-corpus re-run confirmed the near-null.
7. **Sprint 3 plan (next).** Event study (depegging) → walk-forward backtesting →
   **panel of coins + power analysis** → identification/feature work → new model families →
   economic-significance + honest-null reporting. See `improvement.md` §3 for the step order.

## Methodology 
- Portfolio: 50% BTC + 50% ETH, VWAP-based daily returns
- Sentiment tool: VADER (primary), TextBlob (validation)
- Article source: scraped corpus — 1,778 articles, 2021–2026 (GDELT + CoinDesk RSS); *(v1 used 12 hand-picked articles)*
- Extreme event definition (current, Sprint 2): weekly bottom-decile forward return **and** not 50%-recovered within 15 days ("scar event"); *(v1 "±2.5 rolling SD" was mis-calibrated and replaced in Sprint 1)*
- Model: logistic regression → odds ratio, AUC-ROC, McFadden R²
- Primary window: 3 months, robustness check: 48-hour window
- Framing: early warning signal, not causal claim

## Output metrics
- Odds ratio: exponentiate the logistic regression β₁ coefficient
- AUC-ROC: sklearn roc_auc_score
- McFadden R²: 1 - (log-likelihood model / log-likelihood null)

## Pilot period
February 2024, mentor's reference composite sentiment score was -6
*(Resolved in Sprint 2: on the dense scraped corpus our composite is +0.041 vs −0.06 — gap 0.101 ✓. The v1 gap of +0.275 was an article-selection artifact.)*

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

exp-hl7 — "exponentially weighted, 7-day half-life"
lag — "use last month's sentiment to predict this month"

### A. Logistic-regression metrics (sentiment → extreme event)

| Metric | What it indicates (reminder) | v1 (before) | Now (best spec) | Direction |
|---|---|---|---|---|
| **Odds Ratio** | How much the odds of an extreme month multiply per 1-unit rise in sentiment. **1.0 = no effect**; >1 = positive sentiment → more downward extremes; <1 = positive sentiment → fewer downward extremes. 
| **AUC-ROC** | How well sentiment *ranks* months by extremeness. **0.5 = random coin-flip**, 1.0 = perfect. | 0.537 | **0.600** | ⬆ **Better.** Moved from "barely above random" to "modest but real" ranking ability. |
| **McFadden R²** | Share of the variation in extreme months that sentiment explains. **0 = explains nothing**; >0.2 = strong fit for logistic models. | 0.0057 | **0.0270** | ⬆ **Better** (~5× higher), but still very low — sentiment alone explains very little. |
| **p-value (β₁)** | Chance the effect is just random noise. **<0.05 = statistically significant**; higher = can't rule out chance. | 0.583 | **0.380** | ⬆ Lower (better) but still **not significant** — we cannot yet claim the effect is real. |

**One-line conclusion** every metric moved in the right direction (AUC up, R²
up, p-value down, odds ratio now economically interpretable), but the result is still not statistically significant — the binding constraint is too few extreme events (only 6),
not data quality.

### B. Event-study metrics (CAR around 5 known shocks)

*CAR = cumulative abnormal return over the [−10, +10] trading-day window around each
event — i.e. how far the portfolio moved versus its normal baseline. `pre_sentiment` =
average sentiment in the days just before the event. This lens does **not** depend on the
tiny count of extreme months, so it's the more robust check.*

| Event | Date | Expected | CAR (actual move) | Pre-event sentiment |
|---|---|---|---|---|
| Ronin bridge hack | 2022-03-23 | − | **+0.330** (opposite) | +0.047 |
| Terra/LUNA collapse | 2022-05-09 | − | **−0.344** ✓ | +0.071 |
| FTX bankruptcy (SBF) | 2022-11-11 | − | **−0.321** ✓ | −0.059 |
| Bitcoin ETF approval | 2024-01-10 | + | **−0.079** (opposite) | +0.248 |
| Bitcoin halving | 2024-04-20 | + | **−0.234** (opposite) | +0.026 |

- **CAR matched the expected direction in 2 / 5 events.**
- **corr(pre-event sentiment, CAR) = +0.221** (n = 5 — illustrative only, far too few to test).
- **Qualitative finding the regression can't show:** the two "good news" catalysts (ETF
  approval, halving) both produced **negative** CAR — classic **"sell-the-news."**

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

*(Thin-corpus run, kept for the record. The dense-corpus re-run improved the best
AUC to 0.600 — those are the numbers in the Headline table above.)*

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
   *(✓ RESOLVED after the dense-corpus scrape: +0.041, gap 0.101 — selection artifact confirmed.)*

---

# Sprint 2/3 — FinBERT, the original RWDV metric, and the confirmation run (Jun 2026)

*Responding to the Jun-2026 mentor call: FinBERT, multi-factor model (ARKF/QQQ),
downside-only volatility (Sortino / lower partial moments), flash-crash vs.
sustained moves, and originality as the goal.*

## The original contribution (literature-scan verified)

- **RWDV — Recovery-Weighted Downside Volatility.** A lower partial moment where
  each down-day's squared return is weighted by how long the portfolio took to
  claw back half that day's loss (capped at 10 days). Flash crashes ≈ ignored;
  sustained declines count fully. Verified by literature search: recovery time
  exists only as a descriptive drawdown statistic — nobody uses it to weight a
  volatility measure or define the event label. This is ours.
- **Scar-event label.** Extreme-DOWN week = bottom-decile forward return AND not
  50%-recovered within 15 days of the trough. The scar filter removed 10 of 28
  decile weeks as flash crashes — the label now matches the mentor's "a one-day
  dip is not an extreme move."

## Setup (mentor's whiteboard)

Weekly logit: P(scar event in week w+1) = f(FinBERT sentiment, trailing RWDV,
ARKF 4-week return, QQQ 4-week return) — all at week w. FinBERT replaced VADER
(article-level correlation between the two: only 0.35). Data: 2021-01 → 2026-06.

## Results (pre-registered confirmation run, extended sample)

| | n | events | OR (sentiment) | p | AUC | OOS AUC |
|---|---|---|---|---|---|---|
| Sprint-2 exploratory (2021–24) | 197 | 12 | 1.70 | 0.120 | 0.662 | 0.69 |
| **Confirmation (2021–26)** | **200** | **13** | **1.70** | **0.108** | **0.660** | **0.670** |

- **The effect size replicated exactly (OR 1.70)** — higher *sustained* news
  optimism precedes scarring crashes (euphoria-reversal / sell-the-news), the
  same direction the event study found.
- **Not yet significant at 0.05** (p = 0.108). Honest framing per the mentor:
  in this domain robust p-values are hard with ~13 events; the stable effect
  size + out-of-sample AUC ≈ 0.67 is the stronger evidence.
- FinBERT beats VADER head-to-head (p 0.11 vs 0.40 on the same controls).
- ~~Remaining bottleneck: the article corpus covers only 50 of 65 months.~~
  **UPDATE (Sprint 3, 2026-06-22):** this is now resolved — `articles_scraped.csv`
  covers **66 of 66 months** (2021-01 → 2026-06), 2,382 articles, median ~36/month.
  Only the final month **2026-05 is thin (3 articles)**, which is GDELT recency lag,
  not a gap. So article *coverage* is no longer the bottleneck; the open data levers
  are now **per-coin news** (to unlock the panel) and **more events** (daily / longer
  history), not back-filling months. See Step-4: news *volume* is the strongest predictor.

# Sprint 3 — Event study + walk-forward harness (Jun 2026)

*Execution order per `improvement.md` §3. Steps 1–2 done; Step 3 (panel + power) next.*

## Step 1 — Event study (`event_study.py` → `outputs/event_study.{txt,md}`)

Rebuilt on the full daily data (2021-01 → 2026-06) with proper few-event statistics.
Two samples: **A) named exogenous catalysts** (9; China ban, Terra/UST de-peg,
Celsius/3AC, FTX, USDC/SVB de-peg, carry unwind, ETF approval, halving, Trump win) —
non-circular, used for inference; **B) scar-decile** (objective bottom/top-10% weekly
returns, clustered; 17 down + 16 up) — used for recovery asymmetry + the sentiment
regression, with the selection bias flagged. Five tests per window (Patell, BMP,
generalized sign, Corrado-Zivney rank, bootstrap CI); significant only when a
parametric **and** a non-parametric test agree at 5%.

- **Post-event drift is null.** Named down-catalysts, window [+1,+10]: mean CAR −0.026,
  bootstrap CI [−0.124,+0.094], no test significant. The crash is contemporaneous; no
  robust continuation or rebound. Pre-event window [−10,−1] is significant (−0.127) —
  these catalysts leaked/unfolded over days.
- **Sell-the-news direction, weak.** CAR-on-sentiment slope negative (named −0.56,
  decile-down −0.16) but not significant (perm p = 0.81 / 0.59). Same suggestive-not-
  significant story as the logit.
- **De-peg contrast (answers §1a's "how long does a depeg last"):** algorithmic
  **Terra/UST never recovered** within 180 d (−41%); collateralized **USDC recovered
  50%/80% in 3–4 d** with a positive post-event CAR (+0.235).
- **Recovery asymmetry (RWDV bridge):** named downside events take ~53 d to recover
  half the drawdown, ~76 d for 80%; 7 of 17 decile-down events don't recover 80% in 180 d.

## Step 2 — Walk-forward harness (`analysis_walkforward.py` → `outputs/walkforward.{txt,md}`)

Closes the two look-ahead leaks in the Sprint-2 pipeline: the **scar decile threshold**
(was full-sample) and **feature normalization** (was full-sample) are now computed on the
**training window only**. Single 2022 split → expanding-window, one-step-ahead refit with
exponential decay. 205 OOS weeks (2022-06 → 2026-05), 7 events.

| spec | OOS AUC | PR-AUC | note |
|---|---|---|---|
| C0 FinBERT-4w + RWDV + ARKF + QQQ | 0.578 | 0.048 | primary |
| A1 FinBERT-4w alone | 0.584 | 0.048 | ≈ full model |
| A2 VADER-4w + controls | 0.553 | 0.041 | FinBERT > VADER again |
| A3 controls only (no sentiment) | **0.439** | 0.037 | below chance |

- **Look-ahead optimism = +0.063 AUC** (in-sample leaky 0.641 → leak-free 0.578); the
  leak-free number matches the prior ~0.6.
- **Sentiment carries all of the (weak) signal:** C0 − A3 = +0.139 AUC; FinBERT-alone
  matches the full model, so RWDV/ARKF/QQQ add nothing out-of-sample.
- **Exponential decay hurts:** AUC rises monotonically with half-life (26w 0.439 →
  no-decay 0.627). With ~17 events the data wants long memory; reported the sensitivity
  curve, did not tune to the best.
- **No sharp-end value:** precision@K = 0 — the highest-probability weeks miss every
  crash. AUC > 0.5 is mid-ranking skill, not a usable trigger (matters for Step 6).
- Harness is frequency/panel-agnostic — daily (§G2) or a coin panel (§G1) drop into the
  same loop for Step 3.

## Step 3 — Panel of coins + power analysis (§G1–G2)

**Power analysis** (`analysis_power.py` → `outputs/power.{txt,md}`). At the observed base rate
~0.065, the weekly single series has a **minimum detectable OR ≈ 2.1** (80% power) — far above
the real ~1.5–1.7. Detecting OR 1.5 needs **≈ 779 independent obs** (~3× current); the panel
would need **≈ 42 coins at ICC 0.05** (~80 at 0.10). So the Sprint-2 non-significance is
largely a **power problem, not a null** (§G7).

**Panel logit** (`analysis_panel.py` → `outputs/panel.{txt,md}`). 11 coins incl. delisted LUNA
(survivorship-corrected), **142 scar-down events** (vs ~17 single-series). Sentiment OR **1.39**
— same euphoria / sell-the-news direction as Sprint-2's 1.70.

- Honest SE is **two-way clustered (coin × week)**: CI **[0.90, 2.13], p = 0.136** → per the
  pre-reg decision rule this is **qualified support — suggestive, underpowered, not confirmation
  and not a null.**
- The by-coin-only CI [1.15, 1.67], p = 0.001 is a **Moulton trap**: sentiment is one *shared*
  time series, so by-coin clustering treats the same path as 11 independent draws. Do not report
  that p as the headline.
- **Key finding:** the panel does **not** escape the power ceiling for the *shared* sentiment
  regressor — its effective N stays ~the number of weeks (~269), not coin-weeks (~2771). More
  coins help coin-varying predictors (RWDV, crash frequency) but not the shared-sentiment link.
  **The real unlock is per-coin sentiment.**
- Pre-registration frozen in `PREREGISTRATION.md` before this step ran.

## Step 4 — Identification: momentum control (§G3)  [trimmed]

`analysis_identification.py` → `outputs/identification.{txt,md}`. **Trimmed (2026-06) to the
one load-bearing check — the momentum control** (answers the reviewer's "isn't your sentiment
just price momentum?"):

- **Not a momentum proxy.** Sentiment OR 1.53 is unchanged after adding lagged 1-week and
  trailing-4-week returns; momentum itself is not significant. Sentiment carries information
  independent of recent price action.

*Removed to keep the project lean:* the **lead-lag / Granger** test (it found sentiment mostly
*lags* returns on the mean — but that's the wrong test for a *tail* claim) and the
**volume/disagreement** predictors. The **news-volume finding is preserved and stronger**
elsewhere: severity model `outputs/step5_models.*` (p=0.001) and the walk-forward volume add-on
`outputs/costsensitive.*`. (Disagreement was a flat null.)

## Per-coin news experiment (§G1 unlock — tested, did NOT unlock)

To break the shared-sentiment Moulton ceiling, scraped **coin-specific** GDELT news for 7 coins
(`scrape_news_percoin.py` → `data/articles_percoin.csv`, ~30k articles after a gap-fill re-run),
then re-ran the panel with per-coin sentiment, holding the scorer constant.

Now consolidated into **`analysis_panel.py` Part B** (FinBERT only; output in `outputs/panel.*`).
The interim VADER per-coin run was null (OR 0.89) but VADER on short headlines is noisy, so it
was **retired**. The **FinBERT per-coin** result (scores cached in
`data/articles_percoin_finbert.csv`): partial coverage OR 0.94, but after the gap-fill
**OR 1.19 ≈ shared 1.22** — same weak euphoria direction, CI *wider* not tighter.
- **Conclusion:** per-coin news did **not** unlock extra power; per-coin estimates are
  coverage-sensitive and noisy at this N. Can't separate market-wide from coin-specific — both
  ~1.2, both non-significant. The binding constraint is **events/power, not sentiment
  construction**. (My interim "settled: market-wide" claim was overturned by the gap-fill — a
  lesson in not trusting a single underpowered run.)

## Step 5 — Models on the credible setup (§1c, §G4, §1e, §1d)

- **News-volume OOS check** (`analysis_costsensitive.py` → `outputs/costsensitive.*`): adding news
  volume to C0 lifts the leak-free walk-forward **AUC 0.479→0.664** — the project's **strongest
  volume evidence** (the severity finding confirmed out-of-sample). *Slimmed (2026-06):* the §1c
  class-weighting experiment (no ranking gain; trades precision for recall) is now a one-line
  footnote, not a result.
- **§G4 severity** (`analysis_step5_models.py` → `outputs/step5_models.*`): in the quantile
  (severity) regression **news volume is significant at the tail (coef −0.025, p=0.001 at q=0.10;
  also sig. at q=0.05 and 0.25)** while polarity is borderline — the continuous target is what let
  volume reach significance (in-sample; also survives the walk-forward). *Trimmed (2026-06):* the
  GBT benchmark (§1e — doesn't beat the logit) is a one-line footnote; the null hazard model and
  the GARCH-EVT tail model (§1d) were **dropped** to keep the project lean (GARCH-EVT was a working
  result — fat tails + calibrated VaR — but the most tangential to the sentiment hypothesis).

**Cross-cutting result of Steps 4–5:** across the logit, the volume OOS check, the severity model,
and the tree importance, **news attention/volume — not sentiment polarity — is the predictor that
survives**, and it reaches significance only with a richer (continuous) target.
