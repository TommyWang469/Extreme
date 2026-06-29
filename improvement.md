# Sprint 3 Plan — Model Revision & Event Study (paper comes later)

*Focus per the Jun-19 2026 mentor meeting + this round's literature search: **revise the
model and implement the event study first**; the research paper is deferred until the
modeling is solid. Honesty (walk-forward, pre-registration, no p-hacking, report negative
results) runs through everything.*

> **Picking this up in a new session?** Go straight to **§3 — Step-by-step execution order**
> and begin with **Step 1 (event study)**. §1/§1.5 explain *why* each step; §3 is the *do-this-next* list.

---

## 0. Where we are

Sprint 2 built an original method — RWDV (recovery-weighted downside volatility),
scar-event labels, FinBERT sentiment, a multi-factor weekly logit, and rare-event-correct
stats (Firth + permutation + block-permutation). Honest result: a **near-null** —
sentiment→scarring-crash odds ≈ 1.5 (suggestive, euphoria/sell-the-news direction) but not
robustly significant (Firth p = 0.028 vs permutation/block ≈ 0.12–0.14; OOS AUC ≈ 0.6).
Sprint 3 = better models + the event study, to get more robust, better-understood results.

---

## 1. PRIORITY — model revision & event study

Ordered by leverage. Each item: what, why it can help, reference, honest expectation.

### 1a. Implement the event study (NOT done yet — mentor's priority before the paper)
The mentor's preferred low-event tool; its stats don't depend on the 17-event logit, so it's
the best shot at clean, independent evidence.
- `event_study.py` exists (Sprint 1) but is **stale** (≤2024 events, not re-run on extended data).
- Add **stablecoin de-pegging** events (Terra/UST May-2022 + others) — a depeg is by
  definition an extreme event, and the "how long does the depeg last" debate maps directly
  onto our recovery-time heuristic. Link historically to money-market funds "breaking the
  buck" (2008). Ref: Coinbase "why do stablecoins depeg."
- Event selection by an **objective rule** (named catalyst OR weekly drawdown past the scar
  decile); target ≥ 8–10 events, up and down.
- Stats for few, fat-tailed events: variance-robust standardized tests (Patell,
  Boehmer–Musumeci–Poulsen) **plus** non-parametric (sign, Corrado rank) + bootstrap CI;
  significant only when both agree. Regress each event's CAR on pre-event sentiment; measure
  recovery time (50%/80%) to bridge back to RWDV.
- *Expectation:* likely corroborates the descriptive findings (sell-the-news, recovery
  asymmetry) more cleanly than the regression — a real, reportable result.

### 1b. True walk-forward backtesting (the mentor's #1 rigor ask)
Replace the single train≤2022 split with expanding-window walk-forward + exponential decay
of older data; **strict no look-ahead, including normalization and the decile thresholds**
(compute on the training window only). This is the foundation every model below is scored on.
Refs: StrategyQuant walk-forward optimization; Interactive Brokers walk-forward analysis.

### 1c. Handle the rare-event class imbalance (high-leverage, under-used so far)
Ordinary logistic regression **systematically underestimates** rare-event probability — the
core reason a real weak signal stays hidden. Fixes: **class weights / cost-sensitive loss**
(weight the ~6% crash class up), and we already added Firth. Try cost-sensitive logistic and
report recall/precision on crashes, not just AUC.
Refs: "Weighting methods for rare event identification" (PMC8734962); "Learning Rare Events:
Deep Learning for Extreme Price Prediction" (Forecast 2025, doi:10.3390/forecast8030052).

### 1d. GARCH-EVT tail model (domain-standard, strong originality fit)
Pair an **asymmetric GARCH** (EGARCH / GJR-GARCH — captures the "downside vol clusters more"
effect, same spirit as RWDV) with **Extreme Value Theory** (Generalized Pareto on the tail,
the Peaks-Over-Threshold approach we tried in Sprint 1 but which was unstable monthly — viable
at weekly/daily). Gives a principled crash-probability that we can compare to / combine with
the sentiment model. Refs: eGARCH-EVT-Copula for crypto (arXiv 2407.15766); POT prediction
(arXiv 2504.04602); GARCH-family vs deep learning for crypto vol (mf-journal #370).

### 1e. Nonlinear model benchmark — LightGBM / XGBoost (mentor suggested)
Gradient-boosted trees consistently top crypto forecasting horseraces and capture
interactions a linear logit can't (e.g., "high sentiment AND low volatility"). Run under the
same walk-forward CV; **pre-register, don't p-hack** — expect a slight bump, and use it to see
*which variables* matter (feature importance), not just to chase AUC. Refs: LightGBM crypto
forecasting + anomaly detection (MDPI Appl. Sci. 15/4/1864); Köse 2025 (J. Forecasting).

### 1f. Better features (feed every model above)
- **Combine QQQ + ARKF into one factor** (~0.4 QQQ / 0.6 ARKF) — they're correlated.
- **Recode the target for nonlinear extreme-loss weighting** — blow up losses beyond ±2 SD;
  leverage-aware (50% DD @ 2× = wipeout). Mentor: "change the penalty by recoding the output."
- Add candidate predictors: trading **volume**, **funding rates**, **VIX**, Google Trends.
- *(Optional)* **CryptoBERT** instead of FinBERT — reads crypto slang; one clean swap to test
  whether a domain sentiment model moves anything.

### 1g. Self-exciting tail clustering — Hawkes / 2T-POT-Hawkes *(stretch)*
Models extreme events as self-exciting clusters (one crash raises the odds of the next) — an
original complement to RWDV's clustering idea; test whether sentiment shifts the intensity.
Ref: 2T-POT Hawkes for left/right-tail quantiles (arXiv 2202.01043).

---

## 1.5 Deeper research gaps — usually HIGHER leverage than more models

§1 mostly swaps in fancier models for a slight bump. But with ~17 events, the limiting
factors are **data, identification, and information content — not model class.** A careful
reviewer would push on these first:

**G1. Expand the cross-section (panel of coins) — the #1 power lever.** Today it's one
BTC+ETH series → ~17 events. Run the same design across many coins (BTC, ETH, SOL, top-N),
pooled into a **panel / mixed-effects logistic regression** with coin fixed effects and
SEs clustered by coin. This turns ~17 events into *hundreds* — the single biggest legitimate
route to statistical power. Include **delisted/dead coins (LUNA)** to avoid survivorship bias.
*Refinement (added Sprint-3 Step 3, after running it):* the panel multiplies independent
events only for **coin-varying** predictors (per-coin RWDV, crash frequency). For the **shared
market-sentiment** regressor it does **not** — sentiment is identical across coins each week, so
its effective N stays ~the number of weeks (the **Moulton problem**), and clustering by coin
alone badly understates its SE (use **two-way coin × week**). Confirmed empirically: 142 pooled
events but sentiment OR 1.39 with a two-way CI [0.90, 2.13] (p = 0.14) — still underpowered.
**The real unlock for the *sentiment* link is per-coin sentiment, not just more coins.**

**G2. Power analysis + go daily.** Compute the **minimum detectable effect / required event
count** for ~80% power at odds ≈ 1.5 — quantify how underpowered weekly is instead of just
asserting it. **Daily** resolution raises power *and* makes GARCH-EVT / POT actually estimable
(they need many tail observations).
*Caveat (added Sprint-3 Step 1):* going daily inflates the **nominal** event count but not the
**independent** one — daily tail days cluster (in current data, the 100 bottom-5% days form only
~65 distinct clusters and |return| autocorrelation ≈ 0.27 at lag 1), so discount with
block / cluster-robust SEs and always report the **effective** N, not the raw count. Note too
that the event count is partly a **labeling choice**: the strict scar-binary gives ~17 weekly
events while the bottom-decile rule gives ~29 — loosening the threshold (still principled) is
itself a small power lever, and the headline "17 events" should be stated as threshold-dependent.

**G3. Identification — control for momentum / past returns.** Test whether sentiment is just
a proxy for recent price action: add **lagged returns/volatility** as controls and run a
**lead–lag (Granger) test** to confirm sentiment *leads* rather than *lags*. Without this,
"prediction" may be spurious. (This is the difference between *associated* and *predictive*.)

**G4. Richer target than binary.** Also model crash **severity** (continuous tail loss) or a
**survival / hazard** model for time-to-next-extreme — both use more information per
observation and sidestep the rare-event binary ceiling.

**G5. Use the sentiment signal we discard.** We compute `n_articles` but only use polarity.
Add **news volume (attention spikes)** and **disagreement (dispersion across articles)** as
predictors — both are documented volatility predictors, often stronger than mean sentiment,
and we already have the data.

**G6. Economic-significance test.** Translate the signal into a simple **sentiment risk-off
rule** and report Sortino/Sharpe and drawdown avoided — connects the statistics to the
mentor's trading lens, and is a result even if the p-value is weak.

**G7. Report the null rigorously.** Report the **odds-ratio confidence interval** and the
**minimum detectable effect**, not just "p = 0.14." A wide CI (e.g. [0.8, 2.8]) means
"can't rule out a real effect," which is very different from "no effect."

**G8. Operationalize pre-registration.** Write and freeze an actual pre-registration file
(hypotheses, primary spec, analysis plan) *before* running Sprint 3 — otherwise it's just a word.

**G9. Extend the price history backward — same coins, free *independent* events (added
Sprint-3 Step 1).** `data_collection.py` hard-codes `START="2021-01-01"`; BTC-USD daily exists
back to ~2014 and ETH-USD to ~2015-08. Re-pulling to ~2017 (or 2015) adds genuinely independent
regimes/crashes — 2017 mania, 2018 bear market, the Mar-2020 COVID crash — **without** the
survivorship and cross-coin heterogeneity that the panel (G1) introduces. *Limit:* the
news/FinBERT data only reaches 2021, so this powers the **price-side** tail models (EVT, GARCH,
the event study, unconditional crash frequency) but **not** the **sentiment→crash** link unless
articles are backfilled. Cheaper than G1; do it alongside going daily (G2).

> **Meta-point:** items G1–G3 (cross-section, power/daily, momentum control) would do more
> for the result's credibility than GARCH-EVT or LightGBM. Do them first.
> **Sequencing (added Sprint-3 Step 1):** the **walk-forward harness (Step 2) is the measurement
> instrument** — build it *before* scaling data, or more events just produce confident
> look-ahead-biased numbers. Among the power levers, do the **cheap same-coin ones first**
> (go daily G2, richer target G4, backward history G9); the **coin panel (G1) is the only lever
> that multiplies *independent sentiment-linked* events** — highest power but highest cost
> (survivorship, heterogeneity, per-coin news), so it's the *last* of the data levers, not the first.

---

## 2. Research paper — LATER (don't focus on it yet)

After the modeling above is solid. When the time comes: Overleaf/LaTeX; sections Introduction ·
Background/Related Work · Data & Methods · Results · Discussion · Conclusions(+future) ·
References · Appendices; pre-register; APA citations (Purdue OWL); report negative results.
Venues: ICAIF 2026 **workshop** (icaif2026.org; ~Aug submission, virtual option),
Scholarly Review, Curieux Academic Journal. *(Detail kept brief on purpose — revisit after Sprint 3 modeling.)*

---

## 3. Step-by-step execution order (START HERE in a new session)

**Step 1 — Event study.** Re-run `event_study.py` on current data; add stablecoin-depeg
events (Terra/UST + others) and 2025–26 events by an objective rule (target ≥ 8–10 events);
add Patell/BMP + sign/Corrado tests + bootstrap CI; regress each event's CAR on pre-event
sentiment; measure 50%/80% recovery time. → `outputs/event_study.{txt,md}` + CAR plot. (§1a)
*First command: `python3 event_study.py`, then extend it.*

**Step 2 — Walk-forward harness.** Expanding window + exponential decay; compute all
normalization **and** the scar decile thresholds on the *training window only* (no
look-ahead). Re-score the current model under it. (§1b)

**Step 3 — Panel of coins + power analysis** (the real fix for too few events). Pull top-N
coins incl. delisted ones (LUNA); rebuild the weekly features per coin; pooled
panel / mixed-effects logit with coin fixed effects + clustered SEs. Compute the minimum
detectable effect / events needed for 80% power at odds ≈ 1.5. (§G1–G2)
*Do the cheaper same-coin power levers first/alongside — go daily (§G2), extend BTC+ETH history
back to ~2017 (§G9), richer target (§G4) — and report **effective** (cluster-discounted) N.*

**Step 4 — Identification + free signal.** Add lagged-return / momentum controls + a
lead–lag (Granger) test; add news-volume and disagreement predictors. (§G3, §G5, §1f)

**Step 5 — Models on the credible setup.** Class-imbalance / cost-sensitive logit, richer
target (severity / hazard), then GARCH-EVT and LightGBM. (§1c, §G4, §1d, §1e)

**Step 6 — Economic significance + honest null.** Sentiment risk-off backtest
(Sortino/Sharpe, drawdown avoided); report odds-ratio CI + minimum detectable effect.
Hawkes if time. (§G6–G7, §1g)

**Before Step 3 runs:** freeze a pre-registration file (hypotheses + primary spec). (§G8)

---

## 4. Honesty guardrails
Walk-forward with **no look-ahead** (incl. normalization & thresholds) · **pre-register**
each experiment · **no p-hacking** (no run-many-report-best; correct for multiple testing if
many) · **negative results are valid** · always report n, event count, recall on the crash
class, and the autocorrelation-robust p. With ~17 events the ceiling is real — the aim is
robustness + understanding (which variables/models matter and why), not a forced p < 0.05.

---

## 5. Timeline
Next mentor meeting **Jul 1, 2026 (1 PM PT / 4 PM ET)**; user at **COSMOS camp Jul 5 – Aug 1**.
Pre-Jul-5 priority: **event study + walk-forward harness** running, with the new model
families scaffolded.
