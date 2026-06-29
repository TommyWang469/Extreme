# Project Handoff Note

**Research:** Social Media Sentiment as a Predictor of Extreme Crypto Events
**Status (Jun 2026):** Sprint 3 **Steps 1–5 done** + the per-coin news experiment. Modeling is
complete; **Step 6** (economic-significance backtest + consolidated honest-null writeup) is next.
Pre-registration frozen in `PREREGISTRATION.md`. Headline finding this sprint: **news
attention/volume beats sentiment polarity** (volume significant in the severity model, p=0.001);
the polarity-euphoria effect stays weak/underpowered everywhere. See `revision.md` for a
plain-language recap of the session.

## Where to look
- **`improvement.md`** — the current **Sprint 3 plan** (walk-forward backtesting, nonlinear
  extreme-loss recoding, QQQ+ARKF combined factor, stablecoin-depeg event study, and the
  paper write-up). Start here for "what's next."
- **`CONTEXT.md`** — methodology + full results history (v1 → Sprint 1 → Sprint 2/3).
- **`sprint2_results.md`**, **`outputs/firth_results.md`**, **`outputs/regime_results.md`** —
  the Sprint-2 result tables + interpretation (raw tables live in the matching `.txt`).
- **`README.md`** — how to run the pipeline.

## Current state (one paragraph)
The pipeline is complete and data is full (2,382 articles, 66/66 months, prices to mid-2026).
The original contribution is **RWDV** (recovery-weighted downside volatility) + **scar-event**
labels + **FinBERT** sentiment + a multi-factor weekly logit (sentiment + RWDV + ARKF + QQQ),
tested with **rare-event-correct statistics** (Firth + permutation + block-permutation). The
result is an **honest near-null**: sentiment→scarring-crash odds ≈ 1.5 (suggestive,
euphoria/sell-the-news direction) but **not robustly significant** (Firth p = 0.028 vs
permutation/block p ≈ 0.12–0.14; out-of-sample AUC ≈ 0.6). The regime hypothesis was tested
and **not** supported (interaction p = 0.217). Per the mentor, in this low-event domain the
contribution is the **original method + rigorous, honest analysis**, not a forced p-value.

## Next step
Sprint 3 **Step 6** in `improvement.md` §3: economic-significance backtest (a sentiment/volume
risk-off rule → Sortino/Sharpe, drawdown avoided) + the consolidated honest null (odds-ratio CIs
+ minimum detectable effect). Then **pivot the paper headline to attention/volume**. Data levers
that would actually move power (all deferred): go daily (§G2), extend BTC+ETH history to ~2017
(§G9). Do **not** change the pre-registered primary spec.

## Pipeline scripts
Base: `data_collection.py` → `scrape_articles.py` → `finbert_scoring.py` →
`build_features_v2.py` → `analysis_v2.py` → `analysis_firth.py` → `analysis_regime.py`.
**Sprint-3 (Steps 1→5):** `event_study.py` · `analysis_walkforward.py` · `analysis_power.py` ·
`analysis_panel.py` · `analysis_identification.py` · `analysis_costsensitive.py` ·
`analysis_step5_models.py`.
**Per-coin experiment:** `scrape_news_percoin.py` → folded into `analysis_panel.py` Part B
(FinBERT only; the separate VADER/FinBERT per-coin scripts were consolidated away).
(+ legacy Sprint-1 monthly: `sentiment_scoring.py`, `event_definition.py`, `analysis.py`).
Run with `python3`.
