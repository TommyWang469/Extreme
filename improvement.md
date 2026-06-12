# What Could Be Wrong, and What to Improve Next

**Project:** Social Media Sentiment as a Predictor of Extreme Crypto Events

This file is forward-looking only: possible mistakes in the current approach, and
the questions worth pursuing next. (For what's already built, see `CONTEXT.md`.)

---

## 1. Possible mistakes / weaknesses in the current approach

- **Too few extreme events.** Only ~6 extreme months in the sample. This is the
  binding constraint — with so few positives, no logistic regression can reach
  statistical significance no matter how good the predictor is.
- **We score headlines, not articles.** GDELT returns only the title (the
  `snippet` column is just a copy of the `headline`). So the sentiment input is a
  single short sentence per article — noisy, and never the article body or lede.
- **VADER may be the wrong tool for crypto.** It's a generic social-media lexicon
  and likely misreads crypto jargon ("breaks resistance," "capitulation," "down
  bad"). A finance/crypto-tuned model could flip conclusions.
- **News ≠ social media.** The research question says *social media* sentiment, but
  we score *news articles*. We may be measuring editorial tone, not crowd mood.
- **Pilot benchmark gap.** Our Feb-2024 composite doesn't perfectly match the
  mentor's reference — a sign the article set / scoring may differ from intended.
- **EVT fit is unstable.** The Generalized Pareto fit on ~7 monthly exceedances is
  unreliable (ξ blows up); not enough tail data at monthly resolution.
- **Risk of p-hacking.** We already run 4 specs. As specs multiply, a "significant"
  result could just be luck unless we correct for multiple testing.
- **Possible reverse causality.** Sentiment might *react* to extremes rather than
  *predict* them — we haven't formally tested direction.

---

## 2. Sprint 2 plan — originality + significance (decided Jun 2026)

> Driven by the mentor call (Jun 2026) and a literature scan. Two directions:
> (A) FinBERT sentiment, (B) an **original volatility calculation** — the novelty
> judges look for. "Unique and significant" are the two goals.

### 2.1 What's already taken (literature scan, Jun 2026)

| Crowded — do NOT claim as novel | Where |
|---|---|
| VADER/FinBERT + LSTM/GRU price prediction | FinBERT-BiLSTM (arXiv 2411.12748, Applied Intelligence 2026); many Kaggle/GitHub repos |
| Sentiment → price *jumps* via logistic regression | "Not all words are equal: Sentiment and jumps in the cryptocurrency market" (J. Int. Fin. Markets 2023) |
| Sentiment inside GARCH / stochastic-vol models | BERT+GARCH review (arXiv 2510.16503); sentiment-driven stochastic vol (arXiv 1906.00059) |
| Realized semivariance + Twitter sentiment for jump prediction | ScienceDirect 2026 (realized metrics + sentiment via ML) |

**The open gap (verified):** drawdown *recovery time* appears in the literature only
as a descriptive statistic (how long to climb back to the peak). Nobody uses recovery
persistence to (a) **weight a volatility measure** or (b) **define the extreme-event
label**. That gap matches exactly the mentor's hints: lower partial moments / Sortino,
"flash crash ≠ extreme move," "climbing back 50/80%," stairs-up-elevator-down asymmetry.

### 2.2 The original contribution: Recovery-Weighted Downside Volatility (RWDV)

Standard deviation treats a −8% day that bounces back tomorrow the same as a −8% day
that starts a six-month bear market. RWDV does not:

- For each **down** day *t* (only downside — a lower partial moment, per Sortino logic):
  find τ_t = trading days until the portfolio claws back 50% of that day's loss,
  capped at H (=10) days. Weight w_t = τ_t / H ∈ (0,1].
- **RWDV** = √( 252 · mean over window of [ w_t · r_t² | r_t < 0 ] )
- Flash crash (recovers next day) → w ≈ 0.1 → nearly ignored. Sustained decline
  (no recovery within H days) → w = 1 → full weight. *"Bruises fade; scars count."*
- **No look-ahead:** as a trailing predictor at time T, the window only includes
  down-days whose H-day recovery horizon is fully resolved by T (lag by H).

**Scar-event label (the redefined "extreme event"):** a week is an extreme-DOWN event
if next week's return is in the bottom decile AND the drop is *not* 50%-recovered
within 15 trading days of the trough. Extreme-UP labelled separately (top decile) —
up/down modelled asymmetrically per the mentor. Weekly resolution → ~208 obs and
~3× the events of the monthly design → real statistical power.

### 2.3 The multi-factor model (mentor's whiteboard, verbatim)

One variable alone can't predict ("height alone doesn't predict jump ability"):

| Factor | Role |
|---|---|
| FinBERT sentiment (exp-weighted, lag-1) | the variable under test |
| Trailing RWDV | volatility clustering control — *and* our novel metric |
| ARKF (FinTech ETF) trailing 4-week return | the mentor's FinTech-proxy factor |
| QQQ trailing 4-week return | broad tech-market factor |

Logit: P(extreme-down week w+1) = f(factors at week w). Pre-registered primary test:
the FinBERT coefficient, controlling for everything else. Ablations: VADER vs FinBERT,
RWDV vs plain semideviation vs plain SD, with/without scar condition.
Out-of-sample: train 2021–22, test 2023–24.

### 2.4 Build order

1. `finbert_scoring.py` — FinBERT (ProsusAI/finbert) on scraped headlines → weekly composites
2. `build_features_v2.py` — ARKF/QQQ download + RWDV + scar labels → `data/dataset_weekly.csv`
3. `analysis_v2.py` — multi-factor logit + ablation table + out-of-sample AUC

---

## 3. What to improve next (ranked by leverage)

### Highest leverage
- **Q1. Raise the event count → go weekly/daily.** Monthly resolution caps us at
  ~6 events. Weekly or daily returns give far more tail observations — the single
  biggest unlock for statistical power (and makes EVT viable again).
- **Q2. Swap VADER for a domain model.** Try **FinBERT** or **CryptoBERT** and
  compare AUC. Does a crypto-aware model change any conclusion?
- **Q3. Get real article text, not just headlines.** Pull the lede/first paragraph
  (CoinDesk RSS already provides a description; or fetch the first paragraph per
  URL). Then test headline vs. headline+lede vs. full body.

### Medium leverage
- **Q4. Add a second predictor (the "NAND" idea).** Bring in trading volume,
  realised volatility, VIX, funding rates, or Google Trends. Test multivariable
  logistic regression *and* boolean rules (e.g. `high sentiment AND low volume`).
  Does sentiment survive controls?
- **Q5. Model up-moves and down-moves separately.** Sentiment may predict crashes
  but not rallies. Split the label into extreme-positive vs. extreme-negative.
- **Q6. Find the optimal lead time.** We only tested lag-1. Fit a distributed lag
  (1, 2, 3 months) and find which lead time maximises predictive power, if any.
- **Q7. Use real social media data.** Pull Reddit (r/CryptoCurrency, r/Bitcoin) or
  X/Twitter to answer the actual research question, and compare to news sentiment.

### Validation & honesty
- **Q8. Out-of-sample test.** Train on 2021–2022, test on 2023–2024. Does any
  signal survive, or is it in-sample overfitting?
- **Q9. Granger-causality test** on the monthly series to check direction
  (sentiment → extremes vs. the reverse).
- **Q10. Strengthen the event study.** Add more events, run a formal t-test on the
  CARs, and test whether the pre-event sentiment ↔ CAR correlation is real.
- **Q11. Guard against p-hacking.** Pre-register the primary spec, or apply a
  multiple-testing correction (Bonferroni / Benjamini–Hochberg).
- **Q12. Resolve the pilot benchmark gap.** Reconcile which article set the mentor
  used for Feb-2024.

### Parking lot
- **Q13. Extend to Mt Gox (2014)?** Requires price + news back to 2013. Only worth
  it if current results justify the effort.

---

