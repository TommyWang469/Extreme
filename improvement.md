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

## 2. What to improve next (ranked by leverage)

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

