# Improvement Plan & Roadmap

**Project:** Social Media Sentiment as a Predictor of Extreme Crypto Events
**Owner:** Hongqing Wang
**Mentor:** Ganesh Mani (Indigo Researcher, Crimson)
**Synthesises:** `CONTEXT.md`, `HANDOFF.md`, the mentor chat (Apr 14), and the
earlier `IMPROVEMENT_PLAN.md` (now merged into this single file).
**Window:** 2021-01-01 → 2024-12-31 (Mt Gox / pre-2021 out of scope this sprint)

> **This is the single living improvement file.** Section 2 records what Sprint 1
> set out to do and what is now **done**; Section 8 is the forward-looking list of
> **research questions worth pursuing next** — read that first if you just want to
> know "what could make the result better."

---

## 1. Objective

Move the project from a weak, statistically-insignificant first result
(AUC 0.537, p = 0.58, 32% of months mislabelled "extreme") to a defensible
iteration that addresses the mentor's three core critiques:

1. **Data density is too low** — 1–4 articles/month is below the floor for a
   meaningful monthly composite.
2. **The extreme-event label is mis-calibrated** — 32% of months flagged
   "extreme" is, by definition, not extreme.
3. **The methodology may be wrong** — monthly logistic regression may be the
   wrong frame; an *event study* anchored on known crashes is the standard
   finance approach.

---

## 2. Status — Sprint 1 (DONE) vs. remaining

| Task | Description | Status |
|---|---|---|
| T1 | Watch mentor's YouTube link; note if it's event-study / EVT / pipeline | ☐ user to do |
| T2 | Read Medium "statistics of extreme events" | ☐ user to do |
| T3 | Skim arxiv paper | ☐ user to do |
| T4 | Inspect Zenodo dataset (contents, range, license) | ☐ user to do |
| T5 | Decide event study vs. regression vs. both | ◑ both built; pick after T1 |
| **T6** | **Recalibrate extreme-event threshold** | ✅ **done — 31.8% → 13.0%** |
| **T7** | **Build `scrape_articles.py` (BeautifulSoup + GDELT)** | ✅ **built & smoke-tested**; full backfill = re-run |
| **T8** | **Linear + exponential weighting, snippet scoring** | ✅ **done** |
| **T9** | **Build `event_study.py` (CAR around shocks)** | ✅ **done** |
| **T10** | **Re-run regression: linear/exp × contemporaneous/lag** | ✅ **done — AUC 0.480 → 0.559** |
| **T11** | **Update `CONTEXT.md` / `HANDOFF.md`** | ✅ **done** |
| **T12** | **`outputs/sprint1_summary.md` one-pager** | ✅ **done** |
| T13 | Send mentor: summary + open questions | ☐ user to do |

**Sprint 1 headline:** extreme-event label fixed (the critical bug); best
predictor is **exponentially-weighted, one-month-lagged** VADER sentiment
(AUC 0.559); still **not significant** (p ≈ 0.50) — the gain is in *rigour*, not
a breakthrough. Event study surfaced a **"sell-the-news"** pattern on the ETF
approval and halving. Full numbers in `outputs/sprint1_summary.md` and the
"Sprint 1" section of `CONTEXT.md`.

---

## 3. Mentor chat pointers → actions (traceability)

| Mentor pointer | Interpretation | Action | Status |
|---|---|---|---|
| linear vs. exponential weighting | Sentiment aggregation method | T8 — EW (7-day half-life) added | ✅ |
| theregister + arxiv links | Background reading | T1–T3 | ☐ |
| NAND | Boolean signal combination | Sprint 2 (needs a 2nd signal first) | → §8 |
| Snippet | Score snippets, not full body | T7 + T8 | ✅ |
| Event study | Methodology shift | T9 | ✅ |
| Mt Gox | Pre-2021 anchor event | Out of scope (window 2021–2024) | → §8 |
| SBF | Anchor event | T9 — FTX 2022-11-11 | ✅ |
| Medium + Black swan | EVT framing of extremes | T6 — GPD method available | ✅ |
| YouTube link | Methodology walkthrough | T1 | ☐ |
| BeautifulSoup | Scrape articles | T7 | ✅ |
| Zenodo dataset | Existing dataset to leverage | T4 — cross-check, not primary | ☐ |

---

## 4. What each change did, and why it improves the output

*(Condensed; the mentor-facing version with before/after tables lives in
`CONTEXT.md` → "Sprint 1" section.)*

1. **Extreme-event label (T6).** v1 z-scored overlapping (autocorrelated) forward
   returns against a rolling window, then flagged a month if *any* day crossed
   ±2.5 SD. Both inflated the count. v2 takes **one clean obs/month** thresholded
   against the **whole-sample** distribution; methods = `quantile` (default),
   `global_z`, `evt` (Generalized Pareto). Rate **31.8% → 13.0%** → every
   downstream metric is now interpretable.
2. **Exponential weighting (T8).** Articles near month-end count more (7-day
   half-life). Freshest sentiment is most relevant for predicting next month →
   AUC 0.480 → 0.534.
3. **Predictive lag (T10).** Sentiment at *T* → event at *T+1*. Tests the actual
   "early-warning" question → best AUC **0.559**.
4. **Event study (T9).** Robust to the tiny extreme-month count; surfaced the
   "sell-the-news" qualitative finding the regression can't.
5. **Scraper + snippets (T7).** GDELT historical backfill + CoinDesk RSS; targets
   ≥ 30 articles/month — attacks the biggest source of v1 noise.

---

## 5. Project structure (current)

```
Github Extreme/
├── CONTEXT.md                  # methodology reference + Sprint 1 results
├── HANDOFF.md                  # rolling status
├── improvement.md              # THIS FILE — plan, status, next questions
├── README.md
├── requirements.txt            # + scipy, requests, beautifulsoup4, lxml
├── data/
│   ├── articles.txt            # legacy manual corpus (kept)
│   ├── articles_scraped.csv    # produced by scrape_articles.py (re-run to fill)
│   ├── raw_html/               # scraper cache (git-ignored)
│   ├── price_data.csv
│   ├── sentiment_scores.csv    # vader_linear, vader_exp_hl7, tb_polarity
│   └── labeled_events.csv      # recalibrated labels (13% rate)
├── data_collection.py
├── scrape_articles.py          # NEW — GDELT + CoinDesk RSS (BeautifulSoup)
├── sentiment_scoring.py        # UPDATED — linear + exponential + snippet
├── event_definition.py         # UPDATED — quantile / global_z / EVT, 1 obs/month
├── analysis.py                 # UPDATED — 4 specs + pilot benchmark
├── event_study.py              # NEW — CAR around known shocks
└── outputs/
    ├── roc_curve.png
    ├── event_study_car.png
    └── sprint1_summary.md
```

Run order: `data_collection.py` → `scrape_articles.py` → `sentiment_scoring.py`
→ `event_definition.py` → `analysis.py` → `event_study.py`.

---

## 6. Code style & boundaries

**Always:** keep the pipeline runnable end-to-end at every commit; cache scraped
HTML so re-runs are free; throttle scraping; record exact dates/sources/counts of
ingested articles; save every figure to `outputs/`.

**Ask first:** before extending outside 2021–2024 (Mt Gox); before scraping a
paywalled source; before deleting `articles.txt`; before changing the 50/50
BTC/ETH portfolio definition.

**Never:** commit raw scraped HTML to git; hardcode keys; report a result without
its n, extreme-event rate, and p-value; delete or amend the legacy v1 results in
`CONTEXT.md` (append, don't overwrite).

---

## 7. Testing / validation strategy

- **Scraper sanity:** histogram of articles/month — no month below 10; eyeball 5
  random snippets for parsing errors.
- **Threshold sanity:** extreme-event rate must land in [5%, 15%]. ✅ 13.0%.
- **Sentiment sanity:** spot-check 10 articles with |VADER| > 0.8 against a human read.
- **Pilot benchmark:** Feb-2024 composite should land near the mentor's −0.06.
  ⚠ currently +0.275 — unresolved (article selection).
- **Reproducibility:** a clean run should reproduce the headline numbers to 2 dp.

---

## 8. Open research questions — what could make the result better next

> This is the forward-looking heart of the file. Sprint 1 fixed the *plumbing*;
> these are the questions that could change the *answer*. Grouped by theme, each
> phrased as a testable question with the concrete next step.

### A. Data & measurement
- **Q1. Does denser data sharpen the signal or confirm the null?**
  Finish the GDELT backfill (≥ 30 articles/month), re-run sentiment → analysis.
  *This is the highest-leverage open question — the whole v1/Sprint-1 weakness
  may simply be thin monthly composites.*
- **Q2. Is VADER the wrong tool for crypto text?** VADER is a generic
  social-media lexicon and likely misreads crypto jargon ("Bitcoin breaks
  resistance," "capitulation," "down bad"). Swap in a finance/crypto-tuned
  transformer (**FinBERT** or **CryptoBERT**) and compare. Does a domain model
  flip any conclusion?
- **Q3. News vs. actual social media.** The research question says *social media*
  sentiment, but we score *news articles*. Should we pull real Reddit
  (r/CryptoCurrency, r/Bitcoin) or X/Twitter data? Is there a measurable
  difference between editorial-news sentiment and crowd sentiment?
- **Q4. Snippet vs. headline vs. full body.** We assumed snippets carry the
  cleanest signal — test it. Score all three and compare AUC.

### B. Target / label definition
- **Q5. Resolution vs. rarity trade-off.** At monthly resolution, a true <5%
  black-swan rate gives only 1–2 events — too few to regress. Move to **weekly or
  daily** resolution to raise the event count (at the cost of noisier returns)?
- **Q6. Revisit EVT once we have more tail data.** The GPD fit on ~7 monthly
  exceedances was unstable (ξ ≈ 2). At daily resolution there are hundreds of tail
  observations and the Generalized Pareto fit becomes reliable — the proper Black
  Swan framing the mentor wanted.
- **Q7. Should up-moves and down-moves be modelled separately?** Sentiment may
  predict crashes but not rallies (or vice-versa). Split the binary label into
  extreme-positive and extreme-negative and fit each.
- **Q8. Which forward horizon matters?** We use 3-month forward returns. Run the
  **48-hour robustness check** (already in the methodology) and intermediate
  horizons. Does sentiment predict short-horizon tails better than long-horizon?

### C. Methodology & predictors
- **Q9. Does sentiment add anything beyond market conditions? (the NAND idea.)**
  Add a second predictor — **trading volume, realised volatility, VIX, funding
  rates, or Google Trends** — and test (a) multivariable logistic regression with
  interaction terms and (b) boolean signal-combination rules
  (e.g. `high sentiment AND low volume`). Does sentiment survive controls?
- **Q10. Reverse causality / endogeneity.** Does sentiment *cause* extremes or
  merely *react* to them? Enforce strictly pre-event sentiment (done in the event
  study) and run a **Granger-causality** test on the monthly series.
- **Q11. Optimal lead time.** We only tested lag-1. Fit a **distributed lag**
  (sentiment over the past 1, 2, 3 months) and find the lead time that maximises
  predictive power — if any.
- **Q12. Strengthen the event study.** Add more events, run a formal
  cross-sectional **t-test on the CARs**, and regress |CAR| on pre-event
  sentiment with enough events to have power. Is the +0.32 sentiment–CAR
  correlation real?

### D. Validation & honesty
- **Q13. Out-of-sample test.** Train on 2021–2022, test on 2023–2024. Does any
  signal hold out-of-sample, or is it in-sample overfitting?
- **Q14. Guard against p-hacking.** We already ran 4 specs; as the spec count
  grows, pre-register the primary spec or apply a multiple-testing correction
  (Bonferroni / Benjamini–Hochberg) so a "significant" result isn't just luck.
- **Q15. Resolve the pilot benchmark gap.** Our Feb-2024 composite (+0.275) vs.
  the mentor's −0.06 — reconcile the article set used for the pilot month.

### E. Scope (parking lot)
- **Q16. Extend to Mt Gox (2014)?** Requires pulling price + news back to 2013.
  Only worth it if Sprint 1 results justify the effort.

---

## 9. Direct questions for the mentor (for the next check-in)

1. The YouTube link — is it the methodology walkthrough we should follow, or background?
2. What half-life did you have in mind for exponential weighting (we used 7 days)?
3. Should the event study **replace** the monthly regression, or run alongside it?
4. Was the Zenodo dataset the intended source, or just a reference for data format?
5. Pilot Feb-2024: which article set gave you −0.06? Ours gives +0.275 — likely a
   selection difference.
6. "NAND" — did you mean boolean combination of sentiment + volume + volatility,
   or something else?
7. Given the small-sample tension (Q5), are you open to moving to weekly/daily
   resolution, or do you want to keep monthly for interpretability?

---

## 10. Decisions log

- **D1:** Event study vs. monthly regression — *both built; final primary pending
  T1 (YouTube) and mentor Q3.*
- **D2:** Zenodo as primary / cross-check / ignore — *pending T4 inspection.*
- **D3:** Threshold method — *quantile chosen as Sprint-1 default (stable 6-event
  plateau); EVT revisited at daily resolution per Q6.*
