# Improvement Plan — Based on Mentor Feedback (Ganesh Mani, Apr 14)

This plan translates the chat pointers from the Indigo Researcher meeting into concrete actions for the next iteration of the pipeline.

---

## Mentor's pointers (raw)

From the Zoom meeting chat:

1. **Linear weighting vs. exponential weighting**
2. https://www.theregister.com/2026/04/14/two_button_calculator/
3. https://arxiv.org/html/2603.21852v2
4. **NAND**
5. **Snippet**
6. **Event study**
7. **Mt Gox**
8. **SBF**
9. https://medium.com/@felbuch/the-statistics-of-extreme-events-17acc2335242
10. **Black swan**
11. https://www.youtube.com/watch?v=Dcl_AZqfZVc
12. https://pypi.org/project/beautifulsoup4/
13. https://data.niaid.nih.gov/resources?id=zenodo_7684409

---

## Interpretation

The mentor is pushing the project in three directions:

- **Methodology shift** — move from monthly logistic regression toward an *event-study* framework anchored on known extreme events (Mt Gox, FTX/SBF), with extreme-value statistics / black swan framing.
- **Data quality** — replace the manually curated 66-article corpus with a scraped, snippet-level dataset (BeautifulSoup) and consider an existing Zenodo dataset as a starting point.
- **Sentiment aggregation** — compare linear vs. exponential weighting when collapsing daily/article sentiment into a window-level score; consider how to combine multiple signals (NAND / boolean logic on signal flags).

---

## Action items (prioritised)

### P0 — Methodology refit: switch to event study

**Why:** Logistic regression on 41 monthly observations is underpowered. An event study is the standard finance methodology for measuring whether a signal moves around a known event, and aligns naturally with "extreme events" framing.

**How to apply:**
1. Define a list of anchor events with dates:
   - Mt Gox collapse (Feb 2014) — *note: outside our 2021–2024 window, may need to extend price data back to 2013*
   - Terra/LUNA collapse (May 2022)
   - FTX/SBF bankruptcy (Nov 2022)
   - Bitcoin ETF approval (Jan 2024)
   - Halving (Apr 2024)
2. For each event, define an event window (e.g. ±30 days) and an estimation window (e.g. −120 to −30 days).
3. Compute *abnormal returns* in the event window vs. the expected return from the estimation window.
4. Test whether sentiment in the pre-event window predicts the magnitude of the abnormal return.

**Files to change:**
- New script `event_study.py` (do not delete existing pipeline — keep both as comparison).
- Extend `data_collection.py` to pull data back to 2013 if Mt Gox is in scope.

---

### P1 — Replace manual articles with scraped corpus

**Why:** The single biggest weakness in the current pipeline is article density (1–4/month). The mentor explicitly pointed to BeautifulSoup, which means: scrape.

**How to apply:**
1. Identify target sources with HTML archives:
   - CoinDesk (`coindesk.com/tag/...` archives)
   - The Block (`theblock.co/latest`)
   - Decrypt (`decrypt.co/news`)
   - a16z Crypto blog (smaller volume, higher quality)
2. Write `scrape_articles.py` using `requests` + `beautifulsoup4`:
   - Iterate dates 2021-01-01 → 2024-12-31
   - For each day, pull headline + first paragraph (a "snippet" — per mentor's pointer)
   - Save to `data/articles_scraped.csv` with columns: `date, source, url, headline, snippet`
3. Respect robots.txt and rate-limit (1 req/sec). Cache HTML locally to avoid re-scraping.
4. Target ≥ 30 articles/month so the monthly composite is statistically meaningful.

**Snippet vs. full text:** The mentor wrote "Snippet" as a standalone hint. Snippets (headline + lede) often carry the sentiment signal more cleanly than the full article body, which dilutes with background context. Score the snippet, not the body.

---

### P1 — Check the Zenodo dataset before scraping

**Why:** The mentor linked https://data.niaid.nih.gov/resources?id=zenodo_7684409 — this may already contain a labelled crypto news / sentiment corpus. Using it could save weeks of scraping.

**How to apply:**
1. Visit the link, identify the dataset's contents, license, and date range.
2. If it overlaps our 2021–2024 window with usable sentiment-relevant text, prefer it over scraping.
3. If it is a related-but-different dataset, document why and proceed with scraping.

---

### P2 — Linear vs. exponential weighting for sentiment aggregation

**Why:** Current code uses a simple mean of all articles in a month. Recent articles within a month are likely more predictive of the next month's events than articles from the start of the month. Exponentially weighted averages give more weight to recent observations.

**How to apply:**
In `sentiment_scoring.py`, add two aggregation modes side-by-side:

```python
# Linear (current): equal weight
monthly_linear = df.groupby("month")["vader_compound"].mean()

# Exponential: half-life of N days
df["weight"] = np.exp(-np.log(2) * (month_end - df["date"]).dt.days / HALF_LIFE_DAYS)
monthly_exp = (df["vader_compound"] * df["weight"]).groupby("month").sum() / df["weight"].groupby("month").sum()
```

Run the regression with both and compare AUC / odds ratio. Half-life candidates: 3, 7, 14 days.

---

### P2 — Extreme value statistics framing (Black Swan)

**Why:** The mentor linked the "statistics of extreme events" Medium article and wrote "Black swan." This signals he wants the extreme-event threshold to be grounded in extreme value theory (EVT), not just a rolling ±2.5 SD heuristic.

**How to apply:**
1. Read the Medium article and the arxiv paper.
2. Fit a Generalized Pareto Distribution (GPD) to the upper/lower tails of the portfolio return distribution.
3. Define an "extreme event" as a return exceeding the 99th percentile under the fitted GPD, instead of ±2.5 rolling SD.
4. Re-run the analysis with the new label and compare to the current 14/41 = 34% extreme rate (which is suspiciously high — true black swans should be < 5% of observations).

**Concrete fix to investigate first:** The current 34% extreme rate strongly suggests the threshold is mis-calibrated. ±2.5 SD on rolling forward returns + monthly MAX aggregation is over-flagging. Even before EVT, try ±3 SD or ±3.5 SD and see what fraction of months stay flagged.

---

### P3 — Signal combination (NAND)

**Why:** "NAND" is a boolean logic operator. The mentor likely means: don't use sentiment as a single linear predictor. Combine multiple signals (sentiment, volume, volatility) with logic rules — e.g. "flag a warning only when sentiment is high AND volume is low" — or NOT (sentiment low AND volume high).

**How to apply:** Defer until P0 + P1 are done. Once we have a second predictor (volume or VIX), test:
- Logistic regression with interaction terms
- Decision tree / random forest (naturally captures AND/OR/NAND-style splits)
- A simple hand-crafted rule: `signal = (sentiment > θ_s) AND (volume > θ_v)` and measure precision/recall

---

### P3 — Watch the YouTube walkthrough

**Why:** The mentor linked a YouTube video without comment. Likely a methodology tutorial (event study, EVT, or sentiment-finance walkthrough). Watch it before starting P0 to align on methodology vocabulary.

---

## Ordered execution plan

1. **This week:**
   - [ ] Watch the YouTube link, skim the Medium and arxiv articles
   - [ ] Inspect the Zenodo dataset — decide scrape vs. reuse
   - [ ] Fix the extreme-event threshold (P2 sub-task) — likely the 34% rate is wrong

2. **Next week:**
   - [ ] Build `scrape_articles.py` OR ingest Zenodo dataset
   - [ ] Achieve ≥ 30 articles/month for 2021–2024

3. **Following week:**
   - [ ] Build `event_study.py` around Mt Gox / Terra / FTX / ETF / halving anchors
   - [ ] Add exponential-weighted sentiment aggregation alongside linear

4. **Then:**
   - [ ] Add volume / VIX as a second predictor
   - [ ] Try signal-combination (NAND-style) rules
   - [ ] Re-fit extreme-event label using GPD / EVT

---

## Open questions to bring back to the mentor

1. Is Mt Gox (2014) in scope? It would require extending price data back to 2013 and finding pre-2021 articles.
2. Is the Zenodo dataset the intended source, or just a reference?
3. Should the event study replace the logistic regression entirely, or run alongside it?
4. What half-life does he have in mind for exponential weighting (days, weeks)?
5. What is the YouTube link about — confirm it is the methodology walkthrough we should follow?
