# Revision Log — What We Did This Session (Sprint 3, Steps 1–5 + per-coin news)

---

## PART 1 — The story, step by step

### Step 1 — Event study
**Files:** `event_study.py` → results in `outputs/event_study.txt` (numbers) and
`outputs/event_study.md` (plain-English read), plus two charts (`event_study_car.png`,
`event_study_recovery.png`).

We rebuilt the old, stale event study on the full data (2021→2026). We studied real
crashes two ways: **named catalysts** (China ban, Terra, FTX, the USDC bank-scare, the
2024 carry-trade crash, the ETF approval, the halving, the Trump election) and an
**objective rule** (the worst 10% weeks). For each we measured the abnormal price move
and the recovery time, using proper statistics for a handful of events.

What we found: the crash happens *fast* and there's **no reliable pattern after** it.
A nice clean result came from the two "stablecoin broke its peg" events — the fake-backed
one (Terra) **never recovered** in 180 days, while the real-backed one (USDC) **bounced
back in 3–4 days**. And downside events take ~2 months to recover halfway.

### Step 2 — Walk-forward harness 
**Files:** `analysis_walkforward.py` → `outputs/walkforward.txt` / `outputs/walkforward.md`
+ charts.

The old model was secretly "peeking at the future" in two ways. We rebuilt the test so it
only ever uses past data (an expanding window that re-trains each week). Honest result: the
model is **only a little better than a coin flip** (AUC ≈ 0.58), the old way over-stated it
by +0.06, and the **only ingredient doing any work is the news mood** — the stock-market
and volatility ingredients add nothing. We also found that "fading older data" *hurts* here
(too few events to throw any away), and the model's most confident calls never actually hit
a crash.

### Step 3 — Power analysis + a panel of many coins
**Files:** `analysis_power.py` → `outputs/power.*`, and `analysis_panel.py` →
`outputs/panel.*`.

**Power analysis** is the math for "can we even prove this?" Answer: with our few crashes,
no — the smallest effect we could detect is much bigger than the real one, and we'd need
roughly **40+ coins** to have a fair shot. So the earlier "not significant" is mostly a
*data shortage*, not proof of "no effect."

**The panel** added 9 more coins (including the dead LUNA, so we don't only study
survivors), turning ~17 crashes into **142**. At first the news-mood effect looked
significant — but that was a **trap**: the news mood is one single number applied to every
coin, so counting coins as independent overstates confidence. Done honestly, the effect is
the same weak euphoria signal as before, still not provable.

### How we scraped MORE data — per-coin news
**Files:** `scrape_news_percoin.py` → `data/articles_percoin.csv`; the per-coin panel is now
**Part B of `analysis_panel.py`** (FinBERT scoring, scores cached in
`data/articles_percoin_finbert.csv`) → results in `outputs/panel.*`. *(We originally had two
separate per-coin scripts — a VADER one and a FinBERT one — but consolidated them into the single
`analysis_panel.py` and dropped VADER, since FinBERT gave the same answer more cleanly.)*


