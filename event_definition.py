"""
event_definition.py
Loads price_data.csv, computes 3-month (63 trading-day) forward returns
for the portfolio, labels extreme events using a ±2.5 rolling-SD threshold,
and saves labeled_events.csv.

Extreme event definition (from CONTEXT.md):
    forward_return > +2.5 * rolling_SD  →  label = +1  (extreme positive)
    forward_return < -2.5 * rolling_SD  →  label = -1  (extreme negative)
    otherwise                           →  label =  0
For the binary logistic regression a combined extreme flag (|z| > 2.5) is
also stored as `extreme_binary` (1 = extreme, 0 = normal).
"""

import pandas as pd
import numpy as np

# ── Settings ──────────────────────────────────────────────────────────────────
IN_PATH   = "data/price_data.csv"
OUT_PATH  = "data/labeled_events.csv"
WINDOW    = 63   # ≈ 3 months of trading days (primary window)
SD_THRESH = 2.5  # rolling-SD multiplier

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading price data …")
df = pd.read_csv(IN_PATH, parse_dates=["date"], index_col="date")
print(f"  {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")

# ── 2. 3-month forward return ─────────────────────────────────────────────────
# Sum of log-returns over the next WINDOW days (equivalent to log of cumulative
# price change), then shift backward so each row carries its own future return.
df["forward_return"] = (
    df["portfolio_ret"]
    .rolling(window=WINDOW)
    .sum()
    .shift(-WINDOW)   # align: today's row gets the *next* WINDOW days' return
)

# ── 3. Rolling mean and SD of the forward return ──────────────────────────────
# Use the same WINDOW for consistency; rolling over past observations only.
df["rolling_mean"] = df["forward_return"].rolling(window=WINDOW).mean()
df["rolling_std"]  = df["forward_return"].rolling(window=WINDOW).std()

# ── 4. Z-score of the forward return ──────────────────────────────────────────
df["forward_zscore"] = (df["forward_return"] - df["rolling_mean"]) / df["rolling_std"]

# ── 5. Label extreme events ───────────────────────────────────────────────────
conditions = [
    df["forward_zscore"] >  SD_THRESH,   # extreme positive
    df["forward_zscore"] < -SD_THRESH,   # extreme negative
]
choices = [1, -1]
df["extreme_label"]  = np.select(conditions, choices, default=0)
df["extreme_binary"] = (df["extreme_label"] != 0).astype(int)   # for logistic regression

# ── 6. Drop rows where forward return cannot be computed (tail + NaN rows) ────
df.dropna(subset=["forward_return", "rolling_std"], inplace=True)

# ── 7. Summary ────────────────────────────────────────────────────────────────
n_pos    = (df["extreme_label"] ==  1).sum()
n_neg    = (df["extreme_label"] == -1).sum()
n_normal = (df["extreme_label"] ==  0).sum()
print(f"\nExtreme event breakdown (±{SD_THRESH} SD threshold, {WINDOW}-day window):")
print(f"  Extreme positive (+1): {n_pos}")
print(f"  Extreme negative (-1): {n_neg}")
print(f"  Normal          (  0): {n_normal}")
print(f"  Base rate of extremes: {(n_pos + n_neg) / len(df):.1%}")

# ── 8. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH)
print(f"\nSaved {len(df)} rows → {OUT_PATH}")
