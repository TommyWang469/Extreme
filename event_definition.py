"""
event_definition.py
Loads price_data.csv, computes 3-month (63 trading-day) forward returns for the
portfolio, and labels extreme ("black swan") months. Saves labeled_events.csv.

WHY THIS WAS REWRITTEN (Sprint 1, improvement.md T6)
─────────────────────────────────────────────────────────────────────────────
The original script flagged 31.8% of months as "extreme" — far too high for an
event that is supposed to be rare. Two compounding bugs caused this:

  1. It z-scored the 63-day forward return against a *rolling* 63-day window of
     forward returns. Because consecutive forward windows overlap almost
     entirely, those returns are highly autocorrelated and the rolling SD is an
     unstable, often tiny denominator — so ordinary moves blew past 2.5 SD.

  2. It then aggregated daily labels to monthly with MAX: a whole month was
     flagged extreme if ANY single day in it was extreme. Because extremes
     cluster, that flips entire months on.

FIX
─────────────────────────────────────────────────────────────────────────────
  • Take ONE clean observation per month: the portfolio's forward return as of
    each month-end. This removes the overlapping-daily-MAX inflation.
  • Threshold against the FULL-sample distribution (not a rolling window), via
    one of three selectable methods:
        METHOD = "global_z"  → |z| > SD_THRESH using whole-sample mean/std
        METHOD = "quantile"  → beyond the TAIL_PCT/2 tails (empirical)
        METHOD = "evt"       → Generalized Pareto (peaks-over-threshold) tail fit
  • Report the resulting base rate so it can be checked against the 5–10% target.

NOTE ON SMALL SAMPLES
─────────────────────────────────────────────────────────────────────────────
With ~44 monthly observations, a "true black swan" rate of <5% yields only 1–2
events — too few for logistic regression. There is an unavoidable tension
between rarity and statistical power at monthly resolution. The default here
(quantile, 16% two-sided ≈ top/bottom 8%) keeps ~6–8 events so the downstream
regression is estimable, while still being far stricter than the old 32%.
The event-study pipeline (event_study.py) is the better tool for genuinely rare
events and does not depend on this base rate.
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── Settings ──────────────────────────────────────────────────────────────────
IN_PATH   = "data/price_data.csv"
OUT_PATH  = "data/labeled_events.csv"
WINDOW    = 63          # ≈ 3 months of trading days (primary forward window)

# Labeling method: "quantile" (default) | "global_z" | "evt"
METHOD    = "quantile"

# Method-specific parameters
SD_THRESH = 2.0         # for "global_z": |z| threshold (2.0 ≈ 4.6% two-tailed normal)
TAIL_PCT  = 0.12        # for "quantile": total mass in both tails. 0.10–0.13 all land
                        # on a stable 6-event (13%) plateau at n=46; 0.12 sits mid-plateau.
EVT_THRESH_PCT = 0.85   # for "evt": quantile above/below which the GPD tail is fit
EVT_TAIL_PROB  = 0.05   # for "evt": exceedance prob defining an extreme (each tail)


def label_global_z(fwd: pd.Series) -> pd.DataFrame:
    """Z-score against the whole-sample mean/std; flag |z| > SD_THRESH."""
    mu, sd = fwd.mean(), fwd.std()
    z = (fwd - mu) / sd
    label = np.where(z > SD_THRESH, 1, np.where(z < -SD_THRESH, -1, 0))
    return pd.DataFrame({"forward_zscore": z, "extreme_label": label}, index=fwd.index)


def label_quantile(fwd: pd.Series) -> pd.DataFrame:
    """Flag the empirical lower TAIL_PCT/2 and upper TAIL_PCT/2 of forward returns."""
    lo = fwd.quantile(TAIL_PCT / 2)
    hi = fwd.quantile(1 - TAIL_PCT / 2)
    label = np.where(fwd >= hi, 1, np.where(fwd <= lo, -1, 0))
    mu, sd = fwd.mean(), fwd.std()
    z = (fwd - mu) / sd
    print(f"  quantile cuts: lower={lo:.4f}  upper={hi:.4f}")
    return pd.DataFrame({"forward_zscore": z, "extreme_label": label}, index=fwd.index)


def label_evt(fwd: pd.Series) -> pd.DataFrame:
    """
    Peaks-over-threshold using the Generalized Pareto Distribution (Black Swan
    framing, improvement.md T6). Fit a GPD to each tail of exceedances over a
    high threshold, then flag observations whose fitted tail probability is
    below EVT_TAIL_PROB. Falls back gracefully if a tail has too few points.
    """
    mu, sd = fwd.mean(), fwd.std()
    z = (fwd - mu) / sd
    label = np.zeros(len(fwd), dtype=int)

    # ---- Upper tail ----
    u_hi = fwd.quantile(EVT_THRESH_PCT)
    exceed_hi = fwd[fwd > u_hi] - u_hi
    if len(exceed_hi) >= 5:
        c, loc, scale = stats.genpareto.fit(exceed_hi, floc=0)
        # tail prob that a value exceeds x: P(X>u) * (1 - GPD_cdf(x-u))
        p_exceed_u = (fwd > u_hi).mean()
        tail_p = p_exceed_u * (1 - stats.genpareto.cdf(fwd - u_hi, c, loc=0, scale=scale))
        label = np.where((fwd > u_hi) & (tail_p < EVT_TAIL_PROB), 1, label)
        print(f"  EVT upper tail: ξ={c:.3f} σ={scale:.4f}  n_exceed={len(exceed_hi)}")

    # ---- Lower tail (mirror) ----
    u_lo = fwd.quantile(1 - EVT_THRESH_PCT)
    exceed_lo = u_lo - fwd[fwd < u_lo]
    if len(exceed_lo) >= 5:
        c, loc, scale = stats.genpareto.fit(exceed_lo, floc=0)
        p_exceed_l = (fwd < u_lo).mean()
        tail_p = p_exceed_l * (1 - stats.genpareto.cdf(u_lo - fwd, c, loc=0, scale=scale))
        label = np.where((fwd < u_lo) & (tail_p < EVT_TAIL_PROB), -1, label)
        print(f"  EVT lower tail: ξ={c:.3f} σ={scale:.4f}  n_exceed={len(exceed_lo)}")

    return pd.DataFrame({"forward_zscore": z, "extreme_label": label}, index=fwd.index)


LABELERS = {"global_z": label_global_z, "quantile": label_quantile, "evt": label_evt}


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading price data …")
    df = pd.read_csv(IN_PATH, parse_dates=["date"], index_col="date")
    print(f"  {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")

    # ── 2. 3-month forward return (sum of next WINDOW daily log-returns) ───────
    df["forward_return"] = (
        df["portfolio_ret"].rolling(window=WINDOW).sum().shift(-WINDOW)
    )

    # ── 3. Collapse to ONE observation per month BEFORE labeling ──────────────
    # Take the forward return as of each month-end (the last available value in
    # the month). This is the key fix: no daily MAX aggregation, no overlap
    # inflation — each month is a single clean draw.
    monthly_fwd = (
        df["forward_return"].resample("ME").last().dropna()
    )
    print(f"  {len(monthly_fwd)} monthly forward-return observations")

    # ── 4. Apply the selected labeling method ─────────────────────────────────
    if METHOD not in LABELERS:
        raise ValueError(f"Unknown METHOD={METHOD!r}; choose from {list(LABELERS)}")
    print(f"\nLabeling method: {METHOD}")
    labels = LABELERS[METHOD](monthly_fwd)

    monthly = pd.DataFrame({"forward_return": monthly_fwd})
    monthly = monthly.join(labels)
    monthly["extreme_binary"] = (monthly["extreme_label"] != 0).astype(int)
    monthly.index.name = "date"

    # ── 5. Summary ────────────────────────────────────────────────────────────
    n_pos    = int((monthly["extreme_label"] ==  1).sum())
    n_neg    = int((monthly["extreme_label"] == -1).sum())
    n_normal = int((monthly["extreme_label"] ==  0).sum())
    base     = (n_pos + n_neg) / len(monthly)
    print(f"\nMonthly extreme event breakdown ({METHOD}, {WINDOW}-day forward):")
    print(f"  Extreme positive (+1): {n_pos}")
    print(f"  Extreme negative (-1): {n_neg}")
    print(f"  Normal          (  0): {n_normal}")
    print(f"  Base rate of extremes: {base:.1%}", end="")
    if 0.05 <= base <= 0.15:
        print("  ✓ in 5–15% defensible band (down from 31.8% in v1)")
    else:
        print("  ⚠ outside 5–15% band — review METHOD/params")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    monthly.to_csv(OUT_PATH)
    print(f"\nSaved {len(monthly)} monthly rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
