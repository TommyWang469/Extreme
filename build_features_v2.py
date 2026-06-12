"""
build_features_v2.py
Sprint 2 (Jun 2026) — the ORIGINAL contribution + the mentor's multi-factor setup.

1. RWDV — Recovery-Weighted Downside Volatility (novel metric, lit-scan verified
   unoccupied). A lower partial moment where each down-day's squared return is
   weighted by how long the portfolio took to claw back RECOV_FRAC of that day's
   loss (capped at H_RECOVERY days). Flash crashes ≈ ignored; sustained declines
   get full weight. Trailing feature is lagged by H_RECOVERY days so the weight
   for every included day is fully resolved — no look-ahead.

2. Scar-event labels (novel extreme-event definition). Extreme-DOWN week =
   next-week return in the bottom TAIL_PCT decile AND the drop is NOT 50%-recovered
   within SCAR_RECOV_DAYS of the trough. Extreme-UP = top decile (no scar test —
   the mentor's asymmetry: if you're long, only downside persistence matters).

3. Factor download — ARKF (mentor's FinTech proxy) and QQQ trailing 4-week
   returns via yfinance.

Output: data/dataset_weekly.csv  (one row per W-FRI week)
"""

import os
import numpy as np
import pandas as pd

PRICE_PATH      = "data/price_data.csv"
OUT_PATH        = "data/dataset_weekly.csv"

WINDOW          = 63     # trailing window, calendar days (crypto trades daily)
H_RECOVERY      = 10     # RWDV weight cap: days allowed to recover half the loss
RECOV_FRAC      = 0.5
SCAR_RECOV_DAYS = 15     # label: days from trough to recover half, else "scar"
TAIL_PCT        = 0.10   # decile tails
WEEK_RULE       = "W-FRI"
ANN             = 365    # crypto trades every calendar day

os.makedirs("data", exist_ok=True)


# ── 1. Portfolio index from daily log returns ────────────────────────────────
def load_index() -> pd.Series:
    px = pd.read_csv(PRICE_PATH, parse_dates=["date"]).set_index("date")
    r = px["portfolio_ret"]
    idx = np.exp(r.cumsum())
    idx.name = "level"
    print(f"{len(idx)} daily obs, {idx.index.min().date()} → {idx.index.max().date()}")
    return idx


# ── 2. RWDV — Recovery-Weighted Downside Volatility ──────────────────────────
def recovery_weights(idx: pd.Series) -> pd.Series:
    """w_t = (days to claw back RECOV_FRAC of day-t's loss, capped at H)/H; 0 for up days."""
    v = idx.to_numpy()
    w = np.zeros(len(v))
    for t in range(1, len(v)):
        if v[t] < v[t - 1]:
            target = v[t] + RECOV_FRAC * (v[t - 1] - v[t])
            tau = H_RECOVERY
            stop = min(t + H_RECOVERY, len(v) - 1)
            for s in range(t + 1, stop + 1):
                if v[s] >= target:
                    tau = s - t
                    break
            w[t] = tau / H_RECOVERY
    return pd.Series(w, index=idx.index)


def trailing_vols(idx: pd.Series) -> pd.DataFrame:
    r = np.log(idx).diff()
    w = recovery_weights(idx)
    contrib = (r.clip(upper=0) ** 2) * w
    out = pd.DataFrame(index=idx.index)
    # shift(H) ⇒ at time T the window only holds days whose recovery is resolved
    out["rwdv_63"]    = np.sqrt(ANN * contrib.shift(H_RECOVERY).rolling(WINDOW).mean())
    out["semidev_63"] = np.sqrt(ANN * (r.clip(upper=0) ** 2).rolling(WINDOW).mean())
    out["sd_63"]      = r.rolling(WINDOW).std() * np.sqrt(ANN)
    return out


# ── 3. Scar-event labels on weekly returns ────────────────────────────────────
def scar_labels(idx: pd.Series) -> pd.DataFrame:
    r = np.log(idx).diff()
    wk_ret = r.resample(WEEK_RULE).sum()
    wk_end_level = idx.resample(WEEK_RULE).last()

    lo = wk_ret.quantile(TAIL_PCT)
    hi = wk_ret.quantile(1 - TAIL_PCT)

    rows = []
    weeks = wk_ret.index
    for i in range(len(weeks) - 1):
        wk, nxt = weeks[i], weeks[i + 1]
        fwd = wk_ret.loc[nxt]
        start_level = wk_end_level.loc[wk]
        seg = idx.loc[wk:nxt].iloc[1:]              # days of the NEXT week
        scar = False
        if len(seg) and not np.isnan(start_level):
            trough_day = seg.idxmin()
            trough = seg.min()
            if trough < start_level:                # there was a drop to recover from
                target = trough + RECOV_FRAC * (start_level - trough)
                after = idx.loc[trough_day:].iloc[1:SCAR_RECOV_DAYS + 1]
                scar = not (after >= target).any()
        rows.append({
            "week":         wk,
            "ret_1w":       wk_ret.loc[wk],
            "ret_1w_fwd":   fwd,
            "extreme_down": int(fwd <= lo and scar),
            "extreme_up":   int(fwd >= hi),
            "cand_down":    int(fwd <= lo),         # decile-only, for the ablation
        })
    return pd.DataFrame(rows).set_index("week")


# ── 4. Mentor's factors: ARKF (FinTech proxy) + QQQ ──────────────────────────
def factor_features(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(["ARKF", "QQQ"], start=start, end=end,
                     progress=False, auto_adjust=True)["Close"]
    r = np.log(px).diff()
    f = pd.DataFrame({
        "arkf_ret_4w": r["ARKF"].rolling(20).sum(),
        "qqq_ret_4w":  r["QQQ"].rolling(20).sum(),
    })
    return f.resample(WEEK_RULE).last()


def main():
    idx = load_index()
    vols = trailing_vols(idx).resample(WEEK_RULE).last()
    labels = scar_labels(idx)
    factors = factor_features(str(idx.index.min().date()), str(idx.index.max().date()))

    df = labels.join(vols, how="left").join(factors, how="left")
    df[["arkf_ret_4w", "qqq_ret_4w"]] = df[["arkf_ret_4w", "qqq_ret_4w"]].ffill()
    df = df.dropna(subset=["rwdv_63", "ret_1w_fwd"])
    df.to_csv(OUT_PATH)

    n, nd, nu, nc = len(df), df["extreme_down"].sum(), df["extreme_up"].sum(), df["cand_down"].sum()
    print(f"Saved {n} weekly rows → {OUT_PATH}")
    print(f"  extreme-DOWN (decile+scar): {nd}  ({nd/n:.1%})   "
          f"[decile-only candidates: {nc} — scar filter kept {nd}/{nc}]")
    print(f"  extreme-UP   (decile)     : {nu}  ({nu/n:.1%})")
    print(f"  corr(rwdv, semidev) = {df['rwdv_63'].corr(df['semidev_63']):.3f}  "
          f"(further from 1 ⇒ RWDV carries unique information)")


if __name__ == "__main__":
    main()
