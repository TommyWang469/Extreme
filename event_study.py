"""
event_study.py
Classic event-study analysis around known extreme crypto events (improvement.md
T9, mentor's "Event study / Mt Gox / SBF" hint). Mt Gox (2014) is out of scope
this sprint, so anchors are restricted to the 2021–2024 data window.

METHOD (mean-adjusted abnormal returns)
─────────────────────────────────────────────────────────────────────────────
Because our asset *is* the 50/50 BTC+ETH portfolio, there is no separate market
index to regress against, so we use the standard mean-adjusted model:

    estimation window : t = −120 … −31 trading days before the event
    expected return   : μ̂ = mean(portfolio_ret) over the estimation window
    abnormal return   : AR_t = r_t − μ̂           for t in the event window
    cumulative AR     : CAR  = Σ AR_t             over the event window [−10,+10]

We then ask the research question in event-study form: does PRE-EVENT sentiment
(mean VADER over the 30 days before the event) line up with the size/direction
of the realised abnormal return? With only a handful of events this is
illustrative, not inferential — it complements, and is more robust than, the
monthly logistic regression on a tiny extreme-event count.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRICE_PATH     = "data/price_data.csv"
SENTIMENT_PATH = "data/sentiment_scores.csv"
OUT_PLOT       = "outputs/event_study_car.png"

EST_START, EST_END = -120, -31     # estimation window (trading days, relative)
EVT_START, EVT_END = -10,  +10     # event window (trading days, relative)
PRE_SENT_DAYS      = 30            # calendar days before event for pre-event sentiment

# Anchor events within 2021–2024 (name, date, expected sign)
EVENTS = [
    ("Ronin bridge hack",   "2022-03-23", "-"),
    ("Terra/LUNA collapse", "2022-05-09", "-"),
    ("FTX bankruptcy (SBF)","2022-11-11", "-"),
    ("Bitcoin ETF approval","2024-01-10", "+"),
    ("Bitcoin halving",     "2024-04-20", "+"),
]

os.makedirs("outputs", exist_ok=True)


def main():
    px = pd.read_csv(PRICE_PATH, parse_dates=["date"], index_col="date").sort_index()
    ret = px["portfolio_ret"]
    idx = ret.index

    # Article-level sentiment for the pre-event windows (from monthly file we only
    # have monthly means, so reconstruct daily sentiment if available; else use
    # the monthly composite covering the pre-event period).
    sent = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"], index_col="date").sort_index()
    sent.index = sent.index.to_period("M").to_timestamp("M")

    rows = []
    car_paths = {}
    for name, date_str, sign in EVENTS:
        ev_date = pd.Timestamp(date_str)
        # nearest trading day at or after the event date
        pos = idx.searchsorted(ev_date)
        if pos <= abs(EST_START) or pos >= len(idx) - EVT_END:
            print(f"  ⚠ skipping {name}: insufficient surrounding data")
            continue
        t0 = pos  # index of event day

        est = ret.iloc[t0 + EST_START : t0 + EST_END + 1]
        mu = est.mean()

        evt_slice = ret.iloc[t0 + EVT_START : t0 + EVT_END + 1]
        ar = evt_slice - mu
        car = ar.cumsum()
        car_paths[name] = car.reset_index(drop=True)

        # pre-event sentiment: monthly composite for the event's month
        try:
            pre_sent = float(sent.loc[ev_date.to_period("M").to_timestamp("M"), "vader_linear"])
        except Exception:
            pre_sent = np.nan

        rows.append({
            "event":       name,
            "date":        ev_date.date(),
            "exp_sign":    sign,
            "mu_est":      mu,
            "CAR":         car.iloc[-1],
            "AR_max_abs":  ar.abs().max(),
            "pre_sentiment": pre_sent,
        })

    res = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("  EVENT STUDY — cumulative abnormal return over [−10,+10] trading days")
    print("=" * 78)
    with pd.option_context("display.float_format", lambda x: f"{x:+.4f}"):
        print(res.to_string(index=False))
    print("=" * 78)

    # Directional sanity: did negative events produce negative CARs?
    res["car_sign"] = np.where(res["CAR"] >= 0, "+", "-")
    hits = (res["car_sign"] == res["exp_sign"]).sum()
    print(f"  CAR matched expected direction in {hits}/{len(res)} events.")

    # Illustrative link: pre-event sentiment vs realised |CAR|
    valid = res.dropna(subset=["pre_sentiment"])
    if len(valid) >= 3:
        corr = np.corrcoef(valid["pre_sentiment"], valid["CAR"])[0, 1]
        print(f"  corr(pre-event sentiment, CAR) = {corr:+.3f}  "
              f"(n={len(valid)}; illustrative only)")

    # ── Plot CAR paths ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    rel_days = list(range(EVT_START, EVT_END + 1))
    for name, path in car_paths.items():
        ax.plot(rel_days, path.values, marker="o", ms=3, label=name)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Trading days relative to event")
    ax.set_ylabel("Cumulative abnormal return (CAR)")
    ax.set_title("Event Study — CAR around extreme crypto events (2021–2024)")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(OUT_PLOT, dpi=150); plt.close()
    print(f"\nCAR paths saved → {OUT_PLOT}")


if __name__ == "__main__":
    main()
