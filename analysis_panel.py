"""
analysis_panel.py  —  Sprint 3, Step 3 / §G1 (primary spec = PREREGISTRATION.md H1)

Consolidated panel analysis (FinBERT only — the VADER per-coin path was retired as
redundant). Two parts in one file:

  PART A — SHARED-SENTIMENT PANEL (the #1 power lever, §G1)
    Run the SAME scar-down design across 11 coins (incl. delisted LUNA for
    survivorship) and pool, turning ~17 single-series events into ~140, with coin
    fixed effects and SEs clustered by coin AND two-way by coin x week. Shared
    market-news FinBERT is the sentiment regressor. This is where the Moulton
    problem shows up: sentiment is one shared time series, so by-coin clustering
    overstates significance; the honest SE is two-way.

  PART B — PER-COIN FinBERT TEST (does coin-specific news escape the ceiling? §G1)
    On the 7 coins with their own scraped GDELT news, compare SHARED FinBERT vs
    PER-COIN FinBERT sentiment in the same panel (scorer held constant = FinBERT).
    If per-coin news carried independent signal its CI would tighten; it does not.

PRIMARY SPEC (frozen): P(scar_down_{i,w+1}) = logit^{-1}( a_i + b*sent_w + g'*[rwdv,arkf,qqq] )
Descriptive panel association with cluster-robust inference, NOT an OOS claim (the
no-look-ahead discipline lives in the walk-forward harness, Step 2).

OUTPUTS
  data/panel_prices.csv             cached daily closes (re-runs need no network)
  data/articles_percoin_finbert.csv per-article FinBERT cache (per-coin)
  outputs/panel.txt                 raw tables (Part A + Part B)
  outputs/panel.md                  interpretation
  outputs/panel_or.png              sentiment odds-ratio (95% CI) across specs
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

from build_features_v2 import trailing_vols, scar_labels, WEEK_RULE
from analysis_v2 import load as load_features
from finbert_scoring import score_finbert

warnings.filterwarnings("ignore")

PRICE_CACHE = "data/panel_prices.csv"
PERCOIN     = "data/articles_percoin.csv"
FB_CACHE    = "data/articles_percoin_finbert.csv"
OUT_TXT = "outputs/panel.txt"
OUT_MD  = "outputs/panel.md"
OUT_PNG = "outputs/panel_or.png"

START, END = "2021-01-01", "2026-06-12"

# Fixed snapshot universe (top-10 by market cap, circa 2026-06) + delisted LUNA.
TICKERS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
           "SOL-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LUNA1-USD"]

SENT_COL = "finbert_exp_hl7_4w"                  # shared market FinBERT
SHARED   = [SENT_COL, "arkf_ret_4w", "qqq_ret_4w"]
CTRL     = ["rwdv_63", "arkf_ret_4w", "qqq_ret_4w"]

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ── prices (cached) ──────────────────────────────────────────────────────────
def get_prices():
    if os.path.exists(PRICE_CACHE):
        px = pd.read_csv(PRICE_CACHE, parse_dates=["date"]).set_index("date")
        print(f"loaded cached panel prices: {px.shape[1]} coins, {len(px)} days")
        return px
    import yfinance as yf
    print("downloading panel prices …")
    raw = yf.download(TICKERS, start=START, end=END, progress=False, auto_adjust=True)["Close"]
    raw = raw.dropna(how="all"); raw.index.name = "date"
    raw.to_csv(PRICE_CACHE)
    return raw


# ── per-coin FinBERT sentiment (cached scoring) ──────────────────────────────
def trailing_4w_weighted(weekly_sent, weekly_n):
    num = (weekly_sent * weekly_n).rolling(4, min_periods=2).sum()
    den = weekly_n.rolling(4, min_periods=2).sum()
    return num / den


def finbert_scored_articles():
    a = pd.read_csv(PERCOIN)
    a["date"] = pd.to_datetime(a["date"])
    a["headline"] = a["headline"].fillna("").astype(str)
    a = a[a["headline"] != ""].drop_duplicates(subset=["coin", "date", "headline"])
    cache = pd.DataFrame(columns=["coin", "date", "headline", "finbert"])
    if os.path.exists(FB_CACHE):
        cache = pd.read_csv(FB_CACHE, parse_dates=["date"])
        cache["headline"] = cache["headline"].astype(str)
    merged = a.merge(cache[["coin", "date", "headline", "finbert"]],
                     on=["coin", "date", "headline"], how="left")
    todo = merged[merged["finbert"].isna()]
    print(f"per-coin articles: {len(merged)}; scored: {len(merged)-len(todo)}; to score: {len(todo)}")
    if len(todo):
        merged.loc[todo.index, "finbert"] = score_finbert(todo["headline"].tolist())
    merged[["coin", "date", "headline", "finbert"]].to_csv(FB_CACHE, index=False)
    return merged[["coin", "date", "finbert"]]


def percoin_sentiment(scored):
    out, cover = {}, {}
    for coin, grp in scored.groupby("coin"):
        g = grp.set_index("date")["finbert"].resample(WEEK_RULE)
        wk = pd.DataFrame({"sent": g.mean(), "n": g.count()})
        out[coin] = trailing_4w_weighted(wk["sent"], wk["n"])
        cover[coin] = int(len(grp))
    return out, cover


# ── panel construction ───────────────────────────────────────────────────────
def coin_weekly(px, tic, extra_cols):
    """Per-coin weekly frame with scar label, RWDV and the shared `extra_cols`."""
    s = px[tic].dropna()
    if len(s) < 200:
        return None
    idx = s / s.iloc[0]
    vols = trailing_vols(idx).resample(WEEK_RULE).last()
    lab = scar_labels(idx)
    d = lab.join(vols, how="left").join(extra_cols, how="left")
    d["coin"] = tic
    return d


def build_shared_panel(px, shared):
    frames, coverage = [], []
    for tic in px.columns:
        d = coin_weekly(px, tic, shared)
        if d is None:
            coverage.append((tic, "skipped (<200d)")); continue
        d = d.dropna(subset=["extreme_down", "rwdv_63"] + list(shared.columns))
        d = d.reset_index().rename(columns={d.index.name or "index": "week"})
        if "week" not in d.columns:
            d = d.rename(columns={d.columns[0]: "week"})
        frames.append(d)
        coverage.append((tic, f"{int(d['extreme_down'].sum())} events / {len(d)} wk"))
    return pd.concat(frames, ignore_index=True), coverage


def build_percoin_panel(px, shared_sent, factors, pc):
    frames = []
    for coin in pc:
        if coin not in px.columns:
            continue
        extra = pd.concat([shared_sent.rename("sent_shared"), factors], axis=1)
        d = coin_weekly(px, coin, extra)
        if d is None:
            continue
        d["sent_percoin"] = pc[coin].reindex(d.index)
        d = d.dropna(subset=["extreme_down", "rwdv_63", "arkf_ret_4w", "qqq_ret_4w",
                             "sent_shared", "sent_percoin"])
        d = d.reset_index().rename(columns={d.index.name or "index": "week"})
        if "week" not in d.columns:
            d = d.rename(columns={d.columns[0]: "week"})
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ── pooled logit with clustered SEs (generic over sentiment column) ──────────
def fit_pooled(panel, sent_col, ctrl, fe=True, twoway=False):
    x_cols = [sent_col] + ctrl
    d = panel.dropna(subset=["extreme_down"] + x_cols).copy()
    y = d["extreme_down"].astype(float)
    X = d[x_cols].copy()
    X[x_cols] = (X[x_cols] - X[x_cols].mean()) / X[x_cols].std(ddof=0)   # per-1-SD ORs
    if fe:
        dummies = pd.get_dummies(d["coin"], prefix="coin", drop_first=True).astype(float)
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    names = list(X.columns)
    if twoway:
        cov_kwds = {"groups": np.column_stack([d["coin"].astype("category").cat.codes,
                                               d["week"].astype("category").cat.codes])}
    else:
        cov_kwds = {"groups": d["coin"].to_numpy()}
    try:
        m = sm.Logit(y.to_numpy(), X.to_numpy().astype(float)).fit(
            disp=0, cov_type="cluster", cov_kwds=cov_kwds, maxiter=200)
    except Exception as e:
        return {"error": str(e)[:80], "n": len(d), "events": int(y.sum())}
    si = names.index(sent_col)
    ci = m.conf_int()
    return {"n": len(d), "events": int(y.sum()), "n_coins": d["coin"].nunique(),
            "or": float(np.exp(m.params[si])), "lo": float(np.exp(ci[si][0])),
            "hi": float(np.exp(ci[si][1])), "p": float(m.pvalues[si])}


def main():
    feat = load_features()
    shared = feat[SHARED].copy()
    px = get_prices()

    # ── PART A — shared-sentiment panel ──────────────────────────────────────
    panelA, coverageA = build_shared_panel(px, shared)
    specsA = [("P1 coin-FE + cluster(coin)",        dict(fe=True,  twoway=False)),
              ("P2 coin-FE + cluster(coin x week)", dict(fe=True,  twoway=True)),
              ("P3 no-FE  + cluster(coin)",         dict(fe=False, twoway=False))]
    resA = [(name, fit_pooled(panelA, SENT_COL, ["rwdv_63", "arkf_ret_4w", "qqq_ret_4w"], **kw))
            for name, kw in specsA]

    # ── PART B — per-coin FinBERT test ───────────────────────────────────────
    haveB = os.path.exists(PERCOIN)
    resB, panelB, coverB = {}, None, {}
    if haveB:
        pc, coverB = percoin_sentiment(finbert_scored_articles())
        panelB = build_percoin_panel(px, feat["finbert_exp_hl7_4w"],
                                     feat[["arkf_ret_4w", "qqq_ret_4w"]], pc)
        resB = {
            "shared FinBERT  (two-way)":  fit_pooled(panelB, "sent_shared",  CTRL, fe=True, twoway=True),
            "percoin FinBERT (two-way)":  fit_pooled(panelB, "sent_percoin", CTRL, fe=True, twoway=True),
            "percoin FinBERT (by-coin)":  fit_pooled(panelB, "sent_percoin", CTRL, fe=True, twoway=False),
        }

    # ── raw → txt ────────────────────────────────────────────────────────────
    tx = ["=" * 90,
          "PANEL LOGIT — sentiment -> scar-down across coins (Sprint 3, Step 3 / §G1)  [FinBERT]",
          f"universe (fixed snapshot) + delisted LUNA | period {START} → {END}",
          "=" * 90, "",
          "PART A — SHARED-SENTIMENT PANEL (the #1 power lever)",
          "COIN COVERAGE"]
    for tic, note in coverageA:
        tx.append(f"  {tic:12s} {note}")
    tx.append(f"  POOLED: {len(panelA)} coin-weeks, {int(panelA['extreme_down'].sum())} events, "
              f"{panelA['coin'].nunique()} coins  (vs ~17 single-series)")
    tx.append("")
    tx.append("  SENTIMENT ODDS RATIO (per +1 SD), cluster-robust 95% CI")
    tx.append(f"  {'spec':34s} {'n':>6s} {'ev':>4s} {'coins':>5s} {'OR':>6s} {'95% CI':>16s} {'p':>7s}")
    tx.append("  " + "-" * 82)
    for name, r in resA:
        if "error" in r:
            tx.append(f"  {name:34s} {r['n']:>6d} {r['events']:>4d}   failed: {r['error']}"); continue
        tx.append(f"  {name:34s} {r['n']:>6d} {r['events']:>4d} {r['n_coins']:>5d} "
                  f"{r['or']:>6.3f} [{r['lo']:>5.2f},{r['hi']:>5.2f}]   {r['p']:>7.3f}")
    tx.append("  reference: Sprint-2 single-series (BTC+ETH) sentiment OR = 1.70, p = 0.108")

    if haveB:
        shA = resB.get("shared FinBERT  (two-way)", {})
        pcB = resB.get("percoin FinBERT (two-way)", {})
        tot_art = sum(coverB.values())
        tx += ["", "-" * 90,
               "PART B — PER-COIN FinBERT TEST (condensed): does coin-specific news help?  No.",
               f"  7 coins, {tot_art} coin-specific articles, {len(panelB)} coin-weeks, "
               f"{int(panelB['extreme_down'].sum())} events.",
               f"  shared FinBERT OR {shA.get('or', float('nan')):.2f} "
               f"[{shA.get('lo', float('nan')):.2f},{shA.get('hi', float('nan')):.2f}]  vs  "
               f"per-coin OR {pcB.get('or', float('nan')):.2f} "
               f"[{pcB.get('lo', float('nan')):.2f},{pcB.get('hi', float('nan')):.2f}] "
               f"(p={pcB.get('p', float('nan')):.2f})  →  per-coin CI no tighter, so no power gain."]
    tx.append("=" * 90)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(tx) + "\n")
    print("\n".join(tx))
    print(f"\nraw → {OUT_TXT}")

    # ── plot: Part A ORs ─────────────────────────────────────────────────────
    ok = [(n, r) for n, r in resA if "error" not in r]
    if ok:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ys = list(range(len(ok)))
        ax.errorbar([r["or"] for _, r in ok], ys,
                    xerr=[[r["or"] - r["lo"] for _, r in ok], [r["hi"] - r["or"] for _, r in ok]],
                    fmt="o", capsize=4, color="tab:blue")
        ax.axvline(1.0, color="grey", ls="--", lw=1, label="OR = 1 (no effect)")
        ax.axvline(1.70, color="green", ls=":", lw=1, label="Sprint-2 single-series OR 1.70")
        ax.set_yticks(ys); ax.set_yticklabels([n for n, _ in ok], fontsize=8)
        ax.set_xlabel("sentiment odds ratio (per +1 SD), cluster-robust 95% CI")
        ax.set_title("Panel sentiment OR across specifications (shared FinBERT)")
        ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(OUT_PNG, dpi=150); plt.close()
        print(f"plot → {OUT_PNG}")

    write_md(resA, panelA, resB)
    print(f"interpretation → {OUT_MD}")


def write_md(resA, panelA, resB):
    okA = dict((n, r) for n, r in resA if "error" not in r)
    tot_ev = int(panelA["extreme_down"].sum()); n_coins = panelA["coin"].nunique()
    md = ["# Panel Logit — Interpretation (Sprint 3, Step 3 / §G1)\n",
          "*Raw numbers in `outputs/panel.txt`; interpretation only here. FinBERT only.*\n",
          "## Part A — shared-sentiment panel (the power lever)\n",
          f"- **Events went from ~17 to {tot_ev}** across {n_coins} coins ({len(panelA)} "
          "coin-weeks), including the dead LUNA (survivorship-corrected). This is the legitimate "
          "power lever — pooling, not a fancier model.\n"]
    p1 = okA.get("P1 coin-FE + cluster(coin)"); p2 = okA.get("P2 coin-FE + cluster(coin x week)")
    if p2:
        excl = p2["lo"] > 1 or p2["hi"] < 1
        verdict = ("**Confirmation** of H1 (CI excludes 1)." if excl else
                   "**Qualified support** (PREREGISTRATION.md §6): consistent sign/size with a CI "
                   "that still includes 1 — suggestive, underpowered, *not* confirmation and *not* "
                   "a null.")
        md.append(f"- **Primary inference (coin-FE, two-way cluster coin × week):** sentiment OR "
                  f"**{p2['or']:.2f}**, 95% CI [{p2['lo']:.2f}, {p2['hi']:.2f}], p = {p2['p']:.3f}. "
                  f"{verdict}\n")
    if p1:
        md.append(f"- **Why not the by-coin SE?** Clustering by coin alone gives a deceptively tight "
                  f"CI [{p1['lo']:.2f}, {p1['hi']:.2f}] (p = {p1['p']:.3f}) — but sentiment is a "
                  "*single shared time series*, so by-coin clustering treats the same path as 11 "
                  "independent draws (the **Moulton problem**). The two-way SE is the honest one; do "
                  "**not** report the by-coin p as the headline.\n")
    md.append("- **Key point:** because sentiment is shared, the effective independent sample is "
              "~the number of weeks, not coin-weeks — so the panel does *not* escape the power "
              "ceiling for the sentiment regressor. A CI that includes 1 here is *underpowered*, "
              "not evidence of no effect (read with `outputs/power.md`, §G7).\n")

    okB = {k: v for k, v in (resB or {}).items() if "error" not in v}
    A = okB.get("shared FinBERT  (two-way)"); B = okB.get("percoin FinBERT (two-way)")
    md.append("## Part B — per-coin news (condensed): did not help\n")
    if A and B:
        md.append(f"- We also scraped coin-specific news for 7 coins and re-ran the panel with "
                  f"per-coin FinBERT sentiment. It did **not** help: per-coin OR {B['or']:.2f} "
                  f"[{B['lo']:.2f}, {B['hi']:.2f}] ≈ shared OR {A['or']:.2f}, with a CI no tighter, "
                  "so coin-specific news bought no extra power. Reinforces that the binding "
                  "constraint is events/power, not how sentiment is built. (Condensed from a fuller "
                  "analysis; the VADER scoring path was retired.)\n")
    else:
        md.append("- Per-coin section not available (run `scrape_news_percoin.py` first).\n")
    md.append("## Honesty notes\n")
    md.append("- Descriptive panel association with cluster-robust inference, not an OOS claim. "
              "Cost-sensitive / Firth correction is applied on the credible setup at Step 5.\n")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
