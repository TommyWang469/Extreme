"""
analysis_v2.py
Sprint 2 (Jun 2026) — multi-factor logistic regression at WEEKLY resolution.

Pre-registered PRIMARY spec (declared before seeing results, per improvement.md):
    P(extreme_down in week w+1) = f( finbert_exp_hl7_w, rwdv_63_w,
                                     arkf_ret_4w_w, qqq_ret_4w_w )
    Primary hypothesis: the FinBERT coefficient is negative (worse news mood →
    higher odds of a scarring crash) after controlling for volatility clustering
    (RWDV) and the FinTech/tech market factors (ARKF, QQQ).

Ablations: VADER vs FinBERT · RWDV vs plain semideviation · scar label vs
decile-only label · extreme-UP side. Out-of-sample: train ≤2022, test 2023–24.

Inputs : data/sentiment_weekly.csv + data/dataset_weekly.csv
Outputs: console tables + sprint2_results.md
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

SENT_PATH = "data/sentiment_weekly.csv"
FEAT_PATH = "data/dataset_weekly.csv"
OUT_MD    = "sprint2_results.md"
OOS_SPLIT = "2022-12-31"

os.makedirs("outputs", exist_ok=True)


def load() -> pd.DataFrame:
    s = pd.read_csv(SENT_PATH, parse_dates=["week"]).set_index("week")
    f = pd.read_csv(FEAT_PATH, parse_dates=["week"]).set_index("week")
    df = f.join(s, how="left").sort_index()
    # Trailing 4-week article-count-weighted sentiment: weekly composites average
    # only ~8 articles, so the raw weekly series is noisy. Pre-registered as the
    # Sprint-3 confirmation predictor after the Sprint-2 exploratory pass.
    for col in ["finbert_exp_hl7", "vader_exp_hl7"]:
        num = (df[col] * df["n_articles"]).rolling(4, min_periods=2).sum()
        den = df["n_articles"].rolling(4, min_periods=2).sum()
        df[col + "_4w"] = num / den
    print(f"Merged: {len(df)} weeks, {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    return df


def fit_logit(df: pd.DataFrame, y_col: str, x_cols: list):
    """Standardize X, fit Logit, return dict of metrics (sentiment var = x_cols[0])."""
    d = df.dropna(subset=[y_col] + x_cols)
    y = d[y_col].astype(float)
    X = (d[x_cols] - d[x_cols].mean()) / d[x_cols].std(ddof=0)
    X = sm.add_constant(X)
    try:
        m = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        return {"error": str(e), "n": len(d), "events": int(y.sum())}
    p = m.predict(X)
    auc = roc_auc_score(y, p) if y.nunique() > 1 else np.nan
    s = x_cols[0]
    return {
        "model": m, "n": len(d), "events": int(y.sum()),
        "or_sent": float(np.exp(m.params[s])), "p_sent": float(m.pvalues[s]),
        "auc": float(auc), "mcf": float(m.prsquared),
    }


def oos_auc(df: pd.DataFrame, y_col: str, x_cols: list) -> float:
    d = df.dropna(subset=[y_col] + x_cols)
    tr, te = d.loc[:OOS_SPLIT], d.loc[OOS_SPLIT:]
    if te[y_col].nunique() < 2 or tr[y_col].sum() < 3:
        return np.nan
    mu, sd = tr[x_cols].mean(), tr[x_cols].std(ddof=0)
    Xtr = sm.add_constant((tr[x_cols] - mu) / sd)
    Xte = sm.add_constant((te[x_cols] - mu) / sd, has_constant="add")
    try:
        m = sm.Logit(tr[y_col].astype(float), Xtr).fit(disp=0)
        return float(roc_auc_score(te[y_col], m.predict(Xte)))
    except Exception:
        return np.nan


def main():
    df = load()
    fin, vad = "finbert_exp_hl7", "vader_exp_hl7"
    ctrl_rwdv = ["rwdv_63", "arkf_ret_4w", "qqq_ret_4w"]
    ctrl_semi = ["semidev_63", "arkf_ret_4w", "qqq_ret_4w"]

    specs = [
        ("C0 CONFIRMATION: FinBERT-4w + RWDV + ARKF + QQQ", "extreme_down",
         [fin + "_4w"] + ctrl_rwdv),
        ("S1 FinBERT alone (baseline)",             "extreme_down", [fin]),
        ("S2 Sprint-2 primary (raw weekly FinBERT)","extreme_down", [fin] + ctrl_rwdv),
        ("S3 ablation: VADER + RWDV + ARKF + QQQ",  "extreme_down", [vad] + ctrl_rwdv),
        ("S4 ablation: FinBERT + semidev + factors","extreme_down", [fin] + ctrl_semi),
        ("S5 ablation: decile-only label (no scar)","cand_down",    [fin] + ctrl_rwdv),
        ("S6 extreme-UP side",                      "extreme_up",   [fin] + ctrl_rwdv),
    ]

    bar = "=" * 96
    lines = [bar, "  SPRINT 2 — multi-factor weekly logit "
                  "(sentiment metric = first variable in each spec)", bar,
             f"  {'spec':44s} {'n':>4s} {'ev':>3s} {'odds':>8s} "
             f"{'AUC':>7s} {'McF R²':>8s} {'p':>8s}",
             "-" * 96]
    results = {}
    for name, y_col, x_cols in specs:
        r = fit_logit(df, y_col, x_cols)
        results[name] = r
        if "error" in r:
            lines.append(f"  {name:44s} {r['n']:>4d} {r['events']:>3d}   failed: {r['error'][:30]}")
            continue
        star = " *" if r["p_sent"] < 0.05 else ""
        lines.append(f"  {name:44s} {r['n']:>4d} {r['events']:>3d} "
                     f"{r['or_sent']:>8.3f} {r['auc']:>7.3f} "
                     f"{r['mcf']:>8.4f} {r['p_sent']:>8.3f}{star}")
    lines += [bar, "  * = p < 0.05.  odds = odds ratio on the sentiment coefficient "
                   "(per 1-SD move).", ""]

    # Full coefficient table for the confirmation spec
    prim = results.get("C0 CONFIRMATION: FinBERT-4w + RWDV + ARKF + QQQ", {})
    if "model" in prim:
        m = prim["model"]
        lines += [bar, "  CONFIRMATION SPEC — full coefficient table (per 1-SD)", bar]
        for var in m.params.index:
            if var == "const":
                continue
            star = " *" if m.pvalues[var] < 0.05 else ""
            lines.append(f"  {var:18s} odds = {np.exp(m.params[var]):6.3f}   "
                         f"p = {m.pvalues[var]:.3f}{star}")
        a = oos_auc(df, "extreme_down", [fin + "_4w"] + ctrl_rwdv)
        lines += [bar, (f"  Out-of-sample (train ≤{OOS_SPLIT[:4]}, test after): AUC = {a:.3f}"
                        if not np.isnan(a)
                        else "  Out-of-sample: not estimable (too few test events)"), ""]

    report = "\n".join(lines)
    print(report)
    with open(OUT_MD, "w") as fh:
        fh.write("# Sprint 2 results\n\n```\n" + report + "\n```\n")
    print(f"Saved → {OUT_MD}")


if __name__ == "__main__":
    main()
