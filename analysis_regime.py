"""
analysis_regime.py
Sprint 2 step 2 (sprint2plan.md §3C): test whether the sentiment→crash relationship
is REGIME-DEPENDENT — strong in the 2021–22 retail/chaos era, faded after crypto
institutionalised. This is the mentor's "3–4 regimes; how do you recognise the shift?"
question, and the place a real within-regime effect is most likely to live.

Three things, all rare-event-correct (Firth, reused from analysis_firth.py):
  1. Fit the confirmation spec separately in each regime (split 2023-01-01).
  2. For each regime report a FIRTH penalized-LR p-value AND a BLOCK-permutation
     p-value. Block permutation shuffles contiguous blocks of weeks, preserving the
     autocorrelation of the rolling-window predictors — so it does NOT over-state
     significance the way the step-1 plain permutation / asymptotic χ² can.
  3. A formal sentiment × regime INTERACTION test on the full sample: is the
     sentiment coefficient statistically different between the two regimes?

Inputs : data/sentiment_weekly.csv + data/dataset_weekly.csv  (via analysis_firth.load)
Outputs: console table + outputs/regime_results.txt (raw table; interpretation in regime_results.md)
"""

import os
import numpy as np
import pandas as pd
from analysis_firth import load, firth_fit, firth_plr_pvalues, design

SPLIT   = "2023-01-01"      # retail/chaos (before) vs institutional (after)
BLOCK   = 8                 # weeks per block ≈ 2 months ≈ rwdv autocorrelation length
N_PERM  = 3000
SEED    = 7
OUT_TXT = "outputs/regime_results.txt"   # raw table → .txt; interpretation lives in the .md

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(SEED)

FIN  = "finbert_exp_hl7_4w"
CTRL = ["rwdv_63", "arkf_ret_4w", "qqq_ret_4w"]
YCOL = "extreme_down"


def block_perm_p(X, y, block=BLOCK, n_perm=N_PERM):
    """Permutation p-value for the sentiment coef (col 1), preserving y autocorrelation
    by shuffling contiguous blocks rather than individual weeks."""
    beta, _ = firth_fit(X, y)
    obs = abs(beta[1])
    n = len(y)
    nb = int(np.ceil(n / block))
    blocks = [np.arange(b * block, min((b + 1) * block, n)) for b in range(nb)]
    count = 0
    for _ in range(n_perm):
        order = rng.permutation(nb)
        idx = np.concatenate([blocks[b] for b in order])
        b2, _ = firth_fit(X, y[idx])
        if abs(b2[1]) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def fit_block(df, label):
    X, y = design(df, [FIN] + CTRL, YCOL)
    beta, pvals = firth_plr_pvalues(X, y)
    bp = block_perm_p(X, y)
    return {
        "label": label, "n": len(y), "ev": int(y.sum()),
        "odds": float(np.exp(beta[1])), "p_firth": float(pvals[1]), "p_block": float(bp),
    }


def interaction_test(df):
    """Full-sample Firth with sentiment × late-regime interaction. PLR p on the
    interaction term = formal test that the sentiment effect differs by regime."""
    d = df.dropna(subset=[YCOL, FIN] + CTRL).copy()
    late = (d.index >= SPLIT).astype(float)
    cols = [FIN] + CTRL
    Z = (d[cols] - d[cols].mean()) / d[cols].std(ddof=0)
    sent = Z[FIN].to_numpy()
    inter = sent * late
    X = np.column_stack([np.ones(len(d)), sent, late, inter, Z[CTRL].to_numpy()])
    y = d[YCOL].to_numpy(float)
    beta, pvals = firth_plr_pvalues(X, y)
    # columns: 0 const,1 sent,2 late,3 inter,4 rwdv,5 arkf,6 qqq
    return {"odds_inter": float(np.exp(beta[3])), "p_inter": float(pvals[3]),
            "n": len(d), "ev": int(y.sum())}


def main():
    df = load()
    full  = df
    early = df[df.index < SPLIT]
    late  = df[df.index >= SPLIT]

    rows = [fit_block(full,  "Full sample 2021-2026"),
            fit_block(early, f"Early regime  < {SPLIT} (retail/chaos)"),
            fit_block(late,  f"Late regime  >= {SPLIT} (institutional)")]

    bar = "=" * 96
    L = [bar, "  REGIME-SPLIT TEST — sentiment -> scarring-crash, Firth + block-permutation", bar,
         f"  {'regime':40s} {'n':>4s} {'ev':>3s} {'odds(sent)':>11s} "
         f"{'p(Firth)':>9s} {'p(block)':>9s}",
         "-" * 96]
    for r in rows:
        s1 = " *" if r["p_firth"] < 0.05 else ""
        s2 = " *" if r["p_block"] < 0.05 else ""
        L.append(f"  {r['label']:40s} {r['n']:>4d} {r['ev']:>3d} {r['odds']:>11.3f} "
                 f"{r['p_firth']:>8.3f}{s1:2s} {r['p_block']:>8.3f}{s2:2s}")
    L += [bar,
          "  p(Firth) = penalized likelihood-ratio (parametric).",
          "  p(block) = block-permutation, "
          f"{BLOCK}-week blocks, {N_PERM} shuffles (autocorrelation-robust — the honest one).",
          ""]

    it = interaction_test(df)
    star = " *" if it["p_inter"] < 0.05 else ""
    L += [bar, "  FORMAL REGIME-SHIFT TEST  (sentiment x late-regime interaction, Firth)", bar,
          f"  interaction odds = {it['odds_inter']:.3f}   p(PLR) = {it['p_inter']:.3f}{star}"
          f"   (n={it['n']}, events={it['ev']})",
          "  interaction p < 0.05 ⇒ the sentiment effect is statistically different "
          "between regimes.", ""]

    report = "\n".join(L)
    print(report)
    with open(OUT_TXT, "w") as fh:
        fh.write(report + "\n")
    print(f"Saved → {OUT_TXT}")


if __name__ == "__main__":
    main()
