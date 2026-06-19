"""
analysis_firth.py
Sprint 2 step 1 (sprint2plan.md §3A1): re-estimate the weekly multi-factor model
with FIRTH'S PENALIZED LOGISTIC REGRESSION instead of ordinary logistic regression.

Why: with only ~17 extreme events, ordinary maximum-likelihood logistic regression
is biased and its Wald p-values are unreliable. Firth (1993) adds a Jeffreys-prior
penalty (+0.5·log det(X'WX)) to the log-likelihood, which removes the small-sample
bias and yields trustworthy inference for rare events. We report PENALIZED
LIKELIHOOD-RATIO p-values (refit with the coefficient constrained to 0), which behave
far better than Wald p-values at small n, and cross-check the sentiment coefficient
with a label-permutation test (a fully assumption-free p-value).

Implemented from scratch (no external Firth package) so every step is auditable.

Inputs : data/sentiment_weekly.csv + data/dataset_weekly.csv
Outputs: console table + outputs/firth_results.txt (raw table; interpretation in firth_results.md)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

SENT_PATH = "data/sentiment_weekly.csv"
FEAT_PATH = "data/dataset_weekly.csv"
OUT_TXT   = "outputs/firth_results.txt"   # raw table → .txt; interpretation lives in the .md
N_PERM    = 5000
SEED      = 42

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(SEED)


# ── data (same construction as analysis_v2) ──────────────────────────────────
def load() -> pd.DataFrame:
    s = pd.read_csv(SENT_PATH, parse_dates=["week"]).set_index("week")
    f = pd.read_csv(FEAT_PATH, parse_dates=["week"]).set_index("week")
    df = f.join(s, how="left").sort_index()
    for col in ["finbert_exp_hl7", "vader_exp_hl7"]:
        num = (df[col] * df["n_articles"]).rolling(4, min_periods=2).sum()
        den = df["n_articles"].rolling(4, min_periods=2).sum()
        df[col + "_4w"] = num / den
    return df


# ── Firth penalized logistic regression (IRLS) ───────────────────────────────
def firth_fit(X: np.ndarray, y: np.ndarray, max_iter=1000, tol=1e-10):
    """Return (beta, penalized_loglik). X includes an intercept column."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta
        p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        w = p * (1.0 - p)
        XtWX = X.T @ (X * w[:, None])
        XtWX_inv = np.linalg.pinv(XtWX)
        # hat-matrix diagonal: h_i = w_i * x_i' (X'WX)^-1 x_i
        hii = w * np.einsum("ij,jk,ik->i", X, XtWX_inv, X)
        U = X.T @ (y - p + hii * (0.5 - p))      # Firth-modified score
        step = XtWX_inv @ U
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    w = p * (1.0 - p)
    _, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
    return beta, ll + 0.5 * logdet               # penalized log-likelihood


def firth_plr_pvalues(X: np.ndarray, y: np.ndarray):
    """Penalized likelihood-ratio p-value for every non-intercept column."""
    beta_full, pll_full = firth_fit(X, y)
    pvals = np.full(X.shape[1], np.nan)
    for j in range(1, X.shape[1]):               # skip intercept (col 0)
        keep = [c for c in range(X.shape[1]) if c != j]
        _, pll_red = firth_fit(X[:, keep], y)
        lr = 2.0 * (pll_full - pll_red)
        pvals[j] = stats.chi2.sf(max(lr, 0.0), df=1)
    return beta_full, pvals


def design(df: pd.DataFrame, x_cols: list, y_col: str):
    d = df.dropna(subset=[y_col] + x_cols)
    y = d[y_col].to_numpy(float)
    Z = (d[x_cols] - d[x_cols].mean()) / d[x_cols].std(ddof=0)   # per-1-SD odds
    X = np.column_stack([np.ones(len(d)), Z.to_numpy()])
    return X, y


def permutation_p(X, y, n_perm=N_PERM):
    """Assumption-free p-value for the sentiment coef (column 1) via label shuffling."""
    beta, _ = firth_fit(X, y)
    obs = abs(beta[1])
    count = 0
    yp = y.copy()
    for _ in range(n_perm):
        rng.shuffle(yp)
        b, _ = firth_fit(X, yp)
        if abs(b[1]) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    df = load()
    fin = "finbert_exp_hl7_4w"
    ctrl = ["rwdv_63", "arkf_ret_4w", "qqq_ret_4w"]

    specs = [
        ("CONFIRMATION: FinBERT-4w + RWDV + ARKF + QQQ", "extreme_down", [fin] + ctrl),
        ("FinBERT-4w alone",                             "extreme_down", [fin]),
        ("decile-only label (no scar)",                  "cand_down",    [fin] + ctrl),
        ("extreme-UP side",                              "extreme_up",   [fin] + ctrl),
    ]

    bar = "=" * 92
    lines = [bar, "  FIRTH PENALIZED LOGISTIC REGRESSION — rare-event-correct inference", bar,
             f"  {'spec':46s} {'n':>4s} {'ev':>3s} {'odds(sent)':>10s} {'p(PLR)':>8s}",
             "-" * 92]
    conf = None
    for name, y_col, x_cols in specs:
        X, y = design(df, x_cols, y_col)
        beta, pvals = firth_plr_pvalues(X, y)
        odds_sent, p_sent = np.exp(beta[1]), pvals[1]
        if name.startswith("CONFIRMATION"):
            conf = (X, y, beta, pvals, x_cols)
        star = " *" if p_sent < 0.05 else ""
        lines.append(f"  {name:46s} {len(y):>4d} {int(y.sum()):>3d} "
                     f"{odds_sent:>10.3f} {p_sent:>8.3f}{star}")
    lines += [bar, "  * = p < 0.05 (penalized likelihood-ratio). odds per 1-SD of the sentiment composite.", ""]

    # full coefficient table + permutation backstop for the confirmation spec
    if conf is not None:
        X, y, beta, pvals, x_cols = conf
        names = ["(intercept)"] + x_cols
        lines += [bar, "  CONFIRMATION SPEC — Firth coefficients (per 1-SD)", bar]
        for j, nm in enumerate(names):
            if j == 0:
                continue
            star = " *" if pvals[j] < 0.05 else ""
            lines.append(f"  {nm:18s} odds = {np.exp(beta[j]):6.3f}   p(PLR) = {pvals[j]:.3f}{star}")
        pperm = permutation_p(X, y)
        lines += [bar,
                  f"  Permutation test on sentiment coef ({N_PERM} shuffles): p = {pperm:.4f}"
                  + ("  *" if pperm < 0.05 else ""),
                  "  (assumption-free cross-check of the Firth p-value above)", ""]

    report = "\n".join(lines)
    print(report)
    with open(OUT_TXT, "w") as fh:
        fh.write(report + "\n")
    print(f"Saved → {OUT_TXT}")


if __name__ == "__main__":
    main()
