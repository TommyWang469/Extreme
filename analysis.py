"""
analysis.py
Merges monthly sentiment with labeled extreme events and evaluates sentiment as
an early-warning predictor via logistic regression.

SPRINT 1 CHANGES (improvement.md T10)
─────────────────────────────────────────────────────────────────────────────
  • Compares several predictor SPECS in one run, each reporting odds ratio,
    AUC-ROC, McFadden R² and p-value:
        1. vader_linear            (contemporaneous, equal-weight)
        2. vader_exp_hl7           (contemporaneous, recency-weighted)
        3. vader_linear  (lag 1)   (sentiment in month T → event in T+1)
        4. vader_exp_hl7 (lag 1)   (recency-weighted, lagged)
    Lagging tests the actual research question: does sentiment *precede* events?
  • Prints a ranked comparison table and saves the ROC curve for the best spec.
  • Pilot-period benchmark check: compares our Feb-2024 composite against the
    mentor's reference (≈ −0.06 on the raw VADER scale).

Robustness note: with ~6 extreme months in ~45 observations the estimates are
noisy; treat every metric as directional, not definitive.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# ── Settings ──────────────────────────────────────────────────────────────────
EVENTS_PATH    = "data/labeled_events.csv"
SENTIMENT_PATH = "data/sentiment_scores.csv"
ROC_OUT        = "outputs/roc_curve.png"
PILOT_MONTH    = "2024-02"
PILOT_REF      = -0.06     # mentor's reference composite for the pilot month

# (column, lag_in_months, label)
SPECS = [
    ("vader_linear",  0, "VADER linear (contemporaneous)"),
    ("vader_exp_hl7", 0, "VADER exp-hl7 (contemporaneous)"),
    ("vader_linear",  1, "VADER linear (lag 1 → predictive)"),
    ("vader_exp_hl7", 1, "VADER exp-hl7 (lag 1 → predictive)"),
]

os.makedirs("outputs", exist_ok=True)


def load_merged() -> pd.DataFrame:
    events    = pd.read_csv(EVENTS_PATH,    parse_dates=["date"], index_col="date")
    sentiment = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"], index_col="date")
    # Normalise both indices to a month-end anchor so the join is robust.
    events.index    = events.index.to_period("M").to_timestamp("M")
    sentiment.index = sentiment.index.to_period("M").to_timestamp("M")
    df = events[["extreme_binary"]].join(sentiment, how="inner")
    return df.sort_index()


def fit_spec(df: pd.DataFrame, col: str, lag: int):
    """Fit logistic regression for one predictor spec; return metrics dict or None."""
    d = df[["extreme_binary", col]].copy()
    if lag:
        d[col] = d[col].shift(lag)      # sentiment at T-lag predicts event at T
    d = d.dropna()
    y = d["extreme_binary"].to_numpy()
    if y.sum() < 2 or len(d) < 10:
        return None

    X = StandardScaler().fit_transform(d[[col]].to_numpy())
    X_sm = sm.add_constant(X)
    try:
        res = sm.Logit(y, X_sm).fit(disp=False)
    except Exception:
        return None

    prob = res.predict(X_sm)
    return {
        "n":         len(d),
        "n_events":  int(y.sum()),
        "odds":      float(np.exp(res.params[1])),
        "auc":       float(roc_auc_score(y, prob)),
        "mcfadden":  float(1 - res.llf / res.llnull),
        "pval":      float(res.pvalues[1]),
        "_y":        y,
        "_prob":     prob,
    }


def main():
    print("Loading + merging …")
    df = load_merged()
    print(f"  Merged dataset: {len(df)} monthly rows, "
          f"{int(df['extreme_binary'].sum())} extreme months "
          f"({df['extreme_binary'].mean():.1%})")
    if df["extreme_binary"].sum() < 2:
        raise ValueError("Fewer than 2 extreme months — relax the threshold in event_definition.py")

    # ── Evaluate every spec ───────────────────────────────────────────────────
    results = []
    for col, lag, label in SPECS:
        m = fit_spec(df, col, lag)
        if m:
            m.update(col=col, lag=lag, label=label)
            results.append(m)

    # ── Ranked comparison table (rank by AUC) ─────────────────────────────────
    results.sort(key=lambda r: r["auc"], reverse=True)
    print("\n" + "=" * 78)
    print("  PREDICTOR COMPARISON  (ranked by AUC)")
    print("=" * 78)
    print(f"  {'spec':<34}{'n':>4}{'ev':>4}{'odds':>8}{'AUC':>7}{'McF R²':>8}{'p':>8}")
    print("-" * 78)
    for r in results:
        star = "*" if r["pval"] < 0.05 else " "
        print(f"  {r['label']:<34}{r['n']:>4}{r['n_events']:>4}"
              f"{r['odds']:>8.3f}{r['auc']:>7.3f}{r['mcfadden']:>8.4f}{r['pval']:>7.3f}{star}")
    print("=" * 78)
    print("  * = p < 0.05.  ev = number of extreme months in that spec.")

    # ── ROC curve for the best spec ───────────────────────────────────────────
    best = results[0]
    fpr, tpr, _ = roc_curve(best["_y"], best["_prob"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"{best['label']} (AUC = {best['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Best Sentiment Predictor of Extreme Crypto Events")
    ax.legend(loc="lower right")
    ax.annotate(f"McFadden R² = {best['mcfadden']:.4f}\nOdds Ratio = {best['odds']:.3f}\n"
                f"p = {best['pval']:.3f}",
                xy=(0.5, 0.1), xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"))
    plt.tight_layout(); plt.savefig(ROC_OUT, dpi=150); plt.close()
    print(f"\nROC curve (best spec) saved → {ROC_OUT}")

    # ── Pilot-period benchmark check ──────────────────────────────────────────
    try:
        sentiment = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"], index_col="date")
        sentiment.index = sentiment.index.to_period("M")
        row = sentiment.loc[pd.Period(PILOT_MONTH, "M")]
        ours = float(row["vader_linear"])
        print("\n" + "-" * 50)
        print(f"  PILOT BENCHMARK ({PILOT_MONTH})")
        print(f"    Our composite (vader_linear): {ours:+.3f}")
        print(f"    Mentor reference            : {PILOT_REF:+.3f}")
        print(f"    Gap                         : {abs(ours - PILOT_REF):.3f}"
              f"  {'✓ aligned' if abs(ours - PILOT_REF) < 0.15 else '⚠ investigate article selection'}")
        print("-" * 50)
    except Exception as e:
        print(f"\n(Pilot check skipped: {e})")

    print("\nFraming: early-warning signal, not a causal claim. "
          "Metrics are directional given the small sample.")


if __name__ == "__main__":
    main()
