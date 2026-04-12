"""
analysis.py
Merges sentiment scores with labeled extreme events, runs a logistic regression
(sentiment → extreme binary), and reports:
    • Odds ratio  (exp of the β₁ coefficient)
    • AUC-ROC     (sklearn roc_auc_score)
    • McFadden R² (1 − LL_model / LL_null)

Also saves an ROC curve plot to outputs/roc_curve.png.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# ── Settings ──────────────────────────────────────────────────────────────────
EVENTS_PATH    = "data/labeled_events.csv"
SENTIMENT_PATH = "data/sentiment_scores.csv"
ROC_OUT        = "outputs/roc_curve.png"
SENTIMENT_COL  = "vader_compound"   # primary predictor per CONTEXT.md

os.makedirs("outputs", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data …")
events    = pd.read_csv(EVENTS_PATH,    parse_dates=["date"], index_col="date")
sentiment = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"], index_col="date")

# ── 2. Align both series to month-end period, then merge ──────────────────────
# Both files are already monthly (resampled with "ME" = month-end).
# Normalise the index to the same month-end anchor so the join works even if
# one file's timestamps differ by a day due to weekends/holidays.
events.index    = events.index.to_period("M").to_timestamp("M")
sentiment.index = sentiment.index.to_period("M").to_timestamp("M")

df = events[["extreme_binary"]].join(sentiment[[SENTIMENT_COL, "tb_polarity"]], how="inner")
df.dropna(inplace=True)
print(f"  Merged dataset: {len(df)} monthly rows")

if len(df) == 0:
    raise ValueError(
        "No overlapping months between sentiment_scores.csv and labeled_events.csv.\n"
        "Make sure your articles cover months present in price_data.csv."
    )

# ── 3. Prepare X and y ────────────────────────────────────────────────────────
y = df["extreme_binary"].values
X_raw = df[[SENTIMENT_COL]].values

# Standardise the predictor so the coefficient is interpretable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# statsmodels requires an explicit intercept column
X_sm = sm.add_constant(X_scaled)

# ── 4. Logistic regression via statsmodels ────────────────────────────────────
print("\nFitting logistic regression …")
model   = sm.Logit(y, X_sm)
result  = model.fit(disp=False)

# ── 5. Odds ratio ─────────────────────────────────────────────────────────────
# β₁ is the coefficient on the (standardised) sentiment score
beta1      = result.params[1]
odds_ratio = np.exp(beta1)

# ── 6. AUC-ROC ────────────────────────────────────────────────────────────────
y_pred_prob = result.predict(X_sm)
auc         = roc_auc_score(y, y_pred_prob)

# ── 7. McFadden R² ────────────────────────────────────────────────────────────
# statsmodels computes this directly; we also verify with the formula.
ll_model  = result.llf                  # log-likelihood of fitted model
ll_null   = result.llnull               # log-likelihood of intercept-only model
mcfadden  = 1 - (ll_model / ll_null)

# ── 8. ROC curve plot ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y, y_pred_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"VADER sentiment (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Sentiment as Predictor of Extreme Crypto Events", fontsize=12)
ax.legend(loc="lower right", fontsize=11)
ax.annotate(
    f"McFadden R² = {mcfadden:.4f}\nOdds Ratio = {odds_ratio:.4f}",
    xy=(0.55, 0.12), xycoords="axes fraction", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"),
)
plt.tight_layout()
plt.savefig(ROC_OUT, dpi=150)
plt.close()
print(f"ROC curve saved → {ROC_OUT}")

# ── 9. Clean summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  RESULTS SUMMARY")
print("=" * 55)
print(f"  Predictor              : {SENTIMENT_COL} (VADER compound)")
print(f"  N observations         : {len(df)}")
print(f"  Extreme events (y = 1) : {y.sum()} ({y.mean():.1%})")
print("-" * 55)
print(f"  Odds Ratio (exp β₁)    : {odds_ratio:.4f}")
print(f"  AUC-ROC                : {auc:.4f}")
print(f"  McFadden R²            : {mcfadden:.4f}")
print("-" * 55)
p_value = result.pvalues[1]
print(f"  p-value (β₁)           : {p_value:.4f}  {'*' if p_value < 0.05 else '(n.s.)'}")
print("=" * 55)
print("\nInterpretation:")
print(f"  A 1-SD increase in VADER compound score multiplies the")
print(f"  odds of an extreme event by {odds_ratio:.3f}.")
print(f"  AUC = {auc:.3f}: {'better' if auc > 0.5 else 'no better'} than chance at ranking extreme days.")
print(f"  McFadden R² = {mcfadden:.4f} (>0.2 is considered good fit in logistic models).")
print("\nNote: framing is early-warning signal, not causal claim.")
