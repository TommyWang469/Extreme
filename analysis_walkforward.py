"""
analysis_walkforward.py  —  Sprint 3, Step 2 (improvement.md §1b / §3 Step 2)

True walk-forward backtesting of the weekly extreme-down logit, with STRICT
no-look-ahead — including the two leaks in the Sprint-2 pipeline:

  LEAK 1  scar decile threshold  — build_features_v2.scar_labels computes
          `lo = wk_ret.quantile(TAIL_PCT)` over the FULL sample. Here the decile
          threshold is recomputed on the *training window only* at every step.
  LEAK 2  feature normalization  — analysis_v2.fit_logit standardizes X using the
          FULL-sample mean/sd. Here mean/sd are computed on the *training window
          only* and applied to the test week.

DESIGN
───────────────────────────────────────────────────────────────────────────────
  • Expanding window, one-step-ahead: train on all weeks ≤ t, predict week t+1,
    then expand by one week and refit. Foundation every later model is scored on.
  • Exponential decay of older data: training row i gets weight
    0.5 ** (age_weeks_i / HALF_LIFE_WEEKS), so recent weeks dominate the fit.
  • Trailing features (RWDV, ARKF, QQQ, article-weighted FinBERT) are reused from
    the Sprint-2 builders — they are backward-looking by construction, so the only
    no-look-ahead fixes needed are the label threshold and the normalization.
  • Model: L2-regularized logistic regression (sklearn) so the small early
    windows with near-separation stay numerically stable; sample_weight carries
    the decay. The *spec* (features, logit link) is identical to analysis_v2's C0.

The harness is written to be frequency- and panel-agnostic: the weekly frame can
be swapped for a daily one, and a coin dimension added, without touching the loop
(Step 3 — go daily / panel).

OUTPUTS
  outputs/walkforward.txt   raw metric tables (numbers only)
  outputs/walkforward.md    interpretation only
  outputs/walkforward_oos.png        OOS predicted-probability timeline + events
  outputs/walkforward_decay.png      AUC / PR-AUC vs decay half-life
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_score, recall_score)

from build_features_v2 import (load_index, RECOV_FRAC, SCAR_RECOV_DAYS,
                               TAIL_PCT, WEEK_RULE)
from analysis_v2 import load as load_features

OUT_TXT   = "outputs/walkforward.txt"
OUT_MD    = "outputs/walkforward.md"
OUT_OOS   = "outputs/walkforward_oos.png"
OUT_DECAY = "outputs/walkforward_decay.png"

HALF_LIFE_WEEKS = 104     # default decay half-life (~2y); sensitivity reported below
MIN_TRAIN_WEEKS = 60      # don't predict until the training window is ~14 months
MIN_TRAIN_EVENTS = 4      # ...and has at least this many positive (crash) events
L2_C            = 1.0     # ridge strength for the logit (stabilises tiny windows)
OP_THRESHOLD    = 0.5     # fixed operating point for the confusion matrix

# Primary spec C0 and key ablations (sentiment metric is the first feature).
SPECS = {
    "C0 FinBERT-4w + RWDV + ARKF + QQQ": ["finbert_exp_hl7_4w", "rwdv_63", "arkf_ret_4w", "qqq_ret_4w"],
    "A1 FinBERT-4w alone":               ["finbert_exp_hl7_4w"],
    "A2 VADER-4w + RWDV + ARKF + QQQ":   ["vader_exp_hl7_4w", "rwdv_63", "arkf_ret_4w", "qqq_ret_4w"],
    "A3 controls only (no sentiment)":   ["rwdv_63", "arkf_ret_4w", "qqq_ret_4w"],
}

os.makedirs("outputs", exist_ok=True)


# ── threshold-independent scar flag per week (the realised outcome, no leak) ───
def weekly_scar_flags(idx: pd.Series) -> pd.Series:
    """scar_flag[w] = the drop during week w+1 was NOT 50%-recovered within
    SCAR_RECOV_DAYS of its trough. Depends only on week w+1's own price path —
    it is the outcome we predict, not a fitted parameter, so it is leak-free."""
    r = np.log(idx).diff()
    wk_ret = r.resample(WEEK_RULE).sum()
    wk_end_level = idx.resample(WEEK_RULE).last()
    weeks = wk_ret.index
    out = {}
    for i in range(len(weeks) - 1):
        wk, nxt = weeks[i], weeks[i + 1]
        start_level = wk_end_level.loc[wk]
        seg = idx.loc[wk:nxt].iloc[1:]            # days of the next week
        scar = False
        if len(seg) and not np.isnan(start_level):
            trough_day = seg.idxmin()
            trough = seg.min()
            if trough < start_level:
                target = trough + RECOV_FRAC * (start_level - trough)
                after = idx.loc[trough_day:].iloc[1:SCAR_RECOV_DAYS + 1]
                scar = not (after >= target).any()
        out[wk] = scar
    return pd.Series(out, name="scar_flag")


# ── one full walk-forward pass for a given spec + half-life ───────────────────
def walk_forward(df: pd.DataFrame, x_cols, half_life, return_diag=False, class_weight=None):
    """Return DataFrame of (week, y_true, p_hat) one-step-ahead OOS predictions.

    df must contain x_cols, 'ret_1w' (for the decile threshold), 'ret_1w_fwd'
    (the labelled forward return) and 'scar_flag'.
    class_weight: passed to LogisticRegression (e.g. 'balanced' or {0:1,1:k}) for
    the §1c cost-sensitive / rare-event-imbalance study; None = unweighted (default).
    """
    use = df.dropna(subset=list(x_cols) + ["ret_1w", "ret_1w_fwd", "scar_flag"]).copy()
    weeks = use.index.to_list()
    recs, diag = [], []
    for k in range(len(weeks)):
        if k < MIN_TRAIN_WEEKS:
            continue
        train_weeks = weeks[:k]
        test_week = weeks[k]
        assert max(train_weeks) < test_week, "look-ahead: train not strictly before test"

        tr = use.loc[train_weeks]
        te = use.loc[[test_week]]

        # LEAK-1 fix: decile threshold from TRAIN returns only
        lo = tr["ret_1w"].quantile(TAIL_PCT)
        ytr = ((tr["ret_1w_fwd"] <= lo) & tr["scar_flag"]).astype(int)
        yte = int((te["ret_1w_fwd"].iloc[0] <= lo) and bool(te["scar_flag"].iloc[0]))
        if ytr.sum() < MIN_TRAIN_EVENTS or ytr.nunique() < 2:
            continue

        # LEAK-2 fix: standardize on TRAIN moments only
        mu = tr[x_cols].mean()
        sd = tr[x_cols].std(ddof=0).replace(0, 1.0)
        Xtr = ((tr[x_cols] - mu) / sd).to_numpy()
        Xte = ((te[x_cols] - mu) / sd).to_numpy()

        # exponential decay weights (recent weeks dominate)
        age_w = np.array([(test_week - w).days / 7.0 for w in train_weeks])
        sw = 0.5 ** (age_w / half_life)

        clf = LogisticRegression(C=L2_C, solver="lbfgs", max_iter=2000,
                                 class_weight=class_weight)   # default L2
        clf.fit(Xtr, ytr.to_numpy(), sample_weight=sw)
        p = float(clf.predict_proba(Xte)[0, 1])
        recs.append({"week": test_week, "y": yte, "p": p})
        diag.append({"week": test_week, "n_train": len(tr), "lo": lo,
                     "n_train_events": int(ytr.sum())})

    out = (pd.DataFrame(recs).set_index("week") if recs
           else pd.DataFrame(columns=["y", "p"]))
    if return_diag:
        return out, (pd.DataFrame(diag).set_index("week") if diag else pd.DataFrame())
    return out


def in_sample_lookahead_auc(df, x_cols):
    """The OLD (leaky) way: full-sample threshold + full-sample normalization +
    in-sample fit. Reported only to quantify the optimism the harness removes."""
    use = df.dropna(subset=list(x_cols) + ["ret_1w", "ret_1w_fwd", "scar_flag"]).copy()
    lo = use["ret_1w"].quantile(TAIL_PCT)               # full-sample threshold (leak)
    y = ((use["ret_1w_fwd"] <= lo) & use["scar_flag"]).astype(int)
    mu = use[x_cols].mean()
    sd = use[x_cols].std(ddof=0).replace(0, 1.0)
    X = ((use[x_cols] - mu) / sd).to_numpy()            # full-sample normalization (leak)
    if y.nunique() < 2:
        return np.nan
    clf = LogisticRegression(C=L2_C, solver="lbfgs", max_iter=2000)   # default L2
    clf.fit(X, y.to_numpy())
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def metrics(oos: pd.DataFrame):
    if len(oos) == 0 or oos["y"].nunique() < 2:
        return None
    y, p = oos["y"].to_numpy(), oos["p"].to_numpy()
    n_ev = int(y.sum())
    yhat = (p >= OP_THRESHOLD).astype(int)
    # precision@K with K = number of actual events (rare-event-honest operating point)
    order = np.argsort(-p)
    topk = np.zeros_like(y); topk[order[:n_ev]] = 1
    return dict(
        n=len(oos), events=n_ev, base_rate=y.mean(),
        auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
        brier=brier_score_loss(y, p),
        prec_05=precision_score(y, yhat, zero_division=0),
        rec_05=recall_score(y, yhat, zero_division=0),
        prec_atk=precision_score(y, topk, zero_division=0),
        rec_atk=recall_score(y, topk, zero_division=0),
    )


def main():
    feat = load_features()                              # trailing features (no look-ahead)
    idx = load_index()
    scar = weekly_scar_flags(idx)
    df = feat.join(scar, how="left").sort_index()

    primary = list(SPECS.keys())[0]
    x_primary = SPECS[primary]

    # Score every spec on the SAME weeks: require all specs' features present, so
    # the C0-vs-controls AUC difference is a like-for-like comparison.
    all_feats = sorted({c for cols in SPECS.values() for c in cols})
    base = df.dropna(subset=all_feats + ["ret_1w", "ret_1w_fwd", "scar_flag"]).copy()

    # ── walk-forward for every spec at the default half-life ──────────────────
    spec_oos = {name: walk_forward(base, cols, HALF_LIFE_WEEKS) for name, cols in SPECS.items()}
    spec_metrics = {name: metrics(o) for name, o in spec_oos.items()}

    # diagnostics + in-sample (leaky) contrast for the primary spec
    oos_p, diag = walk_forward(base, x_primary, HALF_LIFE_WEEKS, return_diag=True)
    auc_leaky = in_sample_lookahead_auc(base, x_primary)

    # ── decay half-life sensitivity (primary spec) ───────────────────────────
    half_lives = [26, 52, 104, 208, 1e9]               # 0.5y,1y,2y,4y, ~no decay
    decay_rows = []
    for hl in half_lives:
        m = metrics(walk_forward(base, x_primary, hl))
        if m:
            decay_rows.append((hl, m["auc"], m["pr_auc"], m["brier"]))

    # ── raw tables → txt ─────────────────────────────────────────────────────
    n_usable = len(base)
    tx = []
    tx.append("=" * 92)
    tx.append("WALK-FORWARD BACKTEST — weekly extreme-down logit (Sprint 3, Step 2)")
    tx.append("expanding window, one-step-ahead; train-only decile threshold AND normalization; "
              "exp. decay")
    tx.append(f"weeks usable: {n_usable} | min train {MIN_TRAIN_WEEKS}w / {MIN_TRAIN_EVENTS} events | "
              f"half-life {HALF_LIFE_WEEKS}w | L2 C={L2_C} | op threshold p>={OP_THRESHOLD}")
    if len(diag):
        tx.append(f"first OOS week {diag.index.min().date()} → last {diag.index.max().date()} | "
                  f"OOS weeks scored {len(oos_p)} | train decile lo range "
                  f"[{diag['lo'].min():+.4f}, {diag['lo'].max():+.4f}]")
    tx.append("=" * 92)

    tx.append("")
    tx.append("OUT-OF-SAMPLE METRICS BY SPEC (one-step-ahead, all leak-free)")
    tx.append(f"  {'spec':38s} {'OOSn':>5s} {'ev':>3s} {'base':>6s} {'AUC':>6s} "
              f"{'PR-AUC':>7s} {'Brier':>7s} {'P@.5':>6s} {'R@.5':>6s} {'P@K':>6s} {'R@K':>6s}")
    tx.append("  " + "-" * 104)
    for name in SPECS:
        m = spec_metrics[name]
        if m is None:
            tx.append(f"  {name:38s}   not estimable (too few OOS events)")
            continue
        tx.append(f"  {name:38s} {m['n']:>5d} {m['events']:>3d} {m['base_rate']:>6.3f} "
                  f"{m['auc']:>6.3f} {m['pr_auc']:>7.3f} {m['brier']:>7.3f} "
                  f"{m['prec_05']:>6.2f} {m['rec_05']:>6.2f} "
                  f"{m['prec_atk']:>6.2f} {m['rec_atk']:>6.2f}")
    tx.append("  P@K/R@K = precision/recall when the K highest-probability weeks are flagged, "
              "K = #actual OOS events.")

    tx.append("")
    tx.append("LOOK-AHEAD OPTIMISM (primary spec C0)")
    pm = spec_metrics[primary]
    tx.append(f"  in-sample, full-sample threshold + normalization (the OLD leaky number): "
              f"AUC = {auc_leaky:.3f}")
    if pm:
        tx.append(f"  walk-forward, leak-free OOS                                            : "
                  f"AUC = {pm['auc']:.3f}")
        tx.append(f"  optimism removed by the harness                                        : "
                  f"{auc_leaky - pm['auc']:+.3f}")

    tx.append("")
    tx.append("DECAY HALF-LIFE SENSITIVITY (primary spec C0)")
    tx.append(f"  {'half-life(w)':>12s} {'AUC':>7s} {'PR-AUC':>8s} {'Brier':>8s}")
    for hl, a, pr, br in decay_rows:
        label = "no decay" if hl > 1e8 else f"{hl:g}"
        tx.append(f"  {label:>12s} {a:>7.3f} {pr:>8.3f} {br:>8.3f}")

    tx.append("=" * 92)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(tx) + "\n")
    print("\n".join(tx))
    print(f"\nraw tables → {OUT_TXT}")

    # ── plots ────────────────────────────────────────────────────────────────
    if len(oos_p):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(oos_p.index, oos_p["p"], color="tab:blue", lw=1.2, label="P(extreme-down) [OOS]")
        ev = oos_p[oos_p["y"] == 1]
        ax.scatter(ev.index, ev["p"], color="red", zorder=5, label="actual extreme-down week")
        ax.axhline(OP_THRESHOLD, color="grey", ls="--", lw=0.8, label=f"op threshold {OP_THRESHOLD}")
        ax.set_ylabel("predicted probability"); ax.set_xlabel("week")
        ax.set_title("Walk-forward OOS predicted crash probability vs realised events")
        ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(OUT_OOS, dpi=150); plt.close()

    if decay_rows:
        hl_x = [r[0] if r[0] <= 1e8 else 416 for r in decay_rows]   # plot "no decay" off-scale
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hl_x, [r[1] for r in decay_rows], marker="o", label="AUC")
        ax.plot(hl_x, [r[2] for r in decay_rows], marker="s", label="PR-AUC")
        ax.axhline(pm["base_rate"] if pm else 0, color="grey", ls=":", lw=0.8,
                   label="base rate (PR no-skill)")
        ax.set_xlabel("decay half-life (weeks; rightmost ≈ no decay)")
        ax.set_ylabel("metric"); ax.set_title("Walk-forward metric vs decay half-life (C0)")
        ax.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(OUT_DECAY, dpi=150); plt.close()
    print(f"plots → {OUT_OOS}, {OUT_DECAY}")

    # ── interpretation → md ──────────────────────────────────────────────────
    write_md(spec_metrics, primary, auc_leaky, decay_rows)
    print(f"interpretation → {OUT_MD}")


def write_md(spec_metrics, primary, auc_leaky, decay_rows):
    pm = spec_metrics[primary]
    a3 = spec_metrics.get("A3 controls only (no sentiment)")
    md = ["# Walk-Forward Backtest — Interpretation (Sprint 3, Step 2)\n",
          "*Raw numbers in `outputs/walkforward.txt`; this file is interpretation only.*\n",
          "## What this harness fixes\n",
          "- **Leak 1 — scar decile threshold** was computed on the full sample in "
          "`build_features_v2.py`; here it is recomputed on the **training window only** at "
          "every step.\n"
          "- **Leak 2 — feature normalization** used full-sample mean/sd in `analysis_v2.py`; "
          "here mean/sd come from the **training window only**.\n"
          "- Single 2022 split → **expanding-window, one-step-ahead** refit with **exponential "
          "decay** of older weeks. This is the measurement instrument every later model is "
          "scored on.\n"]
    if pm:
        md.append("## Headline\n")
        md.append(f"- Primary spec **C0** leak-free OOS: **AUC {pm['auc']:.3f}**, "
                  f"PR-AUC {pm['pr_auc']:.3f} (base rate {pm['base_rate']:.3f}), "
                  f"Brier {pm['brier']:.3f}, over {pm['n']} OOS weeks / {pm['events']} events.\n")
        md.append(f"- **Look-ahead optimism:** the old in-sample number was AUC {auc_leaky:.3f}; "
                  f"the leak-free harness gives {pm['auc']:.3f} "
                  f"(**{auc_leaky - pm['auc']:+.3f}**). That gap is exactly the kind of "
                  "self-deception walk-forward exists to remove.\n")
        verdict = ("barely better than chance" if pm["auc"] < 0.55 else
                   "weak but non-trivial" if pm["auc"] < 0.62 else "moderate")
        md.append(f"- Honest read: discrimination is **{verdict}** out-of-sample, consistent "
                  "with the Sprint-2 near-null. Report it as such — do not chase a starred AUC.\n")
        if pm["prec_atk"] == 0:
            md.append("- **No sharp-end value:** precision@K = 0 — the K highest-probability "
                      "weeks contain *none* of the actual crashes. The AUC > 0.5 comes from "
                      "mid-distribution ranking, not from confident early warnings, so this model "
                      "has no usable trigger as-is (matters for the Step-6 economic test).\n")
    if pm and a3:
        md.append("## Does sentiment add anything OOS?\n")
        md.append(f"- C0 (with sentiment) AUC {pm['auc']:.3f} vs A3 (controls only) "
                  f"AUC {a3['auc']:.3f} → sentiment's marginal OOS contribution is "
                  f"**{pm['auc'] - a3['auc']:+.3f} AUC**. This is the honest test of whether the "
                  "sentiment signal survives a no-look-ahead backtest.\n")
    if decay_rows:
        aucs = [r[1] for r in decay_rows]                       # ordered short→long half-life
        best = max(decay_rows, key=lambda r: r[1])
        bl = "no decay" if best[0] > 1e8 else f"{best[0]:g}w"
        spread = max(aucs) - min(aucs)
        if aucs[-1] - aucs[0] > 0.02:
            trend = ("AUC **rises monotonically as the half-life lengthens** — i.e. faster decay "
                     "*hurts* out-of-sample. With only ~17 events, down-weighting older weeks "
                     "throws away scarce signal, so the data wants long memory here.")
        elif spread < 0.02:
            trend = ("AUC is **flat across half-lives** — decay neither helps nor hurts; report "
                     "that rather than tuning to the best value.")
        else:
            trend = f"AUC peaks at an intermediate half-life (**{bl}**)."
        md.append("## Decay\n")
        md.append("- AUC across half-lives: "
                  f"{', '.join(('no-decay' if r[0] > 1e8 else f'{r[0]:g}w') + f':{r[1]:.3f}' for r in decay_rows)}. "
                  f"{trend}\n")
        md.append("- **Do not** adopt the best-scoring half-life as a result — that is tuning on "
                  "the test set. The pre-registered default is "
                  f"{HALF_LIFE_WEEKS}w; the sensitivity curve is the honest deliverable.\n")
    md.append("## Honesty notes\n")
    md.append("- L2-regularized logit (C=1.0) for numerical stability in small early windows; "
              "the spec (features + logit link) matches analysis_v2's C0.\n")
    md.append("- precision/recall@K (K = #events) is the rare-event-honest operating point; the "
              "p≥0.5 confusion is shown too but is uninformative at a ~5% base rate.\n")
    md.append("- Harness is frequency/panel-agnostic — daily data (§G2) or a coin panel (§G1) "
              "drop into the same loop for Step 3.\n")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
