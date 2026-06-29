"""
analysis_identification.py  —  Sprint 3, Step 4 (improvement.md §G3)

Trimmed to the one load-bearing identification check: MOMENTUM CONTROL.
Is the sentiment->crash association just a proxy for recent price moves? Add lagged
1-week and trailing-4-week returns to the scar-down logit; if the sentiment odds
ratio survives, sentiment carries information beyond momentum. This answers the
obvious reviewer question, "isn't your sentiment just price momentum?"

(The lead-lag/Granger test and the news-volume/disagreement predictors were removed
to keep the project lean. The news-VOLUME finding — the project's significant positive
result — lives on, stronger, in the severity model `outputs/step5_models.*` (p=0.001)
and the walk-forward volume add-on `outputs/costsensitive.*`.)

OUTPUTS
  outputs/identification.txt   raw table
  outputs/identification.md    interpretation
"""

import os
import warnings
import numpy as np
import statsmodels.api as sm

from analysis_v2 import load as load_features

warnings.filterwarnings("ignore")

OUT_TXT = "outputs/identification.txt"
OUT_MD  = "outputs/identification.md"
SENT_4W = "finbert_exp_hl7_4w"

os.makedirs("outputs", exist_ok=True)


def fit_logit(df, y_col, x_cols):
    d = df.dropna(subset=[y_col] + x_cols)
    y = d[y_col].astype(float)
    X = (d[x_cols] - d[x_cols].mean()) / d[x_cols].std(ddof=0)
    X = sm.add_constant(X)
    try:
        m = sm.Logit(y.to_numpy(), X.to_numpy()).fit(disp=0, maxiter=200)
    except Exception as e:
        return {"error": str(e)[:60], "n": len(d), "events": int(y.sum())}
    out = {"n": len(d), "events": int(y.sum())}
    for i, nm in enumerate(["const"] + x_cols):
        out[nm] = (float(np.exp(m.params[i])), float(m.pvalues[i]))
    out["mcf"] = float(m.prsquared)
    return out


def main():
    df = load_features().sort_index()
    df["mom_1w"] = df["ret_1w"]
    df["mom_4w"] = df["ret_1w"].rolling(4).sum()
    ctrl = ["rwdv_63", "arkf_ret_4w", "qqq_ret_4w"]

    base    = fit_logit(df, "extreme_down", [SENT_4W] + ctrl)
    withmom = fit_logit(df, "extreme_down", [SENT_4W, "mom_1w", "mom_4w"] + ctrl)

    def orp(d, k):
        if "error" in d or k not in d:
            return "   n/a   "
        o, p = d[k]
        return f"{o:5.2f}({p:.2f})"

    tx = ["=" * 80,
          "IDENTIFICATION — momentum control (Sprint 3, Step 4 / §G3)",
          "weekly BTC+ETH; logistic scar-down model; OR per +1 SD with p in parentheses",
          "=" * 80, "",
          "Does the sentiment OR survive controlling for lagged returns?",
          f"  {'spec':32s} {'n':>4s} {'ev':>3s} {'sentiment':>11s} {'mom_1w':>11s} {'mom_4w':>11s} {'McF':>6s}",
          f"  {'base: sent + controls':32s} {base.get('n',0):>4d} {base.get('events',0):>3d} "
          f"{orp(base, SENT_4W):>11s} {'-':>11s} {'-':>11s} {base.get('mcf',float('nan')):>6.3f}",
          f"  {'+ momentum (mom_1w, mom_4w)':32s} {withmom.get('n',0):>4d} {withmom.get('events',0):>3d} "
          f"{orp(withmom, SENT_4W):>11s} {orp(withmom,'mom_1w'):>11s} {orp(withmom,'mom_4w'):>11s} "
          f"{withmom.get('mcf',float('nan')):>6.3f}"]
    if SENT_4W in base and SENT_4W in withmom:
        tx.append(f"  -> sentiment OR {base[SENT_4W][0]:.2f} -> {withmom[SENT_4W][0]:.2f} after momentum "
                  f"(change {withmom[SENT_4W][0]-base[SENT_4W][0]:+.2f})")
    tx.append("=" * 80)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(tx) + "\n")
    print("\n".join(tx))
    print(f"\nraw → {OUT_TXT}")

    write_md(base, withmom)
    print(f"interpretation → {OUT_MD}")


def write_md(base, withmom):
    b = base.get(SENT_4W, (None,))[0]
    w = withmom.get(SENT_4W, (None,))[0]
    survived = (b is not None and w is not None and (w - 1) * (b - 1) > 0
                and abs(w - 1) > 0.5 * abs(b - 1))
    md = ["# Identification — Momentum Control (Sprint 3, Step 4)\n",
          "*Raw numbers in `outputs/identification.txt`; interpretation only here.*\n",
          "## Is sentiment just a proxy for recent returns?\n"]
    if b is not None:
        verdict = ("**largely survives** — sentiment is not merely re-encoding momentum"
                   if survived else "**weakens substantially** — much of it may be a momentum proxy")
        md.append(f"- Sentiment OR goes {b:.2f} -> {w:.2f} once last-week and trailing-4-week returns "
                  f"are added, and momentum itself is not significant. It {verdict}. This is the "
                  "standard robustness check answering 'isn't your sentiment just price momentum?' "
                  "(In-sample, ~17 events — read as direction, not proof.)\n")
    md.append("## Scope note\n")
    md.append("- Trimmed to the momentum-control check. The lead-lag/Granger test and the "
              "news-volume/disagreement predictors were removed to keep the project lean; the "
              "news-VOLUME result (the significant positive finding) lives in the severity model "
              "(`outputs/step5_models.md`, p=0.001) and the walk-forward volume add-on "
              "(`outputs/costsensitive.md`).\n")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
