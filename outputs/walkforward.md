# Walk-Forward Backtest — Interpretation (Sprint 3, Step 2)

*Raw numbers in `outputs/walkforward.txt`; this file is interpretation only.*

## What this harness fixes

- **Leak 1 — scar decile threshold** was computed on the full sample in `build_features_v2.py`; here it is recomputed on the **training window only** at every step.
- **Leak 2 — feature normalization** used full-sample mean/sd in `analysis_v2.py`; here mean/sd come from the **training window only**.
- Single 2022 split → **expanding-window, one-step-ahead** refit with **exponential decay** of older weeks. This is the measurement instrument every later model is scored on.

## Headline

- Primary spec **C0** leak-free OOS: **AUC 0.578**, PR-AUC 0.048 (base rate 0.034), Brier 0.034, over 205 OOS weeks / 7 events.

- **Look-ahead optimism:** the old in-sample number was AUC 0.641; the leak-free harness gives 0.578 (**+0.063**). That gap is exactly the kind of self-deception walk-forward exists to remove.

- Honest read: discrimination is **weak but non-trivial** out-of-sample, consistent with the Sprint-2 near-null. Report it as such — do not chase a starred AUC.

- **No sharp-end value:** precision@K = 0 — the K highest-probability weeks contain *none* of the actual crashes. The AUC > 0.5 comes from mid-distribution ranking, not from confident early warnings, so this model has no usable trigger as-is (matters for the Step-6 economic test).

## Does sentiment add anything OOS?

- C0 (with sentiment) AUC 0.578 vs A3 (controls only) AUC 0.439 → sentiment's marginal OOS contribution is **+0.139 AUC**. This is the honest test of whether the sentiment signal survives a no-look-ahead backtest.

## Decay

- AUC across half-lives: 26w:0.439, 52w:0.508, 104w:0.578, 208w:0.607, no-decay:0.627. AUC **rises monotonically as the half-life lengthens** — i.e. faster decay *hurts* out-of-sample. With only ~17 events, down-weighting older weeks throws away scarce signal, so the data wants long memory here.

- **Do not** adopt the best-scoring half-life as a result — that is tuning on the test set. The pre-registered default is 104w; the sensitivity curve is the honest deliverable.

## Honesty notes

- L2-regularized logit (C=1.0) for numerical stability in small early windows; the spec (features + logit link) matches analysis_v2's C0.

- precision/recall@K (K = #events) is the rare-event-honest operating point; the p≥0.5 confusion is shown too but is uninformative at a ~5% base rate.

- Harness is frequency/panel-agnostic — daily data (§G2) or a coin panel (§G1) drop into the same loop for Step 3.

