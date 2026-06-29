# News-Volume OOS Check (Sprint 3, Step 5)

*Raw numbers in `outputs/costsensitive.txt`; interpretation only here.*

## Does news volume survive the honest walk-forward?

- On identical out-of-sample weeks, adding news volume to the primary model moves leak-free OOS **AUC 0.479 → 0.664** (+0.185; PR-AUC 0.047 → 0.080). This is the project's strongest volume evidence — the in-sample severity finding (p=0.001) confirmed under a no-look-ahead backtest.

## Footnote — cost-sensitive class weighting (§1c)

- Up-weighting the rare crash class barely moves ranking (AUC 0.578 → 0.587) and just trades precision for recall (recall@0.5 0.00 → 0.71). It is a reporting/operating-point dial, not a way to add skill — kept only as this one-line check.

