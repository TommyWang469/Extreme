# Event Study — Interpretation (Sprint 3, Step 1)

*Raw numbers live in `outputs/event_study.txt`; this file is interpretation only.*

## What was tested

- **Sample A — named exogenous catalysts** (6 down, 3 up): selected by catalyst, not by return, so abnormal returns here are **not circular**. This is the sample we trust for inference.
- **Sample B — scar-decile events** (objective bottom/top-10% weekly returns, clustered): more events, but the contemporaneous CAR is **mechanically** large because events are *selected* on the return. Used for recovery asymmetry and the sentiment regression, not for the day-0 CAR claim.

- Five statistics per window (Patell, BMP, generalized sign, Corrado-Zivney rank, bootstrap CI). A window counts as significant **only when a parametric and a non-parametric test agree at 5%** — deliberately conservative.

## Headline reading

- **Post-event drift (named down-catalysts, [+1,+10]):** mean CAR -0.026, bootstrap 95% CI [-0.124, +0.094] — NOT robustly significant. This is the recovery/continuation question that bridges to RWDV.

- **Pre-event drift (named down-catalysts, [-10,-1]):** mean CAR -0.126 — significant. Pre-drift would hint the market partly anticipated the catalyst.

## Does pre-event sentiment line up with the move?

- **Named catalysts (n=9):** CAR-on-sentiment slope -0.559 (negative), Spearman ρ=-0.10 (perm p=0.81). A negative slope is the euphoria / 'sell-the-news' direction — higher prior sentiment, worse subsequent abnormal return.

- **Decile down-events (n=17):** slope -0.163, Spearman ρ=-0.14 (perm p=0.59).

- With single-digit-to-~20 events these correlations are **suggestive, not confirmatory**; report the direction and the CI, not a starred p-value.

## Recovery asymmetry (RWDV bridge)

- See `outputs/event_study.txt` for median 50%/80% recovery times and censoring counts. Slow/incomplete recovery after downside events is the empirical content behind recovery-weighted downside volatility.

## Honesty notes

- 8 windows were tested per sample; Bonferroni reference threshold is p<0.013. No window was selected after the fact — the four sub-windows are fixed in the script.

- Mean-adjusted model (no crypto market index available). De-peg events (Terra, USDC) are included as canonical extreme events per §1a.

- This is descriptive/independent evidence; it complements, and does not replace, the walk-forward logit (Step 2).

