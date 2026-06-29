# Panel Logit — Interpretation (Sprint 3, Step 3 / §G1)

*Raw numbers in `outputs/panel.txt`; interpretation only here. FinBERT only.*

## Part A — shared-sentiment panel (the power lever)

- **Events went from ~17 to 142** across 11 coins (2771 coin-weeks), including the dead LUNA (survivorship-corrected). This is the legitimate power lever — pooling, not a fancier model.

- **Primary inference (coin-FE, two-way cluster coin × week):** sentiment OR **1.39**, 95% CI [0.90, 2.13], p = 0.136. **Qualified support** (PREREGISTRATION.md §6): consistent sign/size with a CI that still includes 1 — suggestive, underpowered, *not* confirmation and *not* a null.

- **Why not the by-coin SE?** Clustering by coin alone gives a deceptively tight CI [1.15, 1.67] (p = 0.001) — but sentiment is a *single shared time series*, so by-coin clustering treats the same path as 11 independent draws (the **Moulton problem**). The two-way SE is the honest one; do **not** report the by-coin p as the headline.

- **Key point:** because sentiment is shared, the effective independent sample is ~the number of weeks, not coin-weeks — so the panel does *not* escape the power ceiling for the sentiment regressor. A CI that includes 1 here is *underpowered*, not evidence of no effect (read with `outputs/power.md`, §G7).

## Part B — per-coin news (condensed): did not help

- We also scraped coin-specific news for 7 coins and re-ran the panel with per-coin FinBERT sentiment. It did **not** help: per-coin OR 1.19 [0.64, 2.22] ≈ shared OR 1.22, with a CI no tighter, so coin-specific news bought no extra power. Reinforces that the binding constraint is events/power, not how sentiment is built. (Condensed from a fuller analysis; the VADER scoring path was retired.)

## Honesty notes

- Descriptive panel association with cluster-robust inference, not an OOS claim. Cost-sensitive / Firth correction is applied on the credible setup at Step 5.

