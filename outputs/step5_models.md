# Step 5 — Severity Model — Interpretation (Sprint 3)

*Raw numbers in `outputs/step5_models.txt`; interpretation only here.*

## Severity (richer target than the binary)

- At the 10th-percentile (tail) quantile, the sentiment coefficient on next-week return is -0.008 (p=0.28) — euphoria‑direction but only borderline.

- **News volume is the stronger, significant tail predictor:** q=0.10 coefficient -0.0245 (**p=0.001**), i.e. higher news attention precedes materially worse tail losses, and it stays significant across the bad tail (q=0.05 and 0.25 too). This is the in‑sample confirmation of the **attention/volume beats polarity** finding; the continuous‑severity target is what let it clear significance where the 17‑event binary logit could not. (In‑sample association; it also survives the walk‑forward — see `outputs/costsensitive.md`.)

## Footnotes

- *Nonlinear benchmark (§1e):* GBT OOS AUC 0.416 vs logit 0.716 — a gradient‑boosted‑tree model does **not** beat the linear logit on this tiny sample; nonlinearity can't manufacture signal. (sklearn HistGradientBoosting stand‑in; LightGBM unavailable/libomp.)

- *Dropped:* the time‑to‑next‑crash hazard model (null) and the GARCH‑EVT tail model (§1d) — both cut to keep the project lean (GARCH‑EVT was a working result but the most tangential to the sentiment hypothesis).

