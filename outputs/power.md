# Power Analysis — Interpretation (Sprint 3, §G2/§G7)

*Raw numbers in `outputs/power.txt`; interpretation only here.*

- **The weekly single series is badly underpowered.** At the observed base rate **0.066** (18/273), the minimum detectable odds ratio at 80% power is **OR ≈ 2.1** — far above the real effect (~1.5–1.7). With 273 weeks we essentially *cannot* detect the signal we think is there, so the Sprint-2 non-significance is largely a power problem, not evidence of no effect (§G7).

- **To detect OR = 1.5 at 80% power needs ≈ 779 independent observations** (OR = 1.7 needs ≈ 487); the Hsieh closed form agrees to order of magnitude. That is ≈ 3× the current weekly sample.

- **The panel is the way there, but clustering taxes it.** If within-coin weeks were independent (ICC 0) the required observations divide cleanly by coins; at a realistic ICC of 0.05–0.10 the design effect is large (each coin's ~270 weeks count for far fewer *effective* observations), so more coins are needed than the naive N/weeks suggests. See the panel table for coins-needed under each ICC.

- **Takeaway for sequencing:** report the OR **confidence interval** and this MDE, not a bare p-value; and prioritise raising *effective* N (panel breadth + daily) over swapping in fancier models, which cannot manufacture power.

