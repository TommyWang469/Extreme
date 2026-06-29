# Identification — Momentum Control (Sprint 3, Step 4)

*Raw numbers in `outputs/identification.txt`; interpretation only here.*

## Is sentiment just a proxy for recent returns?

- Sentiment OR goes 1.53 -> 1.53 once last-week and trailing-4-week returns are added, and momentum itself is not significant. It **largely survives** — sentiment is not merely re-encoding momentum. This is the standard robustness check answering 'isn't your sentiment just price momentum?' (In-sample, ~17 events — read as direction, not proof.)

## Scope note

- Trimmed to the momentum-control check. The lead-lag/Granger test and the news-volume/disagreement predictors were removed to keep the project lean; the news-VOLUME result (the significant positive finding) lives in the severity model (`outputs/step5_models.md`, p=0.001) and the walk-forward volume add-on (`outputs/costsensitive.md`).

