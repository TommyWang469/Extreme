# Sprint 2 results

```
================================================================================================
  SPRINT 2 — multi-factor weekly logit (sentiment metric = first variable in each spec)
================================================================================================
  spec                                            n  ev  OR(sent)  p(sent)    AUC     McF
------------------------------------------------------------------------------------------------
  C0 CONFIRMATION: FinBERT-4w + RWDV + ARKF + QQQ  200  13     1.700    0.108    0.660  0.0368
  S1 FinBERT alone (baseline)                   196  13     0.872    0.630    0.567  0.0024
  S2 Sprint-2 primary (raw weekly FinBERT)      196  13     0.907    0.744    0.596  0.0087
  S3 ablation: VADER + RWDV + ARKF + QQQ        196  13     0.780    0.398    0.669  0.0151
  S4 ablation: FinBERT + semidev + factors      196  13     0.900    0.726    0.604  0.0130
  S5 ablation: decile-only label (no scar)      196  21     0.745    0.228    0.610  0.0236
  S6 extreme-UP side                            196  19     0.886    0.640    0.611  0.0253
================================================================================================
  * = p < 0.05 on the sentiment coefficient. OR per 1-SD move in the sentiment composite.

================================================================================================
  CONFIRMATION SPEC — full coefficient table (per 1-SD)
================================================================================================
  finbert_exp_hl7_4w OR =  1.700   p = 0.108
  rwdv_63            OR =  0.875   p = 0.684
  arkf_ret_4w        OR =  0.760   p = 0.578
  qqq_ret_4w         OR =  0.888   p = 0.819
================================================================================================
  Out-of-sample (train ≤2022, test after): AUC = 0.670

```
