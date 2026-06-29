"""
event_study.py  —  Sprint 3, Step 1 (improvement.md §1a / §3 Step 1)

Event study of extreme crypto events on the 50/50 BTC+ETH portfolio, rebuilt on
the extended daily data (2021-01 → 2026-06) with proper inferential statistics
for *few, fat-tailed* events.

WHY THIS DESIGN
───────────────────────────────────────────────────────────────────────────────
The weekly logit has only ~17 events, so its p-values are fragile. An event
study's statistics do NOT depend on that logit, so it is our best shot at clean,
independent evidence (mentor's preferred low-event tool). We run TWO samples
because they answer different questions and have different biases:

  • Sample A — NAMED CATALYSTS (selected by an *exogenous* catalyst, not by the
    return). This is the non-circular sample used for the standardized tests.
    Includes both stablecoin de-peg events (Terra/UST May-2022, USDC/SVB
    Mar-2023) — a de-peg is by definition an extreme event and maps onto our
    recovery-time heuristic (cf. money-market funds "breaking the buck", 2008).

  • Sample B — SCAR-DECILE (objective rule: weekly portfolio return in the
    bottom/top 10% of the full sample, clustered into distinct events). More
    events ⇒ more power, but selecting on the return makes the *contemporaneous*
    CAR mechanically large. We therefore use Sample B for recovery-time
    asymmetry and the CAR~sentiment regression, and flag the selection bias; we
    do NOT claim the contemporaneous decile CAR as evidence of anything.

MODEL (mean-adjusted abnormal returns)
───────────────────────────────────────────────────────────────────────────────
The asset *is* the BTC+ETH portfolio; there is no separate market index, so we
use the standard mean-adjusted model.
    estimation window : t = -120 … -31   (L = 90 calendar days; crypto trades
                                          7d/wk, so "trading day" = calendar day)
    expected return   : μ̂ = mean(r) over the estimation window
    abnormal return   : AR_t = r_t - μ̂
    event window      : t = -10 … +10    (τ = 21 days)
    sub-windows       : pre [-10,-1], event-day [0], post [+1,+10]

STATISTICS (report-only-when-they-agree)
───────────────────────────────────────────────────────────────────────────────
  • Patell (1976) standardized-residual Z      — parametric
  • Boehmer-Musumeci-Poulsen (1991) Z          — parametric, event-induced-variance robust
  • Generalized sign test (Cowan 1992)         — non-parametric
  • Corrado-Zivney (1992) rank test            — non-parametric
  • Bootstrap percentile CI on mean CAR        — distribution-free
A window is called significant only when ≥1 parametric AND ≥1 non-parametric test
agree at 5%. Multiple-window testing is flagged with a Bonferroni reference line.

OUTPUTS
───────────────────────────────────────────────────────────────────────────────
  outputs/event_study.txt   raw tables (numbers only — results-file convention)
  outputs/event_study.md    interpretation only
  outputs/event_study_car.png        CAR paths (named catalysts)
  outputs/event_study_recovery.png   recovery-time / asymmetry illustration
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── paths / config ───────────────────────────────────────────────────────────
PRICE_PATH        = "data/price_data.csv"
SENT_WEEKLY_PATH  = "data/sentiment_weekly.csv"     # FinBERT + VADER, weekly, + n_articles
OUT_TXT           = "outputs/event_study.txt"
OUT_MD            = "outputs/event_study.md"
OUT_PLOT_CAR      = "outputs/event_study_car.png"
OUT_PLOT_REC      = "outputs/event_study_recovery.png"

EST_START, EST_END = -120, -31     # estimation window (rel. days);  L = 90
EVT_START, EVT_END = -10,  +10     # event window (rel. days);       tau = 21
L   = EST_END - EST_START + 1      # 90
TAU = EVT_END - EVT_START + 1      # 21
PRE_SENT_DAYS = 28                 # 4 weeks of weekly sentiment before the event
REC_HORIZON   = 180                # days to look for recovery after the trough
BOOT_B        = 10000
RNG           = np.random.default_rng(20260621)

# Sample A — named exogenous catalysts (name, catalyst date, expected sign, tag).
# Selected on the CATALYST, never on the realised return ⇒ non-circular.
NAMED_EVENTS = [
    ("China mining/trading ban",   "2021-05-19", "-", "regulatory"),
    ("Terra/UST de-peg + LUNA",    "2022-05-09", "-", "depeg"),
    ("Celsius freeze / 3AC",       "2022-06-13", "-", "contagion"),
    ("FTX collapse (SBF)",         "2022-11-08", "-", "contagion"),
    ("USDC de-peg (SVB)",          "2023-03-10", "-", "depeg"),
    ("Yen carry-trade unwind",     "2024-08-05", "-", "macro"),
    ("Spot BTC ETF approval",      "2024-01-10", "+", "scheduled"),
    ("Bitcoin 4th halving",        "2024-04-19", "+", "scheduled"),
    ("US election (Trump win)",    "2024-11-06", "+", "macro"),
]

DECILE_Q    = 0.10                 # bottom/top decile = "scar decile"
CLUSTER_GAP = 30                   # days; merge nearby extreme weeks into 1 event

os.makedirs("outputs", exist_ok=True)


# ── data loading ─────────────────────────────────────────────────────────────
def load_prices():
    px = pd.read_csv(PRICE_PATH, parse_dates=["date"]).set_index("date").sort_index()
    ret = px["portfolio_ret"].astype(float)
    price = (1.0 + ret).cumprod()          # synthetic price index, P_0 = 1+r_0
    return ret, price


def load_sentiment():
    s = pd.read_csv(SENT_WEEKLY_PATH, parse_dates=["week"]).set_index("week").sort_index()
    return s


def pre_event_sentiment(sent, ev_date):
    """Mean weekly FinBERT sentiment and mean news volume over the 4 weeks before."""
    lo = ev_date - pd.Timedelta(days=PRE_SENT_DAYS)
    win = sent.loc[(sent.index > lo) & (sent.index <= ev_date)]
    if len(win) == 0:
        return np.nan, np.nan
    return float(win["finbert_linear"].mean()), float(win["n_articles"].mean())


# ── objective scar-decile event selection ────────────────────────────────────
def scar_decile_events(ret):
    wk = (1.0 + ret).resample("W-FRI").prod() - 1.0
    wk = wk.dropna()
    q_lo, q_hi = wk.quantile(DECILE_Q), wk.quantile(1 - DECILE_Q)

    def cluster(pairs, worst):
        pairs = sorted(pairs)
        groups, cur = [], [pairs[0]]
        for d, v in pairs[1:]:
            if (d - cur[-1][0]).days <= CLUSTER_GAP:
                cur.append((d, v))
            else:
                groups.append(cur); cur = [(d, v)]
        groups.append(cur)
        pick = (min if worst else max)
        return [pick(g, key=lambda x: x[1]) for g in groups]

    down = cluster([(d, v) for d, v in wk.items() if v <= q_lo], worst=True)
    up   = cluster([(d, v) for d, v in wk.items() if v >= q_hi], worst=False)
    ev = ([(f"down {d.date()}", d, "-", "decile") for d, v in down] +
          [(f"up {d.date()}",   d, "+", "decile") for d, v in up])
    return ev, q_lo, q_hi, len(wk)


# ── per-event abnormal returns ───────────────────────────────────────────────
def event_ar(ret, ev_date):
    """Return dict with AR series over estimation+event windows, or None if short."""
    idx = ret.index
    pos = idx.searchsorted(pd.Timestamp(ev_date))
    if pos <= abs(EST_START) or pos >= len(idx) - EVT_END - 1:
        return None
    t0 = pos
    est = ret.iloc[t0 + EST_START: t0 + EST_END + 1]
    mu = est.mean()
    sd = est.std(ddof=1)                    # mean-adjusted residual sd, df = L-1
    if sd == 0 or np.isnan(sd):
        return None
    evt = ret.iloc[t0 + EVT_START: t0 + EVT_END + 1]
    ar_est = (est - mu).to_numpy()
    ar_evt = (evt - mu).to_numpy()
    rel = np.arange(EVT_START, EVT_END + 1)
    return dict(t0=t0, mu=mu, sd=sd, ar_est=ar_est, ar_evt=ar_evt, rel=rel,
                car=np.cumsum(ar_evt), date=idx[t0])


def car_in_window(e, a, b):
    """Cumulative AR over relative-day window [a,b] inclusive."""
    mask = (e["rel"] >= a) & (e["rel"] <= b)
    return float(e["ar_evt"][mask].sum())


# ── inferential tests over a window [a,b] across a list of events ─────────────
def patell_z(events, a, b):
    """Patell (1976) standardized-residual Z for cumulative AR over [a,b]."""
    n_days = b - a + 1
    csar = []
    for e in events:
        mask = (e["rel"] >= a) & (e["rel"] <= b)
        s_forecast = e["sd"] * np.sqrt(1.0 + 1.0 / L)      # mean-adjusted prediction sd
        sar = e["ar_evt"][mask] / s_forecast
        csar.append(sar.sum())
    csar = np.array(csar)
    N = len(csar)
    var_csar = n_days * (L - 1) / (L - 3)                  # Var(t_{L-1}) per std resid
    z = csar.sum() / np.sqrt(N * var_csar)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def bmp_z(events, a, b):
    """Boehmer-Musumeci-Poulsen (1991) standardized cross-sectional Z."""
    n_days = b - a + 1
    scar = []
    for e in events:
        car = car_in_window(e, a, b)
        s_car = e["sd"] * np.sqrt(n_days) * np.sqrt(1.0 + 1.0 / L)
        scar.append(car / s_car)
    scar = np.array(scar)
    N = len(scar)
    z = np.sqrt(N) * scar.mean() / scar.std(ddof=1)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def generalized_sign_z(events, a, b):
    """Cowan (1992) generalized sign test; p̂ from estimation-window positives."""
    p_hat = np.mean([(e["ar_est"] > 0).mean() for e in events])
    cars = np.array([car_in_window(e, a, b) for e in events])
    w = int((cars > 0).sum())
    N = len(cars)
    denom = np.sqrt(N * p_hat * (1 - p_hat))
    z = (w - N * p_hat) / denom if denom > 0 else np.nan
    return z, 2 * (1 - stats.norm.cdf(abs(z))), w, N, p_hat


def corrado_rank_z(events, a, b):
    """Corrado-Zivney (1992) standardized-rank test over the cumulative window."""
    # Build standardized ranks within each event's (estimation + event) series.
    per_event_ranks = []
    for e in events:
        series = np.concatenate([e["ar_est"], e["ar_evt"]])
        ranks = stats.rankdata(series)
        k = ranks / (len(series) + 1) - 0.5          # mean-zero standardized rank
        rel_all = np.concatenate([np.arange(EST_START, EST_END + 1), e["rel"]])
        per_event_ranks.append(pd.Series(k, index=rel_all))
    K = pd.concat(per_event_ranks, axis=1)           # rows=rel day, cols=events
    abar = K.mean(axis=1)                            # mean std-rank across events per day
    s_k = np.sqrt((abar ** 2).mean())                # sd of daily mean std-rank over all days
    win_days = abar.loc[(abar.index >= a) & (abar.index <= b)]
    n_days = len(win_days)
    z = win_days.sum() / (np.sqrt(n_days) * s_k)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def bootstrap_ci(events, a, b, alpha=0.05):
    cars = np.array([car_in_window(e, a, b) for e in events])
    means = np.array([RNG.choice(cars, size=len(cars), replace=True).mean()
                      for _ in range(BOOT_B)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return cars.mean(), lo, hi


def run_window_tests(events, a, b):
    pz, pp = patell_z(events, a, b)
    bz, bp = bmp_z(events, a, b)
    sz, sp, w, N, phat = generalized_sign_z(events, a, b)
    cz, cp = corrado_rank_z(events, a, b)
    mean_car, lo, hi = bootstrap_ci(events, a, b)
    param_sig    = (pp < 0.05) or (bp < 0.05)
    nonparam_sig = (sp < 0.05) or (cp < 0.05)
    return dict(window=f"[{a:+d},{b:+d}]", N=N, mean_car=mean_car,
                boot_lo=lo, boot_hi=hi,
                patell_z=pz, patell_p=pp, bmp_z=bz, bmp_p=bp,
                sign_z=sz, sign_p=sp, sign_w=w, sign_phat=phat,
                corrado_z=cz, corrado_p=cp,
                agree_sig=(param_sig and nonparam_sig))


# ── recovery-time analysis (bridge to RWDV) ──────────────────────────────────
def recovery_time(price, ev_date):
    """Days from trough to recover 50% / 80% of the peak→trough drawdown.

    peak   = max price in [-30, 0]      (pre-event high)
    trough = min price in [-7, +30]     (the crash low)
    Returns drawdown depth and days-from-trough to 50%/80% recovery (NaN = not
    recovered within REC_HORIZON ⇒ censored).
    """
    idx = price.index
    pos = idx.searchsorted(pd.Timestamp(ev_date))
    if pos < 30 or pos >= len(idx) - 1:
        return None
    peak_win   = price.iloc[max(0, pos - 30): pos + 1]
    trough_win = price.iloc[max(0, pos - 7): min(len(idx), pos + 30)]
    peak = peak_win.max()
    trough_val = trough_win.min()
    trough_pos = idx.searchsorted(trough_win.idxmin())
    depth = (trough_val - peak) / peak
    if depth >= 0:
        return dict(depth=depth, t50=np.nan, t80=np.nan)
    targets = {50: trough_val + 0.50 * (peak - trough_val),
               80: trough_val + 0.80 * (peak - trough_val)}
    out = {"depth": depth}
    fwd = price.iloc[trough_pos: min(len(idx), trough_pos + REC_HORIZON + 1)]
    for pct, tgt in targets.items():
        hit = fwd[fwd >= tgt]
        out[f"t{pct}"] = float((hit.index[0] - idx[trough_pos]).days) if len(hit) else np.nan
    return out


# ── CAR ~ pre-event sentiment regression ─────────────────────────────────────
def car_sentiment_regression(rows):
    """OLS slope + Spearman + permutation p of full-window CAR on pre-event sentiment."""
    df = pd.DataFrame(rows).dropna(subset=["pre_sent", "car_full"])
    if len(df) < 4:
        return None
    x = df["pre_sent"].to_numpy(); y = df["car_full"].to_numpy()
    slope, intercept, r, p_ols, se = stats.linregress(x, y)
    rho, p_spear = stats.spearmanr(x, y)
    obs = abs(rho); count = 0; B = 20000           # permutation p for Spearman (small n)
    for _ in range(B):
        if abs(stats.spearmanr(x, RNG.permutation(y))[0]) >= obs:
            count += 1
    p_perm = (count + 1) / (B + 1)
    return dict(n=len(df), slope=slope, intercept=intercept, r=r, r2=r**2,
                p_ols=p_ols, rho=rho, p_spear=p_spear, p_perm=p_perm)


# ── assembly ─────────────────────────────────────────────────────────────────
def build_event_records(ret, price, sent, raw_events):
    """raw_events: list of (name, date, sign, tag). Returns (rows, ars)."""
    rows, ars = [], []
    for name, date, sign, tag in raw_events:
        ts = pd.Timestamp(date)
        e = event_ar(ret, ts)
        if e is None:
            continue
        ps, nv = pre_event_sentiment(sent, e["date"])
        rec = recovery_time(price, e["date"])
        rows.append(dict(
            name=name, date=e["date"].date(), sign=sign, tag=tag,
            mu=e["mu"], car_full=car_in_window(e, EVT_START, EVT_END),
            car_pre=car_in_window(e, EVT_START, -1),
            car_day0=car_in_window(e, 0, 0),
            car_post=car_in_window(e, 1, EVT_END),
            pre_sent=ps, pre_news=nv,
            depth=(rec["depth"] if rec else np.nan),
            rec50=(rec["t50"] if rec else np.nan),
            rec80=(rec["t80"] if rec else np.nan),
        ))
        e["name"] = name; e["sign"] = sign; e["tag"] = tag
        ars.append(e)
    return rows, ars


def fmt_window_block(title, results):
    lines = [f"  {title}"]
    lines.append("    window      N   meanCAR   boot95%CI            "
                 "Patell(p)      BMP(p)        sign(p)       Corrado(p)   agree?")
    for r in results:
        lines.append(
            f"    {r['window']:>9} {r['N']:>3} {r['mean_car']:+8.4f}  "
            f"[{r['boot_lo']:+.3f},{r['boot_hi']:+.3f}]  "
            f"{r['patell_z']:+6.2f}({r['patell_p']:.3f}) "
            f"{r['bmp_z']:+6.2f}({r['bmp_p']:.3f}) "
            f"{r['sign_z']:+6.2f}({r['sign_p']:.3f}) "
            f"{r['corrado_z']:+6.2f}({r['corrado_p']:.3f})  "
            f"{'YES' if r['agree_sig'] else 'no'}")
    return "\n".join(lines)


def write_md(R):
    """Interpretation only (results-file convention: no raw tables here)."""
    def sig_windows(res):
        return [r["window"] for r in res if r["agree_sig"]]

    nd, nu = R["n_named_down"], R["n_named_up"]
    md = []
    md.append("# Event Study — Interpretation (Sprint 3, Step 1)\n")
    md.append("*Raw numbers live in `outputs/event_study.txt`; this file is interpretation only.*\n")
    md.append("## What was tested\n")
    md.append(f"- **Sample A — named exogenous catalysts** ({nd} down, {nu} up): selected by "
              "catalyst, not by return, so abnormal returns here are **not circular**. This is "
              "the sample we trust for inference.\n"
              "- **Sample B — scar-decile events** (objective bottom/top-10% weekly returns, "
              "clustered): more events, but the contemporaneous CAR is **mechanically** large "
              "because events are *selected* on the return. Used for recovery asymmetry and the "
              "sentiment regression, not for the day-0 CAR claim.\n")
    md.append("- Five statistics per window (Patell, BMP, generalized sign, Corrado-Zivney "
              "rank, bootstrap CI). A window counts as significant **only when a parametric "
              "and a non-parametric test agree at 5%** — deliberately conservative.\n")

    md.append("## Headline reading\n")
    nd_post = next((r for r in R["res_named_down"] if r["window"] == "[+1,+10]"), None)
    nd_pre  = next((r for r in R["res_named_down"] if r["window"] == "[-10,-1]"), None)
    if nd_post:
        verdict = "significant (parametric+non-parametric agree)" if nd_post["agree_sig"] else "NOT robustly significant"
        md.append(f"- **Post-event drift (named down-catalysts, [+1,+10]):** mean CAR "
                  f"{nd_post['mean_car']:+.3f}, bootstrap 95% CI "
                  f"[{nd_post['boot_lo']:+.3f}, {nd_post['boot_hi']:+.3f}] — {verdict}. "
                  "This is the recovery/continuation question that bridges to RWDV.\n")
    if nd_pre:
        verdict = "significant" if nd_pre["agree_sig"] else "not significant"
        md.append(f"- **Pre-event drift (named down-catalysts, [-10,-1]):** mean CAR "
                  f"{nd_pre['mean_car']:+.3f} — {verdict}. Pre-drift would hint the market "
                  "partly anticipated the catalyst.\n")

    rn = R["reg_named"]; rd = R["reg_dec_down"]
    md.append("## Does pre-event sentiment line up with the move?\n")
    if rn:
        direction = "negative" if rn["slope"] < 0 else "positive"
        md.append(f"- **Named catalysts (n={rn['n']}):** CAR-on-sentiment slope {rn['slope']:+.3f} "
                  f"({direction}), Spearman ρ={rn['rho']:+.2f} (perm p={rn['p_perm']:.2f}). "
                  "A negative slope is the euphoria / 'sell-the-news' direction — higher prior "
                  "sentiment, worse subsequent abnormal return.\n")
    if rd:
        md.append(f"- **Decile down-events (n={rd['n']}):** slope {rd['slope']:+.3f}, "
                  f"Spearman ρ={rd['rho']:+.2f} (perm p={rd['p_perm']:.2f}).\n")
    md.append("- With single-digit-to-~20 events these correlations are **suggestive, not "
              "confirmatory**; report the direction and the CI, not a starred p-value.\n")

    md.append("## Recovery asymmetry (RWDV bridge)\n")
    md.append("- See `outputs/event_study.txt` for median 50%/80% recovery times and censoring "
              "counts. Slow/incomplete recovery after downside events is the empirical content "
              "behind recovery-weighted downside volatility.\n")

    md.append("## Honesty notes\n")
    md.append(f"- {len(R['res_named_down']) + len(R['res_named_up'])} windows were tested per "
              f"sample; Bonferroni reference threshold is p<{R['bonf']:.3f}. No window was "
              "selected after the fact — the four sub-windows are fixed in the script.\n")
    md.append("- Mean-adjusted model (no crypto market index available). De-peg events (Terra, "
              "USDC) are included as canonical extreme events per §1a.\n")
    md.append("- This is descriptive/independent evidence; it complements, and does not replace, "
              "the walk-forward logit (Step 2).\n")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")


def main():
    ret, price = load_prices()
    sent = load_sentiment()

    decile_events, q_lo, q_hi, n_weeks = scar_decile_events(ret)

    named_rows, named_ars = build_event_records(ret, price, sent, NAMED_EVENTS)
    dec_rows,   dec_ars   = build_event_records(ret, price, sent, decile_events)

    named_down = [e for e in named_ars if e["sign"] == "-"]
    named_up   = [e for e in named_ars if e["sign"] == "+"]
    dec_down   = [e for e in dec_ars if e["sign"] == "-"]
    dec_up     = [e for e in dec_ars if e["sign"] == "+"]

    windows = [(EVT_START, -1), (0, 0), (1, EVT_END), (EVT_START, EVT_END)]

    def all_windows(evs):
        return [run_window_tests(evs, a, b) for a, b in windows] if len(evs) >= 3 else []

    res_named_down = all_windows(named_down)
    res_named_up   = all_windows(named_up)
    res_dec_down   = all_windows(dec_down)
    res_dec_up     = all_windows(dec_up)

    reg_named = car_sentiment_regression(named_rows)
    reg_dec_down = car_sentiment_regression([r for r in dec_rows if r["sign"] == "-"])

    bonf = 0.05 / len(windows)

    # ── raw tables → .txt ────────────────────────────────────────────────────
    tx = []
    tx.append("=" * 92)
    tx.append("EVENT STUDY — extreme crypto events on 50/50 BTC+ETH portfolio (Sprint 3, Step 1)")
    tx.append(f"data: daily {ret.index.min().date()} → {ret.index.max().date()} | "
              f"est window [{EST_START},{EST_END}] L={L} | event window [{EVT_START},{EVT_END}] tau={TAU}")
    tx.append(f"scar decile: weekly bottom/top {int(DECILE_Q*100)}% over n={n_weeks} weeks | "
              f"q_lo={q_lo:+.4f} q_hi={q_hi:+.4f} | cluster gap={CLUSTER_GAP}d")
    tx.append(f"significance rule: >=1 parametric AND >=1 non-parametric at p<0.05 | "
              f"Bonferroni ref over {len(windows)} windows: p<{bonf:.4f}")
    tx.append("=" * 92)

    def event_table(title, rows):
        out = ["", title]
        df = pd.DataFrame(rows)[
            ["name", "date", "sign", "tag", "car_pre", "car_day0", "car_post",
             "car_full", "pre_sent", "pre_news", "depth", "rec50", "rec80"]]
        with pd.option_context("display.width", 220, "display.max_columns", 30,
                               "display.float_format", lambda v: f"{v:+.4f}"):
            out.append(df.to_string(index=False))
        return "\n".join(out)

    tx.append(event_table("SAMPLE A — NAMED CATALYSTS (per-event)", named_rows))
    tx.append(event_table("SAMPLE B — SCAR-DECILE (per-event)", dec_rows))

    tx.append("")
    tx.append("-" * 92)
    tx.append("WINDOW TESTS  (CAR aggregated across events; z(p) per test)")
    tx.append("-" * 92)
    if res_named_down: tx.append(fmt_window_block(f"A1. Named DOWN-catalysts (N={len(named_down)})", res_named_down))
    if res_named_up:   tx.append(fmt_window_block(f"A2. Named UP-catalysts (N={len(named_up)})", res_named_up))
    if res_dec_down:   tx.append(fmt_window_block(f"B1. Decile DOWN-events (N={len(dec_down)}) [day-0 CAR selection-biased]", res_dec_down))
    if res_dec_up:     tx.append(fmt_window_block(f"B2. Decile UP-events (N={len(dec_up)}) [day-0 CAR selection-biased]", res_dec_up))

    tx.append("")
    tx.append("-" * 92)
    tx.append("CAR ~ PRE-EVENT SENTIMENT (full-window CAR on mean FinBERT, 4wk pre)")
    tx.append("-" * 92)
    def reg_block(title, r):
        if r is None:
            return f"  {title}: n<4, not estimated"
        return (f"  {title}: n={r['n']} slope={r['slope']:+.4f} r={r['r']:+.3f} "
                f"R2={r['r2']:.3f} OLS_p={r['p_ols']:.3f} | "
                f"Spearman rho={r['rho']:+.3f} p={r['p_spear']:.3f} perm_p={r['p_perm']:.3f}")
    tx.append(reg_block("Named catalysts (all signs)", reg_named))
    tx.append(reg_block("Decile DOWN-events", reg_dec_down))

    tx.append("")
    tx.append("-" * 92)
    tx.append("RECOVERY TIME (days from trough to recover 50%/80% of peak->trough drawdown)")
    tx.append("-" * 92)
    def rec_summary(title, rows):
        d = pd.DataFrame(rows)
        d = d[d["sign"] == "-"]
        r50 = d["rec50"].dropna(); r80 = d["rec80"].dropna()
        return (f"  {title} DOWN: depth median={d['depth'].median():+.3f} | "
                f"rec50 median={(r50.median() if len(r50) else float('nan')):.0f}d "
                f"(censored {d['rec50'].isna().sum()}/{len(d)}) | "
                f"rec80 median={(r80.median() if len(r80) else float('nan')):.0f}d "
                f"(censored {d['rec80'].isna().sum()}/{len(d)})")
    tx.append(rec_summary("Named", named_rows))
    tx.append(rec_summary("Decile", dec_rows))
    tx.append("=" * 92)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(tx) + "\n")
    print("\n".join(tx))
    print(f"\nraw tables → {OUT_TXT}")

    # ── plots ────────────────────────────────────────────────────────────────
    rel = list(range(EVT_START, EVT_END + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    for e in named_ars:
        ax.plot(rel, e["car"], marker="o", ms=2.5, label=f"{e['name']} ({e['sign']})")
    ax.axvline(0, color="grey", ls="--", lw=1); ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Days relative to catalyst"); ax.set_ylabel("CAR")
    ax.set_title("Event study — CAR around named crypto catalysts (2021–2026)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(OUT_PLOT_CAR, dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    for evs, lab, col in [(dec_down, "decile DOWN", "tab:red"),
                          (named_down, "named DOWN", "darkred")]:
        if len(evs) >= 3:
            M = np.mean([e["car"] for e in evs], axis=0)
            ax.plot(rel, M, marker="o", ms=3, color=col, label=f"{lab} mean (N={len(evs)})")
    ax.axvline(0, color="grey", ls="--", lw=1); ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Days relative to event"); ax.set_ylabel("mean CAR")
    ax.set_title("Mean CAR — downside events (recovery asymmetry)")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(OUT_PLOT_REC, dpi=150); plt.close()
    print(f"plots → {OUT_PLOT_CAR}, {OUT_PLOT_REC}")

    R = dict(named_rows=named_rows, dec_rows=dec_rows,
             res_named_down=res_named_down, res_named_up=res_named_up,
             res_dec_down=res_dec_down, res_dec_up=res_dec_up,
             reg_named=reg_named, reg_dec_down=reg_dec_down,
             bonf=bonf, n_weeks=n_weeks,
             n_named_down=len(named_down), n_named_up=len(named_up),
             n_dec_down=len(dec_down), n_dec_up=len(dec_up))
    write_md(R)
    print(f"interpretation → {OUT_MD}")
    return R


if __name__ == "__main__":
    main()
