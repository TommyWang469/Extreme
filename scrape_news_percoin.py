"""
scrape_news_percoin.py  —  Sprint 3, Step 3 follow-up (§G1 unlock: PER-COIN news)

The panel (analysis_panel.py) showed that adding coins does NOT escape the power
ceiling for the *shared* market-sentiment regressor (the Moulton problem): sentiment
is one time series applied to every coin, so its effective N stays ~the number of
weeks. The fix is a *coin-specific* sentiment series — that is what this scrapes.

Queries GDELT DOC 2.0 (free, no key) per coin, per quarter (to keep the throttled
call count tractable), maxrecords=250/quarter. Cached + resumable: cached quarters
are read instantly, so a re-run only fills quarters that previously failed/empty.

Output: data/articles_percoin.csv  with columns  date, coin, source, url, headline
"""

import os
import time
import json
import hashlib
import datetime as dt

import requests
import pandas as pd

OUT_PATH   = "data/articles_percoin.csv"
CACHE_DIR  = "data/raw_html_percoin"
START      = dt.date(2021, 1, 1)
END        = dt.date(2026, 6, 12)
MAXRECORDS = 250
THROTTLE   = 8.0
HEADERS    = {"User-Agent": "Mozilla/5.0 (research-bot; contact: student project)"}

# 7 coins with unambiguous news identities + decent event counts (subset of the
# panel universe). Avalanche/Polkadot/Chainlink/LUNA are skipped first pass because
# their names are noisy ("avalanche", "polkadot") or coverage is sparse (LUNA).
COINS = {
    "BTC-USD":  "bitcoin",
    "ETH-USD":  "ethereum",
    "SOL-USD":  "solana",
    "XRP-USD":  "(ripple OR XRP)",
    "DOGE-USD": "dogecoin",
    "ADA-USD":  "cardano",
    "BNB-USD":  "(binance coin OR BNB)",
}

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


def _cache_path(key):
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


def _get_json(url, key):
    cp = _cache_path(key)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    for attempt in range(4):
        time.sleep(THROTTLE)
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 200 and r.content[:1] in (b"{", b"["):
                data = r.json()
                with open(cp, "w") as f:
                    json.dump(data, f)
                return data
            print(f"      retry {attempt} status {r.status_code}")
        except Exception as e:
            print(f"      retry {attempt} {type(e).__name__}")
    return {}


def quarter_iter(start, end):
    y, q = start.year, (start.month - 1) // 3
    while (y, q) <= (end.year, (end.month - 1) // 3):
        first = dt.date(y, q * 3 + 1, 1)
        ny, nq = (y + 1, 0) if q == 3 else (y, q + 1)
        nxt = dt.date(ny, nq * 3 + 1, 1)
        yield first, nxt
        y, q = ny, nq


def main():
    rows = []
    for coin, query in COINS.items():
        got = 0
        for first, nxt in quarter_iter(START, END):
            s = first.strftime("%Y%m%d000000")
            e = nxt.strftime("%Y%m%d000000")
            url = ("https://api.gdeltproject.org/api/v2/doc/doc"
                   f"?query={requests.utils.quote(query)}"
                   f"&mode=artlist&format=json&maxrecords={MAXRECORDS}"
                   "&sort=hybridrel&sourcelang=english"
                   f"&startdatetime={s}&enddatetime={e}")
            data = _get_json(url, key=f"{coin}:{s}:{e}")
            arts = data.get("articles", []) if data else []
            for a in arts:
                seen = a.get("seendate", "")
                try:
                    date = dt.datetime.strptime(seen[:8], "%Y%m%d").date()
                except ValueError:
                    continue
                rows.append({"date": date.isoformat(), "coin": coin,
                             "source": a.get("domain", "gdelt"),
                             "url": a.get("url", ""),
                             "headline": (a.get("title") or "").strip()})
            got += len(arts)
        print(f"  {coin}: {got} articles")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No per-coin articles fetched — check network/cache."); return
    df = (df.dropna(subset=["headline"]).query("headline != ''")
            .drop_duplicates(subset=["coin", "date", "headline"])
            .sort_values(["coin", "date"]).reset_index(drop=True))
    df.to_csv(OUT_PATH, index=False)
    cov = df.groupby("coin")["date"].count()
    print(f"\nSaved {len(df)} per-coin articles → {OUT_PATH}")
    print(cov.to_string())


if __name__ == "__main__":
    main()
