"""
scrape_articles.py
Builds a dense crypto-news corpus for 2021–2024 to replace the hand-curated
~63-article file (improvement.md T7; mentor's "BeautifulSoup / Snippet" hint).

The single biggest weakness of the v1 pipeline was density: 1–4 articles/month
is too thin for a stable monthly sentiment composite. This script targets
≥ 30 articles/month from two complementary sources:

  1. GDELT DOC 2.0 API  (PRIMARY, historical backfill)
       Free, no API key, indexes worldwide news back to 2017 with timestamps —
       the only practical way to backfill 2021–2024. Queried month-by-month.
       Returns headline + source domain + URL + date.

  2. CoinDesk RSS via BeautifulSoup  (ongoing / recency, mentor's named tool)
       RSS only carries ~50 recent items, so it cannot backfill history, but it
       demonstrates the BeautifulSoup path and lets the corpus grow forward if
       the script is re-run on a schedule.

SNIPPET (mentor's hint): sentiment concentrates in the headline + lede. GDELT
gives the headline directly. If ENRICH_SNIPPETS is on, we additionally fetch a
capped sample of article pages and pull the <meta name="description"> with
BeautifulSoup to use as a richer snippet; otherwise the headline doubles as the
snippet (an honest, reproducible fallback).

Politeness: GDELT requires ≤ 1 request / 5 s — we throttle to 1 / 8 s with
exponential-backoff retry. Raw JSON responses are cached under data/raw_html/ so
re-runs are free and never re-hit the source.

Output: data/articles_scraped.csv  with columns
    date, source, url, headline, snippet
"""

import os
import re
import time
import json
import hashlib
import datetime as dt

import requests
import pandas as pd
from bs4 import BeautifulSoup

# Prefer lxml's XML parser for RSS; fall back to the always-available html.parser
# so the script runs with zero extra installs.
try:
    import lxml  # noqa: F401
    _XML_PARSER = "xml"
except ImportError:
    _XML_PARSER = "html.parser"

# ── Settings ──────────────────────────────────────────────────────────────────
OUT_PATH        = "data/articles_scraped.csv"
CACHE_DIR       = "data/raw_html"
START           = dt.date(2021, 1, 1)
END             = dt.date(2024, 12, 31)
ARTICLES_PER_MONTH = 40          # cap pulled per month from GDELT
QUERY           = "(bitcoin OR ethereum OR crypto)"
GDELT_THROTTLE  = 12.0           # seconds between GDELT calls. The documented
                                 # limit is 1/5s but it temporarily blocks after
                                 # bursts, so we stay well clear. Failed months
                                 # are NOT cached, so simply re-running the script
                                 # resumes and fills any months that came back empty.
HEADERS         = {"User-Agent": "Mozilla/5.0 (research-bot; contact: student project)"}

ENRICH_SNIPPETS = False          # if True, fetch meta-description snippets (slow)
ENRICH_CAP      = 50             # max pages to enrich when ENRICH_SNIPPETS is on

COINDESK_RSS    = "https://www.coindesk.com/arc/outboundfeeds/rss/"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cache_path(key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


def _get_json(url: str, cache_key: str, throttle: float) -> dict:
    """GET with on-disk cache + throttle + exponential-backoff retry."""
    cp = _cache_path(cache_key)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    for attempt in range(5):
        time.sleep(throttle)
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 200 and r.content[:1] in (b"{", b"["):
                data = r.json()
                with open(cp, "w") as f:
                    json.dump(data, f)
                return data
            wait = throttle * (attempt + 1)
            print(f"    retry {attempt} (status {r.status_code}); waiting {wait:.0f}s")
        except Exception as e:
            print(f"    retry {attempt} ({type(e).__name__})")
    return {}


def month_iter(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        first = dt.date(y, m, 1)
        nxt = dt.date(y + (m == 12), (m % 12) + 1, 1)
        yield first, nxt
        y, m = nxt.year, nxt.month


# ── 1. GDELT historical backfill ──────────────────────────────────────────────
def fetch_gdelt() -> pd.DataFrame:
    rows = []
    for first, nxt in month_iter(START, END):
        s = first.strftime("%Y%m%d000000")
        e = nxt.strftime("%Y%m%d000000")
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={requests.utils.quote(QUERY)}"
            f"&mode=artlist&format=json&maxrecords={ARTICLES_PER_MONTH}"
            "&sort=hybridrel&sourcelang=english"
            f"&startdatetime={s}&enddatetime={e}"
        )
        data = _get_json(url, cache_key=f"gdelt:{s}:{e}", throttle=GDELT_THROTTLE)
        arts = data.get("articles", []) if data else []
        for a in arts:
            seen = a.get("seendate", "")
            try:
                date = dt.datetime.strptime(seen[:8], "%Y%m%d").date()
            except ValueError:
                continue
            rows.append({
                "date":     date.isoformat(),
                "source":   a.get("domain", "gdelt"),
                "url":      a.get("url", ""),
                "headline": (a.get("title") or "").strip(),
                "snippet":  "",
            })
        print(f"  {first:%Y-%m}: {len(arts)} articles")
    return pd.DataFrame(rows)


# ── 2. CoinDesk RSS via BeautifulSoup (mentor's named tool) ───────────────────
def fetch_coindesk_rss() -> pd.DataFrame:
    rows = []
    try:
        r = requests.get(COINDESK_RSS, timeout=20, headers=HEADERS)
        soup = BeautifulSoup(r.content, _XML_PARSER)
        for item in soup.find_all("item"):
            title = (item.title.text if item.title else "").strip()
            # html.parser treats <link> as a void element, so item.link.text is
            # empty; recover the URL from the raw item markup via regex instead.
            link_tag = item.find("link")
            link = (link_tag.text if link_tag and link_tag.text else "").strip()
            if not link:
                m = re.search(r"<link>(.*?)</link>", str(item), re.S)
                link = m.group(1).strip() if m else ""
            desc  = (item.description.text if item.description else "").strip()
            pub   = (item.pubDate.text if item.pubDate else "")
            try:
                date = pd.to_datetime(pub).date().isoformat()
            except Exception:
                continue
            # strip any HTML in the RSS description to get a clean snippet
            snippet = BeautifulSoup(desc, "html.parser").get_text(" ", strip=True)
            rows.append({"date": date, "source": "coindesk.com",
                         "url": link, "headline": title, "snippet": snippet[:240]})
        print(f"  CoinDesk RSS: {len(rows)} recent articles")
    except Exception as e:
        print(f"  CoinDesk RSS failed: {type(e).__name__}")
    return pd.DataFrame(rows)


# ── 3. Optional snippet enrichment via BeautifulSoup meta description ─────────
def enrich_snippets(df: pd.DataFrame) -> pd.DataFrame:
    todo = df[df["snippet"].str.len() == 0].head(ENRICH_CAP)
    print(f"  enriching {len(todo)} snippets via meta-description …")
    for i, row in todo.iterrows():
        try:
            r = requests.get(row["url"], timeout=10, headers=HEADERS)
            soup = BeautifulSoup(r.content, "html.parser")
            tag = soup.find("meta", attrs={"name": "description"}) or \
                  soup.find("meta", attrs={"property": "og:description"})
            if tag and tag.get("content"):
                df.at[i, "snippet"] = tag["content"].strip()[:240]
        except Exception:
            pass
        time.sleep(1)
    return df


def main():
    print("GDELT historical backfill …")
    gdelt = fetch_gdelt()
    print("CoinDesk RSS (BeautifulSoup) …")
    rss = fetch_coindesk_rss()

    df = pd.concat([gdelt, rss], ignore_index=True)
    if df.empty:
        print("No articles fetched — check network / cache."); return

    # Snippet fallback: headline doubles as snippet when none was captured.
    empty = df["snippet"].str.len() == 0
    if ENRICH_SNIPPETS:
        df = enrich_snippets(df)
        empty = df["snippet"].str.len() == 0
    df.loc[empty, "snippet"] = df.loc[empty, "headline"]

    # Dedupe + sort
    df = (df.dropna(subset=["headline"])
            .query("headline != ''")
            .drop_duplicates(subset=["date", "headline"])
            .sort_values("date")
            .reset_index(drop=True))
    df.to_csv(OUT_PATH, index=False)

    # Density report (the metric that matters most)
    per_month = (pd.to_datetime(df["date"]).dt.to_period("M").value_counts().sort_index())
    below = (per_month < 10).sum()
    print(f"\nSaved {len(df)} articles → {OUT_PATH}")
    print(f"  months covered: {len(per_month)}  |  median articles/month: {per_month.median():.0f}")
    print(f"  months below 10 articles: {below}  (target: 0)")


if __name__ == "__main__":
    main()
