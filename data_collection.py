"""
data_collection.py
Fetches daily VWAP data for BTC and ETH via yfinance, constructs a 50/50
equal-weighted portfolio of daily returns, and saves to data/price_data.csv.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

# ── Settings ──────────────────────────────────────────────────────────────────
TICKERS   = ["BTC-USD", "ETH-USD"]
START     = "2021-01-01"   # wide window so rolling SD has enough history
END       = "2024-12-31"
OUT_PATH  = "data/price_data.csv"

os.makedirs("data", exist_ok=True)

# ── 1. Download OHLCV data ─────────────────────────────────────────────────────
print("Downloading price data …")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)

# ── 2. Approximate VWAP per day: (High + Low + Close) / 3 ────────────────────
# True intraday VWAP requires tick data; this is the standard daily proxy.
btc_vwap = (raw["High"]["BTC-USD"] + raw["Low"]["BTC-USD"] + raw["Close"]["BTC-USD"]) / 3
eth_vwap = (raw["High"]["ETH-USD"] + raw["Low"]["ETH-USD"] + raw["Close"]["ETH-USD"]) / 3

# ── 3. Daily log returns from VWAP ───────────────────────────────────────────
btc_ret = np.log(btc_vwap / btc_vwap.shift(1))
eth_ret = np.log(eth_vwap / eth_vwap.shift(1))

# ── 4. 50 / 50 equal-weighted portfolio return ────────────────────────────────
portfolio_ret = 0.5 * btc_ret + 0.5 * eth_ret

# ── 5. Assemble and save ──────────────────────────────────────────────────────
df = pd.DataFrame({
    "btc_vwap":      btc_vwap,
    "eth_vwap":      eth_vwap,
    "btc_ret":       btc_ret,
    "eth_ret":       eth_ret,
    "portfolio_ret": portfolio_ret,
}, index=raw.index)

df.index.name = "date"
df.dropna(inplace=True)
df.to_csv(OUT_PATH)

print(f"Saved {len(df)} rows → {OUT_PATH}")
print(df.tail())
