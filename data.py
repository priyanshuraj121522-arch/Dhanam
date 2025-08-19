import time
from typing import List, Dict
import pandas as pd
import yfinance as yf

BENCHMARKS = {"IN":"^NSEI","US":"^GSPC"}

def is_india_ticker(t: str) -> bool:
    t = (t or "").upper()
    return t.endswith(".NS") or t.endswith(".BO")

def infer_market(tickers: List[str]) -> str:
    has_in = any(is_india_ticker(t) for t in tickers)
    has_us = any(not is_india_ticker(t) for t in tickers)
    if has_in and has_us: return "MIX"
    return "IN" if has_in else "US"

def download_prices(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    if not tickers: return pd.DataFrame()
    for _ in range(2):
        try:
            df = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by='column')
            break
        except Exception:
            time.sleep(1.0)
    prices = df['Close'].copy() if isinstance(df.columns, pd.MultiIndex) else df['Close'].to_frame()
    return prices.dropna(how='all')

def latest_prices(prices: pd.DataFrame) -> pd.Series:
    return prices.ffill().iloc[-1]

def fetch_benchmark(market: str, period: str = "1y") -> pd.Series:
    sym = BENCHMARKS.get(market, "^GSPC")
    df = yf.download(sym, period=period, auto_adjust=True, progress=False)
    return df['Close'].dropna()

def get_fx_series(period: str = "1y") -> pd.Series:
    df = yf.download("USDINR=X", period=period, auto_adjust=True, progress=False)
    return df['Close'].dropna()

def fetch_sector_for_tickers(tickers: List[str]) -> Dict[str, str]:
    out = {}
    for t in tickers:
        sector = None
        try:
            info = yf.Ticker(t).info or {}
            sector = info.get('sector')
        except Exception:
            pass
        out[t] = sector or "Unknown"
    return out
