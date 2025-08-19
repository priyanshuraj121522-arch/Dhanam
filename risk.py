from typing import Dict
import numpy as np
import pandas as pd
from .data import is_india_ticker

def value_from_positions(latest_prices: pd.Series, positions: pd.DataFrame, usd_inr: float, base: str) -> pd.Series:
    vals = []
    for _, row in positions.iterrows():
        sym = row['symbol']; px = latest_prices.get(sym, np.nan); q = row.get('quantity', 0.0) or 0.0
        if pd.isna(px): vals.append(np.nan); continue
        if base == "INR" and not is_india_ticker(sym): px *= (usd_inr or 0.0)
        if base == "USD" and is_india_ticker(sym): px /= (usd_inr or np.nan)
        vals.append(px * q)
    return pd.Series(vals, index=positions.index)

def weights(values: pd.Series) -> pd.Series:
    total = values.sum(skipna=True); return values/total if total and total>0 else values*0

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame: return prices.ffill().pct_change().dropna(how='all')

def portfolio_series(returns: pd.DataFrame, w: pd.Series) -> pd.Series:
    w = w.reindex(returns.columns).fillna(0.0); return (returns * w).sum(axis=1)

def drawdown(series: pd.Series) -> pd.Series:
    cum = (1 + series).cumprod(); peak = cum.cummax(); return (cum - peak) / peak

def herfindahl_hirschman_index(w: pd.Series) -> float: return float((w.fillna(0.0)**2).sum())

def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame: return returns.corr().fillna(0.0)

def risk_score(metrics: Dict[str, float]) -> float:
    vol = metrics.get('volatility',0.0); beta = metrics.get('beta',1.0); dd = metrics.get('max_drawdown',0.0); hhi = metrics.get('hhi',0.0)
    vol_s = min(vol/0.02,3)/3; beta_s = min(abs(beta-1.0)/0.5,3)/3; dd_s = min(abs(dd)/0.2,3)/3; hhi_s = min(hhi/0.2,3)/3
    return float(round(100*(0.35*vol_s+0.25*beta_s+0.25*dd_s+0.15*hhi_s),1))

SECTOR_SENSITIVITY = {"Energy":3.0,"Industrials":-0.8,"Consumer Discretionary":-1.2,"Materials":0.5,"Utilities":-0.2,
"Financials":0.4,"Information Technology":-0.6,"Health Care":-0.1,"Communication Services":-0.4,"Real Estate":-0.6,"Unknown":-0.3}
RATE_SENSITIVITY = {"Financials":0.6,"Real Estate":-0.8,"Utilities":-0.5,"Information Technology":-0.4,"Consumer Discretionary":-0.3,
"Energy":-0.1,"Industrials":-0.2,"Health Care":-0.2,"Communication Services":-0.3,"Materials":-0.1,"Unknown":-0.2}

def scenario_impact(weights_by_sector: pd.Series, scenario: str, magnitude: float=1.0) -> float:
    if scenario=="Oil +10%": sens=SECTOR_SENSITIVITY
    elif scenario=="Rate +25 bps": sens=RATE_SENSITIVITY
    elif scenario=="Rate -25 bps": sens={k:-v for k,v in RATE_SENSITIVITY.items()}
    else: return 0.0
    return float(round(sum(weights_by_sector.get(s,0.0)*sens.get(s,0.0) for s in weights_by_sector.index)*magnitude,2))

def variance_contributions(returns: pd.DataFrame, w: pd.Series) -> pd.Series:
    w = w.reindex(returns.columns).fillna(0.0).values
    cov = returns.cov().values
    port_var = float(np.dot(w, np.dot(cov, w)))
    if port_var<=0: return pd.Series(0.0, index=returns.columns)
    mrc = np.dot(cov, w)  # marginal contribution
    rc = w * mrc / port_var
    return pd.Series(rc, index=returns.columns)

def sharpe_ratio(port_ret: pd.Series, rf_daily: float=0.0) -> float:
    excess = port_ret - rf_daily
    if excess.std(ddof=0)==0: return 0.0
    daily_sharpe = excess.mean()/excess.std(ddof=0)
    return float(daily_sharpe*np.sqrt(252))

def tracking_error(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    idx = port_ret.index.intersection(bench_ret.index)
    diff = port_ret.loc[idx]-bench_ret.loc[idx]
    return float(diff.std(ddof=0)*np.sqrt(252))
