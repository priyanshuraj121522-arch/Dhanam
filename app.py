import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os

from utils.data import (
    infer_market, download_prices, latest_prices, fetch_benchmark,
    get_fx_series, fetch_sector_for_tickers, is_india_ticker
)
from utils.risk import (
    value_from_positions, weights, daily_returns, portfolio_series, drawdown,
    herfindahl_hirschman_index, compute_correlation, risk_score, scenario_impact,
    variance_contributions, sharpe_ratio, tracking_error
)
from utils import visuals
from utils.report import build_pdf

st.set_page_config(page_title="DHANAM v2 — Risk Dashboard", page_icon="💹", layout="wide")

# Header
cl, ct, cr = st.columns([0.12,0.58,0.30])
with cl: st.image("assets/logo.svg")
with ct: st.markdown("### **DHANAM v2 — Smart Portfolio Risk Dashboard**")
with cr: base_ccy = st.selectbox("Base Currency", ["INR","USD"], index=0)

st.caption("India + USA multi-asset portfolio risk — Bloomberg-inspired UI")
st.markdown("---")

# Sidebar
st.sidebar.markdown("## Portfolio")
up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if up is not None:
    positions = pd.read_csv(up)
else:
    if st.sidebar.button("Load sample"):
        positions = pd.read_csv("sample_portfolio.csv")
    else:
        positions = pd.DataFrame(columns=["exchange","symbol","quantity","avg_price","sector"])

if st.sidebar.checkbox("Edit in app", value=False):
    positions = st.sidebar.data_editor(positions, num_rows="dynamic")

if positions.empty or positions["symbol"].dropna().empty:
    st.info("Upload a CSV with `symbol, quantity, [avg_price], [sector]`. Use .NS/.BO for India tickers.")
    st.stop()

# Sector autofill
if "sector" not in positions.columns or positions["sector"].fillna("").eq("").any():
    st.sidebar.caption("Filling sector labels (best-effort)…")
    sectors = fetch_sector_for_tickers(positions["symbol"].tolist())
    positions["sector"] = positions.apply(lambda r: r.get("sector") if pd.notna(r.get("sector")) and str(r.get("sector")).strip() else sectors.get(r["symbol"], "Unknown"), axis=1)

period = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1)
prices = download_prices(positions["symbol"].tolist(), period=period)
if prices.empty:
    st.error("Could not fetch prices; check symbols or try again."); st.stop()

bench_market = infer_market(positions["symbol"].tolist())
bench = fetch_benchmark("IN" if bench_market in ("IN","MIX") and base_ccy=="INR" else "US", period=period)
bench_r = bench.pct_change().dropna()

# FX
fx = get_fx_series(period=period)
fx_latest = float(fx.ffill().iloc[-1]) if not fx.empty else None

# Valuation
latest = latest_prices(prices)
values = value_from_positions(latest, positions, fx_latest, base_ccy)
w = weights(values)

# Tabs
tab_dash, tab_reb, tab_scen, tab_watch = st.tabs(["Dashboard","Rebalance Ideas","Scenario Lab","Watchlist"])

with tab_dash:
    total_value = float(values.sum(skipna=True))
    c1,c2,c3,c4 = st.columns([0.26,0.26,0.26,0.22])
    rets = daily_returns(prices)
    port_r = portfolio_series(rets, w)
    vol_d = float(port_r.std())
    idx = port_r.index.intersection(bench_r.index)
    beta = float(np.cov(port_r.loc[idx], bench_r.loc[idx])[0,1] / np.var(bench_r.loc[idx])) if len(idx)>5 and np.var(bench_r.loc[idx])>0 else 1.0
    dd = float(drawdown(port_r).min())
    hhi = float(herfindahl_hirschman_index(w))
    score = risk_score({"volatility": vol_d, "beta": beta, "max_drawdown": dd, "hhi": hhi})
    sharpe = sharpe_ratio(port_r)
    te = tracking_error(port_r, bench_r)

    with c1: st.metric("Portfolio Value", f"{total_value:,.2f} {base_ccy}")
    with c2: st.metric("Risk Score (0-100)", score)
    with c3: st.metric("Sharpe (ann.)", f"{sharpe:.2f}")
    with c4: st.metric("Tracking Error (ann.)", f"{te:.2f}%")

    st.markdown("---")
    a,b = st.columns([0.5,0.5])
    with a:
        by_sector = positions.assign(value=values).groupby(positions["sector"].replace({np.nan:"Unknown"}))["value"].sum()
        w_sector = (by_sector / by_sector.sum()).fillna(0.0)
        fig1 = visuals.donut_series(w_sector, "Exposure by Sector")
        st.pyplot(fig1, use_container_width=True)
    with b:
        corr = compute_correlation(rets)
        fig2 = visuals.heatmap_corr(corr, "Correlation Heatmap")
        st.pyplot(fig2, use_container_width=True)

    c,d = st.columns([0.6,0.4])
    with c:
        perf = pd.DataFrame({"Portfolio":(1+port_r).cumprod(),"Benchmark":(1+bench_r.reindex_like(port_r).fillna(0)).cumprod()}).dropna()
        fig3 = visuals.line_series(perf, "Performance vs Benchmark")
        st.pyplot(fig3, use_container_width=True)
    with d:
        # P&L column
        table = positions.copy()
        last_conv = []
        for s in positions["symbol"]:
            px = latest.get(s, np.nan)
            if base_ccy=="INR" and not is_india_ticker(s): px*=fx_latest
            if base_ccy=="USD" and is_india_ticker(s): px/=fx_latest if fx_latest else np.nan
            last_conv.append(px)
        table["last_price_base"] = last_conv
        table["value"] = values
        table["weight_%"] = (w*100).round(2)
        if "avg_price" in table.columns:
            ap = []
            for s,avg in zip(positions["symbol"], positions["avg_price"]):
                avgp = avg
                if base_ccy=="INR" and not is_india_ticker(s): avgp = avg * fx_latest if pd.notna(avg) else np.nan
                if base_ccy=="USD" and is_india_ticker(s): avgp = (avg / fx_latest) if (pd.notna(avg) and fx_latest) else np.nan
                ap.append(avgp)
            table["avg_price_base"] = ap
            table["pnl_%"] = ((table["last_price_base"]-table["avg_price_base"]) / table["avg_price_base"] * 100).round(2)
        st.markdown("**Holdings**")
        st.dataframe(table, use_container_width=True)

        # PDF export
        st.markdown("**Export**")
        # Save charts to disk for PDF
        import matplotlib.pyplot as plt
        img1, img2, img3 = "sector.png","corr.png","perf.png"
        fig1.savefig(img1, bbox_inches="tight"); fig2.savefig(img2, bbox_inches="tight"); fig3.savefig(img3, bbox_inches="tight")
        metrics = {
            "Portfolio Value": f"{total_value:,.2f} {base_ccy}",
            "Risk Score": f"{score:.1f}",
            "Beta vs Benchmark": f"{beta:.2f}",
            "Volatility (daily)": f"{vol_d:.4f}",
            "Max Drawdown": f"{dd:.2%}",
            "Sharpe (ann.)": f"{sharpe:.2f}",
            "Tracking Error (ann.)": f"{te:.2f}%",
        }
        pdf_path = "dhanam_report.pdf"
        build_pdf(pdf_path, "DHANAM Risk Report", metrics, [("Exposure by Sector", img1),("Correlation Heatmap", img2),("Performance vs Benchmark", img3)])
        with open(pdf_path,"rb") as f:
            st.download_button("Download PDF Risk Report", f, file_name="dhanam_report.pdf")

with tab_reb:
    st.markdown("### Rebalance Ideas")
    rets = daily_returns(prices)
    port_r = portfolio_series(rets, w)
    contrib = variance_contributions(rets, w).sort_values(ascending=False)
    st.markdown("**Top variance contributors (positions):**")
    st.dataframe(contrib.to_frame("risk_contribution").head(10))

    # Sector contributions
    pos = positions.copy(); pos["value"] = values
    by_sector = pos.groupby(pos["sector"].replace({np.nan:"Unknown"}))["value"].sum()
    w_sector = (by_sector/by_sector.sum()).fillna(0)
    # naive sector contribution = sum of position contributions per sector
    sector_map = dict(zip(positions["symbol"], positions["sector"].fillna("Unknown")))
    contrib_sector = pd.Series(0.0, index=w_sector.index)
    for sym, rc in contrib.items():
        sec = sector_map.get(sym, "Unknown"); contrib_sector[sec] = contrib_sector.get(sec,0.0) + rc
    st.markdown("**Top variance contributors (sectors):**")
    st.dataframe(contrib_sector.sort_values(ascending=False).to_frame("risk_contribution"))

    # Simple suggestions
    suggestions = []
    if w.max()>0.30: suggestions.append("Trim the largest position below 30% weight to reduce concentration risk.")
    if w_sector.max()>0.50: suggestions.append("Diversify: dominant sector exceeds 50% of portfolio.")
    if port_r.std()>bench_r.std()*1.3: suggestions.append("Overall volatility > benchmark; consider adding cash or low-beta names.")
    if suggestions:
        st.markdown("#### Suggestions")
        for s in suggestions: st.write("- " + s)
    st.caption("Hedge examples (US): SH (S&P inverse), PSQ (Nasdaq inverse). In India, consider index put options or raising cash.")

with tab_scen:
    st.markdown("### Scenario Lab")
    pos = positions.copy(); pos["value"] = values
    by_sector = pos.groupby(pos["sector"].replace({np.nan:"Unknown"}))["value"].sum()
    w_sector = (by_sector/by_sector.sum()).fillna(0)

    col1, col2 = st.columns(2)
    with col1: scen = st.selectbox("Preset", ["Oil +10%","Rate +25 bps","Rate -25 bps","USDINR +2%"], index=0)
    with col2: mag = st.slider("Magnitude (x)", 0.5, 2.0, 1.0, 0.1)

    if scen=="USDINR +2%":
        us_weight = float(pos[~pos['symbol'].str.upper().str.endswith(('.NS','.BO'))]["value"].sum() / pos["value"].sum())
        in_weight = 1 - us_weight
        impact = (us_weight if base_ccy=="INR" else -in_weight) * 2.0 * mag
    else:
        impact = scenario_impact(w_sector, scen, magnitude=mag)
    st.metric("Estimated Portfolio Impact", f"{impact:+.2f}%")

    st.markdown("#### Custom Sector Shock")
    custom_imp = 0.0
    for sec in w_sector.index:
        val = st.slider(f"{sec}", -5.0, 5.0, 0.0, 0.1, key=f"custom_{sec}")
        custom_imp += w_sector.get(sec,0.0) * val
    st.write(f"Custom impact: **{custom_imp:+.2f}%**")

with tab_watch:
    st.markdown("### Watchlist")
    syms = st.text_input("Enter tickers (.NS/.BO for India):","RELIANCE.NS, TCS.NS, AAPL, MSFT")
    if st.button("Fetch Watchlist"):
        lst = [s.strip() for s in syms.split(",") if s.strip()]
        wp = download_prices(lst, period="6mo")
        if wp.empty: st.warning("No data for watchlist.")
        else:
            last = wp.ffill().iloc[-1]; chg = wp.ffill().pct_change().iloc[-1]*100
            st.dataframe(pd.DataFrame({"Last":last,"1D %":chg.round(2)}), use_container_width=True)
