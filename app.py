"""Stock Analyzer: run with `streamlit run app.py`"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from analyzer import DEFAULT_WEIGHTS, add_indicators, analyze
from backtest import run_backtest, split_metrics
from charts import equity_chart, price_chart, trades_chart
from screener import MOVER_MODES, UNIVERSES, download_history, get_universe, rank_movers
from utils import (ETF_FOR, INDEX_NAMES, INTERVALS, OVERVIEW, PRESETS, QUICK_PICKS, benchmark_for,
                   is_index, preset_start, resample_ohlcv, strip_tz)

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# Colors are semi-transparent / inherited so the styling works in both light and dark themes.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricLabel"] p { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 500; }
.muted { opacity: .65; font-size: .85rem; }
.ctl { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6; margin: 0 0 4px 2px; }
.sym-title { display: flex; align-items: baseline; gap: 10px; margin: 4px 0 14px; flex-wrap: wrap; }
.sym-title .name { font-size: 1.7rem; font-weight: 700; letter-spacing: -.02em; }
.sym-title .tick { font-size: 1rem; opacity: .6; font-weight: 500; }
.badge { font-size: .68rem; font-weight: 600; letter-spacing: .06em; padding: 2px 8px; border-radius: 999px;
         border: 1px solid rgba(128,128,128,.45); opacity: .85; }
.card { border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 14px 16px; min-height: 108px; }
.card .lbl { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7; }
.pill { display: inline-block; color: #fff; font-weight: 700; font-size: 1.05rem; padding: 4px 14px;
        border-radius: 999px; margin: 6px 0 12px; letter-spacing: .02em; }
.gauge { position: relative; height: 8px; border-radius: 999px; margin: 2px 0 4px;
         background: linear-gradient(90deg, #dc2626 0%, #f59e0b 50%, #16a34a 100%); }
.gauge .dot { position: absolute; top: 50%; width: 16px; height: 16px; border-radius: 50%;
              background: #fff; border: 3px solid #111827; transform: translate(-50%, -50%); }
.gauge-labels { display: flex; justify-content: space-between; font-size: .68rem; opacity: .55; }
</style>
""", unsafe_allow_html=True)

VERDICT_COLORS = {"STRONG BUY": "#0a7d33", "BUY": "#16a34a", "HOLD": "#b8860b",
                  "SELL": "#dc2626", "STRONG SELL": "#991b1b"}


# ---------- small UI helpers (written to work across Streamlit versions) ----------
def show_df(df: pd.DataFrame, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def segmented(label: str, options: list[str], default: str, key: str) -> str:
    fn = getattr(st, "segmented_control", None)
    if fn:
        return fn(label, options, default=default, key=key, label_visibility="collapsed") or default
    return st.radio(label, options, index=options.index(default), key=key, horizontal=True,
                    label_visibility="collapsed")


def ctl(text: str):
    st.markdown(f"<div class='ctl'>{text}</div>", unsafe_allow_html=True)


def fmt_big(n) -> str:
    if n is None:
        return "n/a"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(n) >= div:
            return f"{n / div:,.2f}{suf}"
    return f"{n:,.0f}"


def signal_card(res: dict) -> str:
    color = VERDICT_COLORS[res["verdict"]]
    return (f"<div class='card'><div class='lbl'>Signal</div>"
            f"<div class='pill' style='background:{color}'>{res['verdict']}</div>"
            f"<div class='gauge'><div class='dot' style='left:{res['score_pct']}%'></div></div>"
            f"<div class='gauge-labels'><span>Sell</span><span>Hold</span><span>Buy</span></div>"
            f"<div class='muted' style='margin-top:6px'>Score {res['score_pct']}/100 · "
            f"Confidence {res['confidence']}%</div></div>")


# ---------- cached data loaders ----------
@st.cache_data(ttl=900, show_spinner=False)
def load_daily(sym: str) -> pd.DataFrame:
    h = yf.Ticker(sym).history(period="max", auto_adjust=True)
    return strip_tz(h) if not h.empty else h


@st.cache_data(ttl=900, show_spinner=False)
def load_meta(sym: str):
    t = yf.Ticker(sym)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        news = t.news or []
    except Exception:
        news = []
    return info, news


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(sym: str) -> dict:
    try:
        return yf.Ticker(sym).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_overview():
    return download_history([s for _, s in OVERVIEW], "10d")


@st.cache_data(ttl=900, show_spinner=False)
def load_universe_data(tickers: tuple, period: str):
    return download_history(list(tickers), period)


def get_bench(ticker: str):
    bsym = benchmark_for(ticker)
    if not bsym:
        return None
    b = load_daily(bsym)
    return b["Close"] if not b.empty else None


# ---------- session state + sidebar ----------
st.session_state.setdefault("ticker", "AAPL")
st.session_state.setdefault("quick", "—")


def apply_quick():
    sym = QUICK_PICKS.get(st.session_state["quick"])
    if sym:
        st.session_state["ticker"] = sym
    st.session_state["quick"] = "—"


def set_ticker(sym: str):
    st.session_state["ticker"] = sym


with st.sidebar:
    st.markdown("### Symbol")
    st.text_input("Ticker or index", key="ticker",
                  help="Stocks (AAPL), ETFs (SPY) or indexes (^GSPC = S&P 500, ^NDX = Nasdaq-100).")
    st.selectbox("Quick picks", list(QUICK_PICKS), key="quick", on_change=apply_quick)
    ticker = st.session_state["ticker"].upper().strip()
    with st.expander("Scoring weights"):
        st.caption("Relative importance of each factor. They're normalized automatically.")
        weights = {k: st.slider(k, 0.0, 0.5, v, 0.05, key=f"w_{k}") for k, v in DEFAULT_WEIGHTS.items()}
    st.caption("Data: Yahoo Finance via yfinance, cached ~15 min. Educational only, not financial advice.")


# ---------- header + market snapshot ----------
st.markdown("<h1 style='margin-bottom:2px'>Stock Analyzer</h1>"
            "<div class='muted' style='margin-bottom:14px'>Technicals, valuation and analyst data combined into a "
            "transparent Buy / Hold / Sell score.</div>", unsafe_allow_html=True)


def market_snapshot():
    try:
        data = load_overview()
    except Exception:
        return
    cols = st.columns(len(OVERVIEW))
    for col, (name, sym) in zip(cols, OVERVIEW):
        d = data.get(sym)
        if d is None or len(d) < 2:
            continue
        last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
        col.metric(name, f"{last:,.2f}", f"{last / prev - 1:+.2%}",
                   delta_color="inverse" if sym == "^VIX" else "normal")


market_snapshot()
st.write("")


# ======================= ANALYZE =======================
def render_analyze(ticker: str, weights: dict):
    if not ticker:
        st.info("Enter a symbol in the sidebar.")
        return
    idx = is_index(ticker)
    daily = load_daily(ticker)
    if daily.empty or len(daily) < 30:
        st.error("No data found for that symbol. Check the spelling (indexes start with ^, e.g. ^GSPC).")
        return
    bench = get_bench(ticker)
    info, news = load_meta(ticker)
    res = analyze(daily, {} if idx else info, bench, weights)

    name = INDEX_NAMES.get(ticker) or info.get("longName") or info.get("shortName") or ticker
    badge = "<span class='badge'>INDEX</span>" if idx else ""
    st.markdown(f"<div class='sym-title'><span class='name'>{name}</span>"
                f"<span class='tick'>{ticker}</span>{badge}</div>", unsafe_allow_html=True)
    if idx:
        st.caption(f"An index can't be bought directly. Tracking funds include "
                   f"{ETF_FOR.get(ticker, 'a matching ETF')}. Valuation and analyst factors don't apply, "
                   "so the signal uses price technicals only.")

    kpi = st.container()
    first_date, last_date = daily.index[0], daily.index[-1]

    # --- date / interval controls ---
    c1, c2, c3 = st.columns([5, 3, 2])
    with c1:
        ctl("Date range")
        rng = segmented("Date range", PRESETS + ["Custom"], "1Y", "rng")
    with c2:
        ctl("Interval")
        interval = segmented("Interval", INTERVALS, "Daily", "interval")
    with c3:
        ctl("Chart")
        ctype = segmented("Chart", ["Candles", "Line"], "Candles", "ctype")

    if rng == "Custom":
        d1, d2 = st.columns(2)
        default_from = max(first_date, last_date - pd.DateOffset(years=1))
        s_in = d1.date_input("From", value=default_from.date(), min_value=first_date.date(),
                             max_value=last_date.date(), key=f"c_from_{ticker}")
        e_in = d2.date_input("To", value=last_date.date(), min_value=first_date.date(),
                             max_value=last_date.date(), key=f"c_to_{ticker}")
        start, end = pd.Timestamp(s_in), pd.Timestamp(e_in)
        if start >= end:
            st.warning("The start date must be before the end date.")
            return
    else:
        start, end = preset_start(rng, last_date, first_date), last_date

    chart_df = add_indicators(resample_ohlcv(daily, interval))
    view = chart_df.loc[start:end]
    if len(view) < 2:
        st.warning("Too few bars in that range for this interval. Try a longer range or a finer interval.")
        return

    o1, o2 = st.columns(2)
    overlays = o1.multiselect("Overlays", ["SMA 9", "SMA 20", "SMA 50", "SMA 200", "Bollinger bands"],
                              default=["SMA 9", "SMA 20", "SMA 50"], key="overlays")
    has_vol = float(view["Volume"].sum()) > 0
    panels = o2.multiselect("Panels", ["Volume", "RSI", "MACD"] if has_vol else ["RSI", "MACD"],
                            default=["Volume", "RSI"] if has_vol else ["RSI"], key=f"panels_{has_vol}")

    # --- KPI row (filled now that the range is known) ---
    pre = "" if idx else "$"
    last_px = float(daily["Close"].iloc[-1])
    day_chg = float(daily["Close"].pct_change().iloc[-1])
    rng_ret = float(view["Close"].iloc[-1] / view["Close"].iloc[0] - 1)
    with kpi:
        k1, k2, k3, k4 = st.columns([1, 1, 1.3, 1.5])
        k1.metric("Last price", f"{pre}{last_px:,.2f}", f"{day_chg:+.2%} today")
        k2.metric(f"Return ({rng})", f"{rng_ret:+.1%}")
        k3.metric("Range high / low", f"{view['High'].max():,.2f} / {view['Low'].min():,.2f}")
        k4.markdown(signal_card(res), unsafe_allow_html=True)

    show_chart(price_chart(chart_df, start, end, ctype, overlays, panels, interval))
    st.caption(f"{len(view)} {interval.lower()} bars. Moving averages are measured in {interval.lower()} bars. "
               "The Buy/Hold/Sell signal always uses daily data, whatever interval you chart.")

    st.markdown("### Why this signal")
    total_w = sum(weights.values()) or 1

    def reading(f):
        if not f.available:
            return "n/a"
        return "Bullish" if f.score > 0.5 else "Bearish" if f.score < -0.5 else "Neutral"

    rows = [{"Factor": f.name, "Reading": reading(f), "Score (-2 to +2)": f.score if f.available else None,
             "Weight": weights.get(f.name, 0) / total_w * 100, "Reasoning": f.reason} for f in res["factors"]]
    show_df(pd.DataFrame(rows), column_config={
        "Weight": st.column_config.NumberColumn(format="%.0f%%")})

    left, right = st.columns(2)
    with left:
        st.markdown("### Key stats")
        d252 = daily.iloc[-252:]
        stats = {"52-week high": f"{pre}{d252['High'].max():,.2f}", "52-week low": f"{pre}{d252['Low'].min():,.2f}"}
        if not idx:
            if info.get("marketCap"): stats["Market cap"] = "$" + fmt_big(info["marketCap"])
            if info.get("forwardPE"): stats["Forward P/E"] = f"{info['forwardPE']:.1f}"
            if info.get("beta"): stats["Beta"] = f"{info['beta']:.2f}"
            if info.get("targetMeanPrice"): stats["Analyst target"] = f"${info['targetMeanPrice']:,.2f}"
        show_df(pd.DataFrame({"Metric": list(stats), "Value": list(stats.values())}))
    with right:
        st.markdown("### Recent headlines")
        shown = 0
        for n in news[:8]:
            c = n.get("content", n)  # yfinance news schema varies by version
            title = c.get("title")
            link = (c.get("canonicalUrl") or {}).get("url") or n.get("link")
            if title:
                st.markdown(f"- [{title}]({link})" if link else f"- {title}")
                shown += 1
        if not shown:
            st.write("No headlines available.")


# ======================= BACKTEST =======================
def fmt_metrics(a: dict, b: dict, an="Strategy", bn="Buy & hold") -> pd.DataFrame:
    spec = [("Total return", "total_return", "{:+.1%}"), ("CAGR", "cagr", "{:+.1%}"),
            ("Max drawdown", "max_drawdown", "{:.1%}"), ("Volatility", "volatility", "{:.1%}"),
            ("Sharpe", "sharpe", "{:.2f}")]
    return pd.DataFrame({"Metric": [s[0] for s in spec],
                         an: [s[2].format(a[s[1]]) for s in spec],
                         bn: [s[2].format(b[s[1]]) for s in spec]})


def render_backtest(ticker: str, weights: dict):
    if not ticker:
        st.info("Enter a symbol in the sidebar.")
        return
    st.markdown(f"Replays the **technical** part of the score on **{ticker}**: buy when the score reaches the buy "
                "threshold, sell to cash when it drops to the sell threshold. Signals are read at the close and "
                "filled at the *next* close.")
    daily = load_daily(ticker)
    if daily.empty or len(daily) < 260:
        st.error("Need at least about a year of history for this symbol.")
        return
    bench = get_bench(ticker)
    first_date, last_date = daily.index[0], daily.index[-1]

    ctl("Test period")
    rng = segmented("Test period", ["3Y", "5Y", "10Y", "Max", "Custom"], "10Y", "bt_rng")
    if rng == "Custom":
        d1, d2 = st.columns(2)
        s_in = d1.date_input("From", value=max(first_date, last_date - pd.DateOffset(years=5)).date(),
                             min_value=first_date.date(), max_value=last_date.date(), key=f"bt_from_{ticker}")
        e_in = d2.date_input("To", value=last_date.date(), min_value=first_date.date(),
                             max_value=last_date.date(), key=f"bt_to_{ticker}")
        start, end = pd.Timestamp(s_in), pd.Timestamp(e_in)
    else:
        start, end = preset_start(rng, last_date, first_date), last_date

    b1, b2, b3, b4 = st.columns(4)
    buy_at = b1.slider("Buy at score ≥", -1.0, 2.0, 0.4, 0.1, key="bt_buy")
    sell_at = b2.slider("Sell at score ≤", -2.0, 1.0, -0.4, 0.1, key="bt_sell")
    cost_bps = b3.number_input("Cost per trade (bps)", 0.0, 100.0, 5.0, 1.0, key="bt_cost")
    split = b4.slider("In-sample share", 50, 90, 70, 5, format="%d%%", key="bt_split")

    if st.button("Run backtest", type="primary", key="bt_run"):
        try:
            with st.spinner("Running…"):
                st.session_state["bt"] = {"ticker": ticker, "split": split / 100,
                                          "res": run_backtest(daily, bench, buy_at, sell_at, cost_bps,
                                                              weights, start, end)}
        except Exception as e:
            st.session_state.pop("bt", None)
            st.error(str(e))

    bt = st.session_state.get("bt")
    if not bt:
        return
    r = bt["res"]
    sp = split_metrics(r["strat_ret"], r["bh_ret"], bt["split"])
    st.markdown(f"### {bt['ticker']}: {r['start']:%b %Y} → {r['close'].index[-1]:%b %Y}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trades", r["n_trades"])
    m2.metric("Win rate", "n/a" if pd.isna(r["win_rate"]) else f"{r['win_rate']:.0%}")
    m3.metric("Avg trade", "n/a" if pd.isna(r["avg_trade"]) else f"{r['avg_trade']:+.1%}")
    m4.metric("Time invested", f"{r['exposure']:.0%}")
    show_df(fmt_metrics(r["strategy"], r["buyhold"]))
    show_chart(equity_chart(r["equity"]))
    show_chart(trades_chart(r["close"], r["trades"]))

    st.markdown("### In-sample vs out-of-sample")
    if sp["boundary"] is not None:
        st.caption(f"Split at {sp['boundary']:%Y-%m-%d}. If you tuned the thresholds by looking at the first part, "
                   "only the second part is an honest test.")
        a = fmt_metrics(*sp["in"], "In-sample strategy", "In-sample buy & hold")
        b = fmt_metrics(*sp["out"], "Out-of-sample strategy", "Out-of-sample buy & hold")
        show_df(a.merge(b, on="Metric"))

    with st.expander("Trade log"):
        t = r["trades"].copy()
        if not t.empty:
            t["Return"] = t["Return"].map("{:+.1%}".format)
        show_df(t)

    st.info("**Read this before trusting it.** Only the technical factors are tested: valuation and analyst data "
            "exist only as today's snapshot, so using them on past dates would be look-ahead bias. One symbol's "
            "history is a small sample; tuning thresholds until the chart looks good is overfitting; and past "
            "results don't predict future returns.")


# ======================= TOP MOVERS =======================
def render_movers(weights: dict):
    st.markdown("Finds what's moving, drops illiquid names, then **ranks the movers by the analyzer's score** so you "
                "can see which big moves have technical backing and which look stretched.")
    u1, u2, u3 = st.columns([2, 2, 1])
    uni = u1.selectbox("Universe", list(UNIVERSES), key="mv_uni")
    mode = u2.selectbox("Rank movers by", MOVER_MODES, key="mv_mode")
    top_n = u3.number_input("How many", 5, 50, 20, 5, key="mv_n")
    custom = (st.text_input("Tickers (comma-separated)", "AAPL, NVDA, TSLA, AMD, PLTR", key="mv_custom")
              if UNIVERSES[uni] == "custom" else "")
    f1, f2, f3 = st.columns(3)
    min_price = f1.number_input("Min price ($)", 0.0, 1000.0, 5.0, 1.0, key="mv_price")
    min_vol = f2.number_input("Min avg daily volume", 0, 50_000_000, 500_000, 100_000, key="mv_vol")
    with_fund = f3.checkbox("Include fundamentals (slower)", value=False, key="mv_fund",
                            help="Fetches P/E, growth and analyst data for each ranked stock.")

    if st.button("Scan market", type="primary", key="mv_run"):
        try:
            with st.spinner("Downloading prices…"):
                syms, note = get_universe(uni, custom)
                if not syms:
                    st.error("No valid tickers.")
                else:
                    data = load_universe_data(tuple(syms), "2y")
                    bdf = load_daily("SPY")
                    bench = bdf["Close"].iloc[-600:] if not bdf.empty else None
                    ranked, stats = rank_movers(data, bench, mode, int(top_n), min_price, min_vol,
                                                load_info if with_fund else None, weights)
                    st.session_state["movers"] = {"df": ranked, "stats": stats, "note": note, "fund": with_fund}
        except Exception as e:
            st.session_state.pop("movers", None)
            st.error(f"Scan failed: {e}")

    mv = st.session_state.get("movers")
    if not mv:
        return
    if mv["note"]:
        st.warning(mv["note"])
    s, ranked = mv["stats"], mv["df"]
    st.caption(f"Screened {s['screened']} tickers · {s['skipped']} skipped (too little history, price or volume) · "
               f"top {s['ranked']} movers ranked by score. "
               + ("Scores include fundamentals." if mv["fund"] else "Scores are technical-only."))
    if ranked.empty:
        st.info("Nothing passed the filters. Try lowering the minimum price or volume.")
        return
    show_df(ranked, column_config={
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "1d %": st.column_config.NumberColumn(format="%+.1f%%"),
        "5d %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Rel. volume": st.column_config.NumberColumn(format="%.1fx"),
        "RSI": st.column_config.NumberColumn(format="%.0f"),
        "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "Confidence": st.column_config.NumberColumn(format="%d%%"),
    })
    p1, p2 = st.columns([1, 3])
    pick = p1.selectbox("Open in analyzer", list(ranked["Ticker"]), key="mv_pick")
    p2.write("")
    p2.write("")
    p2.button("Load into Analyze / Backtest tabs", on_click=set_ticker, args=(pick,), key="mv_load")
    st.info("A big move usually has a news catalyst the score can't see (earnings, a deal, a downgrade). A high "
            "score on a stock that's up 15% today often means *the move already happened*; check the headlines "
            "and relative volume before acting.")


# ---------- layout ----------
tab_analyze, tab_backtest, tab_movers = st.tabs(["Analyze", "Backtest", "Top movers"])
with tab_analyze:
    try:
        render_analyze(ticker, weights)
    except Exception as e:
        st.error(f"Could not load data: {e}")
with tab_backtest:
    try:
        render_backtest(ticker, weights)
    except Exception as e:
        st.error(f"Backtest error: {e}")
with tab_movers:
    render_movers(weights)

st.caption("This tool scores public data with simple rules. It can't see the future, ignores your goals, taxes and "
           "risk tolerance, and is not financial advice. Do your own research.")
