"""Stock Analyzer: run with `streamlit run app.py`"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from analyzer import WEIGHTS, analyze
from backtest import run_backtest, split_metrics
from screener import MOVER_MODES, UNIVERSES, download_history, get_universe, rank_movers

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")
st.title("📈 Stock Analyzer")
st.caption("Technicals + valuation + analyst data → a transparent Buy / Hold / Sell score. Educational only, not financial advice.")

VERDICT_COLORS = {"STRONG BUY": "#0a7d33", "BUY": "#2e9d57", "HOLD": "#b8860b",
                  "SELL": "#c0392b", "STRONG SELL": "#8e1b10"}


# ---------- cached data loaders ----------
@st.cache_data(ttl=900, show_spinner=False)
def load(ticker: str, period: str):
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=True)
    info = t.info or {}
    bench = yf.Ticker("SPY").history(period=period, auto_adjust=True)["Close"]
    news = getattr(t, "news", []) or []
    return hist, info, bench, news


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, period: str):
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    bench = yf.Ticker("SPY").history(period=period, auto_adjust=True)["Close"]
    return hist, bench


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_universe_data(tickers: tuple, period: str):
    return download_history(list(tickers), period)


def set_ticker(t: str):
    st.session_state["ticker"] = t


# ---------- sidebar ----------
with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL", key="ticker").upper().strip()
    period = st.selectbox("History (Analyze tab)", ["1y", "2y", "5y"], index=1)
    st.markdown("**Factor weights** (shared by all tabs)")
    for k, v in WEIGHTS.items():
        WEIGHTS[k] = st.slider(k, 0.0, 0.5, v, 0.05)

tab_analyze, tab_backtest, tab_movers = st.tabs(["🔍 Analyze", "⏪ Backtest", "🚀 Top movers"])


# ======================= ANALYZE =======================
with tab_analyze:
    try:
        with st.spinner(f"Fetching {ticker}…"):
            hist, info, bench, news = load(ticker, period)
        if hist.empty:
            st.error("No data found. Check the ticker symbol.")
        else:
            res = analyze(hist, info, bench)
            df, price = res["df"], res["price"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(info.get("longName", ticker), f"${price:,.2f}", f"{df['Close'].pct_change().iloc[-1]:+.2%} today")
            c2.markdown(f"<div style='font-size:14px;color:gray'>Verdict</div>"
                        f"<div style='font-size:32px;font-weight:700;color:{VERDICT_COLORS[res['verdict']]}'>{res['verdict']}</div>",
                        unsafe_allow_html=True)
            c3.metric("Score", f"{res['score_pct']}/100")
            c4.metric("Confidence", f"{res['confidence']}%")

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                         close=df["Close"], name="Price"), row=1, col=1)
            for col, clr in [("SMA50", "orange"), ("SMA200", "purple"), ("BB_high", "#999"), ("BB_low", "#999")]:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1, color=clr)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="teal")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD hist"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"), row=3, col=1)
            fig.update_layout(height=720, xaxis_rangeslider_visible=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Why this verdict")
            rows = [{"Factor": f.name, "Score (-2…+2)": f.score if f.available else None,
                     "Weight": WEIGHTS[f.name], "Reasoning": f.reason} for f in res["factors"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.subheader("Key stats")
                stats = {"Market cap": info.get("marketCap"), "Forward P/E": info.get("forwardPE"),
                         "52-wk high": info.get("fiftyTwoWeekHigh"), "52-wk low": info.get("fiftyTwoWeekLow"),
                         "Dividend yield": info.get("dividendYield"), "Beta": info.get("beta"),
                         "Analyst target": info.get("targetMeanPrice")}
                st.table(pd.Series({k: v for k, v in stats.items() if v is not None}, name="Value").astype(str))
            with right:
                st.subheader("Recent headlines")
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
    except Exception as e:
        st.error(f"Could not load data: {e}")


# ======================= BACKTEST =======================
def fmt_metrics(a: dict, b: dict, an="Strategy", bn="Buy & hold") -> pd.DataFrame:
    spec = [("Total return", "total_return", "{:+.1%}"), ("CAGR", "cagr", "{:+.1%}"),
            ("Max drawdown", "max_drawdown", "{:.1%}"), ("Volatility", "volatility", "{:.1%}"),
            ("Sharpe", "sharpe", "{:.2f}")]
    return pd.DataFrame({"Metric": [s[0] for s in spec],
                         an: [s[2].format(a[s[1]]) for s in spec],
                         bn: [s[2].format(b[s[1]]) for s in spec]})


with tab_backtest:
    st.markdown(f"Replays the **technical** part of the score on **{ticker}**'s history: buy when the score "
                "reaches the buy threshold, sell to cash when it drops to the sell threshold. Signals are read at "
                "the close and filled at the *next* close.")
    b1, b2, b3, b4, b5 = st.columns(5)
    bt_period = b1.selectbox("History", ["3y", "5y", "10y", "max"], index=2)
    buy_at = b2.slider("Buy at score ≥", -1.0, 2.0, 0.4, 0.1)
    sell_at = b3.slider("Sell at score ≤", -2.0, 1.0, -0.4, 0.1)
    cost_bps = b4.number_input("Cost per trade (bps)", 0.0, 100.0, 5.0, 1.0)
    split = b5.slider("In-sample share", 50, 90, 70, 5, format="%d%%")

    if st.button("Run backtest", type="primary"):
        try:
            with st.spinner("Running…"):
                h, bm = load_prices(ticker, bt_period)
                if h.empty:
                    st.error("No data found. Check the ticker symbol.")
                else:
                    st.session_state["bt"] = {"ticker": ticker, "res": run_backtest(h, bm, buy_at, sell_at, cost_bps),
                                              "split": split / 100}
        except Exception as e:
            st.session_state.pop("bt", None)
            st.error(str(e))

    bt = st.session_state.get("bt")
    if bt:
        r, sp = bt["res"], split_metrics(bt["res"]["strat_ret"], bt["res"]["bh_ret"], bt["split"])
        st.subheader(f"{bt['ticker']}: {r['start']:%b %Y} → {r['close'].index[-1]:%b %Y}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", r["n_trades"])
        m2.metric("Win rate", "n/a" if pd.isna(r["win_rate"]) else f"{r['win_rate']:.0%}")
        m3.metric("Avg trade", "n/a" if pd.isna(r["avg_trade"]) else f"{r['avg_trade']:+.1%}")
        m4.metric("Time invested", f"{r['exposure']:.0%}")
        st.dataframe(fmt_metrics(r["strategy"], r["buyhold"]), hide_index=True, use_container_width=True)

        eq = r["equity"]
        f1 = go.Figure()
        for col in eq.columns:
            f1.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col))
        f1.update_layout(title="Growth of $1", height=340, margin=dict(t=40, b=10))
        st.plotly_chart(f1, use_container_width=True)

        t = r["trades"]
        f2 = go.Figure(go.Scatter(x=r["close"].index, y=r["close"], name="Close", line=dict(color="#888", width=1)))
        if not t.empty:
            f2.add_trace(go.Scatter(x=t["Entry date"], y=t["Entry price"], mode="markers", name="Buy",
                                    marker=dict(symbol="triangle-up", size=11, color="green")))
            ex = t[t["Status"] == "closed"]
            f2.add_trace(go.Scatter(x=ex["Exit date"], y=ex["Exit price"], mode="markers", name="Sell",
                                    marker=dict(symbol="triangle-down", size=11, color="red")))
        f2.update_layout(title="Entries and exits", height=340, margin=dict(t=40, b=10))
        st.plotly_chart(f2, use_container_width=True)

        st.subheader("In-sample vs out-of-sample")
        if sp["boundary"] is not None:
            st.caption(f"Split at {sp['boundary']:%Y-%m-%d}. If you tuned the thresholds by looking at the "
                       "first part, only the second part is an honest test.")
            a, b = fmt_metrics(*sp["in"], "In-sample strategy", "In-sample buy & hold"), \
                   fmt_metrics(*sp["out"], "Out-of-sample strategy", "Out-of-sample buy & hold")
            st.dataframe(a.merge(b, on="Metric"), hide_index=True, use_container_width=True)

        with st.expander("Trade log"):
            show = t.copy()
            if not show.empty:
                show["Return"] = show["Return"].map("{:+.1%}".format)
            st.dataframe(show, hide_index=True, use_container_width=True)

        st.info("**Read this before trusting it.** Only the technical factors are tested: valuation and analyst data "
                "exist only as today's snapshot, so using them on past dates would be look-ahead bias. One stock's "
                "history is a small sample; tuning thresholds until the chart looks good is overfitting; and "
                "past results don't predict future returns. A strategy that loses to buy & hold on most "
                "tickers isn't adding value just because it has a smaller drawdown.")


# ======================= TOP MOVERS =======================
with tab_movers:
    st.markdown("Finds what's moving, drops illiquid names, then **ranks the movers by the analyzer's score** so you "
                "can see which big moves have the technical backing and which look stretched.")
    u1, u2, u3 = st.columns([2, 2, 1])
    uni = u1.selectbox("Universe", list(UNIVERSES))
    mode = u2.selectbox("Rank movers by", MOVER_MODES)
    top_n = u3.number_input("How many", 5, 50, 20, 5)
    custom = st.text_input("Tickers (comma-separated)", "AAPL, NVDA, TSLA, AMD, PLTR") if UNIVERSES[uni] == "custom" else ""
    f1c, f2c, f3c = st.columns(3)
    min_price = f1c.number_input("Min price ($)", 0.0, 1000.0, 5.0, 1.0)
    min_vol = f2c.number_input("Min avg daily volume", 0, 50_000_000, 500_000, 100_000)
    with_fund = f3c.checkbox("Include fundamentals (slower)", value=False,
                             help="Fetches P/E, growth and analyst data for each ranked stock.")

    if st.button("Scan market", type="primary"):
        try:
            with st.spinner("Downloading prices…"):
                syms, note = get_universe(uni, custom)
                if not syms:
                    st.error("No valid tickers.")
                else:
                    data = load_universe_data(tuple(syms), "2y")
                    bench = yf.Ticker("SPY").history(period="2y", auto_adjust=True)["Close"]
                    ranked, stats = rank_movers(data, bench, mode, int(top_n), min_price, min_vol,
                                                load_info if with_fund else None)
                    st.session_state["movers"] = {"df": ranked, "stats": stats, "note": note,
                                                  "mode": mode, "fund": with_fund}
        except Exception as e:
            st.session_state.pop("movers", None)
            st.error(f"Scan failed: {e}")

    mv = st.session_state.get("movers")
    if mv:
        if mv["note"]:
            st.warning(mv["note"])
        s, ranked = mv["stats"], mv["df"]
        st.caption(f"Screened {s['screened']} tickers · {s['skipped']} skipped (too little history, price or volume) · "
                   f"top {s['ranked']} movers ranked by score. "
                   + ("Scores include fundamentals." if mv["fund"] else "Scores are technical-only."))
        if ranked.empty:
            st.info("Nothing passed the filters. Try lowering the minimum price or volume.")
        else:
            st.dataframe(
                ranked, hide_index=True, use_container_width=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "1d %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "5d %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Rel. volume": st.column_config.NumberColumn(format="%.1fx"),
                    "RSI": st.column_config.NumberColumn(format="%.0f"),
                    "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                    "Confidence": st.column_config.NumberColumn(format="%d%%"),
                })
            p1, p2 = st.columns([1, 3])
            pick = p1.selectbox("Open in analyzer", list(ranked["Ticker"]))
            p2.write("")
            p2.button("Load into Analyze / Backtest tabs", on_click=set_ticker, args=(pick,))
        st.info("A big move usually has a news catalyst the score can't see (earnings, a deal, a downgrade). "
                "A high score on a stock that's up 15% today often means *the move already happened*; "
                "check the headlines and relative volume before acting.")

st.warning("This tool scores public data with simple rules. It can't see the future, ignores your goals, "
           "taxes and risk tolerance, and is not financial advice. Do your own research.")
