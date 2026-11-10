"""Stock Analyzer: run with `streamlit run app.py`"""
from __future__ import annotations

import hashlib
import re

import pandas as pd
import streamlit as st
import yfinance as yf

APP_VERSION = "5.0"
st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

try:
    from analyzer import DEFAULT_WEIGHTS, add_indicators, analyze
    from backtest import run_backtest, split_metrics
    from charts import (equity_chart, group_bars, intraday_chart, price_chart, regime_chart,
                        trades_chart)
    from market import GROUPS, GROUP_OF, REGIME_WEIGHTS, all_symbols, compute_pulse, flags, signal_label, symbol_stats
    from screener import MOVER_MODES, UNIVERSES, download_history, get_universe, rank_movers
    from utils import (ETF_FOR, INDEX_NAMES, INTERVALS, INTRADAY, OVERVIEW, PRESETS, QUICK_PICKS, benchmark_for,
                       is_index, live_params, preset_start, resample_ohlcv, strip_tz)
except ImportError as e:
    st.error(f"A file is missing or out of date: {e}. Upload ALL of these together (same folder / same repo level): "
             "app.py, analyzer.py, backtest.py, screener.py, charts.py, utils.py, market.py, requirements.txt.")
    st.stop()

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
.muted { opacity: .65; font-size: .85rem; }
.ctl { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6; margin: 0 0 4px 2px; }
.sym-title { display: flex; align-items: baseline; gap: 10px; margin: 4px 0 14px; flex-wrap: wrap; }
.sym-title .name { font-size: 1.7rem; font-weight: 700; letter-spacing: -.02em; }
.sym-title .tick { font-size: 1rem; opacity: .6; font-weight: 500; }
.badge { font-size: .68rem; font-weight: 600; letter-spacing: .06em; padding: 2px 8px; border-radius: 999px;
         border: 1px solid rgba(128,128,128,.45); opacity: .85; }
.card { border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 14px 16px; min-height: 108px; }
.card .lbl { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7; }
.card .big { font-size: 1.25rem; font-weight: 700; margin: 8px 0 6px; }
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
LIVE_TTL = 20  # seconds: how long short-lived market data is reused before asking Yahoo again
REFRESH_OPTIONS = {"Off": 0, "10 sec": 10, "30 sec": 30, "60 sec": 60}
SECTIONS = ["Analyze", "Favorites", "Market pulse", "Backtest", "Top movers"]


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


def now_et() -> str:
    return pd.Timestamp.now(tz="America/New_York").strftime("%H:%M:%S")


# ---------- cached data loaders ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load_hist_max(sym: str) -> pd.DataFrame:
    h = yf.Ticker(sym).history(period="max", auto_adjust=True)
    return strip_tz(h) if not h.empty else h


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def load_recent_daily(sym: str) -> pd.DataFrame:
    h = yf.Ticker(sym).history(period="5d", interval="1d", auto_adjust=True)
    return strip_tz(h) if not h.empty else h


def get_daily(sym: str) -> pd.DataFrame:
    """Full daily history (cached 1h) with the latest bars refreshed every few seconds."""
    base = load_hist_max(sym)
    if base.empty:
        return base
    try:
        rec = load_recent_daily(sym)
    except Exception:
        return base
    if rec.empty:
        return base
    return pd.concat([base[~base.index.isin(rec.index)], rec]).sort_index()


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def load_intraday(sym: str, interval: str, period: str, prepost: bool) -> pd.DataFrame:
    h = yf.Ticker(sym).history(period=period, interval=interval, prepost=prepost, auto_adjust=True)
    return strip_tz(h) if not h.empty else h


def _num(v):
    try:
        v = float(v)
        return None if v != v else v
    except Exception:
        return None


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def load_quote(sym: str) -> dict:
    t = yf.Ticker(sym)
    q = {"price": None, "prev": None, "high": None, "low": None, "age_min": None}
    try:
        fi = t.fast_info
        for key, attr in (("price", "last_price"), ("prev", "previous_close"),
                          ("high", "day_high"), ("low", "day_low")):
            try:
                q[key] = _num(getattr(fi, attr))
            except Exception:
                pass
    except Exception:
        pass
    try:
        h = t.history(period="1d", interval="1m")
        if not h.empty:
            q["age_min"] = (pd.Timestamp.now(tz=h.index.tz) - h.index[-1]).total_seconds() / 60
            q["price"] = q["price"] or _num(h["Close"].iloc[-1])
            q["high"] = q["high"] or _num(h["High"].max())
            q["low"] = q["low"] or _num(h["Low"].min())
    except Exception:
        pass
    return q


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=30, show_spinner=False)
def load_overview():
    return download_history([s for _, s in OVERVIEW], "10d")


@st.cache_data(ttl=60, show_spinner=False)
def load_universe_data(tickers: tuple, period: str):
    return download_history(list(tickers), period)


@st.cache_data(ttl=60, show_spinner=False)
def cached_universe(label: str, custom: str):
    return get_universe(label, custom)


@st.cache_data(ttl=60, show_spinner="Scanning about 200 stocks and ETFs (the first scan can take up to a minute)…")
def load_pulse(weights_key: tuple):
    return compute_pulse(download_history(all_symbols(), "2y"), dict(weights_key))


@st.cache_data(ttl=30, show_spinner=False)
def load_favorites(symbols: tuple, weights_key: tuple):
    data = download_history(list(symbols), "2y")
    stats = symbol_stats(data, dict(weights_key), include_indexes=True)
    have = set(stats.index) if not stats.empty else set()
    return stats, [s for s in symbols if s not in have]


def refresh_now():
    for fn in (load_recent_daily, load_intraday, load_quote, load_overview, load_universe_data, cached_universe,
               load_pulse, load_favorites):
        getattr(fn, "clear", lambda: None)()


def get_bench(ticker: str):
    bsym = benchmark_for(ticker)
    if not bsym:
        return None
    b = get_daily(bsym)
    return b["Close"] if not b.empty else None


# ---------- session state + sidebar ----------
st.session_state.setdefault("ticker", "AAPL")
st.session_state.setdefault("quick", "—")
st.session_state.setdefault("section", "Analyze")


def apply_quick():
    sym = QUICK_PICKS.get(st.session_state["quick"])
    if sym:
        st.session_state["ticker"] = sym
    st.session_state["quick"] = "—"


def open_in_analyzer(sym: str):
    st.session_state["ticker"] = sym
    st.session_state["section"] = "Analyze"


# ---------- favorites (saved in the page address, so every visitor has their own list) ----------
MAX_FAVS = 50
_SYM_RE = re.compile(r"^\^?[A-Z0-9]{1,8}(?:[.\-=][A-Z0-9]{1,4})?$")


def parse_symbols(text) -> list[str]:
    if isinstance(text, (list, tuple)):
        text = ",".join(text)
    out: list[str] = []
    for tok in re.split(r"[\s,;]+", str(text or "").upper()):
        if tok and _SYM_RE.match(tok) and tok not in out:
            out.append(tok)
    return out


def save_favorites(favs: list[str]):
    favs = favs[:MAX_FAVS]
    st.session_state["favorites"] = favs
    try:  # mirror into the URL (?fav=AAPL,NVDA) so a bookmark brings the list back
        if favs:
            st.query_params["fav"] = ",".join(favs)
        elif "fav" in st.query_params:
            del st.query_params["fav"]
    except Exception:
        pass


def add_favorites(syms):
    save_favorites(list(dict.fromkeys(st.session_state["favorites"] + parse_symbols(syms))))


def remove_favorite(sym: str):
    save_favorites([s for s in st.session_state["favorites"] if s != sym])
    st.session_state.pop("fav_pick", None)


def toggle_current_favorite():
    sym = st.session_state["ticker"].upper().strip()
    if sym in st.session_state["favorites"]:
        remove_favorite(sym)
    else:
        add_favorites([sym])


def add_from_box():
    add_favorites(st.session_state.get("fav_add", ""))
    st.session_state["fav_add"] = ""


def apply_list(key: str):
    save_favorites(parse_symbols(st.session_state.get(key, "")))


def clear_favorites():
    save_favorites([])


def open_favorite():
    sym = st.session_state.get("fav_menu")
    if sym and sym != "—":
        open_in_analyzer(sym)
    st.session_state["fav_menu"] = "—"


if "favorites" not in st.session_state:
    try:
        st.session_state["favorites"] = parse_symbols(st.query_params.get("fav", ""))[:MAX_FAVS]
    except Exception:
        st.session_state["favorites"] = []


with st.sidebar:
    st.markdown("### Symbol")
    st.text_input("Ticker or index", key="ticker",
                  help="Stocks (AAPL), ETFs (SPY) or indexes (^GSPC = S&P 500, ^NDX = Nasdaq-100).")
    cur = st.session_state["ticker"].upper().strip()
    is_fav = cur in st.session_state["favorites"]
    st.button("★ Remove from favorites" if is_fav else "☆ Add to favorites", on_click=toggle_current_favorite,
              key="fav_toggle", disabled=not cur)
    st.selectbox("Quick picks", list(QUICK_PICKS), key="quick", on_change=apply_quick)
    if st.session_state["favorites"]:
        st.selectbox("★ Favorites", ["—"] + st.session_state["favorites"], key="fav_menu", on_change=open_favorite)
    ticker = cur
    st.markdown("### Live data")
    refresh_label = st.selectbox("Auto-refresh", list(REFRESH_OPTIONS), index=2, key="refresh")
    st.button("Refresh now", on_click=refresh_now, key="refresh_now")
    with st.expander("Scoring weights"):
        st.caption("Relative importance of each factor. They're normalized automatically.")
        weights = {k: st.slider(k, 0.0, 0.5, v, 0.05, key=f"w_{k}") for k, v in DEFAULT_WEIGHTS.items()}
    st.caption(f"Data: Yahoo Finance via yfinance (free feed: usually near real-time, but some exchanges lag up "
               f"to ~15 min). Educational only, not financial advice. · App v{APP_VERSION}")

REFRESH_SECS = REFRESH_OPTIONS.get(refresh_label, 0)


def refreshing(fn, every: int | None = None):
    """Wrap `fn` so it re-runs on its own every N seconds without reloading the whole page."""
    secs = REFRESH_SECS if every is None else every
    frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if frag is None:
        return fn
    try:
        return frag(run_every=f"{secs}s" if secs else None)(fn)
    except TypeError:
        return frag(fn)


# ---------- header ----------
st.markdown("<h1 style='margin-bottom:2px'>Stock Analyzer</h1>"
            "<div class='muted' style='margin-bottom:14px'>Live prices, technicals, valuation and analyst data "
            "combined into a transparent Buy / Hold / Sell score.</div>", unsafe_allow_html=True)


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


refreshing(market_snapshot)()
st.write("")
ctl("Section")
section = segmented("Section", SECTIONS, "Analyze", "section")
st.write("")


# ======================= ANALYZE =======================
def live_kpis(ticker: str, weights: dict):
    """Price strip + signal. Auto-refreshes."""
    idx = is_index(ticker)
    daily = get_daily(ticker)
    if daily.empty or len(daily) < 30:
        st.error("No data found for that symbol. Check the spelling (indexes start with ^, e.g. ^GSPC).")
        return
    info, _ = load_meta(ticker)
    res = analyze(daily, {} if idx else info, get_bench(ticker), weights)
    q = load_quote(ticker)
    close = daily["Close"]
    price = q.get("price") or float(close.iloc[-1])
    prev = q.get("prev") or float(close.iloc[-2])
    hi = q.get("high") or float(daily["High"].iloc[-1])
    lo = q.get("low") or float(daily["Low"].iloc[-1])
    pre = "" if idx else "$"
    age = q.get("age_min")
    live = age is not None and age < 20
    status = ("<span style='color:#16a34a'>● LIVE</span>" if live
              else "<span style='opacity:.75'>● Market closed</span>")

    k1, k2, k3, k4 = st.columns([1.2, 1, 1.2, 1.5])
    k1.metric("Last price", f"{pre}{price:,.2f}", f"{price - prev:+,.2f} ({price / prev - 1:+.2%})")
    k2.metric("Day range", f"{lo:,.2f} – {hi:,.2f}")
    k3.markdown(f"<div class='card'><div class='lbl'>Market</div><div class='big'>{status}</div>"
                f"<div class='muted'>Updated {now_et()} ET</div></div>", unsafe_allow_html=True)
    k4.markdown(signal_card(res), unsafe_allow_html=True)


def why_table(ticker: str, weights: dict):
    daily = get_daily(ticker)
    if daily.empty or len(daily) < 30:
        return
    info, _ = load_meta(ticker)
    res = analyze(daily, {} if is_index(ticker) else info, get_bench(ticker), weights)
    total_w = sum(weights.values()) or 1

    def reading(f):
        if not f.available:
            return "n/a"
        return "Bullish" if f.score > 0.5 else "Bearish" if f.score < -0.5 else "Neutral"

    rows = [{"Factor": f.name, "Reading": reading(f), "Score (-2 to +2)": f.score if f.available else None,
             "Weight": weights.get(f.name, 0) / total_w * 100, "Reasoning": f.reason} for f in res["factors"]]
    show_df(pd.DataFrame(rows), column_config={"Weight": st.column_config.NumberColumn(format="%.0f%%")})


def live_chart(ticker: str, interval: str, rng: str, prepost: bool, ctype: str,
               overlays: list, panels: list):
    """Intraday chart. Auto-refreshes."""
    period, eff, note = live_params(interval, rng)
    df = load_intraday(ticker, eff, period, prepost)
    if df.empty or len(df) < 2:
        st.info("No intraday bars available yet (the market may be closed, or this symbol has no intraday data).")
        return
    cdf = add_indicators(df)
    us = not ("." in ticker or ticker.endswith(("-USD", "=X", "=F")))
    show_chart(price_chart(cdf, cdf.index[0], cdf.index[-1], ctype, overlays, panels, eff,
                           intraday=True, us_hours=us, prepost=prepost))
    st.caption(f"{len(cdf)} {eff} bars · last bar {cdf.index[-1]:%b %d, %H:%M} (exchange time). "
               "Moving averages are measured in bars. " + note)


def history_view(ticker: str, daily: pd.DataFrame, idx: bool, ctype: str, rng: str, interval: str,
                 overlays: list, panels: list):
    first_date, last_date = daily.index[0], daily.index[-1]
    if rng == "Custom":
        d1, d2 = st.columns(2)
        s_in = d1.date_input("From", value=max(first_date, last_date - pd.DateOffset(years=1)).date(),
                             min_value=first_date.date(), max_value=last_date.date(), key=f"c_from_{ticker}")
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
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Return ({rng})", f"{view['Close'].iloc[-1] / view['Close'].iloc[0] - 1:+.1%}")
    r2.metric("Range high", f"{view['High'].max():,.2f}")
    r3.metric("Range low", f"{view['Low'].min():,.2f}")
    show_chart(price_chart(chart_df, start, end, ctype, overlays, panels, interval))
    st.caption(f"{len(view)} {interval.lower()} bars. Moving averages are measured in {interval.lower()} bars. "
               "The Buy/Hold/Sell signal always uses daily data, whatever interval you chart.")


def render_analyze(ticker: str, weights: dict):
    if not ticker:
        st.info("Enter a symbol in the sidebar.")
        return
    idx = is_index(ticker)
    daily = get_daily(ticker)
    if daily.empty or len(daily) < 30:
        st.error("No data found for that symbol. Check the spelling (indexes start with ^, e.g. ^GSPC).")
        return
    info, news = load_meta(ticker)
    name = INDEX_NAMES.get(ticker) or info.get("longName") or info.get("shortName") or ticker
    badge = "<span class='badge'>INDEX</span>" if idx else ""
    if ticker in st.session_state["favorites"]:
        badge += "<span class='badge'>★ FAVORITE</span>"
    st.markdown(f"<div class='sym-title'><span class='name'>{name}</span>"
                f"<span class='tick'>{ticker}</span>{badge}</div>", unsafe_allow_html=True)
    if idx:
        st.caption(f"An index can't be bought directly. Tracking funds include "
                   f"{ETF_FOR.get(ticker, 'a matching ETF')}. Valuation and analyst factors don't apply, "
                   "so the signal uses price technicals only.")

    refreshing(live_kpis)(ticker, weights)
    st.write("")

    # --- chart controls: Live (intraday, auto-refreshing) or History (day / week / month ...) ---
    c0, c1, c2, c3 = st.columns([2, 4, 4, 2])
    with c0:
        ctl("View")
        view = segmented("View", ["Live", "History"], "Live", "view")
    if view == "Live":
        with c1:
            ctl("Range")
            lrng = segmented("Range", ["1D", "5D", "1M"], "1D", "live_rng")
        with c2:
            ctl("Bar size")
            lint = segmented("Bar size", INTRADAY, "5m", "live_int")
    else:
        with c1:
            ctl("Date range")
            rng = segmented("Date range", PRESETS + ["Custom"], "1Y", "rng")
        with c2:
            ctl("Interval")
            interval = segmented("Interval", INTERVALS, "Daily", "interval")
    with c3:
        ctl("Chart")
        ctype = segmented("Chart", ["Candles", "Line"], "Candles", "ctype")

    o1, o2, o3 = st.columns([4, 4, 2])
    overlays = o1.multiselect("Overlays", ["SMA 9", "SMA 20", "SMA 50", "SMA 200", "Bollinger bands"],
                              default=["SMA 9", "SMA 20", "SMA 50"], key="overlays")
    has_vol = float(daily["Volume"].iloc[-60:].sum()) > 0
    panels = o2.multiselect("Panels", ["Volume", "RSI", "MACD"] if has_vol else ["RSI", "MACD"],
                            default=["Volume", "RSI"] if has_vol else ["RSI"], key=f"panels_{has_vol}")
    prepost = o3.checkbox("Extended hours", value=False, key="ext") if view == "Live" else False

    if view == "Live":
        refreshing(live_chart)(ticker, lint, lrng, prepost, ctype, overlays, panels)
    else:
        history_view(ticker, daily, idx, ctype, rng, interval, overlays, panels)

    st.markdown("### Why this signal")
    refreshing(why_table)(ticker, weights)

    left, right = st.columns(2)
    with left:
        st.markdown("### Key stats")
        pre = "" if idx else "$"
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
    daily = get_daily(ticker)
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
def movers_results(weights: dict):
    """Runs the scan with the last-submitted settings. Can auto-refresh."""
    p = st.session_state.get("mv_params")
    if not p:
        return
    try:
        syms, note = cached_universe(p["uni"], p["custom"])
        if not syms:
            st.error("No valid tickers.")
            return
        data = load_universe_data(tuple(syms), "2y")
        bdf = get_daily("SPY")
        bench = bdf["Close"] if not bdf.empty else None
        ranked, stats = rank_movers(data, bench, p["mode"], p["top_n"], p["min_price"], p["min_vol"],
                                    load_info if p["fund"] else None, weights)
    except Exception as e:
        st.error(f"Scan failed: {e}")
        return
    st.session_state["mv_tickers"] = list(ranked["Ticker"]) if not ranked.empty else []
    if note:
        st.warning(note)
    st.caption(f"Updated {now_et()} ET · screened {stats['screened']} tickers · {stats['skipped']} skipped (too "
               f"little history, price or volume) · top {stats['ranked']} movers ranked by score. "
               + ("Scores include fundamentals." if p["fund"] else "Scores are technical-only."))
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


def render_movers(weights: dict):
    st.markdown("Finds what's moving, drops illiquid names, then **ranks the movers by the analyzer's score** so you "
                "can see which big moves have technical backing and which look stretched.")
    u1, u2, u3 = st.columns([2, 2, 1])
    uni = u1.selectbox("Universe", list(UNIVERSES), key="mv_uni")
    mode = u2.selectbox("Rank movers by", MOVER_MODES, key="mv_mode")
    top_n = u3.number_input("How many", 5, 50, 20, 5, key="mv_n")
    custom = (st.text_input("Tickers (comma-separated)", "AAPL, NVDA, TSLA, AMD, PLTR", key="mv_custom")
              if UNIVERSES[uni] == "custom" else "")
    f1, f2, f3, f4 = st.columns(4)
    min_price = f1.number_input("Min price ($)", 0.0, 1000.0, 5.0, 1.0, key="mv_price")
    min_vol = f2.number_input("Min avg daily volume", 0, 50_000_000, 500_000, 100_000, key="mv_vol")
    with_fund = f3.checkbox("Include fundamentals (slower)", value=False, key="mv_fund",
                            help="Fetches P/E, growth and analyst data for each ranked stock.")
    auto = f4.checkbox("Auto-refresh every minute", value=False, key="mv_auto")

    if st.button("Scan market", type="primary", key="mv_run"):
        st.session_state["mv_params"] = {"uni": uni, "custom": custom, "mode": mode, "top_n": int(top_n),
                                         "min_price": min_price, "min_vol": min_vol, "fund": with_fund}
    if "mv_params" not in st.session_state:
        return
    with st.spinner("Scanning…"):
        refreshing(movers_results, every=60 if auto else 0)(weights)

    tickers = st.session_state.get("mv_tickers", [])
    if tickers:
        p1, p2, p3 = st.columns([2, 1, 1])
        pick = p1.selectbox("Open in analyzer", tickers, key="mv_pick")
        for col in (p2, p3):
            col.write("")
            col.write("")
        p2.button("Open in Analyze", on_click=open_in_analyzer, args=(pick,), key="mv_load")
        p3.button("☆ Add to favorites", on_click=add_favorites, args=([pick],), key="mv_fav")
    st.info("A big move usually has a news catalyst the score can't see (earnings, a deal, a downgrade). A high "
            "score on a stock that's up 15% today often means *the move already happened*; check the headlines "
            "and relative volume before acting.")


# ======================= MARKET PULSE =======================
REGIME_COLORS = {"STRONGLY BULLISH": "#0a7d33", "BULLISH": "#16a34a", "NEUTRAL / MIXED": "#b8860b",
                 "BEARISH": "#dc2626", "STRONGLY BEARISH": "#991b1b"}
VOL_COLORS = {"LOW": "#16a34a", "NORMAL": "#2563eb", "ELEVATED": "#d97706", "HIGH": "#dc2626", "EXTREME": "#991b1b"}
PULSE_REFRESH = {"Off": 0, "1 min": 60, "5 min": 300}
PULSE_SORTS = {"Trend rating": ("Rating", False), "Biggest 1D gain": ("1d %", False),
               "Biggest 1D loss": ("1d %", True), "Unusual volume": ("Rel. volume", False),
               "Most volatile": ("Volatility %", False)}


def pulse_results(wkey: tuple, selected: list, flt: dict):
    """Everything on the Market pulse page. Can auto-refresh."""
    try:
        p = load_pulse(wkey)
    except Exception as e:
        st.error(f"The market scan failed: {e}")
        return
    reg, vol, stocks = p["regime"], p["vol"], p["stocks"]
    if not reg.get("ok"):
        st.error("Not enough market data came back from Yahoo to rate the market right now. "
                 "Try 'Refresh now' in the sidebar in a minute.")
        return
    nan = float("nan")

    # --- headline cards ---
    arrow = {"Improving": "▲", "Deteriorating": "▼", "Stable": "►"}.get(reg["direction"], "")
    delta_txt = "" if reg["delta"] != reg["delta"] else f" ({reg['delta']:+.1f} vs 5 sessions ago)"
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<div class='card'><div class='lbl'>Market rating</div>"
        f"<div class='pill' style='background:{REGIME_COLORS[reg['label']]}'>{reg['label']}</div>"
        f"<div class='gauge'><div class='dot' style='left:{reg['pct']}%'></div></div>"
        f"<div class='gauge-labels'><span>Bearish</span><span>Neutral</span><span>Bullish</span></div>"
        f"<div class='muted' style='margin-top:6px'>{reg['pct']}/100 · {arrow} {reg['direction']}{delta_txt}</div></div>",
        unsafe_allow_html=True)
    if vol.get("ok"):
        vix_txt = (f"VIX {vol['vix']:.1f} ({vol['vix_chg']:+.1%} today)" if vol["vix"] is not None else "VIX n/a")
        rv_txt = f"S&P 500 realized volatility {vol['realized']:.0f}%/yr" if vol["realized"] is not None else ""
        c2.markdown(
            f"<div class='card'><div class='lbl'>Volatility</div>"
            f"<div class='pill' style='background:{VOL_COLORS[vol['level']]}'>{vol['level']}</div>"
            f"<div class='muted'>{vix_txt}<br>{rv_txt}</div></div>", unsafe_allow_html=True)
    b = reg["breadth"]
    highs, lows = int(stocks["new_high"].sum()), int(stocks["new_low"].sum())
    c3.markdown(
        f"<div class='card'><div class='lbl'>Breadth</div>"
        f"<div class='big'>{b.get('p50', nan):.0f}% above 50-day</div>"
        f"<div class='muted'>{b.get('p200', nan):.0f}% above 200-day · {b.get('padv', nan):.0f}% up today<br>"
        f"{highs} new 20-day highs · {lows} new lows</div></div>", unsafe_allow_html=True)

    # --- volatility warnings ---
    if vol.get("ok"):
        msg = "\n".join(f"- {w}" for w in vol["warnings"])
        if vol["idx"] >= 3:
            st.error(f"**Volatility is {vol['level']}.** Expect larger and faster price swings than usual.\n{msg}")
        elif vol["idx"] == 2 or vol["warnings"]:
            st.warning(f"**Volatility is {vol['level']}.**\n{msg}")
        else:
            st.success("**Volatility is calm.** No warnings right now.")

    # --- trend over time + today's tape ---
    a, bcol = st.columns(2)
    with a:
        st.markdown("#### Market rating, last 3 months")
        show_chart(regime_chart(reg["history"]))
    with bcol:
        st.markdown("#### S&P 500 (SPY) today")
        try:
            idf = load_intraday("SPY", "5m", "1d", False)
            if len(idf) >= 2:
                show_chart(intraday_chart(idf, load_quote("SPY").get("prev")))
            else:
                st.info("No intraday bars yet. The market may be closed.")
        except Exception:
            st.info("Intraday data isn't available right now.")

    st.markdown("#### What's driving the rating")
    total_w = sum(REGIME_WEIGHTS.values())

    def reading(c):
        if not c["available"]:
            return "n/a"
        return "Bullish" if c["score"] > 0.5 else "Bearish" if c["score"] < -0.5 else "Neutral"

    show_df(pd.DataFrame([{"Factor": c["Factor"], "Reading": reading(c), "Score (-2 to +2)": c["score"],
                           "Weight": c["weight"] / total_w * 100, "Detail": c["reason"]}
                          for c in reg["components"]]),
            column_config={"Weight": st.column_config.NumberColumn(format="%.0f%%")})

    # --- business types ---
    st.markdown("### Business types")
    g = p["groups"]
    g = g[g["Business type"].isin(selected)] if not g.empty else g
    if g.empty:
        st.info("Select at least one business type above.")
    else:
        ctl("Heat map timeframe")
        tf = segmented("Heat map timeframe", ["1D", "5D", "1M", "3M"], "1D", "pulse_tf")
        show_chart(group_bars(g, tf))
        show_df(g[["Business type", "Signal", "Rating", "1D", "5D", "1M", "3M", "% above 50-day", "% up today",
                   "Volatility %", "Leader today", "Laggard today", "ETF"]].sort_values("Rating", ascending=False),
                column_config={
                    "Rating": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                    "1D": st.column_config.NumberColumn(format="%+.1f%%"),
                    "5D": st.column_config.NumberColumn(format="%+.1f%%"),
                    "1M": st.column_config.NumberColumn(format="%+.1f%%"),
                    "3M": st.column_config.NumberColumn(format="%+.1f%%"),
                    "% above 50-day": st.column_config.NumberColumn(format="%.0f%%"),
                    "% up today": st.column_config.NumberColumn(format="%.0f%%"),
                    "Volatility %": st.column_config.NumberColumn(format="%.0f%%")})

    # --- stock scanner ---
    st.markdown("### Stock scanner")
    s = stocks.copy()
    s["Business types"] = s["Ticker"].map(lambda t: ", ".join(GROUP_OF.get(t, [])))
    s = s[s["Ticker"].map(lambda t: any(x in selected for x in GROUP_OF.get(t, [])))]
    total = len(s)
    s["Signal"] = s["Score"].map(signal_label)
    s = s[(s["Price"] >= flt["min_price"]) & (s["avg_vol"] >= flt["min_vol"])]
    if flt["signal"] != "All":
        s = s[s["Signal"] == flt["signal"]]
    if flt["unusual"]:
        s = s[s["Rel. volume"] >= 1.5]
    if flt["hide_vol"]:
        s = s[s["Volatility %"] < flt["vol_limit"]]
    if flt["search"]:
        q = flt["search"].upper()
        s = s[s["Ticker"].str.contains(q, regex=False)]
    col, asc = PULSE_SORTS[flt["sort"]]
    s = s.sort_values(col, ascending=asc, na_position="last").head(flt["limit"])
    if s.empty:
        st.info("No stocks match these filters.")
        st.session_state["pulse_tickers"] = []
    else:
        s["Flags"] = s.apply(lambda r: flags(r, flt["vol_limit"]), axis=1)
        st.session_state["pulse_tickers"] = list(s["Ticker"])
        show_df(s[["Ticker", "Business types", "Signal", "Rating", "Price", "1d %", "5d %", "1m %", "Rel. volume",
                   "RSI", "Volatility %", "Flags"]],
                column_config={
                    "Rating": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "1d %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "5d %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "1m %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Rel. volume": st.column_config.NumberColumn(format="%.1fx"),
                    "RSI": st.column_config.NumberColumn(format="%.0f"),
                    "Volatility %": st.column_config.NumberColumn(format="%.0f%%")})
    st.caption(f"Updated {now_et()} ET · showing {len(s)} of {total} stocks in the selected business types · "
               f"scanned {p['loaded']} of {p['requested']} symbols.")


def render_pulse(weights: dict):
    st.markdown("Scans about 200 stocks and ETFs and rates whether the market is trending **bullish or bearish**, "
                "warns when **volatility** is rising, and breaks the market out by **business type**. The market-wide "
                "rating always uses the full scan; the filters below only change which business types and stocks "
                "are shown.")
    f1, f2 = st.columns([5, 1.5])
    selected = f1.multiselect("Business types", list(GROUPS), default=list(GROUPS), key="pulse_groups")
    refresh = f2.selectbox("Auto-refresh", list(PULSE_REFRESH), index=1, key="pulse_refresh")
    g1, g2, g3 = st.columns([3, 2, 2])
    with g1:
        ctl("Signal")
        sig = segmented("Signal", ["All", "Bullish", "Neutral", "Bearish"], "All", "pulse_sig")
    sort = g2.selectbox("Sort stocks by", list(PULSE_SORTS), key="pulse_sort")
    search = g3.text_input("Find ticker", key="pulse_search").strip()
    with st.expander("More filters"):
        m1, m2, m3, m4 = st.columns(4)
        min_price = m1.number_input("Min price ($)", 0.0, 1000.0, 5.0, 1.0, key="pulse_price")
        min_vol = m2.number_input("Min avg daily volume", 0, 50_000_000, 500_000, 100_000, key="pulse_vol")
        vol_limit = m3.slider("Flag volatility above (%/yr)", 20, 150, 60, 5, key="pulse_vlim")
        limit = m4.number_input("Rows to show", 10, 200, 50, 10, key="pulse_limit")
        h1, h2 = st.columns(2)
        unusual = h1.checkbox("Only unusual volume (1.5x or more)", value=False, key="pulse_unusual")
        hide_vol = h2.checkbox("Hide high-volatility stocks", value=False, key="pulse_hide")
    flt = {"signal": sig, "sort": sort, "search": search, "min_price": min_price, "min_vol": min_vol,
           "vol_limit": vol_limit, "limit": int(limit), "unusual": unusual, "hide_vol": hide_vol}

    wkey = tuple(sorted(weights.items()))
    refreshing(pulse_results, every=PULSE_REFRESH[refresh])(wkey, selected, flt)

    tickers = st.session_state.get("pulse_tickers", [])
    if tickers:
        p1, p2, p3 = st.columns([2, 1, 1])
        pick = p1.selectbox("Open in analyzer", tickers, key="pulse_pick")
        for col in (p2, p3):
            col.write("")
            col.write("")
        p2.button("Open in Analyze", on_click=open_in_analyzer, args=(pick,), key="pulse_load")
        p3.button("☆ Add to favorites", on_click=add_favorites, args=([pick],), key="pulse_fav")
    st.info("The rating is rule-based: it combines index trends, how many stocks are above their moving averages, "
            "the VIX and sector rotation, using Yahoo's data on a curated set of about 200 stocks and ETFs, not the "
            "whole market. It describes current conditions. It does not predict what happens next, and "
            "the business-type lists are examples you can edit in market.py.")


# ======================= FAVORITES =======================
FAV_REFRESH = {"Off": 0, "30 sec": 30, "1 min": 60, "5 min": 300}
FAV_SORTS = {"Saved order": ("_pos", True), "Biggest 1D gain": ("1d %", False),
             "Biggest 1D loss": ("1d %", True), "Trend rating": ("Rating", False)}


@st.cache_data(ttl=30, show_spinner=False)
def load_favorites(symbols: tuple, weights_key: tuple):
    data = download_history(list(symbols), "2y")
    stats = symbol_stats(data, dict(weights_key), include_indexes=True)
    have = set(stats.index) if not stats.empty else set()
    return stats, [s for s in symbols if s not in have]


def favorites_results(wkey: tuple, favs: list, order: str):
    """The watchlist table. Can auto-refresh."""
    try:
        stats, missing = load_favorites(tuple(favs), wkey)
    except Exception as e:
        st.error(f"Couldn't load your favorites: {e}")
        return
    if stats.empty:
        st.warning("No data came back for your favorites. Check the symbols, or try 'Refresh now' in the sidebar.")
        return
    s = stats.reset_index(drop=True)
    s["Signal"] = s["Score"].map(signal_label)
    s["Flags"] = s.apply(lambda r: flags(r), axis=1)
    pos = {t: i for i, t in enumerate(favs)}
    s["_pos"] = s["Ticker"].map(pos)
    key, asc = FAV_SORTS[order]
    s = s.sort_values(key, ascending=asc, na_position="last")
    n_bull, n_bear = int((s["Signal"] == "Bullish").sum()), int((s["Signal"] == "Bearish").sum())
    st.caption(f"Updated {now_et()} ET · {n_bull} bullish · {len(s) - n_bull - n_bear} neutral/n-a · {n_bear} bearish · "
               f"average move today {s['1d %'].mean():+.2f}%. Signals here use price technicals only; open a symbol "
               "in Analyze for the full signal.")
    show_df(s[["Ticker", "Signal", "Rating", "Price", "1d %", "5d %", "1m %", "3m %", "Rel. volume", "RSI",
               "Volatility %", "Flags"]],
            column_config={
                "Rating": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "1d %": st.column_config.NumberColumn(format="%+.1f%%"),
                "5d %": st.column_config.NumberColumn(format="%+.1f%%"),
                "1m %": st.column_config.NumberColumn(format="%+.1f%%"),
                "3m %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Rel. volume": st.column_config.NumberColumn(format="%.1fx"),
                "RSI": st.column_config.NumberColumn(format="%.0f"),
                "Volatility %": st.column_config.NumberColumn(format="%.0f%%")})
    if missing:
        st.warning("No data for: " + ", ".join(missing) + ". Check the spelling (indexes start with ^).")


def render_favorites(weights: dict):
    favs = st.session_state["favorites"]
    st.markdown("Save the symbols you follow and see them all in one live table. Add one from the sidebar while "
                "viewing it, or type symbols below.")
    a1, a2 = st.columns([5, 1])
    a1.text_input("Add symbols (comma-separated)", key="fav_add", placeholder="e.g. NVDA, XOM, ^GSPC")
    a2.write("")
    a2.write("")
    a2.button("Add", on_click=add_from_box, key="fav_add_btn")
    note = ("Your favorites are saved in this page's web address (the part after ?fav=). **Bookmark the page** after "
            "adding them and they'll be there next time, on any device. Opening the plain app link starts with an "
            "empty list; use 'Manage / back up' below to copy or restore your list.")
    if not favs:
        st.info("No favorites yet. Add symbols above, or use ☆ Add to favorites in the sidebar.")
        st.button("Add examples (AAPL, MSFT, NVDA, SPY)", on_click=add_favorites,
                  args=(["AAPL", "MSFT", "NVDA", "SPY"],), key="fav_examples")
        st.caption(note)
        return

    o1, o2 = st.columns([2, 1])
    order = o1.selectbox("Sort by", list(FAV_SORTS), key="fav_sort")
    refresh = o2.selectbox("Auto-refresh", list(FAV_REFRESH), index=2, key="fav_refresh")
    refreshing(favorites_results, every=FAV_REFRESH[refresh])(tuple(sorted(weights.items())), list(favs), order)

    c1, c2, c3 = st.columns([2, 1, 1])
    pick = c1.selectbox("Selected favorite", favs, key="fav_pick")
    for col in (c2, c3):
        col.write("")
        col.write("")
    c2.button("Open in Analyze", on_click=open_in_analyzer, args=(pick,), key="fav_open")
    c3.button("★ Remove", on_click=remove_favorite, args=(pick,), key="fav_remove")

    with st.expander("Manage / back up your list"):
        tkey = "fav_text_" + hashlib.md5(",".join(favs).encode()).hexdigest()[:8]
        st.text_area("Your list (copy it to back it up, or paste a list and press Apply)",
                     value=", ".join(favs), key=tkey, height=80)
        m1, m2 = st.columns(2)
        m1.button("Apply list", on_click=apply_list, args=(tkey,), key="fav_apply")
        m2.button("Clear all favorites", on_click=clear_favorites, key="fav_clear")
    st.caption(note)


# ---------- main ----------
if section == "Analyze":
    try:
        render_analyze(ticker, weights)
    except Exception as e:
        st.error(f"Could not load data: {e}")
elif section == "Favorites":
    try:
        render_favorites(weights)
    except Exception as e:
        st.error(f"Favorites error: {e}")
elif section == "Market pulse":
    try:
        render_pulse(weights)
    except Exception as e:
        st.error(f"Market pulse error: {e}")
elif section == "Backtest":
    try:
        render_backtest(ticker, weights)
    except Exception as e:
        st.error(f"Backtest error: {e}")
else:
    render_movers(weights)

st.caption("This tool scores public data with simple rules. It can't see the future, ignores your goals, taxes and "
           "risk tolerance, and is not financial advice. Do your own research.")
