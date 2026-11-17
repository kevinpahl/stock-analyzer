"""Stock Analyzer: run with `streamlit run app.py`"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

APP_VERSION = "7.1"
# Sizes of the other files in this release. Only used to explain a startup problem if an old copy was uploaded.
EXPECTED_SIZES = {"analyzer.py": 14007, "backtest.py": 5320, "charts.py": 18815, "screener.py": 8367, "utils.py": 4601, "market.py": 20549, "accounts.py": 10648, "forecast.py": 10898, "sessions.py": 4028, "crypto.py": 12905}
st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

try:
    from analyzer import DEFAULT_WEIGHTS, add_indicators, analyze, signal_points, technical_score_series
    from backtest import run_backtest, split_metrics
    from charts import (asia_lines, compare_chart, equity_chart, fan_chart, group_bars, intraday_chart, price_chart,
                        regime_chart, session_timeline, trades_chart, volume_waterfall)
    from forecast import (HORIZONS, asia_bucket, asia_composite, asia_lead_table, cone, edge_text, live_lean,
                          model_forecast, range_for, vol_estimates, walk_forward)
    from sessions import ASIA, US, futures_weekend, session_status, timeline_intervals
    from accounts import TOKEN_DAYS, AccountError, build_accounts
    from crypto import (CATEGORIES, CATEGORY_OF, COINS, CRYPTO_REGIME_WEIGHTS, NAME_OF, TOP_COINS, compute_crypto_pulse,
                        crypto_symbols)
    from market import GROUPS, GROUP_OF, REGIME_WEIGHTS, all_symbols, compute_pulse, flags, session_fraction, signal_label, symbol_stats
    from screener import ASSET_TYPES, MOVER_MODES, UNIVERSES, download_history, get_universe, rank_movers
    from utils import (BAR_LABELS, LIVE_RANGES, ETF_FOR, INDEX_NAMES, INTERVALS, OVERVIEW, PRESETS, QUICK_PICKS, benchmark_for,
                       is_crypto, is_index, live_days, live_params, preset_start, resample_ohlcv, strip_tz)
except ImportError as e:
    here = Path(__file__).parent
    found = {p.name: p.stat().st_size for p in here.glob("*.py")}
    rows = []
    for fname, size in EXPECTED_SIZES.items():
        got = found.get(fname)
        rows.append({"File": fname, "Found on the server": "not found" if got is None else f"{got:,} bytes",
                     "This version expects": f"{size:,} bytes",
                     "Status": "✗ missing" if got is None else "✓ matches" if got == size else "✗ different (old version?)"})
    copies = sorted(n for n in found if n not in EXPECTED_SIZES and n != "app.py" and ("(" in n or " " in n))
    st.error(f"**The app can't start: some files are missing or out of date.**\n\nTechnical detail: `{e}`")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if copies:
        st.warning("These look like duplicate downloads and are NOT used by the app: "
                   + ", ".join(f"`{c}`" for c in copies) + ". Delete them from your repo, then upload the file with "
                   "its exact original name (for example `charts.py`, not `charts (1).py`).")
    st.info("**How to fix:** in your GitHub repo, upload the files marked ✗ with exactly the names above, at the top "
            "level (not inside a folder), commit, then reboot the app. A tiny size difference can be harmless, but a "
            "big one means an old copy is still there. Also upload `requirements.txt`.")
    st.stop()

# Warn (without stopping) if any file in the repo is an older copy than this app.py expects.
_stale = []
try:
    _here = Path(__file__).parent
    for _f, _size in EXPECTED_SIZES.items():
        _p = _here / _f
        if not _p.exists() or _p.stat().st_size != _size:
            _stale.append(_f)
except Exception:
    _stale = []
if _stale:
    st.warning("**Some files in your repo look out of date: " + ", ".join(f"`{f}`" for f in _stale) +
               ".** Upload the newest copy of each (same name, top level of the repo), commit, then reboot the app. "
               "Until then you may see errors like \"unexpected keyword argument\".")

# Soft Dusk theme: muted blue-grey surfaces, soft sage / coral for up / down. (The base colors also live in
# .streamlit/config.toml so Streamlit's own widgets match.)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { color-scheme: dark; --bg:#1B2129; --surface:#232B35; --surface2:#1F2730; --text:#E4E8EC; --muted:#98A2AE;
        --border:#323C48; --primary:#7FB2CE; --primary-soft:#2C3C49; --up:#6FBF9C; --down:#E58A82;
        --shadow:0 1px 2px rgba(0,0,0,.25), 0 8px 22px rgba(0,0,0,.22); }
.stApp { background: var(--bg); color: var(--text); font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.015em; font-weight: 650; }
h1 { color: var(--text); }
a { color: var(--primary); }
hr { border-color: var(--border); }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h3 { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
                           padding: 14px 18px; box-shadow: var(--shadow); }
[data-testid="stMetricLabel"] p { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 650; }
.muted { color: var(--muted); font-size: .85rem; }
.ctl { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; margin: 0 0 4px 2px; }
.sym-title { display: flex; align-items: baseline; gap: 10px; margin: 4px 0 14px; flex-wrap: wrap; }
.sym-title .name { font-size: 1.7rem; font-weight: 650; letter-spacing: -.02em; }
.sym-title .tick { font-size: 1rem; color: var(--muted); font-weight: 600; }
.badge { font-size: .68rem; font-weight: 650; letter-spacing: .06em; padding: 3px 10px; border-radius: 999px;
         background: var(--primary-soft); color: var(--primary); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 14px 18px;
        min-height: 108px; box-shadow: var(--shadow); }
.card .lbl { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); font-weight: 600; }
.card .big { font-size: 1.25rem; font-weight: 650; margin: 8px 0 6px; }
.pill { display: inline-block; color: #10202A; font-weight: 700; font-size: 1.05rem; padding: 4px 14px;
        border-radius: 999px; margin: 6px 0 12px; letter-spacing: .02em; }
.gauge { position: relative; height: 8px; border-radius: 999px; margin: 2px 0 4px;
         background: linear-gradient(90deg, #E58A82 0%, #E9B866 50%, #6FBF9C 100%); }
.gauge .dot { position: absolute; top: 50%; width: 16px; height: 16px; border-radius: 50%;
              background: #E4E8EC; border: 3px solid #1B2129; transform: translate(-50%, -50%); }
.gauge-labels { display: flex; justify-content: space-between; font-size: .68rem; color: var(--muted); }
.tape { overflow: hidden; white-space: nowrap; background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 9px 0; box-shadow: var(--shadow); }
.tape-track { display: inline-block; animation: tape-scroll 60s linear infinite; }
.tape:hover .tape-track { animation-play-state: paused; }
.tk { margin: 0 22px; font-size: .92rem; }
.tk .up { color: var(--up); font-weight: 600; }
.tk .dn { color: var(--down); font-weight: 600; }
@keyframes tape-scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }

/* controls */
[data-testid="stSegmentedControl"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
                                     padding: 3px; width: fit-content; max-width: 100%; }
[data-testid="stSegmentedControl"] button, button[data-testid^="stBaseButton-segmented_control"] {
  border: 0 !important; border-radius: 9px !important; background: transparent !important;
  color: var(--muted) !important; font-weight: 550; }
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
  background: var(--primary-soft) !important; color: var(--primary) !important; font-weight: 650; }
.st-key-section [data-testid="stSegmentedControl"] { box-shadow: var(--shadow); border-radius: 14px; padding: 4px; }
.st-key-section [data-testid="stBaseButton-segmented_controlActive"],
.st-key-section [data-testid="stSegmentedControl"] button[aria-checked="true"],
.st-key-section [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
  background: var(--primary) !important; color: #14202A !important; box-shadow: 0 2px 8px rgba(127,178,206,.3); }
.stButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stDownloadButton"] button {
  background: var(--surface2); color: var(--primary); border: 1px solid var(--border); border-radius: 10px; font-weight: 600; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { border-color: var(--primary); color: var(--text); }
[data-testid="stBaseButton-primary"] { background: var(--primary) !important; color: #14202A !important; border: 0 !important; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  background: var(--surface2) !important; border-color: var(--border) !important; border-radius: 10px !important; }
[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }
[data-testid="stAlert"] { border-radius: 12px; border: 0; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
[data-testid="stPlotlyChart"] { background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
                                padding: 6px 6px 2px; box-shadow: var(--shadow); }
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* ---- tablets and small screens ---- */
@media (max-width: 1100px) {
  .block-container { padding: 1.2rem 1rem 3rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .75rem !important; }
  [data-testid="stColumn"], [data-testid="column"] { flex: 1 1 320px !important; min-width: min(320px, 100%) !important; }
  [data-testid="stColumn"]:has([data-testid="stMetric"]), [data-testid="column"]:has([data-testid="stMetric"]) {
    flex: 1 1 150px !important; min-width: 150px !important; }
}
@media (max-width: 640px) {
  h1 { font-size: 1.6rem !important; }
  .sym-title .name { font-size: 1.35rem; }
  [data-testid="stColumn"], [data-testid="column"] { flex: 1 1 100% !important; min-width: 100% !important; }
  [data-testid="stColumn"]:has([data-testid="stMetric"]), [data-testid="column"]:has([data-testid="stMetric"]) {
    flex: 1 1 calc(50% - .75rem) !important; min-width: calc(50% - .75rem) !important; }
  [data-testid="stMetricValue"] { font-size: 1.2rem; }
  [data-testid="stMetric"] { padding: 10px 12px; }
  .tk { margin: 0 14px; font-size: .85rem; }
}
@media (pointer: coarse) {  /* touch screens: bigger tap targets */
  .stButton > button, [data-testid="stFormSubmitButton"] > button { min-height: 2.75rem; }
  [data-baseweb="select"] > div { min-height: 2.75rem; }
  [data-baseweb="tab"] { min-height: 2.75rem; }
  [data-testid="stSegmentedControl"] button { min-height: 2.5rem; }
}
</style>
""", unsafe_allow_html=True)

# If .streamlit/config.toml (with the dark base theme) isn't in the repo, Streamlit's own widgets stay light and the
# text on them turns invisible. This extra CSS repaints them so the page still looks right; the config file is the
# cleaner fix and makes this block unnecessary.
try:
    _base_dark = st.get_option("theme.base") == "dark"
except Exception:
    _base_dark = False
if not _base_dark:
    st.markdown("""
<style>
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp { background: #1B2129 !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #232B35 !important; }
h1, h2, h3, h4, [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMetricValue"], [data-testid="stMetricValue"] div,
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stCheckbox"] p,
[data-testid="stToggle"] p, [data-testid="stRadio"] p { color: #E4E8EC !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, [data-testid="stMetricLabel"] p,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .stApp small { color: #98A2AE !important; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-baseweb="select"] div,
[data-baseweb="select"] span, [data-baseweb="select"] input { color: #E4E8EC !important;
  -webkit-text-fill-color: #E4E8EC !important; }
[data-baseweb="input"] input::placeholder, [data-baseweb="textarea"] textarea::placeholder { color: #6F7A86 !important; }
[data-baseweb="select"] svg, [data-baseweb="input"] svg { fill: #98A2AE !important; }
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="menu"] {
  background: #232B35 !important; color: #E4E8EC !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] li { color: #E4E8EC !important; background: transparent !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover { background: #2C3C49 !important; }
[data-baseweb="tag"] { background: #2C3C49 !important; color: #7FB2CE !important; }
[data-baseweb="tag"] span { color: #7FB2CE !important; }
[data-testid="stAlert"] { background: #232B35 !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: #E4E8EC !important; }
[data-testid="stDataFrame"] { filter: invert(.9) hue-rotate(180deg); }
[data-baseweb="slider"] [role="slider"] { background: #7FB2CE !important; }
[data-baseweb="checkbox"] [data-checked="true"], [data-baseweb="checkbox"] input:checked + div { background: #7FB2CE !important; }
.stButton > button p, [data-testid="stFormSubmitButton"] > button p { color: #7FB2CE !important; }
[data-testid="stBaseButton-primary"] p { color: #14202A !important; }
</style>
""", unsafe_allow_html=True)

VERDICT_COLORS = {"STRONG BUY": "#4E9F82", "BUY": "#6FBF9C", "HOLD": "#D4AE5F",
                  "SELL": "#E58A82", "STRONG SELL": "#C25F58"}
LIVE_TTL = 20  # seconds: how long short-lived market data is reused before asking Yahoo again
REFRESH_OPTIONS = {"Off": 0, "10 sec": 10, "30 sec": 30, "60 sec": 60}
SECTIONS = ["Analyze", "Trading", "Forecast", "Crypto", "Favorites", "Market pulse", "Backtest", "Top movers"]


# ---------- small UI helpers (written to work across Streamlit versions) ----------
def show_df(df: pd.DataFrame, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def show_chart(fig):
    kw = {}
    if st.session_state.get("compact"):  # shorter charts; one-finger swipes scroll the page instead of panning the chart
        try:
            fig.update_layout(height=max(240, int((fig.layout.height or 400) * 0.72)), dragmode=False,
                              margin=dict(l=4, r=4, t=26, b=4))
        except Exception:
            pass
        kw["config"] = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, **kw)


def segmented(label: str, options: list[str], default: str, key: str) -> str:
    fn = getattr(st, "segmented_control", None)
    if fn:
        return fn(label, options, default=default, key=key, label_visibility="collapsed") or default
    return st.radio(label, options, index=options.index(default), key=key, horizontal=True,
                    label_visibility="collapsed")


def fmt_price(x: float) -> str:
    """Two decimals for prices of $1 or more, four for sub-dollar coins."""
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4f}"


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
    t = yf.Ticker(sym)
    if period.endswith("d") and period not in ("1d", "5d"):   # e.g. "60d": Yahoo wants a start date for these
        h = t.history(start=pd.Timestamp.now() - pd.Timedelta(days=int(period[:-1]) - 1), interval=interval,
                      prepost=prepost, auto_adjust=True)
    else:
        h = t.history(period=period, interval=interval, prepost=prepost, auto_adjust=True)
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
               load_pulse, load_favorites, load_board, load_intraday_et, load_asia_hist, forecast_index,
               load_crypto_board, load_crypto_pulse, forecast_coin, forecast_symbol):
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
st.session_state.setdefault("user", None)


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
    user = st.session_state.get("user")
    if user:  # signed in: the account is the source of truth; keep the list out of the URL
        try:
            get_accounts().set_favorites(user, favs)
        except Exception as e:
            st.session_state["acct_msg"] = ("error", f"Couldn't save to your account: {e}")
        return
    try:  # not signed in: mirror into the URL (?fav=AAPL,NVDA) so a bookmark brings the list back
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


# ---------- accounts (favorites follow you when you log back in) ----------
def get_secret(name: str):
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    return v or os.environ.get(name)


@st.cache_resource(show_spinner=False)
def get_accounts():
    return build_accounts(get_secret("UPSTASH_REDIS_REST_URL"), get_secret("UPSTASH_REDIS_REST_TOKEN"),
                          Path(__file__).with_name("accounts_local.json"))


def clear_url_favorites():
    try:
        if "fav" in st.query_params:
            del st.query_params["fav"]
    except Exception:
        pass


def _forget_login_fields():
    for k in ("acct_user", "acct_pw", "acct_pw2"):
        try:
            st.session_state.pop(k, None)
        except Exception:
            pass


COOKIE = "sa_login"


def saved_login_token() -> str:
    """The stay-signed-in token the browser sent: from the address (?s=...) or from the cookie."""
    try:
        t = st.query_params.get("s")
        if isinstance(t, list):
            t = t[0] if t else ""
        if t:
            return str(t)
    except Exception:
        pass
    try:
        return st.context.cookies.get(COOKIE) or ""
    except Exception:
        return ""


def drop_url_token():
    try:
        if "s" in st.query_params:
            del st.query_params["s"]
    except Exception:
        pass


# Runs once per visit when nobody is signed in: if this browser kept a login token in its own storage, reload the page
# once with the token in the address so the server can read it (the server can't see browser storage directly).
BRIDGE_JS = """var w=window.parent,ls=null,ss=null;try{ls=w.localStorage;ss=w.sessionStorage}catch(e){}
var t=ls&&ls.getItem('sa_login'),u=new URL(APP.location.href);
if(!u.searchParams.get('s'))toast(t?'Restoring your saved login...':'No saved login found in this browser.');
if(t&&!u.searchParams.get('s')&&ss&&!ss.getItem('sa_tried')){ss.setItem('sa_tried','1');u.searchParams.set('s',t);APP.location.replace(u.toString());}"""


def queue_cookie(token: str | None):
    """Ask the browser to store (or, with None, erase) the stay-signed-in cookie on the next render."""
    st.session_state["cookie_op"] = token if token else "clear"


# A small note at the bottom-left of the page so you can see whether the browser accepted the saved login.
TOAST_JS = """function toast(m){var o=d.getElementById('sa_toast');if(o)o.remove();var e=d.createElement('div');e.id='sa_toast';
e.style.cssText='position:fixed;left:12px;bottom:12px;z-index:99999;background:#1B2129;color:#fff;padding:8px 12px;border-radius:8px;font:13px sans-serif;max-width:80%';
e.textContent=m;d.body.appendChild(e);setTimeout(function(){e.remove()},12000);}"""


def run_js(js: str):
    """Run a snippet of JavaScript on the page. Newer Streamlit runs it directly via st.html; older versions use a
    tiny iframe (which Streamlit is retiring), and either way the snippet reaches the page through `window.parent`."""
    # APP = the window that holds the app itself (it may reload itself, but a sandboxed app can't navigate its parent)
    def body(app: str) -> str:
        return f"<script>try{{var APP={app},d=window.parent.document;{TOAST_JS}{js}}}catch(e){{}}</script>"
    try:
        st.html(body("window"), unsafe_allow_javascript=True)
    except Exception:
        pass
    try:      # also via the older iframe route when it still exists; the snippets are safe to run twice
        import streamlit.components.v1 as components
        components.html(body("window.parent"), height=0)
    except Exception:
        pass


def emit_cookie_script():
    op = st.session_state.pop("cookie_op", None)
    if not op:
        return
    if op == "clear":
        js = (f"d.cookie='{COOKIE}=; Max-Age=0; Path=/; SameSite=Lax; Secure';"
              "try{window.parent.localStorage.removeItem('sa_login')}catch(e){}")
    else:
        js = (f"d.cookie='{COOKIE}={op}; Max-Age={TOKEN_DAYS * 86400}; Path=/; SameSite=Lax; Secure';"
              f"try{{window.parent.localStorage.setItem('sa_login','{op}')}}catch(e){{}}"
              "var c=d.cookie.indexOf('sa_login=')>=0,l=false;try{l=!!window.parent.localStorage.getItem('sa_login')}catch(e){}"
              "toast('Login remembered on this browser. Cookie: '+(c?'saved':'blocked')+', browser storage: '+(l?'saved':'blocked'));")
    run_js(js)


def try_resume_login():
    """Returning visitor: sign them back in from the saved cookie."""
    ss = st.session_state
    if ss.get("user") or ss.get("login_checked"):
        return
    ss["login_checked"] = True
    token = saved_login_token()
    ss["token_seen"] = "address" if st.query_params.get("s") else "cookie" if token else "none"
    if not token:
        run_js(BRIDGE_JS)
        return
    try:
        found = get_accounts().resume(token)
    except Exception:
        return
    if found:
        user, saved = found
        ss["user"], ss["favorites"] = user, list(dict.fromkeys(saved + ss["favorites"]))[:MAX_FAVS]
        ss["resume_token"] = token
        ss["auto_login"] = True
        drop_url_token()
        run_js("try{window.parent.sessionStorage.removeItem('sa_tried')}catch(e){}"
               f"toast('Signed in automatically.');")
    else:
        drop_url_token()
        queue_cookie(None)   # expired or revoked: stop sending it


def do_login(username: str, password: str):
    ss = st.session_state
    user, saved = get_accounts().login(username, password)
    merged = list(dict.fromkeys(saved + ss["favorites"]))[:MAX_FAVS]  # keep anything added while signed out
    ss["user"], ss["favorites"] = user, merged
    if merged != saved:
        get_accounts().set_favorites(user, merged)
    clear_url_favorites()
    if ss.get("stay_signed_in", True):
        ss["resume_token"] = get_accounts().issue_token(user)
        queue_cookie(ss["resume_token"])
    ss["acct_msg"] = ("success", f"Welcome back, {user}. {len(merged)} favorite{'s' if len(merged) != 1 else ''} loaded.")


def do_signup(username: str, password: str):
    ss = st.session_state
    user, favs = get_accounts().create(username, password, ss["favorites"])
    ss["user"], ss["favorites"] = user, favs
    clear_url_favorites()
    if ss.get("stay_signed_in", True):
        ss["resume_token"] = get_accounts().issue_token(user)
        queue_cookie(ss["resume_token"])
    ss["acct_msg"] = ("success", f"Account created. You're signed in as {user}.")


def do_logout():
    ss = st.session_state
    try:
        get_accounts().revoke_token(ss.pop("resume_token", ""))
    except Exception:
        pass
    queue_cookie(None)
    ss["user"], ss["favorites"] = None, []
    clear_url_favorites()
    ss["acct_msg"] = ("info", "You're signed out.")


def render_account_box():
    ss = st.session_state
    user = ss.get("user")
    st.markdown("### Account")
    msg = ss.pop("acct_msg", None)
    if msg:
        {"success": st.success, "error": st.error, "info": st.info}[msg[0]](msg[1])
    if user:
        st.caption(f"Signed in as **{user}**. Your favorites are saved to your account."
                   + (" ✓ Signed in automatically from your saved login." if ss.get("auto_login") else ""))
        st.button("Log out", on_click=do_logout, key="acct_logout")
        with st.expander("Account settings"):
            with st.form("pw_form", clear_on_submit=True):
                old = st.text_input("Current password", type="password")
                new = st.text_input("New password (8+ characters)", type="password")
                if st.form_submit_button("Change password"):
                    try:
                        get_accounts().change_password(user, old, new)
                        st.success("Password changed.")
                    except AccountError as e:
                        st.error(str(e))
            with st.form("del_form", clear_on_submit=True):
                st.caption("Deleting your account permanently removes it and your saved favorites.")
                pw = st.text_input("Password", type="password", key="del_pw")
                sure = st.checkbox("Yes, delete my account", key="del_sure")
                if st.form_submit_button("Delete account"):
                    if not sure:
                        st.error("Tick the box to confirm.")
                    else:
                        try:
                            get_accounts().delete(user, pw)
                            do_logout()
                            ss["acct_msg"] = ("info", "Your account was deleted.")
                            st.rerun()
                        except AccountError as e:
                            st.error(str(e))
    else:
        st.caption("Log in to keep your favorites across visits and devices.")
        mode = segmented("Account action", ["Log in", "Create account"], "Log in", "acct_mode")
        with st.form("acct_form"):
            u = st.text_input("Username", key="acct_user")
            p = st.text_input("Password", type="password", key="acct_pw")
            p2 = st.text_input("Confirm password", type="password", key="acct_pw2") if mode == "Create account" else None
            st.checkbox("Keep me signed in on this device", value=True, key="stay_signed_in")
            submitted = st.form_submit_button(mode)
        if submitted:
            try:
                if mode == "Create account":
                    if p != p2:
                        raise AccountError("The two passwords don't match.")
                    do_signup(u, p)
                else:
                    do_login(u, p)
                _forget_login_fields()
                st.rerun()
            except AccountError as e:
                st.error(str(e))
    with st.expander("Not staying signed in?"):
        try:
            kind = get_accounts().store.kind
        except Exception:
            kind = "unavailable"
        seen = ss.get("token_seen", "n/a")
        st.caption(f"Account storage: **{kind}**. Saved login found on this visit: "
                   f"**{'yes, via ' + seen if seen in ('address', 'cookie') else 'no' if seen == 'none' else 'not checked'}**"
                   f"{' (and it was accepted)' if ss.get('auto_login') else ''}. "
                   "If storage says 'local file' the saved login is erased whenever the app restarts, so set up "
                   "Upstash (see README). Private browsing and blocked site data also prevent it.")
    try:
        if get_accounts().store.kind == "local file":
            st.caption("⚠ Accounts are stored in a local file. On Streamlit Cloud they reset whenever the app "
                       "restarts. Add a free Upstash database in Secrets to keep them for good (see README).")
    except Exception:
        pass


def default_compact() -> bool:
    """Phones and Android tablets say so in their User-Agent (iPads pretend to be Macs, so use the toggle there)."""
    try:
        ua = (st.context.headers.get("User-Agent") or "").lower()
    except Exception:
        return False
    return any(k in ua for k in ("iphone", "android", "mobile", "ipad", "tablet"))


st.session_state.setdefault("compact", default_compact())
COMPACT = bool(st.session_state["compact"])

if COMPACT:  # the "Compact" toggle: stack everything for tablets and phones
    st.markdown("""<style>
.block-container { padding: .8rem .6rem 3rem !important; }
[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .6rem !important; }
[data-testid="stColumn"], [data-testid="column"] { flex: 1 1 100% !important; min-width: 100% !important; }
[data-testid="stColumn"]:has([data-testid="stMetric"]), [data-testid="column"]:has([data-testid="stMetric"]) {
  flex: 1 1 calc(50% - .6rem) !important; min-width: calc(50% - .6rem) !important; }
h1 { font-size: 1.5rem !important; } h3 { font-size: 1.15rem !important; }
</style>""", unsafe_allow_html=True)


def symbol_controls():
    """Symbol box, favorite star, quick picks and favorites menu (sidebar, or on the page in compact mode)."""
    st.text_input("Ticker or index", key="ticker",
                  help="Stocks (AAPL), ETFs (SPY), indexes (^GSPC = S&P 500, ^NDX = Nasdaq-100) or coins (BTC-USD).")
    cur = st.session_state["ticker"].upper().strip()
    is_fav = cur in st.session_state["favorites"]
    st.button("★ Remove from favorites" if is_fav else "☆ Add to favorites", on_click=toggle_current_favorite,
              key="fav_toggle", disabled=not cur)
    st.selectbox("Quick picks", list(QUICK_PICKS), key="quick", on_change=apply_quick)
    if st.session_state["favorites"]:
        st.selectbox("★ Favorites", ["—"] + st.session_state["favorites"], key="fav_menu", on_change=open_favorite)


try_resume_login()
emit_cookie_script()


ICON_180 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAYAAAA9zQYyAAAWfmNhQlgAABZ+anVtYgAAAB5qdW1kYzJwYQARABCAAACqADibcQNjMnBhAAAAFlhqdW1iAAAAR2p1bWRjMm1hABEAEIAAAKoAOJtxA3VybjpjMnBhOjNmZjJlMmU1LTM0ZDEtNDQwMC1hOTRmLTMxNDkxMjExMTRkNQAAAAOTanVtYgAAAClqdW1kYzJhcwARABCAAACqADibcQNjMnBhLmFzc2VydGlvbnMAAAAAuGp1bWIAAABEanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5pbmdyZWRpZW50LnYzAAAAABhjMnNoNYGK58lJ+CxGPB4IQy7NJgAAAGxjYm9yo2lkYzpmb3JtYXRpaW1hZ2UvcG5namluc3RhbmNlSUR4LHhtcDppaWQ6YjcwOGM5ZjQtNDE4Ni00YWM5LTgzMTYtZWIzZDk1MDllNjY1bHJlbGF0aW9uc2hpcGhwYXJlbnRPZgAAAeJqdW1iAAAAQWp1bWRjYm9yABEAEIAAAKoAOJtxE2MycGEuYWN0aW9ucy52MgAAAAAYYzJzaFBYvnTYLoBdBV/vT3m4T0YAAAGZY2JvcqJnYWN0aW9uc4KiZmFjdGlvbmtjMnBhLm9wZW5lZGpwYXJhbWV0ZXJzoWtpbmdyZWRpZW50c4GiY3VybHgtc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5pbmdyZWRpZW50LnYzZGhhc2hYIOxuztBgRqiJ/Lzw9TZ7K0GZ26OgNOZ9m3lrEvwgAY/VpGZhY3Rpb254HWNvbS5hbnRocm9waWMuY2xhdWRlLnByb3ZpZGVkanBhcmFtZXRlcnOheB9jb20uYW50aHJvcGljLm9yaWdpbi1jb25maWRlbmNlZ3Vua25vd25rZGVzY3JpcHRpb254ZkNsYXVkZSBwcm92aWRlZCB0aGlzIGZpbGUgYXQgdGhlIHJlcXVlc3Qgb2YgYSB1c2VyIGFuZCBtYXkgaGF2ZSBjcmVhdGVkIG9yIG1vZGlmaWVkIHRoZSBmaWxlIGNvbnRlbnRzLm1zb2Z0d2FyZUFnZW50oWRuYW1lZkNsYXVkZXJhbGxBY3Rpb25zSW5jbHVkZWT1AAAAyGp1bWIAAABAanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5oYXNoLmRhdGEAAAAAGGMyc2h24guzExU3EeCCRkCbWePIAAAAgGNib3KlY2FsZ2ZzaGEyNTZjcGFkTQAAAAAAAAAAAAAAAABkaGFzaFggWFJVZ4BiylsY4RuhfWDH7Yn0H7CKhnFZzaXqbkQ5UQRkbmFtZW5qdW1iZiBtYW5pZmVzdGpleGNsdXNpb25zgaJlc3RhcnQYIWZsZW5ndGgZFooAAAI+anVtYgAAACdqdW1kYzJjbAARABCAAACqADibcQNjMnBhLmNsYWltLnYyAAAAAg9jYm9ypWNhbGdmc2hhMjU2aXNpZ25hdHVyZXhNc2VsZiNqdW1iZj0vYzJwYS91cm46YzJwYTozZmYyZTJlNS0zNGQxLTQ0MDAtYTk0Zi0zMTQ5MTIxMTE0ZDUvYzJwYS5zaWduYXR1cmVqaW5zdGFuY2VJRHgseG1wOmlpZDo1NWRmMWNlYy1jMDkwLTQwODktOTJlYS0xNjgxMjQyNTAyM2NyY3JlYXRlZF9hc3NlcnRpb25zg6JjdXJseC1zZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jMnBhLmluZ3JlZGllbnQudjNkaGFzaFgg7G7O0GBGqIn8vPD1NnsrQZnbo6A05n2beWsS/CABj9WiY3VybHgqc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5hY3Rpb25zLnYyZGhhc2hYINpdhhAzMwoJ4XJuSkICY/EBZL3t8Ic4vwTa9DaHIzqdomN1cmx4KXNlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2MycGEuaGFzaC5kYXRhZGhhc2hYIKVfboTxWjKmvbPMzEWh3JrbU9EuzWIa43bR89lp/N15dGNsYWltX2dlbmVyYXRvcl9pbmZvo2RuYW1lb0FudGhyb3BpYyBGaWxlc2d2ZXJzaW9uZTEuMC4wa3NwZWNWZXJzaW9uZTIuNC4wAAAQOGp1bWIAAAAoanVtZGMyY3MAEQAQgAAAqgA4m3EDYzJwYS5zaWduYXR1cmUAAAAQCGNib3LShFkCEqIBJhghWQIKMIICBjCCAY2gAwIBAgIUQOWgCu7COdC+uIP6BkIFPWdVEwAwCgYIKoZIzj0EAwMwSTEXMBUGA1UEChMOQW50aHJvcGljLCBQQkMxLjAsBgNVBAMTJUFudGhyb3BpYyBDb250ZW50IENyZWRlbnRpYWxzIFJvb3QgQ0EwHhcNMjYwODA3MTg0MzU2WhcNMjgwODA2MTk0MzU2WjBEMRcwFQYDVQQKEw5BbnRocm9waWMsIFBCQzEpMCcGA1UEAxMgQW50aHJvcGljIENsYXVkZSBDb250ZW50IFNpZ25pbmcwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAASYegpry1AYBRTVNL1CpTlbROnY3dey+UrsF9C3phYrATN3ZHf93Mo8RQN0KOUuOn19P4oWNFWe5n2/She9N7eTo1gwVjAOBgNVHQ8BAf8EBAMCB4AwFQYDVR0lBA4wDAYKKwYBBAGD6F4CATAMBgNVHRMBAf8EAjAAMB8GA1UdIwQYMBaAFM5R4gSBTmRbI/jjxM+aPpzB11zCMAoGCCqGSM49BAMDA2cAMGQCMDFzHRSeAXrSy1WOzkbhPZ6Km2wGTmZ/2gK18k8BQGXyqz88Rdrz6CTX9flAnYNVxgIwcF9c3fVhqmJKpi+UhasNUMko69cyX6STPfta3Q8EjyzDjzoyrol46FP6VFHhvUcJoWNwYWRZDZ4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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"
ICON_192 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMAAAADACAYAAABS3GwHAAAWfmNhQlgAABZ+anVtYgAAAB5qdW1kYzJwYQARABCAAACqADibcQNjMnBhAAAAFlhqdW1iAAAAR2p1bWRjMm1hABEAEIAAAKoAOJtxA3VybjpjMnBhOjA5ZDI0Y2ZjLTE2MTEtNDQwZC1iNDJlLTRhZTUxZTI1ZTg3YgAAAAOTanVtYgAAAClqdW1kYzJhcwARABCAAACqADibcQNjMnBhLmFzc2VydGlvbnMAAAAAuGp1bWIAAABEanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5pbmdyZWRpZW50LnYzAAAAABhjMnNoC335XdroCSNlRxQ7nM0BhAAAAGxjYm9yo2lkYzpmb3JtYXRpaW1hZ2UvcG5namluc3RhbmNlSUR4LHhtcDppaWQ6ZTIwMjJiMTctYjlmMC00MTU3LWI5MzUtZjA5YzNhNWQzZTg0bHJlbGF0aW9uc2hpcGhwYXJlbnRPZgAAAeJqdW1iAAAAQWp1bWRjYm9yABEAEIAAAKoAOJtxE2MycGEuYWN0aW9ucy52MgAAAAAYYzJzaFm7+XLvv+7kq7UnPj0QdA4AAAGZY2JvcqJnYWN0aW9uc4KiZmFjdGlvbmtjMnBhLm9wZW5lZGpwYXJhbWV0ZXJzoWtpbmdyZWRpZW50c4GiY3VybHgtc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5pbmdyZWRpZW50LnYzZGhhc2hYIOsLsdXGHX9fT6NUkE09W5CWRTgaGQEHK+BFB5uRvFSVpGZhY3Rpb254HWNvbS5hbnRocm9waWMuY2xhdWRlLnByb3ZpZGVkanBhcmFtZXRlcnOheB9jb20uYW50aHJvcGljLm9yaWdpbi1jb25maWRlbmNlZ3Vua25vd25rZGVzY3JpcHRpb254ZkNsYXVkZSBwcm92aWRlZCB0aGlzIGZpbGUgYXQgdGhlIHJlcXVlc3Qgb2YgYSB1c2VyIGFuZCBtYXkgaGF2ZSBjcmVhdGVkIG9yIG1vZGlmaWVkIHRoZSBmaWxlIGNvbnRlbnRzLm1zb2Z0d2FyZUFnZW50oWRuYW1lZkNsYXVkZXJhbGxBY3Rpb25zSW5jbHVkZWT1AAAAyGp1bWIAAABAanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5oYXNoLmRhdGEAAAAAGGMyc2ikJjV8wtwPBUhmlUJVuPdoAAAAgGNib3KlY2FsZ2ZzaGEyNTZjcGFkTQAAAAAAAAAAAAAAAABkaGFzaFggImPoPIdfZZ+YMd5O5/CAdkThW716guadC86XQCZf1YdkbmFtZW5qdW1iZiBtYW5pZmVzdGpleGNsdXNpb25zgaJlc3RhcnQYIWZsZW5ndGgZFooAAAI+anVtYgAAACdqdW1kYzJjbAARABCAAACqADibcQNjMnBhLmNsYWltLnYyAAAAAg9jYm9ypWNhbGdmc2hhMjU2aXNpZ25hdHVyZXhNc2VsZiNqdW1iZj0vYzJwYS91cm46YzJwYTowOWQyNGNmYy0xNjExLTQ0MGQtYjQyZS00YWU1MWUyNWU4N2IvYzJwYS5zaWduYXR1cmVqaW5zdGFuY2VJRHgseG1wOmlpZDo0MDQ4MjM2Ni04MWVhLTQyZjQtOTQ5Yy0yMGIwMTQxN2RlZTFyY3JlYXRlZF9hc3NlcnRpb25zg6JjdXJseC1zZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jMnBhLmluZ3JlZGllbnQudjNkaGFzaFgg6wux1cYdf19Po1SQTT1bkJZFOBoZAQcr4EUHm5G8VJWiY3VybHgqc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5hY3Rpb25zLnYyZGhhc2hYILESgeAXXUIzf02RFLBk3qzfkr2+v/dwy3Choj80xSqFomN1cmx4KXNlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2MycGEuaGFzaC5kYXRhZGhhc2hYIIyqMJvp5kCR8mNg3AT2CIy37+l9Jisz5VUWkA1Zrv+HdGNsYWltX2dlbmVyYXRvcl9pbmZvo2RuYW1lb0FudGhyb3BpYyBGaWxlc2d2ZXJzaW9uZTEuMC4wa3NwZWNWZXJzaW9uZTIuNC4wAAAQOGp1bWIAAAAoanVtZGMyY3MAEQAQgAAAqgA4m3EDYzJwYS5zaWduYXR1cmUAAAAQCGNib3LShFkCEqIBJhghWQIKMIICBjCCAY2gAwIBAgIUQOWgCu7COdC+uIP6BkIFPWdVEwAwCgYIKoZIzj0EAwMwSTEXMBUGA1UEChMOQW50aHJvcGljLCBQQkMxLjAsBgNVBAMTJUFudGhyb3BpYyBDb250ZW50IENyZWRlbnRpYWxzIFJvb3QgQ0EwHhcNMjYwODA3MTg0MzU2WhcNMjgwODA2MTk0MzU2WjBEMRcwFQYDVQQKEw5BbnRocm9waWMsIFBCQzEpMCcGA1UEAxMgQW50aHJvcGljIENsYXVkZSBDb250ZW50IFNpZ25pbmcwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAASYegpry1AYBRTVNL1CpTlbROnY3dey+UrsF9C3phYrATN3ZHf93Mo8RQN0KOUuOn19P4oWNFWe5n2/She9N7eTo1gwVjAOBgNVHQ8BAf8EBAMCB4AwFQYDVR0lBA4wDAYKKwYBBAGD6F4CATAMBgNVHRMBAf8EAjAAMB8GA1UdIwQYMBaAFM5R4gSBTmRbI/jjxM+aPpzB11zCMAoGCCqGSM49BAMDA2cAMGQCMDFzHRSeAXrSy1WOzkbhPZ6Km2wGTmZ/2gK18k8BQGXyqz88Rdrz6CTX9flAnYNVxgIwcF9c3fVhqmJKpi+UhasNUMko69cyX6STPfta3Q8EjyzDjzoyrol46FP6VFHhvUcJoWNwYWRZDZ4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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"


def install_as_app():
    """Make 'Add to Home Screen' produce an app-style icon (full screen, own icon, no browser bars).
    The icon and manifest are built right here, so no extra files are needed in the repo."""
    if st.session_state.get("pwa_done"):
        return
    st.session_state["pwa_done"] = True
    js = """var h=d.head;
function add(t,a){var e=d.createElement(t);for(var k in a)e.setAttribute(k,a[k]);h.appendChild(e);}
var m={name:'Stock Analyzer',short_name:'Stocks',start_url:window.parent.location.origin+'/',display:'standalone',
background_color:'#1B2129',theme_color:'#1B2129',icons:[{src:'ICON192',sizes:'192x192',type:'image/png'}]};
add('link',{rel:'manifest',href:URL.createObjectURL(new Blob([JSON.stringify(m)],{type:'application/manifest+json'}))});
add('link',{rel:'apple-touch-icon',href:'ICON180'});
add('meta',{name:'apple-mobile-web-app-capable',content:'yes'});
add('meta',{name:'mobile-web-app-capable',content:'yes'});
add('meta',{name:'apple-mobile-web-app-title',content:'Stocks'});
add('meta',{name:'apple-mobile-web-app-status-bar-style',content:'default'});
add('meta',{name:'theme-color',content:'#1B2129'});
""".replace("ICON192", ICON_192).replace("ICON180", ICON_180)
    run_js(js)


install_as_app()

with st.sidebar:
    render_account_box()
    if not COMPACT:
        st.markdown("### Symbol")
        symbol_controls()
    st.markdown("### Live data")
    refresh_label = st.selectbox("Auto-refresh", list(REFRESH_OPTIONS), index=2, key="refresh")
    st.button("Refresh now", on_click=refresh_now, key="refresh_now")
    with st.expander("Scoring weights"):
        st.caption("Relative importance of each factor. They're normalized automatically.")
        weights = {k: st.slider(k, 0.0, 0.5, v, 0.05, key=f"w_{k}") for k, v in DEFAULT_WEIGHTS.items()}
    st.caption(f"Data: Yahoo Finance via yfinance (free feed: usually near real-time, but some exchanges lag up "
               f"to ~15 min). Educational only, not financial advice. · App v{APP_VERSION}")

REFRESH_SECS = REFRESH_OPTIONS.get(refresh_label, 0)
ticker = st.session_state["ticker"].upper().strip()


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
tg1, tg2 = st.columns([3, 5])
tg1.toggle("Compact layout (tablet / phone)", key="compact",
           help="Stacks columns, shortens charts and moves the symbol box onto the page. Turned on automatically on most phones and tablets.")
if COMPACT:
    with st.expander("Symbol, favorites and quick picks", expanded=False):
        symbol_controls()
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
    crypto = is_crypto(ticker)
    age = q.get("age_min")
    live = crypto or (age is not None and age < 20)
    status = ("<span style='color:#6FBF9C'>● LIVE</span>" if live
              else "<span style='opacity:.75'>● Market closed</span>")
    if crypto:
        status += "<span class='muted'> · trades 24/7</span>"

    k1, k2, k3, k4 = st.columns([1.2, 1, 1.2, 1.5])
    k1.metric("Last price", f"{pre}{fmt_price(price)}", f"{price - prev:+,.2f} ({price / prev - 1:+.2%})")
    k2.metric("Day range", f"{fmt_price(lo)} – {fmt_price(hi)}")
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


def intraday_signals(ticker: str, eff: str, df: pd.DataFrame, prepost: bool, weights: dict, arrow_cfg: tuple):
    """Buy/sell arrows worked out on the intraday bars themselves (warmed up on a longer stretch of bars so the
    200-bar trend measure has data from the first bar shown)."""
    warm = load_intraday(ticker, eff, f"{WARM_DAYS[eff]}d", prepost)
    if len(warm) < 220:
        warm = df
    sc = technical_score_series(warm, None, weights)
    return signal_points(sc.loc[sc.index >= df.index[0]], *arrow_cfg, cooldown=12)


def daily_signals(ticker: str, weights: dict, arrow_cfg: tuple):
    return signal_points(technical_score_series(get_daily(ticker), get_bench(ticker), weights), *arrow_cfg)


def live_chart(ticker: str, interval: str, rng: str, prepost: bool, ctype: str,
               overlays: list, panels: list, signals=None, arrow_cfg=None, weights=None, mini: bool = False,
               tall: bool = False):
    """Intraday chart. Auto-refreshes. With `arrow_cfg` the arrows are computed on the intraday bars themselves.
    `mini` = a compact card for a grid of charts (title on the chart, no long caption)."""
    period, eff, note = live_params(interval, rng)
    df = load_intraday(ticker, eff, period, prepost)
    if df.empty or len(df) < 2:
        if mini:
            st.caption(f"{ticker}: no intraday bars available (market closed, or no intraday data for this symbol).")
        else:
            st.info("No intraday bars available yet (the market may be closed, or this symbol has no intraday data).")
        return
    cdf = add_indicators(df)
    crypto = is_crypto(ticker)
    us = not ("." in ticker or ticker.endswith(("-USD", "=X", "=F")))
    daily_bars = eff == "1d"
    if daily_bars:   # one bar per day: hide weekends for stocks, nothing for 24/7 coins
        fig = price_chart(cdf, cdf.index[0], cdf.index[-1], ctype, overlays, panels, "Daily",
                          trades_24_7=crypto, signals=signals)
    else:
        arrows, n_arrows = None, 0
        if arrow_cfg is not None:
            try:
                arrows = intraday_signals(ticker, eff, df, prepost, weights, arrow_cfg)
                n_arrows = len(arrows)
            except Exception as e:
                st.warning(f"Couldn't work out intraday arrows: {e}")
        fig = price_chart(cdf, cdf.index[0], cdf.index[-1], ctype, overlays, panels, eff,
                          intraday=True, us_hours=us, prepost=prepost, trades_24_7=crypto,
                          signals=arrows if arrows is not None else signals, signals_exact=arrows is not None)
        if arrows is not None:
            note += (f" {n_arrows} buy/sell arrow{'s' if n_arrows != 1 else ''} in view, worked out on these {eff} bars "
                     "(trend, momentum, MACD, Bollinger measured in bars) using only bars up to each point."
                     " Move the Arrow sensitivity setting for more or fewer.")
    if mini:
        grid_style(fig, ticker, len(panels), tall)
        show_chart(fig)
        return
    show_chart(fig)
    last_bar = f"{cdf.index[-1]:%b %d}" if daily_bars else f"{cdf.index[-1]:%b %d, %H:%M} (exchange time)"
    st.caption(f"{len(cdf)} {eff} bars · last bar {last_bar}. Moving averages are measured in bars. " + note)


def history_view(ticker: str, daily: pd.DataFrame, idx: bool, ctype: str, rng: str, interval: str,
                 overlays: list, panels: list, signals=None):
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
    show_chart(price_chart(chart_df, start, end, ctype, overlays, panels, interval, signals=signals))
    st.caption(f"{len(view)} {interval.lower()} bars. Moving averages are measured in {interval.lower()} bars. "
               "The Buy/Hold/Sell signal always uses daily data, whatever interval you chart.")


def grid_style(fig, title: str, n_panels: int, tall: bool = False):
    """Chart with the symbol as its title, for the favorites page. `tall` = full Analyze-size (one chart per row)."""
    fig.update_layout(height=(460 + 150 * n_panels) if tall else (330 + 90 * n_panels), title=dict(text=f"<b>{title}</b>", x=0.01, y=0.98, font=dict(size=15)),
                      margin=dict(l=6, r=6, t=44, b=6))


WARM_DAYS = {"1m": 6, "5m": 30, "15m": 60, "30m": 60, "1h": 200}   # extra bars to warm up the indicators
ARROW_SENS = {"Fewer": (0.8, -0.8, 3), "Normal": (0.5, -0.5, 3), "More": (0.3, -0.3, 2)}


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
    c0, c1, c2, c3 = st.columns([2, 4, 5, 2])
    with c0:
        ctl("View")
        view = segmented("View", ["Live", "History"], "Live", "view")
    if view == "Live":
        with c1:
            ctl("Range")
            lrng = segmented("Range", list(LIVE_RANGES), "1D", "live_rng")
        with c2:
            ctl("Bar size")
            lint = BAR_LABELS.get(segmented("Bar size", list(BAR_LABELS), "5 min", "live_bar"), "5m")
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

    o1, o2, o3, o4 = st.columns([4, 4, 2, 2])
    overlays = o1.multiselect("Overlays", ["SMA 9", "SMA 20", "SMA 50", "SMA 200", "Bollinger bands"],
                              default=["SMA 9", "SMA 20", "SMA 50"], key="overlays")
    has_vol = float(daily["Volume"].iloc[-60:].sum()) > 0
    panels = o2.multiselect("Panels", ["Volume", "RSI", "MACD"] if has_vol else ["RSI", "MACD"],
                            default=["Volume", "RSI"] if has_vol else ["RSI"], key=f"panels_{has_vol}")
    prepost = (o3.checkbox("Extended hours", value=False, key="ext")
               if view == "Live" and not is_crypto(ticker) else False)

    show_arrows = o4.checkbox("Buy/sell arrows", value=True, key="arrows",
                              help="Green up arrow: the technical score moved into Buy territory. Red down arrow: Sell.")
    signals = None
    if show_arrows:
        sens = o4.selectbox("Arrow sensitivity", list(ARROW_SENS), index=1, key="arrow_sens",
                            help="More = earlier but noisier arrows. Fewer = only the strongest turns.")
        try:
            hi, lo, conf = ARROW_SENS[sens]
            signals = signal_points(technical_score_series(daily, get_bench(ticker), weights), hi, lo, conf)
        except Exception as e:
            signals = None
            st.warning(f"Couldn't work out the buy/sell arrows for this symbol: {e}")
    if view == "Live":
        cfg = ARROW_SENS[sens] if show_arrows and signals is not None and lint != "1d" else None
        refreshing(live_chart)(ticker, lint, lrng, prepost, ctype, overlays, panels, signals, cfg, weights)
    else:
        history_view(ticker, daily, idx, ctype, rng, interval, overlays, panels, signals)
    intraday_arrows = view == "Live" and lint != "1d"
    if show_arrows and signals is not None and not intraday_arrows:
        if len(signals):
            last = signals.iloc[-1]
            try:
                win_start = (pd.Timestamp.now() - pd.Timedelta(days=live_days(lrng, lint)) if view == "Live"
                             else None if rng == "Custom" else preset_start(rng, daily.index[-1], daily.index[0]))
            except Exception:
                win_start = None
            if win_start is not None and not (signals.index >= win_start.normalize()).any():
                st.info(f"No arrows fall inside this chart window. The latest signal was a **{last['side'].upper()}** on "
                        f"{signals.index[-1]:%b %d, %Y}. Choose a longer range (for example 3M or 1Y, or the History "
                        "view) to see the arrows.")
            st.caption(f"Arrows mark where the technical score (trend, momentum, MACD, Bollinger, relative strength; "
                       f"the same one the backtester uses) moved into Buy or Sell territory and held, "
                       f"using only data available on that day. Latest: {last['side'].upper()} on "
                       f"{signals.index[-1]:%b %d, %Y}. Past arrows don't predict the future; many trends hit "
                       f"false signals. Not financial advice.")
        else:
            st.caption("No buy/sell arrows yet: the technical score hasn't held in Buy or Sell territory for this symbol.")

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
                                    load_info if p["fund"] else None, weights, p.get("types"))
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
    st.markdown("Finds what's moving across **stocks, ETFs, indexes, futures, currencies and crypto**, drops illiquid "
                "names, then **ranks the movers by the analyzer's score** so you can see which big moves have "
                "technical backing and which look stretched.")
    u1, u2, u3 = st.columns([2, 2, 1])
    uni = u1.selectbox("Universe", list(UNIVERSES), key="mv_uni")
    mode = u2.selectbox("Rank movers by", MOVER_MODES, key="mv_mode")
    top_n = u3.number_input("How many", 5, 50, 20, 5, key="mv_n")
    custom = (st.text_input("Tickers (comma-separated)", "AAPL, NVDA, TSLA, AMD, PLTR", key="mv_custom")
              if UNIVERSES[uni] == "custom" else "")
    types = st.multiselect("Include", ASSET_TYPES, default=ASSET_TYPES, key="mv_types",
                           help="Stocks, ETFs, indexes, futures, currencies and coins. Pick a built-in ETF, index or "
                                "crypto list above to scan those. Yahoo's gainers/losers/most-active lists are stocks only.")
    f1, f2, f3, f4 = st.columns(4)
    min_price = f1.number_input("Min price ($)", 0.0, 1000.0, 5.0, 1.0, key="mv_price",
                                help="Applies to stocks and ETFs only.")
    min_vol = f2.number_input("Min avg daily volume", 0, 50_000_000, 500_000, 100_000, key="mv_vol",
                              help="Applies to stocks and ETFs only. Indexes, futures, currencies and coins skip it.")
    with_fund = f3.checkbox("Include fundamentals (slower)", value=False, key="mv_fund",
                            help="Fetches P/E, growth and analyst data for each ranked stock.")
    auto = f4.checkbox("Auto-refresh every minute", value=False, key="mv_auto")

    if st.button("Scan market", type="primary", key="mv_run"):
        st.session_state["mv_params"] = {"uni": uni, "custom": custom, "mode": mode, "top_n": int(top_n),
                                         "min_price": min_price, "min_vol": min_vol, "fund": with_fund,
                                         "types": types or ASSET_TYPES}
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
REGIME_COLORS = {"STRONGLY BULLISH": "#4E9F82", "BULLISH": "#6FBF9C", "NEUTRAL / MIXED": "#D4AE5F",
                 "BEARISH": "#E58A82", "STRONGLY BEARISH": "#C25F58"}
VOL_COLORS = {"LOW": "#6FBF9C", "NORMAL": "#7FB2CE", "ELEVATED": "#E0A458", "HIGH": "#E58A82", "EXTREME": "#C25F58"}
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

    # --- volume flow ---
    st.markdown("### Where the volume is trading")
    vb = p["volume"]
    v1, v2 = st.columns(2)
    with v1:
        ctl("Measure")
        vmeas = segmented("Measure", ["Dollars traded", "Shares traded"], "Dollars traded", "pulse_vmeasure")
    with v2:
        ctl("View")
        vmode = segmented("View", ["Total today", "Change vs 20-day average"], "Total today", "pulse_vmode")
    measure = "dollars" if vmeas.startswith("Dollars") else "shares"
    mode = "total" if vmode.startswith("Total") else "change"
    ind = vb["industries"]
    ind = ind[ind["name"].isin(selected)] if not ind.empty else ind
    w1, w2 = st.columns([3, 2])
    with w1:
        st.markdown("#### By business type")
        if ind.empty:
            st.info("Select at least one business type above.")
        else:
            show_chart(volume_waterfall(ind, measure, mode, vb["frac"]))
    with w2:
        st.markdown("#### By index (tracking ETFs)")
        if vb["indexes"].empty:
            st.info("Index ETF data isn't available right now.")
        else:
            show_chart(volume_waterfall(vb["indexes"], measure, mode, vb["frac"]))
    pace = (f" About {vb['frac']:.0%} of a normal day's volume has traded so far, so 'change vs average' compares "
            "today with the average pace for this time of day (an estimate)." if vb["frac"] < 1 else
            " The market is closed, so this is the last full session.")
    st.caption("Dollars traded = price × volume, for the stocks this page scans (not the whole market). Each stock is "
               "counted once, in the first business type it's listed under. Index volume uses the ETFs that track "
               "each index." + pace)

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


# ======================= TRADING =======================
US_BOARD = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^NDX": "Nasdaq-100", "^DJI": "Dow", "^RUT": "Russell 2000",
            "^VIX": "VIX"}
FUT_BOARD = {"ES=F": "S&P 500 futures", "NQ=F": "Nasdaq-100 futures", "YM=F": "Dow futures",
             "RTY=F": "Russell 2000 futures"}
MACRO_BOARD = {"^TNX": "10-yr yield", "DX-Y.NYB": "US dollar index", "CL=F": "Crude oil", "GC=F": "Gold",
               "BTC-USD": "Bitcoin"}
TAPE_STOCKS = "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM NFLX AMD V UNH XOM".split()
ASIA_SYMS = [m["sym"] for m in ASIA]
TAPE_LABELS = {**US_BOARD, "ES=F": "S&P fut", "NQ=F": "Nasdaq fut", "YM=F": "Dow fut", "RTY=F": "Russell fut",
               **{m["sym"]: m["name"] for m in ASIA}}
LEAN_COLORS = {"LEANING HIGHER": "#6FBF9C", "LEANING LOWER": "#E58A82", "MIXED": "#D4AE5F"}
STATUS_ICON = {"Open": "🟢 Open", "Lunch break": "🟡 Lunch break", "Pre-open": "🟡 Pre-open", "Closed": "⚪ Closed"}


def board_symbols(extra=()) -> list[str]:
    return list(dict.fromkeys(list(US_BOARD) + list(FUT_BOARD) + list(MACRO_BOARD) + ["^VXN"] + ASIA_SYMS
                              + TAPE_STOCKS + list(extra)))


@st.cache_data(ttl=20, show_spinner=False)
def load_board(extra: tuple = ()):
    return download_history(board_symbols(extra), "5d")


@st.cache_data(ttl=600, show_spinner=False)
def load_asia_hist():
    return download_history(ASIA_SYMS, "10y")


@st.cache_data(ttl=60, show_spinner=False)
def load_intraday_et(sym: str, interval: str, period: str) -> pd.DataFrame:
    """Intraday bars shifted to US Eastern time so markets from different time zones share one clock."""
    h = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=True)
    if not h.empty and h.index.tz is not None:
        h.index = h.index.tz_convert("America/New_York").tz_localize(None)
    return h


@st.cache_data(ttl=300, show_spinner="Training the forecast models…")
def forecast_index(sym: str, vix_sym: str, vol_sym: str, bar_key: str):
    daily = get_daily(sym)
    if len(daily) < 2500:
        return {"ok": False, "reason": "Not enough price history to train a model."}
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    partial = session_fraction(now, daily.index[-1]) < 1   # today's bar is still forming: forecast from the last full close
    close = (daily["Close"].iloc[:-1] if partial else daily["Close"]).iloc[-5200:]
    vdf = get_daily(vix_sym)
    vix = vdf["Close"] if not vdf.empty else None
    idf = get_daily(vol_sym)
    implied = float(idf["Close"].iloc[-1]) if not idf.empty else None
    ve = vol_estimates(close, implied)
    last = float(close.iloc[-1])
    out = {"ok": True, "sym": sym, "last_close": last, "last_date": close.index[-1], "partial": partial,
           "vol": ve, "implied": implied, "history": close.iloc[-90:],
           "cone": cone(last, ve["sigma_daily"], close.index[-1], 10), "horizons": {}}
    for h in HORIZONS:
        out["horizons"][h] = {"model": model_forecast(close, vix, h), "test": walk_forward(close, vix, h),
                              "range": range_for(last, ve["sigma_daily"], h)}
    return out


def quote(board: dict, sym: str):
    d = board.get(sym)
    if d is None:
        return None
    c = d["Close"].dropna()
    if len(c) < 2:
        return None
    last, prev = float(c.iloc[-1]), float(c.iloc[-2])
    return {"last": last, "prev": prev, "chg": last - prev, "pct": (last / prev - 1) * 100}


def tape_html(items: list) -> str:
    cells = "".join(
        f"<span class='tk'><b>{lab}</b> {last:,.2f} <span class='{'up' if pct >= 0 else 'dn'}'>"
        f"{'▲' if pct >= 0 else '▼'} {abs(pct):.2f}%</span></span>" for lab, last, pct in items)
    dur = max(60, len(items) * 3)
    delay = -(time.time() % dur)   # keeps the scroll position continuous across auto-refreshes
    return (f"<div class='tape'><div class='tape-track' style='animation-duration:{dur}s;"
            f"animation-delay:{delay:.1f}s'>{cells}{cells}</div></div>")


def trading_tape(favs: tuple):
    board = load_board(favs)
    order = (list(favs) + ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "ES=F", "NQ=F"] + ASIA_SYMS + TAPE_STOCKS)
    items = []
    for sym in dict.fromkeys(order):
        q = quote(board, sym)
        if q:
            items.append((TAPE_LABELS.get(sym, sym), q["last"], q["pct"]))
    if not items:
        st.info("Live prices aren't available right now.")
        return
    st.markdown(tape_html(items), unsafe_allow_html=True)
    st.caption(f"Updated {now_et()} ET · your favorites come first · hover to pause. Prices come from Yahoo Finance "
               "and can lag by up to ~15 minutes on some exchanges.")


def metric_cards(board: dict, mapping: dict, inverse: tuple = ()):
    cols = st.columns(len(mapping))
    for col, (sym, label) in zip(cols, mapping.items()):
        q = quote(board, sym)
        if q:
            col.metric(label, f"{q['last']:,.2f}", f"{q['pct']:+.2f}%", delta_color="inverse" if sym in inverse else "normal")
        else:
            col.metric(label, "n/a")


def trading_live(wkey: tuple, favs: tuple):
    board = load_board(favs)
    st.markdown("#### US indexes")
    metric_cards(board, US_BOARD, inverse=("^VIX",))
    st.markdown("#### Futures (trade almost 24 hours)")
    metric_cards(board, FUT_BOARD)
    st.markdown("#### Rates, dollar, commodities, crypto")
    metric_cards(board, MACRO_BOARD)

    c1, c2 = st.columns(2)
    for col, sym, nm in ((c1, "^GSPC", "S&P 500"), (c2, "^IXIC", "Nasdaq")):
        with col:
            st.markdown(f"#### {nm} today")
            try:
                df = load_intraday(sym, "5m", "1d", False)
                q = quote(board, sym)
                if len(df) >= 2:
                    show_chart(intraday_chart(df, q["prev"] if q else None, nm))
                else:
                    st.info("No intraday bars yet. The market may be closed.")
            except Exception:
                st.info("Intraday data isn't available right now.")

    st.markdown("### Most active stocks right now")
    try:
        s = load_pulse(wkey)["stocks"]
    except Exception as e:
        st.info(f"The stock scan isn't available right now ({e}).")
        return
    if s.empty:
        st.info("No scan data yet.")
        return
    s = s.copy()
    s["Signal"] = s["Score"].map(signal_label)
    s["Traded today ($M)"] = s["dollar_vol"] / 1e6
    cfg = {"Price": st.column_config.NumberColumn(format="$%.2f"),
           "1d %": st.column_config.NumberColumn(format="%+.2f%%"),
           "Rel. volume": st.column_config.NumberColumn(format="%.1fx"),
           "Traded today ($M)": st.column_config.NumberColumn(format="$%,.0f")}
    cols = ["Ticker", "Price", "1d %", "Rel. volume", "Traded today ($M)", "Signal"]
    show_df(s.sort_values("dollar_vol", ascending=False).head(15)[cols], column_config=cfg)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Biggest gainers")
        show_df(s.sort_values("1d %", ascending=False).head(8)[cols[:4] + ["Signal"]], column_config=cfg)
    with g2:
        st.markdown("#### Biggest losers")
        show_df(s.sort_values("1d %").head(8)[cols[:4] + ["Signal"]], column_config=cfg)
    st.caption(f"Updated {now_et()} ET · from the scan of about 180 stocks on the Market pulse page.")


def trading_asia(favs: tuple):
    board = load_board(favs)
    now = pd.Timestamp.now(tz="UTC")
    rows, chgs, status = [], [], {}
    for m in ASIA:
        stt, detail = session_status(m, now)
        q = quote(board, m["sym"])
        status[m["name"]] = stt
        if q:
            chgs.append(q["pct"])
        rows.append({"Market": m["name"], "Country": m["country"], "Session": STATUS_ICON[stt], "Detail": detail,
                     "Local time": now.tz_convert(m["tz"]).strftime("%a %H:%M"),
                     "Index": q["last"] if q else None, "Change %": q["pct"] if q else None})
    avg = sum(chgs) / len(chgs) if len(chgs) >= 3 else None
    if avg is not None:
        up = sum(1 for c in chgs if c > 0)
        tone = "higher" if avg > 0.25 else "lower" if avg < -0.25 else "mixed"
        open_now = [n for n, s in status.items() if s == "Open"]
        st.markdown(f"**Asia is {tone}:** average {avg:+.2f}%, with {up} of {len(chgs)} markets up. "
                    + (f"Still open: {', '.join(open_now)}." if open_now else "All Asian markets are closed or on break."))
    show_df(pd.DataFrame(rows), column_config={
        "Index": st.column_config.NumberColumn(format="%,.2f"),
        "Change %": st.column_config.NumberColumn(format="%+.2f%%")})

    st.markdown("#### When each market is open (US Eastern time)")
    now_et_ts = now.tz_convert("America/New_York").tz_localize(None)
    colors = {m["name"]: ("#6FBF9C" if status[m["name"]] == "Open" else "#98A2AE") for m in ASIA}
    colors[US["name"]] = "#7FB2CE"
    iv = timeline_intervals(ASIA + [US], now)
    if iv:
        show_chart(session_timeline(iv, now_et_ts, colors))
    else:
        st.info("All of these markets are closed for the weekend.")
    st.caption("Green bars are markets open right now; the dashed line is the current time. Based on regular "
               "scheduled hours. Exchange holidays aren't tracked, so a market can show open on a holiday.")

    st.markdown("#### Asian indexes today, % since each session's open (US Eastern clock)")
    lines = {}
    for m in ASIA:
        try:
            d = load_intraday_et(m["sym"], "5m", "1d")
            if len(d) >= 3:
                lines[m["name"]] = (d["Close"] / d["Open"].iloc[0] - 1) * 100
        except Exception:
            pass
    if lines:
        show_chart(asia_lines(pd.DataFrame(lines).sort_index()))
        st.caption("Shows each market's latest session. Yahoo's intraday data for Asia can be delayed or missing "
                   "for some indexes.")
    else:
        st.info("Intraday data for Asia isn't available right now. Try again in a few minutes.")

    st.markdown("### Does Asia predict the US close?")
    try:
        comp = asia_composite(load_asia_hist())
    except Exception:
        comp = pd.Series(dtype=float)
    cols = st.columns(2)
    cur = asia_bucket(avg) if avg is not None else None
    for col, (nm, sym) in zip(cols, (("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"))):
        with col:
            tab, corr = asia_lead_table(get_daily(sym)["Close"], comp) if len(comp) else (pd.DataFrame(), float("nan"))
            if tab.empty:
                st.info("Not enough history to compare.")
                continue
            tab["Today"] = tab["bucket"].map(lambda b: "◀ today" if b == cur else "")
            show_df(tab.rename(columns={"bucket": f"When Asia was...", "sessions": "Sessions",
                                        "us_up_pct": f"{nm} closed up %", "avg_us_move_pct": "Avg move %"}),
                    column_config={f"{nm} closed up %": st.column_config.NumberColumn(format="%.0f%%"),
                                   "Avg move %": st.column_config.NumberColumn(format="%+.2f%%")})
            st.caption(f"Last 10 years: correlation between Asia's average move and the {nm}'s same-day move is "
                       f"{corr:.2f}. Asia closes before the US opens, so this is known in advance. It's a tilt, not a guarantee.")


def forecast_card(name: str, f: dict, lean: dict | None, lean_note: str, crypto: bool = False):
    st.markdown(f"### {name}")
    if not f.get("ok"):
        st.warning(f.get("reason", "The forecast isn't available right now."))
        return
    unit = "day" if crypto else "session"
    st.caption(f"Forecast from the last full {'daily close (00:00 UTC)' if crypto else 'close'}: "
               f"{fmt_price(f['last_close'])} on {f['last_date']:%b %d}"
               + (" (today's candle is still forming)" if f["partial"] and crypto
                  else " (today's session is still in progress)" if f["partial"] else ""))
    if lean:
        reasons = " · ".join(f"{c['name']}: {c['detail']}" for c in lean["components"])
        st.markdown(f"<div class='card'><div class='lbl'>Live read right now</div>"
                    f"<div class='pill' style='background:{LEAN_COLORS[lean['label']]}'>{lean['label']}</div>"
                    f"<div class='muted'>{reasons}</div></div>", unsafe_allow_html=True)
    else:
        st.info(lean_note)
    labels = {1: "Today's close" if f["partial"] else "Next close", 5: f"In 5 {unit}s", 21: f"In 21 {unit}s"}
    if crypto:
        labels[1] = "Today's close (00:00 UTC)" if f["partial"] else "Next daily close"
    for col, h in zip(st.columns(3), HORIZONS):
        hz = f["horizons"][h]
        mf, rg = hz["model"], hz["range"]
        col.metric(labels[h], f"{mf['p_up'] * 100:.0f}% up" if mf.get("ok") else "n/a")
        col.caption(f"Typical move ±{rg['sigma_pct']:.1f}%. 68% range {fmt_price(rg['lo68']) if crypto else format(rg['lo68'], ',.0f')} to {fmt_price(rg['hi68']) if crypto else format(rg['hi68'], ',.0f')}. "
                    f"{edge_text(hz['test'].get('edge', '')).capitalize()}.")
    show_chart(fan_chart(f["history"], f["cone"], name, weekend_gaps=not crypto))
    with st.expander("How reliable is this? (tested on roughly the last 3 years)"):
        rows = []
        for h in HORIZONS:
            t = f["horizons"][h]["test"]
            if t.get("n"):
                rows.append({"Horizon": labels[h], "Test days": t["n"], "Model accuracy": f"{t['accuracy']:.1%}",
                             "Naive guess": f"{t['baseline_accuracy']:.1%}", "Probability skill": f"{t['skill']:+.1%}",
                             "Verdict": edge_text(t["edge"]).capitalize()})
        if rows:
            show_df(pd.DataFrame(rows))
        mf = f["horizons"][1]["model"]
        if mf.get("ok") and mf.get("drivers"):
            st.write("Biggest influences on the next-close number right now: " + "; ".join(
                f"{d['feature']} (pushes {d['pushes']})" for d in mf["drivers"]) + ".")
        vol = f["vol"]
        imp_txt = "n/a" if f["implied"] is None else f"{f['implied']:.1f}"
        vol_txt = ("Ranges use recent realized volatility only (coins have no implied-volatility index) "
                   if crypto else f"Ranges blend implied volatility ({imp_txt}) with recent realized volatility ")
        st.caption("Method: a logistic regression on trend, momentum, volatility and VIX features, retrained monthly "
                   "using only data available at the time, then scored on what happened next. 'Naive guess' always "
                   "picks the more common outcome. Probabilities are capped at 30% to 70%. " + vol_txt +
                   f"and assume no drift: typical daily move {vol['sigma_daily'] * 100:.2f}%.")


def trading_forecast(favs: tuple):
    board = load_board(favs)
    now_u = pd.Timestamp.now(tz="UTC")
    now_e = now_u.tz_convert("America/New_York")
    weekend = futures_weekend(now_e)
    chgs = [q["pct"] for q in (quote(board, m["sym"]) for m in ASIA) if q]
    asia_avg = sum(chgs) / len(chgs) if len(chgs) >= 3 else None
    vq = quote(board, "^VIX")
    us_open = session_status(US, now_u)[0] == "Open"
    specs = [("S&P 500", "^GSPC", "ES=F", "^VIX"), ("Nasdaq", "^IXIC", "NQ=F", "^VXN")]
    results = {}
    for name, sym, fut, vol_sym in specs:
        try:
            bar = get_daily(sym)
            f = forecast_index(sym, "^VIX", vol_sym, f"{bar.index[-1]:%Y-%m-%d}") if len(bar) else {"ok": False}
        except Exception as e:
            f = {"ok": False, "reason": f"The forecast failed: {e}"}
        fq, iq = quote(board, fut), quote(board, sym)
        lean = None if weekend else live_lean(fq["pct"] if fq else None, asia_avg, vq["pct"] if vq else None,
                                              iq["pct"] if iq and us_open else None)
        results[name] = (f, lean)

    lines = []
    for name, (f, lean) in results.items():
        if f.get("ok") and f["horizons"][1]["model"].get("ok"):
            p = f["horizons"][1]["model"]["p_up"] * 100
            lines.append(f"**{name}:** model {p:.0f}% up for the next close ("
                         f"{edge_text(f['horizons'][1]['test'].get('edge', ''))}); live signals "
                         f"{lean['label'].lower() if lean else 'unavailable'}.")
    if lines:
        st.info("  \n".join(lines))
    note = ("Markets are closed for the weekend, so there are no live signals. They resume when futures reopen "
            "Sunday evening (US Eastern time).") if weekend else "Live signals aren't available right now."
    cols = st.columns(2)
    for col, (name, (f, lean)) in zip(cols, results.items()):
        with col:
            forecast_card(name, f, lean, note)
    st.warning("**Please read:** these are statistical estimates, not predictions you can rely on. Short-term market "
               "moves are mostly unpredictable, and the tests above usually show little or no edge. Use the ranges "
               "to understand how much a market can move, not to time trades. Not financial advice.")


def render_trading(weights: dict):
    favs = tuple(st.session_state.get("favorites", [])[:10])
    wkey = tuple(sorted(weights.items()))
    st.markdown("Near-real-time prices for stocks and the key indexes, the Asian sessions, and a forecast for the "
                "S&P 500 and Nasdaq. Refreshes automatically (set the rate in the sidebar).")
    refreshing(trading_tape)(favs)
    st.write("")
    ctl("Trading view")
    view = segmented("Trading view", ["Live board", "Asia sessions", "Forecast"], "Live board", "trade_view")
    if view == "Live board":
        refreshing(trading_live)(wkey, favs)
    elif view == "Asia sessions":
        refreshing(trading_asia)(favs)
    else:
        refreshing(trading_forecast)(favs)


# ======================= CRYPTO =======================
COIN_OPTIONS = {f"{name} ({sym.replace('-USD', '')})": sym for sym, (name, _) in COINS.items()}


@st.cache_data(ttl=20, show_spinner=False)
def load_crypto_board(extra: tuple = ()):
    return download_history(list(dict.fromkeys(crypto_symbols() + list(extra))), "5d")


@st.cache_data(ttl=60, show_spinner="Scanning the crypto market (the first scan can take up to a minute)…")
def load_crypto_pulse(weights_key: tuple):
    return compute_crypto_pulse(download_history(crypto_symbols(), "2y"), dict(weights_key))


@st.cache_data(ttl=300, show_spinner="Training the forecast models…")
def forecast_coin(sym: str, bar_key: str):
    daily = get_daily(sym)
    if len(daily) < 1000:
        return {"ok": False, "reason": "Not enough price history to train a model."}
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    partial = daily.index[-1].normalize() == now.normalize()   # today's UTC bar is still forming
    close = (daily["Close"].iloc[:-1] if partial else daily["Close"]).iloc[-5200:]
    ve = vol_estimates(close, None)                            # no implied volatility index for coins
    last = float(close.iloc[-1])
    out = {"ok": True, "sym": sym, "last_close": last, "last_date": close.index[-1], "partial": partial,
           "vol": ve, "implied": None, "history": close.iloc[-90:],
           "cone": cone(last, ve["sigma_daily"], close.index[-1], 10, calendar_days=True), "horizons": {}}
    for h in HORIZONS:
        out["horizons"][h] = {"model": model_forecast(close, None, h), "test": walk_forward(close, None, h),
                              "range": range_for(last, ve["sigma_daily"], h)}
    return out


IMPLIED_FOR = {"^GSPC": "^VIX", "^IXIC": "^VXN", "^NDX": "^VXN"}
MIN_FORECAST_ROWS = 1000   # about four years of daily bars


@st.cache_data(ttl=300, show_spinner="Training the forecast model…")
def forecast_symbol(sym: str, bar_key: str):
    """Statistical forecast for any single stock, ETF, index or coin."""
    crypto = is_crypto(sym)
    daily = get_daily(sym)
    if len(daily) < MIN_FORECAST_ROWS:
        return {"ok": False, "reason": f"{sym} has only {len(daily)} days of price history. The model needs about "
                                       f"{MIN_FORECAST_ROWS} (roughly four years) to train, so there is no forecast for it."}
    if crypto:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        partial = daily.index[-1].normalize() == now.normalize()
    else:
        now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        partial = session_fraction(now, daily.index[-1]) < 1
    close = (daily["Close"].iloc[:-1] if partial else daily["Close"]).iloc[-5200:]
    vix = None
    if not crypto:
        vdf = get_daily("^VIX")
        vix = vdf["Close"] if not vdf.empty else None
    implied = None
    if sym in IMPLIED_FOR:
        idf = get_daily(IMPLIED_FOR[sym])
        implied = float(idf["Close"].iloc[-1]) if not idf.empty else None
    ve = vol_estimates(close, implied)
    last = float(close.iloc[-1])
    out = {"ok": True, "sym": sym, "last_close": last, "last_date": close.index[-1], "partial": partial,
           "vol": ve, "implied": implied, "history": close.iloc[-90:],
           "cone": cone(last, ve["sigma_daily"], close.index[-1], 10, calendar_days=crypto), "horizons": {}}
    for h in HORIZONS:
        out["horizons"][h] = {"model": model_forecast(close, vix, h), "test": walk_forward(close, vix, h),
                              "range": range_for(last, ve["sigma_daily"], h)}
    return out


FC_TYPE = "Type my own…"


def fc_options() -> dict:
    opts = {FC_TYPE: None}
    for f in st.session_state["favorites"]:
        opts[f"★ {f}"] = f
    for lab, sym in QUICK_PICKS.items():
        if sym and sym not in opts.values():
            opts[lab] = sym
    for lab, sym in COIN_OPTIONS.items():
        if sym not in opts.values():
            opts[lab] = sym
    return opts


def render_forecast(weights: dict):
    st.markdown("### Forecast one symbol")
    st.caption("Pick any stock, ETF, index or coin. The model estimates the chance of closing up and a typical price "
               "range, and shows how well it did when tested on past data.")
    opts = fc_options()
    cur = st.session_state.get("ticker", "AAPL")
    labels = list(opts)
    default = next((lab for lab, s in opts.items() if s == cur), FC_TYPE)
    if st.session_state.get("fc_pick") not in labels:
        st.session_state["fc_pick"] = default
    c1, c2 = st.columns([3, 2])
    pick = c1.selectbox("Symbol", labels, key="fc_pick")
    typed = c2.text_input("Type a symbol (AAPL, ^GSPC, BTC-USD or just ETH)", value=cur, key="fc_typed",
                          disabled=pick != FC_TYPE).upper().strip()
    sym = opts[pick] if opts[pick] else typed
    if sym and not sym.endswith("-USD") and not sym.startswith("^") and sym in {c.replace("-USD", "") for c in COINS} \
            and pick == FC_TYPE:
        sym += "-USD"
    if not sym:
        st.info("Enter a symbol to forecast.")
        return
    name = NAME_OF.get(sym) or INDEX_NAMES.get(sym) or sym
    b1, b2 = st.columns(2)
    b1.button("Open in Analyze", on_click=open_in_analyzer, args=(sym,), key="fc_open")
    b2.button("★ Remove from favorites" if sym in st.session_state["favorites"] else "☆ Add to favorites",
              on_click=toggle_favorite_symbol, args=(sym,), key="fc_fav")
    try:
        bar = get_daily(sym)
        f = forecast_symbol(sym, f"{bar.index[-1]:%Y-%m-%d}") if len(bar) else {"ok": False, "reason": f"No data found for {sym}."}
    except Exception as e:
        f = {"ok": False, "reason": f"The forecast failed: {e}"}
    crypto = is_crypto(sym)
    q = load_quote(sym) if f.get("ok") else {}
    note = ("Live signals (futures and overnight sessions) are only available for the S&P 500 and Nasdaq on the Trading page, "
            "so this shows the statistical forecast only.")
    if q.get("price") and q.get("prev"):
        note = f"Last price {fmt_price(q['price'])} ({q['price'] / q['prev'] - 1:+.2%} today). " + note
    forecast_card(name if name == sym else f"{name} ({sym})", f, None, note, crypto=crypto)
    st.warning("**Please read:** these are statistical estimates, not predictions you can rely on. Single stocks and "
               "coins are driven by news the model can't see, and the test results above usually show little or no "
               "edge. Use the ranges to see how far a price can move, not to time trades. Not financial advice.")


def toggle_favorite_symbol(sym: str):
    if sym in st.session_state["favorites"]:
        remove_favorite(sym)
    else:
        add_favorites([sym])


def open_coin(sym: str):
    ss = st.session_state
    ss["section"], ss["crypto_view"] = "Crypto", "Analyze a coin"
    label = next((lab for lab, s in COIN_OPTIONS.items() if s == sym), None)
    if label:
        ss["crypto_coin_pick"], ss["crypto_coin_custom"] = label, ""
    else:
        ss["crypto_coin_custom"] = sym


def crypto_tape(favs: tuple):
    board = load_crypto_board(favs)
    items = []
    for sym in dict.fromkeys(list(favs) + crypto_symbols()):
        q = quote(board, sym)
        if q:
            items.append((sym.replace("-USD", ""), q["last"], q["pct"]))
    if not items:
        st.info("Live crypto prices aren't available right now.")
        return
    st.markdown(tape_html(items), unsafe_allow_html=True)
    st.caption(f"Updated {now_et()} ET · your favorite coins come first · hover to pause. Prices are in US dollars; the "
               "change is since 00:00 UTC (Yahoo's daily bars). Crypto trades around the clock.")


def coin_tables(s: pd.DataFrame):
    s = s.copy()
    s["Coin"] = s["Ticker"].map(lambda t: NAME_OF.get(t, t))
    s["Signal"] = s["Score"].map(signal_label)
    s["Traded today ($M)"] = s["dollar_vol"] / 1e6
    cfg = {"Price": st.column_config.NumberColumn(format="$%.6g"),
           "1d %": st.column_config.NumberColumn(format="%+.2f%%"),
           "Rel. volume": st.column_config.NumberColumn(format="%.1fx"),
           "Traded today ($M)": st.column_config.NumberColumn(format="$%,.0f")}
    cols = ["Coin", "Price", "1d %", "Rel. volume", "Traded today ($M)", "Signal"]
    return s, cols, cfg


def crypto_live(wkey: tuple, favs: tuple):
    board = load_crypto_board(favs)
    st.markdown("#### Major coins")
    metric_cards(board, {s: NAME_OF[s] for s in TOP_COINS})
    c1, c2 = st.columns(2)
    for col, sym, nm in ((c1, "BTC-USD", "Bitcoin"), (c2, "ETH-USD", "Ethereum")):
        with col:
            st.markdown(f"#### {nm}, last 24 hours")
            try:
                df = load_intraday(sym, "5m", "1d", False)
                q = quote(board, sym)
                if len(df) >= 2:
                    show_chart(intraday_chart(df, q["prev"] if q else None, nm, breaks="none"))
                else:
                    st.info("No intraday bars available right now.")
            except Exception:
                st.info("Intraday data isn't available right now.")
    st.markdown("### Most active coins right now")
    try:
        coins = load_crypto_pulse(wkey)["coins"]
    except Exception as e:
        st.info(f"The crypto scan isn't available right now ({e}).")
        return
    if coins.empty:
        st.info("No scan data yet.")
        return
    s, cols, cfg = coin_tables(coins)
    active = s.sort_values("dollar_vol", ascending=False)
    st.session_state["crypto_tickers"] = list(active["Ticker"])
    show_df(active.head(12)[cols], column_config=cfg)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Biggest gainers today")
        show_df(s.sort_values("1d %", ascending=False).head(8)[cols], column_config=cfg)
    with g2:
        st.markdown("#### Biggest losers today")
        show_df(s.sort_values("1d %").head(8)[cols], column_config=cfg)
    st.caption(f"Updated {now_et()} ET · volume is as Yahoo reports it, in US dollars, for the UTC day so far.")


def render_crypto_analyze(weights: dict):
    c1, c2 = st.columns([3, 2])
    pick = c1.selectbox("Coin", list(COIN_OPTIONS), key="crypto_coin_pick")
    custom = c2.text_input("Or type another coin (for example ETH or ETH-USD)", key="crypto_coin_custom").strip().upper()
    sym = COIN_OPTIONS[pick]
    if custom:
        sym = custom if custom.endswith("-USD") else custom + "-USD"
    st.button("★ Remove from favorites" if sym in st.session_state["favorites"] else "☆ Add to favorites",
              on_click=toggle_favorite_symbol, args=(sym,), key="crypto_fav")
    render_analyze(sym, weights)


def crypto_pulse_results(wkey: tuple, selected: list, flt: dict):
    try:
        p = load_crypto_pulse(wkey)
    except Exception as e:
        st.error(f"The crypto scan failed: {e}")
        return
    reg, vol, coins = p["regime"], p["vol"], p["coins"]
    if not reg.get("ok") or coins.empty:
        st.error("Not enough crypto data came back from Yahoo to rate the market right now. "
                 "Try 'Refresh now' in the sidebar in a minute.")
        return
    nan = float("nan")
    arrow = {"Improving": "▲", "Deteriorating": "▼", "Stable": "►"}.get(reg["direction"], "")
    delta_txt = "" if reg["delta"] != reg["delta"] else f" ({reg['delta']:+.1f} vs a week ago)"
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<div class='card'><div class='lbl'>Crypto market rating</div>"
        f"<div class='pill' style='background:{REGIME_COLORS[reg['label']]}'>{reg['label']}</div>"
        f"<div class='gauge'><div class='dot' style='left:{reg['pct']}%'></div></div>"
        f"<div class='gauge-labels'><span>Bearish</span><span>Neutral</span><span>Bullish</span></div>"
        f"<div class='muted' style='margin-top:6px'>{reg['pct']}/100 · {arrow} {reg['direction']}{delta_txt}</div></div>",
        unsafe_allow_html=True)
    if vol.get("ok"):
        c2.markdown(
            f"<div class='card'><div class='lbl'>Volatility</div>"
            f"<div class='pill' style='background:{VOL_COLORS[vol['level']]}'>{vol['level']}</div>"
            f"<div class='muted'>Bitcoin 30-day volatility {vol['realized']:.0f}% a year<br>"
            f"Bitcoin today {vol['btc_today']:+.1f}% (UTC day)</div></div>", unsafe_allow_html=True)
    b = reg["breadth"]
    highs, lows = int(coins["new_high"].sum()), int(coins["new_low"].sum())
    c3.markdown(
        f"<div class='card'><div class='lbl'>Breadth</div>"
        f"<div class='big'>{b.get('p50', nan):.0f}% above 50-day</div>"
        f"<div class='muted'>{b.get('p200', nan):.0f}% above 200-day · {b.get('padv', nan):.0f}% up today<br>"
        f"{highs} new 20-day highs · {lows} new lows</div></div>", unsafe_allow_html=True)

    if vol.get("ok"):
        msg = "\n".join(f"- {w}" for w in vol["warnings"])
        if vol["idx"] >= 3:
            st.error(f"**Crypto volatility is {vol['level']}.** Expect very large and fast price swings.\n{msg}")
        elif vol["idx"] == 2 or vol["warnings"]:
            st.warning(f"**Crypto volatility is {vol['level']}.**\n{msg}")
        else:
            st.success("**Crypto volatility is calm for crypto.** No warnings right now.")

    a, bcol = st.columns(2)
    with a:
        st.markdown("#### Crypto market rating, last 4 months")
        show_chart(regime_chart(reg["history"]))
    with bcol:
        st.markdown("#### Bitcoin, last 24 hours")
        try:
            idf = load_intraday("BTC-USD", "5m", "1d", False)
            if len(idf) >= 2:
                show_chart(intraday_chart(idf, load_quote("BTC-USD").get("prev"), "Bitcoin", breaks="none"))
            else:
                st.info("No intraday bars available right now.")
        except Exception:
            st.info("Intraday data isn't available right now.")

    st.markdown("#### What's driving the rating")
    total_w = sum(CRYPTO_REGIME_WEIGHTS.values())

    def reading(c):
        if not c["available"]:
            return "n/a"
        return "Bullish" if c["score"] > 0.5 else "Bearish" if c["score"] < -0.5 else "Neutral"

    show_df(pd.DataFrame([{"Factor": c["Factor"], "Reading": reading(c), "Score (-2 to +2)": c["score"],
                           "Weight": c["weight"] / total_w * 100, "Detail": c["reason"]} for c in reg["components"]]),
            column_config={"Weight": st.column_config.NumberColumn(format="%.0f%%")})

    st.markdown("### Where the volume is trading")
    vb = p["volume"]
    ctl("View")
    vmode = segmented("View", ["Total today", "Change vs 20-day average"], "Total today", "crypto_vmode")
    mode = "total" if vmode.startswith("Total") else "change"
    ind = vb["industries"]
    ind = ind[ind["name"].isin(selected)] if not ind.empty else ind
    w1, w2 = st.columns([3, 2])
    with w1:
        st.markdown("#### By category")
        if ind.empty:
            st.info("Select at least one category above.")
        else:
            show_chart(volume_waterfall(ind, "dollars", mode, vb["frac"]))
    with w2:
        st.markdown("#### By coin")
        if vb["indexes"].empty:
            st.info("No coin data right now.")
        else:
            show_chart(volume_waterfall(vb["indexes"], "dollars", mode, vb["frac"]))
    st.caption(f"Dollars traded in the UTC day so far, as Yahoo reports them (crypto has no open or close). About "
               f"{vb['frac']:.0%} of the day has passed, so 'change vs average' compares with the average pace for this "
               "time of day. Each coin is counted once, in its category. Coverage is the coins this page tracks.")

    st.markdown("### Categories")
    g = p["groups"]
    g = g[g["Business type"].isin(selected)] if not g.empty else g
    if g.empty:
        st.info("Select at least one category above.")
    else:
        ctl("Heat map timeframe")
        tf = segmented("Heat map timeframe", ["1D", "5D", "1M", "3M"], "1D", "crypto_tf")
        show_chart(group_bars(g, tf))
        show_df(g[["Business type", "Signal", "Rating", "1D", "5D", "1M", "3M", "% above 50-day", "% up today",
                   "Volatility %", "Leader today", "Laggard today"]].rename(columns={"Business type": "Category"})
                .sort_values("Rating", ascending=False),
                column_config={
                    "Rating": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                    "1D": st.column_config.NumberColumn(format="%+.1f%%"), "5D": st.column_config.NumberColumn(format="%+.1f%%"),
                    "1M": st.column_config.NumberColumn(format="%+.1f%%"), "3M": st.column_config.NumberColumn(format="%+.1f%%"),
                    "% above 50-day": st.column_config.NumberColumn(format="%.0f%%"),
                    "% up today": st.column_config.NumberColumn(format="%.0f%%"),
                    "Volatility %": st.column_config.NumberColumn(format="%.0f%%")})

    st.markdown("### Coin scanner")
    s = coins.copy()
    s["Coin"] = s["Ticker"].map(lambda t: NAME_OF.get(t, t))
    s["Category"] = s["Ticker"].map(lambda t: CATEGORY_OF.get(t, "Other"))
    s = s[s["Category"].isin(selected)]
    total = len(s)
    s["Signal"] = s["Score"].map(signal_label)
    s = s[s["dollar_vol"] >= flt["min_vol_m"] * 1e6]
    if flt["signal"] != "All":
        s = s[s["Signal"] == flt["signal"]]
    if flt["unusual"]:
        s = s[s["Rel. volume"] >= 1.5]
    if flt["hide_vol"]:
        s = s[s["Volatility %"] < flt["vol_limit"]]
    if flt["search"]:
        q = flt["search"].upper()
        s = s[s["Ticker"].str.contains(q, regex=False) | s["Coin"].str.upper().str.contains(q, regex=False)]
    col, asc = PULSE_SORTS[flt["sort"]]
    s = s.sort_values(col, ascending=asc, na_position="last").head(flt["limit"])
    if s.empty:
        st.info("No coins match these filters.")
        st.session_state["crypto_tickers"] = []
    else:
        s["Flags"] = s.apply(lambda r: flags(r, flt["vol_limit"]), axis=1)
        st.session_state["crypto_tickers"] = list(s["Ticker"])
        show_df(s[["Coin", "Category", "Signal", "Rating", "Price", "1d %", "5d %", "1m %", "Rel. volume", "RSI",
                   "Volatility %", "Flags"]],
                column_config={
                    "Rating": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                    "Price": st.column_config.NumberColumn(format="$%.6g"),
                    "1d %": st.column_config.NumberColumn(format="%+.1f%%"), "5d %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "1m %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Rel. volume": st.column_config.NumberColumn(format="%.1fx"),
                    "RSI": st.column_config.NumberColumn(format="%.0f"),
                    "Volatility %": st.column_config.NumberColumn(format="%.0f%%")})
    st.caption(f"Updated {now_et()} ET · showing {len(s)} of {total} coins in the selected categories · "
               f"loaded {p['loaded']} of {p['requested']} tracked coins.")


def render_crypto_pulse(weights: dict):
    st.markdown("Scans the major coins and rates whether crypto is trending **bullish or bearish**, warns when "
                "**volatility** is rising, and breaks the market out by **category**. The rating always uses the full "
                "scan; the filters only change what's shown.")
    f1, f2 = st.columns([5, 1.5])
    selected = f1.multiselect("Categories", CATEGORIES, default=CATEGORIES, key="cp_cats")
    refresh = f2.selectbox("Auto-refresh", list(PULSE_REFRESH), index=1, key="cp_refresh")
    g1, g2, g3 = st.columns([3, 2, 2])
    with g1:
        ctl("Signal")
        sig = segmented("Signal", ["All", "Bullish", "Neutral", "Bearish"], "All", "cp_sig")
    sort = g2.selectbox("Sort coins by", list(PULSE_SORTS), key="cp_sort")
    search = g3.text_input("Find coin", key="cp_search").strip()
    with st.expander("More filters"):
        m1, m2, m3 = st.columns(3)
        min_vol_m = m1.number_input("Min traded today ($ millions)", 0, 100_000, 0, 50, key="cp_minvol")
        vol_limit = m2.slider("Flag volatility above (% a year)", 40, 300, 120, 10, key="cp_vlim")
        limit = m3.number_input("Rows to show", 10, 100, 40, 10, key="cp_limit")
        h1, h2 = st.columns(2)
        unusual = h1.checkbox("Only unusual volume (1.5x or more)", value=False, key="cp_unusual")
        hide_vol = h2.checkbox("Hide very volatile coins", value=False, key="cp_hide")
    flt = {"signal": sig, "sort": sort, "search": search, "min_vol_m": float(min_vol_m), "vol_limit": vol_limit,
           "limit": int(limit), "unusual": unusual, "hide_vol": hide_vol}
    wkey = tuple(sorted(weights.items()))
    refreshing(crypto_pulse_results, every=PULSE_REFRESH[refresh])(wkey, selected, flt)


def crypto_forecast():
    board = load_crypto_board(())
    results = {}
    for name, sym in (("Bitcoin", "BTC-USD"), ("Ethereum", "ETH-USD")):
        try:
            bar = get_daily(sym)
            f = forecast_coin(sym, f"{bar.index[-1]:%Y-%m-%d}") if len(bar) else {"ok": False}
        except Exception as e:
            f = {"ok": False, "reason": f"The forecast failed: {e}"}
        q = quote(board, sym)
        note = (f"Today so far (UTC day): {q['pct']:+.2f}%. There's no futures or overnight-session signal for crypto, "
                "so this page shows the statistical forecast only." if q else "Live signals aren't available right now.")
        results[name] = (f, note)
    lines = []
    for name, (f, _) in results.items():
        if f.get("ok") and f["horizons"][1]["model"].get("ok"):
            p = f["horizons"][1]["model"]["p_up"] * 100
            lines.append(f"**{name}:** model {p:.0f}% up for today's close at 00:00 UTC "
                         f"({edge_text(f['horizons'][1]['test'].get('edge', ''))}).")
    if lines:
        st.info("  \n".join(lines))
    for col, (name, (f, note)) in zip(st.columns(2), results.items()):
        with col:
            forecast_card(name, f, None, note, crypto=True)
    st.warning("**Please read:** crypto is far more volatile than stocks, and these statistical estimates are not "
               "reliable predictions. The tests above usually show little or no edge, and a single headline can move "
               "prices 10% in minutes. Use the ranges to see how much a coin can move, not to time trades. "
               "Not financial advice.")


def render_crypto(weights: dict):
    favs = tuple(f for f in st.session_state.get("favorites", []) if is_crypto(f))[:10]
    wkey = tuple(sorted(weights.items()))
    st.markdown("Bitcoin, Ethereum and the biggest coins. Crypto trades 24 hours a day, 7 days a week, so there's no "
                "open, close or weekend gap. Refreshes automatically (set the rate in the sidebar).")
    refreshing(crypto_tape)(favs)
    st.write("")
    ctl("Crypto view")
    view = segmented("Crypto view", ["Live board", "Analyze a coin", "Market pulse", "Forecast"], "Live board",
                     "crypto_view")
    if view == "Live board":
        refreshing(crypto_live)(wkey, favs)
    elif view == "Analyze a coin":
        render_crypto_analyze(weights)
    elif view == "Market pulse":
        render_crypto_pulse(weights)
    else:
        refreshing(crypto_forecast)()
    tickers = st.session_state.get("crypto_tickers", [])
    if view in ("Live board", "Market pulse") and tickers:
        p1, p2, p3 = st.columns([2, 1, 1])
        pick = p1.selectbox("Open a coin", tickers, format_func=lambda s: f"{NAME_OF.get(s, s)} ({s.replace('-USD', '')})",
                            key="crypto_pick")
        for col in (p2, p3):
            col.write("")
            col.write("")
        p2.button("Open in Analyze", on_click=open_coin, args=(pick,), key="crypto_open")
        p3.button("☆ Add to favorites", on_click=add_favorites, args=([pick],), key="crypto_addfav")


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


# ---------- favorites: chart several symbols at once ----------
FV_MAX = 12
FV_LAYOUTS = ["Separate charts", "One chart (compare)"]


def to_et(obj):
    """Coins report intraday times in UTC, stocks in New York time. Put coins on New York time so they line up."""
    if obj is None or len(obj) == 0:
        return obj
    obj = obj.copy()
    obj.index = obj.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    return obj


def fav_compare(cfg: dict, weights: dict):
    """One chart, every selected symbol rebased to the same starting point."""
    syms, view = cfg["syms"], cfg["view"]
    series, sigs, notes, missing = {}, {}, [], []
    arrows = cfg["arrow_cfg"]
    intraday = view == "Live" and cfg["lint"] != "1d"
    exact = False
    if view == "Live":
        period, eff, note = live_params(cfg["lint"], cfg["lrng"])
        if note:
            notes.append(note)
        intraday = eff != "1d"
        exact = intraday and arrows is not None
        for sym in syms:
            df = load_intraday(sym, eff, period, cfg["prepost"])
            if df.empty or len(df) < 2:
                missing.append(sym)
                continue
            fix = to_et if (intraday and is_crypto(sym)) else (lambda x: x)
            series[sym] = fix(df["Close"])
            if arrows is not None:
                try:
                    sigs[sym] = fix(intraday_signals(sym, eff, df, cfg["prepost"], weights, arrows) if intraday
                                    else daily_signals(sym, weights, arrows))
                except Exception:
                    pass
    else:
        dailies = {s: get_daily(s) for s in syms}
        ok = {s: d for s, d in dailies.items() if not d.empty and len(d) > 2}
        missing = [s for s in syms if s not in ok]
        if ok:
            last = max(d.index[-1] for d in ok.values())
            first = max(d.index[0] for d in ok.values())     # start where every symbol has data
            if cfg["rng"] == "Custom":
                start, end = pd.Timestamp(cfg["from"]), pd.Timestamp(cfg["to"])
            else:
                start, end = preset_start(cfg["rng"], last, first), last
            for s, d in ok.items():
                v = resample_ohlcv(d, cfg["interval"])["Close"].loc[start:end]
                if len(v) >= 2:
                    series[s] = v
                    if arrows is not None:
                        try:
                            sigs[s] = daily_signals(s, weights, arrows)
                        except Exception:
                            pass
                else:
                    missing.append(s)
    if not series:
        st.info("No data came back for those symbols in this range. Try a longer range or a different bar size.")
        return
    any_crypto = any(is_crypto(s) for s in series)
    breaks = "none" if any_crypto else ("us" if intraday else "daily")
    fig = compare_chart(series, cfg["scale"], breaks, cfg["prepost"], sigs if arrows is not None else None,
                        intraday=intraday, exact=exact)
    show_chart(fig)
    # a small table of how each symbol did over the window shown
    rows = []
    for s, v in series.items():
        rows.append({"Symbol": s, "Start": float(v.iloc[0]), "Latest": float(v.iloc[-1]),
                     "Change %": (float(v.iloc[-1]) / float(v.iloc[0]) - 1) * 100,
                     "High": float(v.max()), "Low": float(v.min()),
                     "Arrows": len(sigs.get(s, [])) if arrows is not None else None})
    t = pd.DataFrame(rows).sort_values("Change %", ascending=False)
    cols = {"Start": st.column_config.NumberColumn(format="%.2f"), "Latest": st.column_config.NumberColumn(format="%.2f"),
            "Change %": st.column_config.NumberColumn(format="%+.2f%%"),
            "High": st.column_config.NumberColumn(format="%.2f"), "Low": st.column_config.NumberColumn(format="%.2f")}
    if arrows is None:
        t = t.drop(columns="Arrows")
    show_df(t, column_config=cols)
    msg = f"Updated {now_et()} ET."
    if any_crypto and intraday:
        msg += " Coins trade 24/7, so nights and weekends are shown for all symbols; coin times are shown in New York time."
    if missing:
        msg += " No data for: " + ", ".join(missing) + "."
    if arrows is not None:
        msg += (" Arrows sit on each line where that symbol's technical score moved into Buy or Sell territory "
                "(hover for details).")
    st.caption(msg + " " + " ".join(notes))


def fav_grid(cfg: dict, weights: dict):
    """One chart per symbol, with the same candles / overlays / panels / arrows as the Analyze page."""
    cols_n = 1 if st.session_state.get("compact") else cfg["ncols"]
    view, arrows = cfg["view"], cfg["arrow_cfg"]
    syms = cfg["syms"]
    for i in range(0, len(syms), cols_n):
        cols = st.columns(cols_n)
        for col, sym in zip(cols, syms[i:i + cols_n]):
            with col:
                try:
                    q = load_quote(sym)
                    if q.get("price") and q.get("prev"):
                        st.markdown(f"**{sym}** &nbsp; {fmt_price(q['price'])} "
                                    f"<span class='muted'>({q['price'] / q['prev'] - 1:+.2%} today)</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{sym}**")
                    daily = get_daily(sym)
                    has_vol = (not daily.empty) and float(daily["Volume"].iloc[-60:].sum()) > 0
                    panels = [p for p in cfg["panels"] if p != "Volume" or has_vol]
                    if view == "Live":
                        sig = None
                        cfg_a = arrows if arrows is not None and cfg["lint"] != "1d" else None
                        if arrows is not None and cfg["lint"] == "1d":
                            sig = daily_signals(sym, weights, arrows)
                        live_chart(sym, cfg["lint"], cfg["lrng"], cfg["prepost"], cfg["ctype"], cfg["overlays"], panels,
                                   sig, cfg_a, weights, mini=True, tall=cols_n == 1)
                    else:
                        if daily.empty or len(daily) < 30:
                            st.caption("No data for this symbol.")
                            continue
                        first, last = daily.index[0], daily.index[-1]
                        if cfg["rng"] == "Custom":
                            start, end = pd.Timestamp(cfg["from"]), pd.Timestamp(cfg["to"])
                        else:
                            start, end = preset_start(cfg["rng"], last, first), last
                        chart_df = add_indicators(resample_ohlcv(daily, cfg["interval"]))
                        if len(chart_df.loc[start:end]) < 2:
                            st.caption("Too few bars in this range for this symbol.")
                            continue
                        sig = daily_signals(sym, weights, arrows) if arrows is not None else None
                        fig = price_chart(chart_df, start, end, cfg["ctype"], cfg["overlays"], panels,
                                          cfg["interval"], signals=sig)
                        grid_style(fig, sym, len(panels), cols_n == 1)
                        show_chart(fig)
                except Exception as e:
                    st.caption(f"{sym}: couldn't draw this chart ({e}).")
    if arrows is not None:
        st.caption("Arrows mark where each symbol's technical score moved into Buy (green up) or Sell (red down) "
                   "territory. Open a symbol in Analyze for the full signal. Not financial advice.")


def render_fav_charts(weights: dict):
    ss = st.session_state
    favs = ss["favorites"]
    st.markdown("### Chart your favorites")
    st.caption("Every favorite you pick gets its own chart, just like the Analyze page, with the same ranges, bar sizes, "
               "indicators and buy/sell arrows. Prefer one chart with all of them on it? Choose **One chart (compare)**.")
    # keep the selection valid when favorites change
    keep = [s for s in ss.get("fv_syms", favs[:4]) if s in favs]
    ss["fv_syms"] = (keep or favs[:4])[:FV_MAX]
    syms = st.multiselect(f"Symbols to chart (up to {FV_MAX})", favs, key="fv_syms", max_selections=FV_MAX)
    if not syms:
        st.info("Pick at least one symbol above.")
        return

    m0, m1 = st.columns([3, 3])
    with m0:
        ctl("Layout")
        layout = segmented("Layout", FV_LAYOUTS, FV_LAYOUTS[0], "fv_layout2")
    grid = layout == FV_LAYOUTS[0]
    with m1:
        if grid:
            ctl("Charts per row")
            ncols = int(segmented("Charts per row", ["1", "2", "3"], "1", "fv_ncols2"))
        else:
            ctl("Show as")
            scale = segmented("Show as", ["% change", "Indexed (start = 100)", "Price"], "% change", "fv_scale")

    c0, c1, c2, c3 = st.columns([2, 4, 5, 2])
    with c0:
        ctl("View")
        view = segmented("View", ["Live", "History"], "Live", "fv_view")
    lrng = lint = rng = interval = None
    d_from = d_to = None
    if view == "Live":
        with c1:
            ctl("Range")
            lrng = segmented("Range", list(LIVE_RANGES), "1D", "fv_live_rng")
        with c2:
            ctl("Bar size")
            lint = BAR_LABELS.get(segmented("Bar size", list(BAR_LABELS), "5 min", "fv_live_bar"), "5m")
    else:
        with c1:
            ctl("Date range")
            rng = segmented("Date range", PRESETS + ["Custom"], "1Y", "fv_rng")
        with c2:
            ctl("Interval")
            interval = segmented("Interval", INTERVALS, "Daily", "fv_interval")
        if rng == "Custom":
            d1, d2 = st.columns(2)
            today = pd.Timestamp.now().normalize()
            d_from = d1.date_input("From", value=(today - pd.DateOffset(years=1)).date(), key="fv_from")
            d_to = d2.date_input("To", value=today.date(), key="fv_to")
            if d_from >= d_to:
                st.warning("The start date must be before the end date.")
                return
    ctype = "Line"
    if grid:
        with c3:
            ctl("Chart")
            ctype = segmented("Chart", ["Candles", "Line"], "Candles", "fv_ctype")

    o1, o2, o3, o4 = st.columns([4, 4, 2, 2])
    overlays, panels = [], []
    if grid:
        overlays = o1.multiselect("Overlays", ["SMA 9", "SMA 20", "SMA 50", "SMA 200", "Bollinger bands"],
                                  default=["SMA 9", "SMA 20", "SMA 50"], key="fv_overlays")
        panels = o2.multiselect("Panels", ["Volume", "RSI", "MACD"], default=["Volume", "RSI"], key="fv_panels2")
    prepost = (o3.checkbox("Extended hours", value=False, key="fv_ext")
               if view == "Live" and not all(is_crypto(s) for s in syms) else False)
    show_arrows = o4.checkbox("Buy/sell arrows", value=True, key="fv_arrows",
                              help="Green up arrow: the technical score moved into Buy territory. Red down arrow: Sell.")
    arrow_cfg = None
    if show_arrows:
        sens = o4.selectbox("Arrow sensitivity", list(ARROW_SENS), index=1, key="fv_arrow_sens",
                            help="More = earlier but noisier arrows. Fewer = only the strongest turns.")
        arrow_cfg = ARROW_SENS[sens]
    if not grid:
        st.caption("Candles, overlays and panels apply when each favorite has its own chart. Choose **Separate charts** to use them.")

    cfg = {"syms": list(syms), "view": view, "lrng": lrng, "lint": lint, "rng": rng, "interval": interval,
           "from": d_from, "to": d_to, "ctype": ctype, "overlays": overlays, "panels": panels, "prepost": prepost,
           "arrow_cfg": arrow_cfg}
    every = None if view == "Live" else 0   # live views refresh with the sidebar setting; history doesn't need to
    if grid:
        cfg["ncols"] = ncols
        refreshing(fav_grid, every=every)(cfg, weights)
    else:
        cfg["scale"] = scale
        refreshing(fav_compare, every=every)(cfg, weights)


def render_favorites(weights: dict):
    favs = st.session_state["favorites"]
    st.markdown("Save the symbols you follow and see them all in one live table. Add one from the sidebar while "
                "viewing it, or type symbols below.")
    if st.session_state.get("user"):
        st.caption(f"✓ Signed in as {st.session_state['user']}. Your favorites are saved to your account and will be "
                   "here when you log back in.")
    else:
        st.caption("You're not signed in. Create an account or log in (sidebar) to keep your favorites across visits "
                   "and devices.")
    a1, a2 = st.columns([5, 1])
    a1.text_input("Add symbols (comma-separated)", key="fav_add", placeholder="e.g. NVDA, XOM, ^GSPC")
    a2.write("")
    a2.write("")
    a2.button("Add", on_click=add_from_box, key="fav_add_btn")
    note = ("Your favorites are saved to your account, so they're here whenever you log in."
            if st.session_state.get("user") else
            "Not signed in: favorites are only kept in this page's web address (the part after ?fav=). Log in from "
            "the sidebar to keep them for good, or bookmark the page.")
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

    render_fav_charts(weights)

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
elif section == "Trading":
    try:
        render_trading(weights)
    except Exception as e:
        st.error(f"Trading page error: {e}")
elif section == "Forecast":
    try:
        render_forecast(weights)
    except Exception as e:
        st.error(f"Forecast error: {e}")
elif section == "Crypto":
    try:
        render_crypto(weights)
    except Exception as e:
        st.error(f"Crypto error: {e}")
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
