"""Small helpers: symbols, benchmarks, date ranges and resampling."""
from __future__ import annotations

import pandas as pd

INDEX_NAMES = {
    "^GSPC": "S&P 500", "^NDX": "Nasdaq-100", "^DJI": "Dow Jones Industrial Average",
    "^IXIC": "Nasdaq Composite", "^RUT": "Russell 2000", "^VIX": "CBOE Volatility Index",
}
ETF_FOR = {
    "^GSPC": "SPY, VOO or IVV", "^NDX": "QQQ", "^DJI": "DIA",
    "^IXIC": "ONEQ or QQQ", "^RUT": "IWM",
}
QUICK_PICKS = {
    "—": None, "S&P 500 (^GSPC)": "^GSPC", "Nasdaq-100 (^NDX)": "^NDX",
    "Dow Jones (^DJI)": "^DJI", "Nasdaq Composite (^IXIC)": "^IXIC",
    "Russell 2000 (^RUT)": "^RUT", "VIX (^VIX)": "^VIX",
    "SPY (S&P 500 ETF)": "SPY", "QQQ (Nasdaq-100 ETF)": "QQQ",
}
OVERVIEW = [("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"), ("Dow", "^DJI"),
            ("Russell 2000", "^RUT"), ("VIX", "^VIX")]

INTERVALS = ["Daily", "Weekly", "Monthly", "Quarterly"]
PRESETS = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y", "Max"]

_RULES = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME", "Quarterly": "QE"}
_OLD_RULES = {"ME": "M", "QE": "Q"}
_OFFSETS = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
            "1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3), "5Y": pd.DateOffset(years=5),
            "10Y": pd.DateOffset(years=10)}


def strip_tz(obj):
    """Drop timezone info from a DataFrame/Series index so dates compare cleanly."""
    if getattr(obj.index, "tz", None) is not None:
        obj = obj.copy()
        obj.index = obj.index.tz_localize(None)
    return obj


def is_index(sym: str) -> bool:
    return sym.startswith("^")


def benchmark_for(sym: str) -> str | None:
    """What to compare a symbol against for relative strength (None = it IS the market)."""
    if sym in {"SPY", "VOO", "IVV", "^GSPC", "^SPX"}:
        return None
    return "^GSPC" if is_index(sym) else "SPY"


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = _RULES[interval]
    if rule is None:
        return df
    agg = {k: v for k, v in {"Open": "first", "High": "max", "Low": "min",
                             "Close": "last", "Volume": "sum"}.items() if k in df.columns}
    try:
        out = df.resample(rule).agg(agg)
    except ValueError:  # older pandas doesn't know "ME"/"QE"
        out = df.resample(_OLD_RULES.get(rule, rule)).agg(agg)
    return out.dropna(subset=["Close"])


def preset_start(label: str, end: pd.Timestamp, first: pd.Timestamp) -> pd.Timestamp:
    if label == "Max":
        return first
    if label == "YTD":
        return max(pd.Timestamp(year=end.year, month=1, day=1), first)
    return max(end - _OFFSETS[label], first)


# ---- live / intraday ----
INTRADAY = ["1m", "5m", "15m", "30m", "1h"]
INTRADAY_MAX_DAYS = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}  # Yahoo's lookback limits
LIVE_RANGES = {"1D": ("1d", 1), "5D": ("5d", 5), "1M": ("1mo", 30)}


def live_params(interval: str, rng: str) -> tuple[str, str, str]:
    """(yfinance period, bar size actually used, note). Coarsens the bars if Yahoo can't go back that far."""
    period, days = LIVE_RANGES[rng]
    note = ""
    if days > INTRADAY_MAX_DAYS[interval]:
        for alt in INTRADAY:
            if INTRADAY_MAX_DAYS[alt] >= days:
                note = (f"{interval} bars only go back {INTRADAY_MAX_DAYS[interval]} days, "
                        f"so {alt} bars are used for this range.")
                interval = alt
                break
    return period, interval, note
