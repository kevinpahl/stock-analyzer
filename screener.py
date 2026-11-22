"""Top-movers screener: find what is moving today, then rank it with the analyzer's score."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yfinance as yf

from analyzer import DEFAULT_WEIGHTS, analyze
from utils import is_crypto, strip_tz

MIN_ROWS = 210  # need ~200 days for the trend factor

BUILTIN = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM LLY V UNH XOM MA COST HD PG JNJ ABBV NFLX BAC CRM "
           "WMT ORCL CVX KO MRK AMD PEP TMO ADBE CSCO ACN LIN MCD ABT DIS WFC GE IBM PM INTU QCOM TXN CAT "
           "AMGN VZ ISRG NOW UBER BKNG GS AXP SPGI PFE T LOW HON NEE RTX BLK DE PLTR SHOP COIN MU PANW "
           "SNOW SMCI ARM DELL INTC UAL CCL F").split()

# Built-in lists beyond stocks. Edit freely: anything Yahoo has no data for is skipped automatically.
ETF_BROAD = ("SPY VOO IVV QQQ DIA IWM VTI MDY IJH IJR RSP VT ITOT SCHX VUG VTV IWF IWD SCHD VYM VIG DVY MTUM QUAL "
             "USMV").split()
ETF_SECTOR = ("XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC VGT SMH SOXX XBI IBB KRE KBE XOP OIH XME ITA JETS ARKK ARKG "
              "ARKW IGV CIBR HACK BOTZ ROBO TAN ICLN LIT URA XHB ITB XRT").split()
ETF_INTL = "EFA EEM VEA VWO IEFA EWJ EWZ FXI MCHI INDA EWW EWY EWT EZU VGK".split()
ETF_ASSETS = ("TLT IEF SHY AGG BND LQD HYG JNK TIP MUB EMB BIL GLD IAU SLV USO UNG DBC PDBC CPER UUP FXE FXY "
              "IBIT FBTC BITO ETHA GBTC").split()
ETF_LEVERAGED = "TQQQ SQQQ SPXL SPXS SOXL SOXS UPRO TNA FAS LABU".split()
INDEXES_FUTURES = ("^GSPC ^NDX ^DJI ^IXIC ^RUT ^VIX ^FTSE ^N225 ^HSI ^GDAXI ^TNX ES=F NQ=F YM=F RTY=F CL=F GC=F SI=F "
                   "NG=F HG=F ZC=F ZW=F ZB=F EURUSD=X GBPUSD=X USDJPY=X").split()
ETF_SET = set(ETF_BROAD + ETF_SECTOR + ETF_INTL + ETF_ASSETS + ETF_LEVERAGED)
LISTS = {
    "etf_all": ETF_BROAD + ETF_SECTOR + ETF_INTL + ETF_ASSETS,
    "etf_sector": ETF_SECTOR,
    "etf_assets": ETF_ASSETS,
    "etf_leveraged": ETF_LEVERAGED,
    "indexes": INDEXES_FUTURES,
}


def _coins() -> list[str]:
    from crypto import COINS   # imported here: crypto.py itself imports market.py, which imports this module
    return list(COINS)

# label -> Yahoo predefined screener key (None = built-in large caps, "custom" = user list, other = LISTS key)
UNIVERSES = {
    "Today's gainers (Yahoo screener)": "day_gainers",
    "Today's losers (Yahoo screener)": "day_losers",
    "Most active (Yahoo screener)": "most_actives",
    "Growth technology stocks (Yahoo screener)": "growth_technology_stocks",
    "Undervalued growth stocks (Yahoo screener)": "undervalued_growth_stocks",
    "Small caps with momentum (Yahoo screener)": "aggressive_small_caps",
    "Most shorted stocks (Yahoo screener)": "most_shorted_stocks",
    "Large-cap watchlist (built-in)": None,
    "Popular ETFs: broad, sector, international, bonds, commodities (built-in)": "etf_all",
    "Sector & thematic ETFs (built-in)": "etf_sector",
    "Bond, gold, oil & crypto ETFs (built-in)": "etf_assets",
    "Leveraged & inverse ETFs (built-in)": "etf_leveraged",
    "Indexes, futures & currencies (built-in)": "indexes",
    "Crypto coins (built-in)": "crypto",
    "Everything: stocks + ETFs + crypto (built-in)": "everything",
    "Custom list": "custom",
}
ASSET_TYPES = ["Stock", "ETF", "Index", "Future", "Currency", "Crypto"]
MOVER_MODES = ["Biggest % gainers", "Biggest % losers", "Largest absolute move", "Highest relative volume"]


_SYMBOL = re.compile(r"\^?[A-Z0-9]{1,6}([.-][A-Z0-9]{1,5})?(=[FX])?")


def _clean(symbols) -> list[str]:
    """Keep anything that looks like a Yahoo symbol: AAPL, BRK-B, SPY, ^GSPC, BTC-USD, GC=F, EURUSD=X."""
    out = []
    for s in symbols:
        s = str(s).upper().strip()
        if _SYMBOL.fullmatch(s) and s not in out:
            out.append(s)
    return out


def asset_type(sym: str) -> str:
    if is_crypto(sym):
        return "Crypto"
    if sym.startswith("^"):
        return "Index"
    if sym.endswith("=F"):
        return "Future"
    if sym.endswith("=X"):
        return "Currency"
    return "ETF" if sym in ETF_SET else "Stock"


def get_universe(label: str, custom_text: str = "", count: int = 50) -> tuple[list[str], str]:
    key = UNIVERSES[label]
    if key == "custom":
        return _clean(re.split(r"[,\s]+", custom_text)), ""
    if key is None:
        return list(BUILTIN), ""
    if key == "everything":
        return list(dict.fromkeys(BUILTIN + LISTS["etf_all"] + _coins())), ""
    if key == "crypto":
        return _coins(), ""
    if key in LISTS:
        return list(LISTS[key]), ""
    try:
        if key in ("growth_technology_stocks", "undervalued_growth_stocks", "aggressive_small_caps", "most_shorted_stocks"):
            count = max(count, 100)   # discovery lists: a wider net is more useful
        quotes = yf.screen(key, count=count).get("quotes", [])
        syms = _clean(q.get("symbol", "") for q in quotes)
        if syms:
            return syms, ""
    except Exception:
        pass
    return list(BUILTIN), "Yahoo's screener wasn't reachable, so the built-in large-cap list was used instead."


def download_history(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = yf.download(tickers, period=period, auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    out: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        have = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t in have:
                d = strip_tz(raw[t].dropna(subset=["Close"]))
                if not d.empty:
                    out[t] = d
    elif len(tickers) == 1:
        out[tickers[0]] = strip_tz(raw.dropna(subset=["Close"]))
    return out


def rank_movers(data: dict[str, pd.DataFrame], bench: pd.Series | None = None,
                mover_by: str = "Biggest % gainers", top_n: int = 20, min_price: float = 5.0,
                min_avg_volume: float = 500_000, info_loader=None,
                weights: dict | None = None, types: list | None = None) -> tuple[pd.DataFrame, dict]:
    """1) keep liquid names, 2) pick the top_n movers, 3) score each, 4) sort by score.
    Works for stocks, ETFs, indexes, futures, currencies and coins. The price and volume minimums only apply to
    stocks and ETFs (coins can cost pennies; indexes, futures and currencies have no real volume)."""
    w = weights or DEFAULT_WEIGHTS
    rows, frames, skipped = [], {}, 0
    for t, raw in data.items():
        d = raw.dropna(subset=["Close"])
        if len(d) < MIN_ROWS:
            skipped += 1
            continue
        kind = asset_type(t)
        if types and kind not in types:
            continue
        c, vol = d["Close"], d["Volume"]
        avg_vol = float(vol.iloc[-21:-1].mean())
        has_vol = avg_vol > 0
        if kind in ("Stock", "ETF") and (float(c.iloc[-1]) < min_price or not avg_vol >= min_avg_volume):
            skipped += 1
            continue
        frames[t] = d
        rows.append({
            "Ticker": t, "Type": kind, "Price": float(c.iloc[-1]),
            "1d %": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
            "5d %": (c.iloc[-1] / c.iloc[-6] - 1) * 100,
            "Rel. volume": float(vol.iloc[-1]) / avg_vol if has_vol else np.nan,
        })
    stats = {"screened": len(data), "skipped": skipped, "ranked": 0}
    if not rows:
        return pd.DataFrame(), stats

    m = pd.DataFrame(rows)
    m["_abs"] = m["1d %"].abs()
    col, asc = {"Biggest % gainers": ("1d %", False), "Biggest % losers": ("1d %", True),
                "Largest absolute move": ("_abs", False), "Highest relative volume": ("Rel. volume", False)}[mover_by]
    m = m.sort_values(col, ascending=asc, na_position="last").head(top_n).drop(columns="_abs")

    out = []
    for r in m.to_dict("records"):
        t = r["Ticker"]
        kind = r["Type"]
        info = info_loader(t) if info_loader and kind in ("Stock", "ETF") else {}
        res = analyze(frames[t], info, bench if kind in ("Stock", "ETF") else None, w)
        live = [f for f in res["factors"] if f.available]
        top = sorted(live, key=lambda f: abs(f.score * w.get(f.name, 0)), reverse=True)[:2]
        r.update({"Verdict": res["verdict"], "Score": res["score_pct"], "Confidence": res["confidence"],
                  "RSI": float(res["df"]["RSI"].iloc[-1]),
                  "Drivers": ", ".join(f"{f.name.split(' (')[0]} {f.score:+.0f}" for f in top)})
        out.append(r)
    ranked = pd.DataFrame(out).sort_values("Score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", ranked.index + 1)
    stats["ranked"] = len(ranked)
    return ranked[["Rank", "Ticker", "Type", "Verdict", "Score", "Confidence", "Price", "1d %", "5d %",
                   "Rel. volume", "RSI", "Drivers"]], stats
