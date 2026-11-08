"""Top-movers screener: find what is moving today, then rank it with the analyzer's score."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yfinance as yf

from analyzer import WEIGHTS, analyze

MIN_ROWS = 210  # need ~200 days for the trend factor

BUILTIN = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM LLY V UNH XOM MA COST HD PG JNJ ABBV NFLX BAC CRM "
           "WMT ORCL CVX KO MRK AMD PEP TMO ADBE CSCO ACN LIN MCD ABT DIS WFC GE IBM PM INTU QCOM TXN CAT "
           "AMGN VZ ISRG NOW UBER BKNG GS AXP SPGI PFE T LOW HON NEE RTX BLK DE PLTR SHOP COIN MU PANW "
           "SNOW SMCI ARM DELL INTC UAL CCL F").split()

# label -> Yahoo predefined screener key (None = built-in list, "custom" = user list)
UNIVERSES = {
    "Today's gainers (Yahoo screener)": "day_gainers",
    "Today's losers (Yahoo screener)": "day_losers",
    "Most active (Yahoo screener)": "most_actives",
    "Large-cap watchlist (built-in)": None,
    "Custom list": "custom",
}
MOVER_MODES = ["Biggest % gainers", "Biggest % losers", "Largest absolute move", "Highest relative volume"]


def _clean(symbols) -> list[str]:
    out = []
    for s in symbols:
        s = str(s).upper().strip()
        if re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", s) and s not in out:
            out.append(s)
    return out


def get_universe(label: str, custom_text: str = "", count: int = 50) -> tuple[list[str], str]:
    key = UNIVERSES[label]
    if key == "custom":
        return _clean(re.split(r"[,\s]+", custom_text)), ""
    if key is None:
        return list(BUILTIN), ""
    try:
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
                d = raw[t].dropna(subset=["Close"])
                if not d.empty:
                    out[t] = d
    elif len(tickers) == 1:
        out[tickers[0]] = raw.dropna(subset=["Close"])
    return out


def rank_movers(data: dict[str, pd.DataFrame], bench: pd.Series | None = None,
                mover_by: str = "Biggest % gainers", top_n: int = 20, min_price: float = 5.0,
                min_avg_volume: float = 500_000, info_loader=None) -> tuple[pd.DataFrame, dict]:
    """1) keep liquid stocks, 2) pick the top_n movers, 3) score each, 4) sort by score."""
    rows, frames, skipped = [], {}, 0
    for t, raw in data.items():
        d = raw.dropna(subset=["Close"])
        if len(d) < MIN_ROWS:
            skipped += 1
            continue
        c, vol = d["Close"], d["Volume"]
        avg_vol = float(vol.iloc[-21:-1].mean())
        if float(c.iloc[-1]) < min_price or not avg_vol >= min_avg_volume:
            skipped += 1
            continue
        frames[t] = d
        rows.append({
            "Ticker": t, "Price": float(c.iloc[-1]),
            "1d %": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
            "5d %": (c.iloc[-1] / c.iloc[-6] - 1) * 100,
            "Rel. volume": float(vol.iloc[-1]) / avg_vol,
        })
    stats = {"screened": len(data), "skipped": skipped, "ranked": 0}
    if not rows:
        return pd.DataFrame(), stats

    m = pd.DataFrame(rows)
    m["_abs"] = m["1d %"].abs()
    col, asc = {"Biggest % gainers": ("1d %", False), "Biggest % losers": ("1d %", True),
                "Largest absolute move": ("_abs", False), "Highest relative volume": ("Rel. volume", False)}[mover_by]
    m = m.sort_values(col, ascending=asc).head(top_n).drop(columns="_abs")

    out = []
    for r in m.to_dict("records"):
        t = r["Ticker"]
        info = info_loader(t) if info_loader else {}
        res = analyze(frames[t], info, bench)
        live = [f for f in res["factors"] if f.available]
        top = sorted(live, key=lambda f: abs(f.score * WEIGHTS[f.name]), reverse=True)[:2]
        r.update({"Verdict": res["verdict"], "Score": res["score_pct"], "Confidence": res["confidence"],
                  "RSI": float(res["df"]["RSI"].iloc[-1]),
                  "Drivers": ", ".join(f"{f.name.split(' (')[0]} {f.score:+.0f}" for f in top)})
        out.append(r)
    ranked = pd.DataFrame(out).sort_values("Score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", ranked.index + 1)
    stats["ranked"] = len(ranked)
    return ranked[["Rank", "Ticker", "Verdict", "Score", "Confidence", "Price", "1d %", "5d %",
                   "Rel. volume", "RSI", "Drivers"]], stats
