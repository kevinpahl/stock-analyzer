"""Live-data tools the AI analyst can call. No Streamlit import: loaders are passed in as `L`."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from analyzer import add_indicators, analyze
from market import GROUP_OF, GROUPS, INDEXES, signal_label
from utils import INDEX_NAMES, is_index, resample_ohlcv

_SYM = re.compile(r"^\^?[A-Z0-9]{1,8}(?:[.\-=][A-Z0-9]{1,4})?$")
_PERIODS = {"1mo": pd.DateOffset(months=1), "3mo": pd.DateOffset(months=3), "6mo": pd.DateOffset(months=6),
            "1y": pd.DateOffset(years=1), "5y": pd.DateOffset(years=5)}


def clean_symbol(s) -> str:
    s = str(s).upper().strip()
    if not _SYM.match(s):
        raise ValueError(f"'{s}' doesn't look like a ticker symbol")
    return s


def _reading(score) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "n/a"
    return "bullish" if score > 0.5 else "bearish" if score < -0.5 else "neutral"


def build_impls(L, weights: dict) -> dict:
    """L needs: get_daily(sym), load_quote(sym), load_meta(sym), get_bench(sym), load_pulse(), load_overview(),
    load_favorites(tuple), favorites(), now_et()."""

    def asof() -> str:
        return f"{L.now_et()} ET"

    def get_quote(symbol: str):
        sym = clean_symbol(symbol)
        daily = L.get_daily(sym)
        if daily.empty:
            return {"error": f"No data found for {sym}."}
        q = L.load_quote(sym)
        c = daily["Close"]
        price = q.get("price") or float(c.iloc[-1])
        prev = q.get("prev") or float(c.iloc[-2])
        age = q.get("age_min")
        return {"symbol": sym, "price": price, "previous_close": prev, "change": price - prev,
                "change_pct": (price / prev - 1) * 100,
                "day_high": q.get("high") or float(daily["High"].iloc[-1]),
                "day_low": q.get("low") or float(daily["Low"].iloc[-1]),
                "market_status": "open (live data)" if age is not None and age < 20 else "closed or delayed",
                "minutes_since_last_trade_bar": age, "as_of": asof()}

    def get_price_history(symbol: str, period: str = "3mo", interval: str = "daily"):
        sym = clean_symbol(symbol)
        daily = L.get_daily(sym)
        if daily.empty:
            return {"error": f"No data found for {sym}."}
        interval = interval if interval in ("daily", "weekly", "monthly") else "daily"
        df = resample_ohlcv(daily, interval.capitalize())
        end = df.index[-1]
        start = df.index[0] if period == "max" else max(df.index[0], end - _PERIODS.get(period, _PERIODS["3mo"]))
        w = df.loc[start:]
        if len(w) < 2:
            return {"error": "Not enough bars for that period and interval."}
        c = w["Close"]
        dret = daily["Close"].loc[start:].pct_change().dropna()
        pick = sorted(set(np.linspace(0, len(w) - 1, min(len(w), 30)).round().astype(int)))
        return {"symbol": sym, "period": period, "interval": interval, "bars": len(w),
                "start": str(w.index[0].date()), "end": str(w.index[-1].date()),
                "first_close": float(c.iloc[0]), "last_close": float(c.iloc[-1]),
                "return_pct": float(c.iloc[-1] / c.iloc[0] - 1) * 100,
                "high": {"price": float(w["High"].max()), "date": str(w["High"].idxmax().date())},
                "low": {"price": float(w["Low"].min()), "date": str(w["Low"].idxmin().date())},
                "annualized_volatility_pct": float(dret.std() * np.sqrt(252) * 100) if len(dret) > 2 else None,
                "max_drawdown_pct": float((c / c.cummax() - 1).min()) * 100,
                "average_volume": float(w["Volume"].mean()),
                "closes_sampled": [[str(w.index[i].date()), round(float(c.iloc[i]), 2)] for i in pick],
                "last_5_closes": [[str(i.date()), round(float(v), 2)] for i, v in c.iloc[-5:].items()],
                "as_of": asof()}

    def get_stock_analysis(symbol: str):
        sym = clean_symbol(symbol)
        idx = is_index(sym)
        daily = L.get_daily(sym)
        if len(daily) < 60:
            return {"error": f"Not enough price history for {sym}."}
        info, _ = L.load_meta(sym)
        res = analyze(daily, {} if idx else info, L.get_bench(sym), weights)
        d = res["df"]
        last = d.iloc[-1]
        width = last["BB_high"] - last["BB_low"]
        c = d["Close"].iloc[-252:]
        out = {"symbol": sym, "name": INDEX_NAMES.get(sym) or info.get("longName") or info.get("shortName") or sym,
               "is_index": idx, "price": res["price"], "verdict": res["verdict"], "score_0_to_100": res["score_pct"],
               "confidence_pct": res["confidence"],
               "factors": [{"factor": f.name, "reading": _reading(f.score) if f.available else "n/a",
                            "score_minus2_to_plus2": f.score if f.available else None, "reason": f.reason}
                           for f in res["factors"]],
               "indicators": {"rsi14": last["RSI"], "sma9": last["SMA9"], "sma20": last["SMA20"],
                              "sma50": last["SMA50"], "sma200": last["SMA200"], "macd": last["MACD"],
                              "macd_signal": last["MACD_signal"], "macd_histogram": last["MACD_hist"],
                              "bollinger_position_pct": float((last["Close"] - last["BB_low"]) / width * 100)
                              if width and not np.isnan(width) else None},
               "range_52w": {"high": float(c.max()), "low": float(c.min()),
                             "pct_below_high": float(res["price"] / c.max() - 1) * 100},
               "as_of": asof()}
        if not idx:
            keys = {"market_cap": "marketCap", "forward_pe": "forwardPE", "trailing_pe": "trailingPE",
                    "peg_ratio": "pegRatio", "profit_margin": "profitMargins", "revenue_growth": "revenueGrowth",
                    "debt_to_equity": "debtToEquity", "beta": "beta", "analyst_target_price": "targetMeanPrice",
                    "analyst_rating_1_strong_buy_to_5_sell": "recommendationMean", "sector": "sector",
                    "industry": "industry"}
            out["fundamentals"] = {k: info.get(v) for k, v in keys.items() if info.get(v) is not None}
        return out

    def get_market_overview():
        p = L.load_pulse()
        reg, vol = p["regime"], p["vol"]
        if not reg.get("ok"):
            return {"error": "Not enough market data is available right now."}
        b = reg["breadth"]
        stocks = p["stocks"]
        out = {"as_of": asof(),
               "market_rating": {"label": reg["label"], "score_0_to_100": reg["pct"],
                                 "direction_vs_5_sessions_ago": reg["direction"],
                                 "factors": [{"factor": c["Factor"], "reading": _reading(c["score"]),
                                              "score_minus2_to_plus2": c["score"], "detail": c["reason"]}
                                             for c in reg["components"]]},
               "breadth": {"pct_stocks_above_50day": b.get("p50"), "pct_stocks_above_200day": b.get("p200"),
                           "pct_stocks_up_today": b.get("padv"), "stocks_scanned": b.get("n"),
                           "new_20day_highs": int(stocks["new_high"].sum()) if not stocks.empty else None,
                           "new_20day_lows": int(stocks["new_low"].sum()) if not stocks.empty else None}}
        if vol.get("ok"):
            out["volatility"] = {"level": vol["level"], "vix": vol["vix"],
                                 "vix_change_today_pct": vol["vix_chg"] * 100 if vol["vix_chg"] is not None else None,
                                 "sp500_realized_volatility_pct": vol["realized"],
                                 "sp500_move_today_pct": vol["today"], "warnings": vol["warnings"]}
        idxs = []
        try:
            for sym, d in L.load_overview().items():
                if len(d) >= 2:
                    idxs.append({"name": INDEXES.get(sym, sym), "symbol": sym, "level": float(d["Close"].iloc[-1]),
                                 "change_today_pct": float(d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100})
        except Exception:
            pass
        out["indexes"] = idxs
        out["coverage_note"] = "Breadth and rating come from about 180 curated stocks and ETFs, not the whole market."
        return out

    def get_sector_performance(timeframe: str = "1D"):
        g = L.load_pulse()["groups"]
        if g.empty:
            return {"error": "No sector data available right now."}
        tf = timeframe if timeframe in ("1D", "5D", "1M", "3M") else "1D"
        g = g.sort_values(tf, ascending=False)
        rows = [{"business_type": r["Business type"], "signal": r["Signal"], "rating_0_to_100": r["Rating"],
                 "avg_change_1d_pct": r["1D"], "avg_change_5d_pct": r["5D"], "avg_change_1m_pct": r["1M"],
                 "avg_change_3m_pct": r["3M"], "pct_above_50day": r["% above 50-day"],
                 "pct_up_today": r["% up today"], "avg_volatility_pct": r["Volatility %"],
                 "leader_today": r["Leader today"], "laggard_today": r["Laggard today"]}
                for _, r in g.iterrows()]
        return {"sorted_by": tf, "business_types": rows, "as_of": asof()}

    def get_top_movers(direction: str = "gainers", business_type: str | None = None, limit: int = 8):
        s = L.load_pulse()["stocks"].copy()
        if s.empty:
            return {"error": "No scan data available right now."}
        if business_type:
            q = business_type.lower().strip()
            names = [n for n in GROUPS if q in n.lower()]
            if not names:
                return {"error": f"No business type matches '{business_type}'.", "available": list(GROUPS)}
            s = s[s["Ticker"].map(lambda t: any(n in names for n in GROUP_OF.get(t, [])))]
        col, asc = {"gainers": ("1d %", False), "losers": ("1d %", True),
                    "unusual_volume": ("Rel. volume", False)}.get(direction, ("1d %", False))
        s = s.sort_values(col, ascending=asc, na_position="last").head(max(1, min(int(limit), 15)))
        return {"direction": direction, "business_type": business_type, "as_of": asof(),
                "stocks": [{"ticker": r["Ticker"], "business_types": GROUP_OF.get(r["Ticker"], []),
                            "price": r["Price"], "change_1d_pct": r["1d %"], "change_5d_pct": r["5d %"],
                            "relative_volume": r["Rel. volume"], "rsi14": r["RSI"],
                            "technical_signal": signal_label(r["Score"]), "volatility_pct": r["Volatility %"]}
                           for _, r in s.iterrows()]}

    def get_news(symbol: str):
        sym = clean_symbol(symbol)
        _, news = L.load_meta(sym)
        items = []
        for n in (news or [])[:8]:
            c = n.get("content", n)
            if c.get("title"):
                items.append({"title": c.get("title"), "publisher": (c.get("provider") or {}).get("displayName")
                              or c.get("publisher"), "published": c.get("pubDate") or c.get("providerPublishTime")})
        return {"symbol": sym, "headlines": items} if items else {"symbol": sym, "headlines": [],
                                                                  "note": "No headlines available."}

    def get_favorites_status():
        favs = L.favorites()
        if not favs:
            return {"favorites": [], "note": "The user has no saved favorites yet."}
        stats, missing = L.load_favorites(tuple(favs))
        rows = []
        if not stats.empty:
            for _, r in stats.iterrows():
                rows.append({"symbol": r["Ticker"], "price": r["Price"], "change_1d_pct": r["1d %"],
                             "change_5d_pct": r["5d %"], "change_1m_pct": r["1m %"], "rsi14": r["RSI"],
                             "technical_signal": signal_label(r["Score"])})
        return {"favorites": rows, "no_data_for": missing, "as_of": asof()}

    return {"get_quote": get_quote, "get_price_history": get_price_history,
            "get_stock_analysis": get_stock_analysis, "get_market_overview": get_market_overview,
            "get_sector_performance": get_sector_performance, "get_top_movers": get_top_movers,
            "get_news": get_news, "get_favorites_status": get_favorites_status}
