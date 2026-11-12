"""Market-wide scan: bullish/bearish rating, volatility warnings, sector and theme breakdown.

Pure logic (no Streamlit) so it can be tested offline. Everything is computed from daily bars;
the newest bar is the live, in-progress session when the market is open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer import DEFAULT_WEIGHTS, rsi, technical_score_series
from screener import BUILTIN

INDEXES = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^RUT": "Russell 2000", "^DJI": "Dow", "^VIX": "VIX"}
CORE_ETFS = ["SPY", "QQQ", "IWM", "DIA"]
INDEX_ETFS = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)", "DIA": "Dow (DIA)", "IWM": "Russell 2000 (IWM)"}
OFFENSIVE = ["XLK", "XLY", "XLC"]   # tech, consumer discretionary, communications
DEFENSIVE = ["XLP", "XLU", "XLV"]   # staples, utilities, health care

# Business types / themes. Curated lists: edit freely. Stocks can appear in more than one group.
GROUPS: dict[str, dict] = {
    "Energy": {"etf": "XLE", "stocks": "XOM CVX COP EOG SLB OXY MPC PSX VLO KMI WMB HAL DVN FANG OKE"},
    "Technology": {"etf": "XLK", "stocks": "AAPL MSFT ORCL CRM ADBE NOW INTU CSCO IBM ACN"},
    "AI": {"etf": "AIQ", "stocks": "NVDA MSFT GOOGL META AMD AVGO PLTR SMCI ARM TSM ORCL AMZN SNOW MU DELL VRT ANET"},
    "Semiconductors": {"etf": "SMH", "stocks": "NVDA AMD AVGO QCOM TXN MU INTC AMAT LRCX KLAC ASML TSM MRVL ON NXPI ADI"},
    "Cybersecurity": {"etf": "CIBR", "stocks": "PANW CRWD FTNT ZS NET OKTA S"},
    "Financials": {"etf": "XLF", "stocks": "JPM BAC WFC GS MS C SCHW BLK AXP V MA"},
    "Health care": {"etf": "XLV", "stocks": "UNH LLY JNJ ABBV MRK PFE TMO ABT ISRG AMGN BMY GILD VRTX REGN"},
    "Consumer discretionary": {"etf": "XLY", "stocks": "AMZN TSLA HD MCD NKE SBUX LOW BKNG TJX ABNB"},
    "Consumer staples": {"etf": "XLP", "stocks": "PG KO PEP COST WMT PM MO MDLZ CL TGT"},
    "Industrials": {"etf": "XLI", "stocks": "CAT GE HON UNP RTX BA DE LMT UPS ETN"},
    "Aerospace & defense": {"etf": "ITA", "stocks": "LMT RTX NOC GD BA HII LHX TDG"},
    "Materials": {"etf": "XLB", "stocks": "LIN SHW FCX NEM APD ECL NUE DOW"},
    "Utilities": {"etf": "XLU", "stocks": "NEE DUK SO D AEP SRE EXC XEL"},
    "Real estate": {"etf": "XLRE", "stocks": "PLD AMT EQIX WELL SPG O PSA"},
    "Communications": {"etf": "XLC", "stocks": "GOOGL META NFLX DIS T VZ CMCSA TMUS"},
    "Clean energy": {"etf": "ICLN", "stocks": "ENPH FSLR RUN PLUG BE SEDG NEE"},
    "Nuclear & uranium": {"etf": "URA", "stocks": "CCJ UEC LEU SMR OKLO"},
    "Gold & miners": {"etf": "GDX", "stocks": "NEM GOLD AEM WPM FNV KGC"},
    "Crypto-related": {"etf": "IBIT", "stocks": "COIN MSTR MARA RIOT CLSK HOOD"},
    "EVs & autos": {"etf": "DRIV", "stocks": "TSLA F GM RIVN LCID NIO"},
}
_used = {t for g in GROUPS.values() for t in g["stocks"].split()}
GROUPS["Other large caps"] = {"etf": None, "stocks": " ".join(t for t in BUILTIN if t not in _used)}
for _g in GROUPS.values():
    _g["stocks"] = _g["stocks"].split()

GROUP_ETFS = [g["etf"] for g in GROUPS.values() if g["etf"]]
ETF_SET = set(GROUP_ETFS) | set(CORE_ETFS)
GROUP_OF: dict[str, list[str]] = {}
for _name, _g in GROUPS.items():
    for _t in _g["stocks"]:
        GROUP_OF.setdefault(_t, []).append(_name)

REGIME_WEIGHTS = {
    "Index trend": 0.25, "Short-term momentum": 0.15, "Breadth: above 50-day": 0.15,
    "Breadth: above 200-day": 0.10, "Participation (up today)": 0.10,
    "Volatility (VIX)": 0.15, "Risk appetite": 0.10,
}
LEVELS = ["LOW", "NORMAL", "ELEVATED", "HIGH", "EXTREME"]


def all_symbols() -> list[str]:
    syms = list(INDEXES) + CORE_ETFS + GROUP_ETFS + list(GROUP_OF)
    return list(dict.fromkeys(syms))


def signal_label(score: float) -> str:
    if score is None or np.isnan(score):
        return "n/a"
    return "Bullish" if score >= 0.4 else "Bearish" if score <= -0.4 else "Neutral"


def regime_label(score: float) -> str:
    if score >= 1.0: return "STRONGLY BULLISH"
    if score >= 0.4: return "BULLISH"
    if score > -0.4: return "NEUTRAL / MIXED"
    if score > -1.0: return "BEARISH"
    return "STRONGLY BEARISH"


# ---------------------------------------------------------------- per-symbol stats
def symbol_stats(data: dict[str, pd.DataFrame], weights: dict | None = None,
                 include_indexes: bool = False) -> pd.DataFrame:
    """One row per stock/ETF (indexes excluded): moves, volatility, RSI, breadth flags, technical score."""
    rows = []
    for t, raw in data.items():
        if t.startswith("^") and not include_indexes:
            continue
        d = raw.dropna(subset=["Close"])
        if len(d) < 60:
            continue
        c, v = d["Close"], d["Volume"]
        r = c.pct_change()
        px = float(c.iloc[-1])

        def chg(n):
            return float(px / c.iloc[-1 - n] - 1) * 100 if len(c) > n else np.nan

        sc = technical_score_series(d, None, weights)
        score = float(sc.iloc[-1]) if len(sc) and not np.isnan(sc.iloc[-1]) else np.nan
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
        avg_vol = float(v.iloc[-21:-1].mean())
        prior = c.iloc[-21:-1]
        lo, hi = c.iloc[-252:].min(), c.iloc[-252:].max()
        rows.append({
            "Ticker": t, "Price": px, "1d %": chg(1), "5d %": chg(5), "1m %": chg(21), "3m %": chg(63),
            "Rel. volume": float(v.iloc[-1]) / avg_vol if avg_vol > 0 else np.nan,
            "RSI": float(rsi(c).iloc[-1]),
            "Volatility %": float(r.iloc[-20:].std() * np.sqrt(252) * 100),
            "Score": score, "Rating": (score + 2) / 4 * 100 if not np.isnan(score) else np.nan,
            "above50": bool(px > sma50) if not np.isnan(sma50) else np.nan,
            "above200": bool(px > sma200) if not np.isnan(sma200) else np.nan,
            "up_today": bool(r.iloc[-1] > 0),
            "new_high": bool(px >= prior.max()), "new_low": bool(px <= prior.min()),
            "pos52": float((px - lo) / (hi - lo)) if hi > lo else np.nan,
            "avg_vol": avg_vol,
            "Volume": float(v.iloc[-1]), "dollar_vol": px * float(v.iloc[-1]),
            "avg_dollar_vol": float((c * v).iloc[-21:-1].mean()),
        })
    df = pd.DataFrame(rows)
    return df.set_index("Ticker", drop=False) if not df.empty else df


def flags(row: pd.Series, vol_limit: float = 60.0) -> str:
    f = []
    if row["Volatility %"] >= vol_limit: f.append("⚠ High volatility")
    if row["RSI"] >= 70: f.append("Overbought")
    elif row["RSI"] <= 30: f.append("Oversold")
    if row["Rel. volume"] >= 2: f.append("Volume spike")
    if row["pos52"] >= 0.95: f.append("Near 52-wk high")
    elif row["pos52"] <= 0.05: f.append("Near 52-wk low")
    if abs(row["1d %"]) >= 5: f.append("Big move today")
    return " · ".join(f)


# ---------------------------------------------------------------- market regime (bullish / bearish)
def _closes(data, syms, tail=420) -> pd.DataFrame:
    cols = {t: data[t]["Close"] for t in syms if t in data and not data[t].empty}
    return pd.DataFrame(cols).sort_index().iloc[-tail:] if cols else pd.DataFrame()


def _trend_series(c: pd.Series) -> pd.Series:
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    out = np.where(c > s200, 1.0, -1.0) + np.where(s50 > s200, 1.0, -1.0)
    return pd.Series(out, index=c.index).where(s200.notna())


def _map(x: pd.Series, bounds: list[float], values: list[float], default: float) -> pd.Series:
    """Descending thresholds: first bound that x >= bound wins."""
    conds = [x >= b for b in bounds]
    return pd.Series(np.select(conds, values, default=default), index=x.index).where(x.notna())


def regime(data: dict[str, pd.DataFrame]) -> dict:
    """Score the market from -2 (strongly bearish) to +2 (strongly bullish), with history."""
    F = pd.DataFrame()
    raw = {}

    idx_syms = [s for s in ("^GSPC", "^IXIC", "^RUT") if s in data] or [s for s in ("SPY", "QQQ", "IWM") if s in data]
    trends = {s: _trend_series(data[s]["Close"].dropna()) for s in idx_syms}
    if trends:
        T = pd.DataFrame(trends)
        F["Index trend"] = T.mean(axis=1, skipna=True)
        raw["trends"] = T.dropna(how="all").iloc[-1].to_dict() if T.notna().any().any() else {}

    base = "^GSPC" if "^GSPC" in data else ("SPY" if "SPY" in data else None)
    if base:
        c = data[base]["Close"].dropna()
        s9, s20 = c.rolling(9).mean(), c.rolling(20).mean()
        F["Short-term momentum"] = pd.Series(np.where(c > s20, 1.0, -1.0) + np.where(s9 > s20, 1.0, -1.0),
                                             index=c.index).where(s20.notna())
        raw["base"] = (base, bool(c.iloc[-1] > s20.iloc[-1]), bool(s9.iloc[-1] > s20.iloc[-1]))

    stock_syms = [t for t in data if not t.startswith("^") and t not in ETF_SET]
    C = _closes(data, stock_syms)
    if not C.empty:
        s50, s200 = C.rolling(50).mean(), C.rolling(200).mean()
        ret = C.pct_change()
        v50, v200, vr = s50.notna(), s200.notna(), ret.notna()
        n50, n200, nr = v50.sum(axis=1), v200.sum(axis=1), vr.sum(axis=1)
        p50 = ((C > s50) & v50).sum(axis=1) / n50.where(n50 >= 15) * 100
        p200 = ((C > s200) & v200).sum(axis=1) / n200.where(n200 >= 15) * 100
        padv = ((ret > 0) & vr).sum(axis=1) / nr.where(nr >= 15) * 100
        F["Breadth: above 50-day"] = _map(p50, [75, 60, 40, 25], [2.0, 1.0, 0.0, -1.0], -2.0)
        F["Breadth: above 200-day"] = _map(p200, [75, 60, 45, 30], [2.0, 1.0, 0.0, -1.0], -2.0)
        F["Participation (up today)"] = _map(padv, [70, 58, 42, 30], [2.0, 1.0, 0.0, -1.0], -2.0)
        raw.update(p50=p50.dropna().iloc[-1] if p50.notna().any() else np.nan,
                   p200=p200.dropna().iloc[-1] if p200.notna().any() else np.nan,
                   padv=padv.dropna().iloc[-1] if padv.notna().any() else np.nan,
                   n=int(nr.iloc[-1]))

    if "^VIX" in data:
        v = data["^VIX"]["Close"].dropna()
        chg5 = v / v.shift(5) - 1
        lvl = np.select([v < 15, v < 20, v < 25, v < 30], [1.5, 0.5, -0.5, -1.5], default=-2.0)
        adj = np.select([chg5 < -0.15, chg5 > 0.25], [0.5, -0.5], default=0.0)
        F["Volatility (VIX)"] = pd.Series(np.clip(lvl + adj, -2, 2), index=v.index).where(chg5.notna())
        raw["vix"] = (float(v.iloc[-1]), float(chg5.iloc[-1]) if not np.isnan(chg5.iloc[-1]) else np.nan)

    E = _closes(data, OFFENSIVE + DEFENSIVE + ["IWM", "SPY"])
    off, dfn = [s for s in OFFENSIVE if s in E], [s for s in DEFENSIVE if s in E]
    if off and dfn and "IWM" in E and "SPY" in E:
        r21 = E / E.shift(21) - 1
        d1 = r21[off].mean(axis=1) - r21[dfn].mean(axis=1)
        d2 = r21["IWM"] - r21["SPY"]
        s1 = np.select([d1 > .03, d1 > .01, d1 < -.03, d1 < -.01], [1, .5, -1, -.5], default=0.0)
        s2 = np.select([d2 > .02, d2 > .005, d2 < -.02, d2 < -.005], [1, .5, -1, -.5], default=0.0)
        F["Risk appetite"] = pd.Series(s1 + s2, index=E.index).where(d1.notna() & d2.notna())
        raw["rot"] = (float(d1.iloc[-1]), float(d2.iloc[-1]))

    if F.empty:
        return {"ok": False}
    W = pd.Series({k: REGIME_WEIGHTS.get(k, 0.0) for k in F.columns})
    den = (F.notna() * W).sum(axis=1)
    score = ((F.fillna(0) * W).sum(axis=1) / den.where(den > 0)).where(den >= 0.5)
    last = score.last_valid_index()
    if last is None:
        return {"ok": False}
    s_now = float(score.loc[last])
    clean = score.dropna()
    prev = float(clean.iloc[-6]) if len(clean) >= 6 else np.nan
    delta = s_now - prev if not np.isnan(prev) else np.nan
    direction = ("Improving" if delta > 0.25 else "Deteriorating" if delta < -0.25 else "Stable") \
        if not np.isnan(delta) else "n/a"

    comps = []
    for name, wgt in REGIME_WEIGHTS.items():
        val = F[name].loc[last] if name in F and last in F.index else np.nan
        comps.append({"Factor": name, "available": not np.isnan(val), "score": None if np.isnan(val) else float(val),
                      "weight": wgt, "reason": _reason(name, raw, val)})
    return {"ok": True, "score": s_now, "pct": round((s_now + 2) / 4 * 100), "label": regime_label(s_now),
            "direction": direction, "delta": delta, "components": comps,
            "history": ((clean + 2) / 4 * 100).iloc[-90:], "breadth": raw, "asof": last}


def _reason(name: str, raw: dict, val) -> str:
    if np.isnan(val):
        return "Not enough data."
    if name == "Index trend":
        parts = ", ".join(f"{INDEXES.get(k, k)} {v:+.0f}" for k, v in raw.get("trends", {}).items() if not np.isnan(v))
        return f"Trend scores (-2 to +2, from 50/200-day averages): {parts}."
    if name == "Short-term momentum":
        sym, a, b = raw["base"]
        return (f"{INDEXES.get(sym, sym)} is {'above' if a else 'below'} its 20-day average; "
                f"9-day is {'above' if b else 'below'} 20-day.")
    if name == "Breadth: above 50-day":
        return f"{raw['p50']:.0f}% of {raw['n']} scanned stocks are above their 50-day average."
    if name == "Breadth: above 200-day":
        return f"{raw['p200']:.0f}% of scanned stocks are above their 200-day average."
    if name == "Participation (up today)":
        return f"{raw['padv']:.0f}% of scanned stocks are up today."
    if name == "Volatility (VIX)":
        v, c5 = raw["vix"]
        return f"VIX {v:.1f}" + (f" ({c5:+.0%} vs 5 sessions ago)." if not np.isnan(c5) else ".")
    if name == "Risk appetite":
        d1, d2 = raw["rot"]
        return (f"Cyclical sectors {d1:+.1%} vs defensive over 1 month; small caps {d2:+.1%} vs large caps.")
    return ""


# ---------------------------------------------------------------- volatility warnings
def volatility(data: dict[str, pd.DataFrame], stocks: pd.DataFrame) -> dict:
    out = {"ok": False, "warnings": []}
    base = "^GSPC" if "^GSPC" in data else ("SPY" if "SPY" in data else None)
    if base is None and "^VIX" not in data:
        return out
    vix = vchg = rv20 = today = z = None
    if "^VIX" in data:
        v = data["^VIX"]["Close"].dropna()
        if len(v) >= 6:
            vix, vchg = float(v.iloc[-1]), float(v.iloc[-1] / v.iloc[-2] - 1)
    if base:
        r = data[base]["Close"].dropna().pct_change()
        if len(r) > 25:
            rv20 = float(r.iloc[-20:].std() * np.sqrt(252) * 100)
            today = float(r.iloc[-1] * 100)
            avg_abs = float(r.iloc[-21:-1].abs().mean() * 100)
            z = abs(today) / avg_abs if avg_abs > 0 else None

    if vix is not None:
        lvl = 0 if vix < 15 else 1 if vix < 20 else 2 if vix < 25 else 3 if vix < 35 else 4
    elif rv20 is not None:
        lvl = 0 if rv20 < 10 else 1 if rv20 < 15 else 2 if rv20 < 22 else 3 if rv20 < 32 else 4
    else:
        return out
    if (vchg is not None and vchg > 0.12) or (rv20 is not None and rv20 > 25):
        lvl = min(4, lvl + 1)

    w = []
    if vix is not None and vix >= 30: w.append(f"VIX is {vix:.1f}, above 30: the market is under stress.")
    if vchg is not None and vchg >= 0.10: w.append(f"VIX jumped {vchg:+.0%} today: fear is rising fast.")
    if z is not None and z >= 2: w.append(f"The S&P 500 moved {today:+.1f}% today, {z:.1f}x its normal daily move.")
    if rv20 is not None and rv20 >= 25: w.append(f"Realized S&P 500 volatility is {rv20:.0f}% annualized, which is high.")
    if not stocks.empty:
        down = (stocks[~stocks.index.isin(ETF_SET)]["1d %"] <= -2).mean() * 100
        if down >= 25: w.append(f"{down:.0f}% of scanned stocks are down more than 2% today: broad selling.")
        hv = (stocks[~stocks.index.isin(ETF_SET)]["Volatility %"] >= 60).mean() * 100
        if hv >= 30: w.append(f"{hv:.0f}% of scanned stocks have 20-day volatility above 60%.")
    if vix is not None and vix < 13 and not w:
        w.append(f"VIX is very low at {vix:.1f}. Calm markets can turn quickly, so don't assume it lasts.")
    return {"ok": True, "level": LEVELS[lvl], "idx": lvl, "vix": vix, "vix_chg": vchg, "realized": rv20,
            "today": today, "z": z, "warnings": w}


# ---------------------------------------------------------------- business types
def group_table(stats: pd.DataFrame) -> pd.DataFrame:
    """One row per business type: how its stocks (and ETF) are doing and trending."""
    rows = []
    for name, g in GROUPS.items():
        s = stats[stats.index.isin(g["stocks"])]
        if s.empty:
            continue
        etf = stats.loc[g["etf"]] if g["etf"] and g["etf"] in stats.index else None
        sc = s["Score"].mean()
        row = {
            "Business type": name, "Stocks": len(s), "Signal": signal_label(sc),
            "Rating": (sc + 2) / 4 * 100 if not np.isnan(sc) else np.nan,
            "1D": s["1d %"].mean(), "5D": s["5d %"].mean(), "1M": s["1m %"].mean(), "3M": s["3m %"].mean(),
            "% above 50-day": s["above50"].astype(float).mean() * 100,
            "% up today": s["up_today"].mean() * 100,
            "Volatility %": s["Volatility %"].mean(),
            "Leader today": s.loc[s["1d %"].idxmax(), "Ticker"],
            "Laggard today": s.loc[s["1d %"].idxmin(), "Ticker"],
            "ETF": g["etf"] or "",
            "ETF 1D": float(etf["1d %"]) if etf is not None else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- volume flow
_CURVE_MIN = [0, 30, 60, 120, 180, 240, 300, 360, 390]               # minutes since the 9:30 ET open
_CURVE_CUM = [0, 0.14, 0.22, 0.37, 0.49, 0.60, 0.72, 0.88, 1.0]      # rough share of a day's volume traded so far


def session_fraction(now: pd.Timestamp, last_bar: pd.Timestamp) -> float:
    """Approximate share of a normal day's volume that has traded so far (naive Eastern times).
    1.0 when the session is over or the newest bar isn't today's."""
    if last_bar.normalize() != now.normalize() or now.weekday() >= 5:
        return 1.0
    mins = now.hour * 60 + now.minute - (9 * 60 + 30)
    if mins <= 0 or mins >= 390:
        return 1.0
    return max(0.05, float(np.interp(mins, _CURVE_MIN, _CURVE_CUM)))  # floor avoids absurd ratios at the open


def volume_breakdown(stats: pd.DataFrame, frac: float = 1.0) -> dict:
    """Dollars and shares traded today vs the 20-day average, by business type and by index ETF.
    Each stock is counted once, in the first business type it is listed under, so the steps add up."""
    empty = pd.DataFrame(columns=["name", "n", "dollars", "avg_dollars", "shares", "avg_shares"])
    if stats.empty:
        return {"industries": empty, "indexes": empty, "frac": frac}
    stocks = stats[~stats.index.isin(ETF_SET)].copy()
    stocks["group"] = stocks["Ticker"].map(lambda t: (GROUP_OF.get(t) or ["Other large caps"])[0])
    agg = stocks.groupby("group").agg(n=("Ticker", "count"), dollars=("dollar_vol", "sum"),
                                      avg_dollars=("avg_dollar_vol", "sum"), shares=("Volume", "sum"),
                                      avg_shares=("avg_vol", "sum")).reset_index().rename(columns={"group": "name"})
    etf = stats[stats.index.isin(INDEX_ETFS)].copy()
    idx = pd.DataFrame({"name": [INDEX_ETFS[t] for t in etf.index], "n": 1, "dollars": etf["dollar_vol"].to_numpy(),
                        "avg_dollars": etf["avg_dollar_vol"].to_numpy(), "shares": etf["Volume"].to_numpy(),
                        "avg_shares": etf["avg_vol"].to_numpy()}) if not etf.empty else empty
    return {"industries": agg, "indexes": idx, "frac": frac}


def compute_pulse(data: dict[str, pd.DataFrame], weights: dict | None = None, now: pd.Timestamp | None = None) -> dict:
    stats = symbol_stats(data, weights)
    stocks = stats[~stats.index.isin(ETF_SET)] if not stats.empty else stats
    out = {"stats": stats, "stocks": stocks, "loaded": len(data), "requested": len(all_symbols()),
           "regime": regime(data), "vol": volatility(data, stocks) if not stocks.empty else {"ok": False, "warnings": []}}
    out["groups"] = group_table(stats) if not stats.empty else pd.DataFrame()
    now = now if now is not None else pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    ref = data.get("SPY")
    if ref is None:
        ref = next(iter(data.values()), None)
    last_bar = ref.index.max() if ref is not None and len(ref) else now
    out["volume"] = volume_breakdown(stats, session_fraction(now, last_bar))
    return out
