"""Scoring engine: turns price history + fundamentals into a Buy / Hold / Sell signal.

Each factor returns a score from -2 (very bearish) to +2 (very bullish) plus a
plain-English reason. Factors are weighted and combined into a final score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Relative importance of each factor (normalised automatically). Treat as read-only:
# pass your own dict via the `weights` argument instead of mutating this one.
DEFAULT_WEIGHTS = {
    "Trend": 0.20,
    "Short-term trend": 0.10,
    "Momentum (RSI)": 0.10,
    "MACD": 0.10,
    "Bollinger": 0.05,
    "Relative strength vs market": 0.15,
    "Valuation": 0.10,
    "Growth & profitability": 0.10,
    "Analyst consensus": 0.10,
}
WEIGHTS = DEFAULT_WEIGHTS  # backwards-compatible alias


@dataclass
class Factor:
    name: str
    score: float  # -2..+2
    reason: str
    available: bool = True


# ---------- Indicators ----------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, window=20, k=2):
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return mid - k * sd, mid, mid + k * sd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["SMA9"] = c.rolling(9).mean()
    out["SMA20"] = c.rolling(20).mean()
    out["SMA50"] = c.rolling(50).mean()
    out["SMA200"] = c.rolling(200).mean()
    out["RSI"] = rsi(c)
    out["MACD"], out["MACD_signal"], out["MACD_hist"] = macd(c)
    out["BB_low"], out["BB_mid"], out["BB_high"] = bollinger(c)
    return out


# ---------- Factors ----------
def f_trend(df: pd.DataFrame) -> Factor:
    last = df.iloc[-1]
    if pd.isna(last["SMA200"]):
        return Factor("Trend", 0, "Not enough history for a 200-day average.", False)
    s, why = 0.0, []
    if last["Close"] > last["SMA200"]:
        s += 1; why.append("price is above the 200-day average (long-term uptrend)")
    else:
        s -= 1; why.append("price is below the 200-day average (long-term downtrend)")
    if last["SMA50"] > last["SMA200"]:
        s += 1; why.append("50-day is above 200-day (golden-cross regime)")
    else:
        s -= 1; why.append("50-day is below 200-day (death-cross regime)")
    return Factor("Trend", s, "; ".join(why).capitalize() + ".")


def f_short_trend(df: pd.DataFrame) -> Factor:
    last = df.iloc[-1]
    if pd.isna(last["SMA20"]):
        return Factor("Short-term trend", 0, "Not enough history for a 20-day average.", False)
    s, why = 0.0, []
    if last["Close"] > last["SMA20"]:
        s += 1; why.append("price is above the 20-day average")
    else:
        s -= 1; why.append("price is below the 20-day average")
    if last["SMA9"] > last["SMA20"]:
        s += 1; why.append("9-day is above 20-day (short-term momentum up)")
    else:
        s -= 1; why.append("9-day is below 20-day (short-term momentum down)")
    return Factor("Short-term trend", s, "; ".join(why).capitalize() + ".")


def f_rsi(df: pd.DataFrame) -> Factor:
    v = df["RSI"].iloc[-1]
    if pd.isna(v):
        return Factor("Momentum (RSI)", 0, "Not enough data.", False)
    if v >= 80: return Factor("Momentum (RSI)", -2, f"RSI {v:.0f}: extremely overbought.")
    if v >= 70: return Factor("Momentum (RSI)", -1, f"RSI {v:.0f}: overbought, pullback risk.")
    if v <= 20: return Factor("Momentum (RSI)", 2, f"RSI {v:.0f}: extremely oversold, bounce potential.")
    if v <= 30: return Factor("Momentum (RSI)", 1, f"RSI {v:.0f}: oversold.")
    if 50 <= v < 70: return Factor("Momentum (RSI)", 0.5, f"RSI {v:.0f}: healthy bullish momentum.")
    return Factor("Momentum (RSI)", -0.5, f"RSI {v:.0f}: soft momentum.")


def f_macd(df: pd.DataFrame) -> Factor:
    h, h_prev = df["MACD_hist"].iloc[-1], df["MACD_hist"].iloc[-2]
    if pd.isna(h):
        return Factor("MACD", 0, "Not enough data.", False)
    if h > 0 and h > h_prev: return Factor("MACD", 2, "Bullish and strengthening.")
    if h > 0: return Factor("MACD", 1, "Bullish but fading.")
    if h < 0 and h < h_prev: return Factor("MACD", -2, "Bearish and strengthening.")
    return Factor("MACD", -1, "Bearish but improving.")


def f_bollinger(df: pd.DataFrame) -> Factor:
    last = df.iloc[-1]
    if pd.isna(last["BB_high"]):
        return Factor("Bollinger", 0, "Not enough data.", False)
    width = last["BB_high"] - last["BB_low"]
    pos = (last["Close"] - last["BB_low"]) / width if width else 0.5
    if pos > 1: return Factor("Bollinger", -1, "Closed above the upper band (stretched).")
    if pos < 0: return Factor("Bollinger", 1, "Closed below the lower band (washed out).")
    return Factor("Bollinger", 0, f"Inside the bands ({pos:.0%} of the range).")


def f_relative_strength(df: pd.DataFrame, bench: pd.Series | None, days=126) -> Factor:
    if bench is None or len(df) <= days or len(bench) <= days:
        return Factor("Relative strength vs market", 0, "No benchmark data.", False)
    stock = df["Close"].iloc[-1] / df["Close"].iloc[-days] - 1
    mkt = bench.iloc[-1] / bench.iloc[-days] - 1
    diff = stock - mkt
    msg = f"6-month return {stock:+.1%} vs market {mkt:+.1%} ({diff:+.1%})."
    if diff > 0.15: return Factor("Relative strength vs market", 2, "Strongly outperforming. " + msg)
    if diff > 0.03: return Factor("Relative strength vs market", 1, "Outperforming. " + msg)
    if diff < -0.15: return Factor("Relative strength vs market", -2, "Badly lagging. " + msg)
    if diff < -0.03: return Factor("Relative strength vs market", -1, "Lagging. " + msg)
    return Factor("Relative strength vs market", 0, "In line. " + msg)


def f_valuation(info: dict) -> Factor:
    pe = info.get("forwardPE") or info.get("trailingPE")
    peg = info.get("pegRatio") or info.get("trailingPegRatio")
    if not pe or pe <= 0:
        return Factor("Valuation", 0, "No meaningful P/E (negative, missing, or not applicable).", False)
    s, why = 0.0, [f"P/E {pe:.1f}"]
    if pe < 15: s += 1
    elif pe > 35: s -= 1
    if peg and peg > 0:
        why.append(f"PEG {peg:.2f}")
        if peg < 1: s += 1
        elif peg > 2.5: s -= 1
    label = "looks cheap" if s > 0 else "looks expensive" if s < 0 else "fairly valued"
    return Factor("Valuation", s, ", ".join(why) + f": {label}.")


def f_growth(info: dict) -> Factor:
    rg, pm, de = info.get("revenueGrowth"), info.get("profitMargins"), info.get("debtToEquity")
    parts, s = [], 0.0
    if rg is not None:
        parts.append(f"revenue growth {rg:.0%}")
        s += 1 if rg > 0.10 else -1 if rg < 0 else 0
    if pm is not None:
        parts.append(f"profit margin {pm:.0%}")
        s += 1 if pm > 0.15 else -1 if pm < 0 else 0
    if de is not None:
        parts.append(f"debt/equity {de:.0f}%")
        s += -0.5 if de > 200 else 0
    if not parts:
        return Factor("Growth & profitability", 0, "No fundamentals available.", False)
    return Factor("Growth & profitability", max(-2, min(2, s)), ", ".join(parts).capitalize() + ".")


def f_analysts(info: dict, price: float) -> Factor:
    tgt, rec = info.get("targetMeanPrice"), info.get("recommendationMean")  # 1=strong buy..5=sell
    if not tgt and not rec:
        return Factor("Analyst consensus", 0, "No analyst data.", False)
    s, why = 0.0, []
    if tgt:
        up = tgt / price - 1
        why.append(f"mean target implies {up:+.0%}")
        s += 1 if up > 0.15 else -1 if up < -0.05 else 0
    if rec:
        why.append(f"rating {rec:.1f}/5 (1 = strong buy)")
        s += 1 if rec <= 2.0 else -1 if rec >= 3.5 else 0
    return Factor("Analyst consensus", max(-2, min(2, s)), ", ".join(why).capitalize() + ".")


# ---------- Combine ----------
def verdict_from_score(score: float) -> str:
    if score >= 1.0: return "STRONG BUY"
    if score >= 0.4: return "BUY"
    if score > -0.4: return "HOLD"
    if score > -1.0: return "SELL"
    return "STRONG SELL"


def analyze(df: pd.DataFrame, info: dict, bench: pd.Series | None = None,
            weights: dict | None = None) -> dict:
    w = weights or DEFAULT_WEIGHTS
    df = add_indicators(df)
    price = float(df["Close"].iloc[-1])
    factors = [
        f_trend(df), f_short_trend(df), f_rsi(df), f_macd(df), f_bollinger(df),
        f_relative_strength(df, bench), f_valuation(info), f_growth(info),
        f_analysts(info, price),
    ]
    live = [f for f in factors if f.available]
    live_w = sum(w.get(f.name, 0) for f in live)
    score = sum(f.score * w.get(f.name, 0) for f in live) / (live_w or 1)  # -2..+2
    pct = round((score + 2) / 4 * 100)  # 0..100
    verdict = verdict_from_score(score)

    # Confidence: how much of the weighting had data, and how much the factors agree.
    total_w = sum(w.get(f.name, 0) for f in factors) or 1
    coverage = live_w / total_w
    signs = [np.sign(f.score) for f in live if f.score != 0]
    agree = abs(sum(signs)) / len(signs) if signs else 0
    confidence = round(100 * (0.5 * coverage + 0.5 * agree))

    return {"df": df, "price": price, "factors": factors, "score": score,
            "score_pct": pct, "verdict": verdict, "confidence": confidence, "weights": w}


def technical_score_series(df: pd.DataFrame, bench: pd.Series | None = None,
                           weights: dict | None = None, rs_days: int = 126) -> pd.Series:
    """Daily score (-2..+2) using only the technical factors.

    Vectorised twin of the single-day factors above. Every value at day t uses
    data up to and including day t only, so it is safe for backtesting.
    Fundamentals and analyst data are excluded on purpose: we only have today's
    snapshot of those, and using them on past dates would be look-ahead bias.
    """
    w = weights or DEFAULT_WEIGHTS
    d = add_indicators(df)
    c = d["Close"]
    F = pd.DataFrame(index=d.index)

    F["Trend"] = np.where(c > d["SMA200"], 1.0, -1.0) + np.where(d["SMA50"] > d["SMA200"], 1.0, -1.0)
    F.loc[d["SMA200"].isna(), "Trend"] = np.nan

    F["Short-term trend"] = np.where(c > d["SMA20"], 1.0, -1.0) + np.where(d["SMA9"] > d["SMA20"], 1.0, -1.0)
    F.loc[d["SMA20"].isna(), "Short-term trend"] = np.nan

    v = d["RSI"]
    F["Momentum (RSI)"] = np.select([v >= 80, v >= 70, v <= 20, v <= 30, v >= 50],
                                    [-2.0, -1.0, 2.0, 1.0, 0.5], default=-0.5)
    F.loc[v.isna(), "Momentum (RSI)"] = np.nan

    h = d["MACD_hist"]
    hp = h.shift(1)
    F["MACD"] = np.select([(h > 0) & (h > hp), h > 0, (h < 0) & (h < hp)], [2.0, 1.0, -2.0], default=-1.0)
    F.loc[h.isna(), "MACD"] = np.nan

    width = d["BB_high"] - d["BB_low"]
    pos = ((c - d["BB_low"]) / width.replace(0, np.nan)).fillna(0.5)
    F["Bollinger"] = np.select([pos > 1, pos < 0], [-1.0, 1.0], default=0.0)
    F.loc[d["BB_high"].isna(), "Bollinger"] = np.nan

    rs = "Relative strength vs market"
    if bench is not None:
        b = bench.reindex(d.index).ffill()
        diff = (c / c.shift(rs_days - 1) - 1) - (b / b.shift(rs_days - 1) - 1)
        F[rs] = np.select([diff > 0.15, diff > 0.03, diff < -0.15, diff < -0.03],
                          [2.0, 1.0, -2.0, -1.0], default=0.0)
        F.loc[diff.isna(), rs] = np.nan
        F.iloc[:rs_days, F.columns.get_loc(rs)] = np.nan
    else:
        F[rs] = np.nan

    W = pd.Series({k: w.get(k, 0.0) for k in F.columns})
    den = (F.notna() * W).sum(axis=1)
    score = (F.fillna(0) * W).sum(axis=1) / den.where(den > 0)
    score[F["Trend"].isna()] = np.nan  # wait for the 200-day average before trusting any signal
    return score
