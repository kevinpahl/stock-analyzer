"""Breakout watch: look for stocks / ETFs / coins that are *set up* to run, before they run.

Two stages, so a scan of ~100 names stays fast:
  1. Price & volume setup, computed for the whole universe from one batch of daily history. This is a vectorised
     time series, so the same code also powers the historical check ("did high scores actually precede runs?").
  2. Catalysts for the best-looking names only: news tone, earnings track record, analyst upgrades and targets,
     insider buying and short interest, all from Yahoo Finance via yfinance.

Nothing here predicts the future. It lists conditions that often come before a move, shows its reasons, and
reports how well the price setup would have worked historically so you can judge it yourself.
"""
from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

from screener import asset_type

# --------------------------------------------------------------------------------------
# 1. Price & volume setup
# --------------------------------------------------------------------------------------
SETUP_WEIGHTS = {"coil": 0.25, "high": 0.20, "trend": 0.20, "rs": 0.20, "accum": 0.15}
SETUP_LABELS = {"coil": "Volatility squeeze", "high": "Near 52-week high", "trend": "Uptrend",
                "rs": "Beating the market", "accum": "Buying pressure"}


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def setup_frame(df: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    """Daily setup components (each 0..1) and a 0..100 score, using only data up to each day (no look-ahead)."""
    d = df.dropna(subset=["Close"])
    c, h, v = d["Close"], d["High"], d["Volume"].fillna(0)
    out = pd.DataFrame(index=d.index)

    ma20, ma50, ma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    width = 4 * c.rolling(20).std() / ma20                                  # Bollinger band width
    pct_rank = width.rolling(126, min_periods=60).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    out["coil"] = _clip01(1 - pct_rank)                                     # 1 = tightest in ~6 months

    hi252 = h.rolling(252, min_periods=120).max()
    out["high"] = _clip01((c / hi252 - 0.85) / 0.15)                        # 85% of the high -> 0, at the high -> 1

    out["trend"] = ((c > ma50).astype(float) + (ma50 > ma200).astype(float)
                    + (ma50 > ma50.shift(10)).astype(float)) / 3
    out.loc[ma200.isna(), "trend"] = np.nan

    if bench is not None and len(bench):
        b = bench.reindex(d.index).ffill()
        rel = c.pct_change(63) - b.pct_change(63)
        out["rs"] = _clip01((rel + 0.05) / 0.20)                            # -5% vs market -> 0, +15% -> 1
    else:
        out["rs"] = np.nan

    chg = c.diff()
    upv = v.where(chg > 0, 0).rolling(20).sum()
    dnv = v.where(chg < 0, 0).rolling(20).sum()
    ratio = upv / dnv.replace(0, np.nan)
    out["accum"] = _clip01((ratio - 0.8) / 0.8) if v.sum() > 0 else np.nan  # up-day volume vs down-day volume
    if v.sum() > 0:
        out.loc[dnv.eq(0) & upv.gt(0), "accum"] = 1.0

    w = pd.Series(SETUP_WEIGHTS)
    avail = out[list(w.index)].notna()
    raw = (out[list(w.index)].fillna(0) * w).sum(axis=1) / (avail * w).sum(axis=1).replace(0, np.nan)

    ext = c / ma50 - 1
    rsi = _rsi(c)
    penalty = (_clip01((ext - 0.12) / 0.20) * 0.45 + _clip01((rsi - 72) / 18) * 0.25)  # stretched = may have run
    out["score"] = (raw * (1 - penalty) * 100).clip(0, 100)

    prior_high = h.shift(1).rolling(20).max()
    avg_v = v.shift(1).rolling(20).mean()
    out["relvol"] = v / avg_v.replace(0, np.nan)
    out["breakout"] = (c > prior_high) & (out["relvol"] >= 1.5)
    out["ext50"] = ext
    out["rsi"] = rsi
    out["from_high"] = c / hi252 - 1
    out["ret20"] = c.pct_change(20)
    out["ret5"] = c.pct_change(5)
    out["ret1"] = c.pct_change(1)
    out["tight"] = (h.rolling(10).max() - d["Low"].rolling(10).min()) / c       # 10-day range as % of price
    return out


def stage_of(row: pd.Series) -> str:
    if bool(row.get("breakout")) and row.get("ext50", 0) < 0.15:
        return "Breaking out now"
    if row.get("ext50", 0) > 0.15 or row.get("rsi", 50) > 75:
        return "Extended (may have already run)"
    if row.get("coil", 0) >= 0.7 and row.get("high", 0) >= 0.7:
        return "Coiled near highs"
    if row.get("coil", 0) >= 0.6 and row.get("trend", 0) >= 0.6:
        return "Basing in an uptrend"
    if row.get("trend", 0) >= 0.6 and row.get("high", 0) >= 0.6:
        return "Trending up"
    return "Not set up"


def setup_reasons(row: pd.Series) -> tuple[list[str], list[str]]:
    good, bad = [], []
    if row.get("coil", 0) >= 0.7:
        good.append(f"Volatility squeeze: price swings are among the tightest of the last 6 months "
                    f"(10-day range just {row.get('tight', float('nan')):.1%} of price). Tight ranges often come "
                    f"before a larger move, in either direction.")
    if row.get("high", 0) >= 0.8:
        good.append(f"Within {abs(row.get('from_high', 0)):.1%} of its 52-week high, with no overhead resistance "
                    f"from trapped buyers above.")
    if row.get("trend", 0) >= 0.99:
        good.append("Clean uptrend: price above its 50-day average, which is above the 200-day and still rising.")
    elif row.get("trend", 0) >= 0.6:
        good.append("Mostly in an uptrend (two of three trend checks pass).")
    if row.get("rs", 0) >= 0.7:
        good.append("Beating the market over the last 3 months (relative strength).")
    if row.get("accum", 0) >= 0.7:
        good.append("More volume on up days than down days lately, a sign of steady buying.")
    if bool(row.get("breakout")):
        good.append(f"Just closed above its 20-day high on {row.get('relvol', 0):.1f}x normal volume (a confirmed "
                    f"breakout).")
    elif row.get("relvol", 0) >= 1.8:
        good.append(f"Volume is {row.get('relvol', 0):.1f}x normal today: someone is paying attention.")
    if row.get("ext50", 0) > 0.15:
        bad.append(f"Stretched: {row['ext50']:.0%} above its 50-day average. The run may already be underway.")
    if row.get("rsi", 50) > 75:
        bad.append(f"RSI {row['rsi']:.0f}: overbought, so a pullback is common.")
    if row.get("trend", 1) < 0.34:
        bad.append("Below its long-term trend averages. A breakout from here is lower-odds.")
    if row.get("rs", 1) < 0.25:
        bad.append("Lagging the market over the last 3 months.")
    return good, bad


# --------------------------------------------------------------------------------------
# 2. News tone
# --------------------------------------------------------------------------------------
_POS = {
    r"\bupgrad(e|ed|es)\b": 2.0, r"\braises? (its |full-year |annual )?(guidance|forecast|outlook|target)": 2.5,
    r"\b(beats?|tops?|crush(es|ed)|smash(es|ed)|exceeds?) (analyst |wall street |earnings |profit |revenue |sales )?"
    r"(estimates?|expectations?|forecasts?|views?|consensus)": 2.5,
    r"\brecord (revenue|sales|earnings|profit|quarter|high|growth|demand|bookings|orders)": 2.0,
    r"\b(fda|regulatory|ema) (approv|clearance)": 3.0, r"\bapproval\b": 1.0,
    r"\b(wins?|awarded|secures?|lands?|signs?) .{0,30}(contract|deal|order|agreement|partnership)": 2.0,
    r"\bpartnership\b": 1.0, r"\bbuy ?back|repurchase program|share repurchase": 1.5,
    r"\bacquisition target|takeover (bid|offer|interest)|buyout offer|to be acquired": 2.5,
    r"\b(strong|robust|surging|soaring|accelerating) (demand|growth|sales|results|earnings|orders)": 1.5,
    r"\b(breakout|breaks out|all-time high|new high|52-week high)\b": 1.0,
    r"\binitiat(es|ed) (coverage )?.{0,20}(buy|outperform|overweight)": 2.0,
    r"\b(price target) (raised|hiked|boosted|lifted|increased)": 2.0,
    r"\b(dividend) (hike|increase|raised)": 1.0, r"\blaunch(es|ed)? (new|its)": 0.7,
    r"\bbullish\b": 1.0, r"\boutperform\b": 1.0, r"\bbeat\b": 1.2, r"\bsoars?|jumps?|surges?|rall(y|ies)\b": 0.8,
    r"\bprofit (jump|surge|soar|rise)": 1.5, r"\bstake (increase|boost)|increases? stake|buys? shares": 1.0,
}
_NEG = {
    r"\bdowngrad(e|ed|es)\b": -2.0, r"\b(cuts?|lowers?|slashes?|trims?) (its |full-year |annual )?"
    r"(guidance|forecast|outlook|target)": -2.5,
    r"\b(miss(es|ed)?|falls? short of|lags?) (analyst |wall street |earnings |profit |revenue |sales )?"
    r"(estimates?|expectations?|forecasts?|views?|consensus)": -2.5,
    r"\b(sec|doj|ftc) (probe|investigation|subpoena|charges?)|\binvestigation\b": -2.0,
    r"\b(lawsuit|sued|class action|fraud|indict)": -1.5, r"\brecall\b": -1.5,
    r"\b(offering|share sale|dilution|convertible notes?)\b": -1.5,
    r"\b(layoffs?|job cuts|restructuring|bankruptcy|chapter 11|going concern)\b": -2.0,
    r"\b(plunges?|tumbles?|sinks?|slumps?|crash(es)?|plummets?)\b": -1.2,
    r"\bwarns?|warning\b": -1.5, r"\bhalts?\b|\bdelist": -1.5,
    r"\b(bearish|underperform|sell rating)\b": -1.0, r"\bweak (demand|results|sales|guidance)": -1.5,
    r"\b(ceo|cfo) (resigns?|steps down|departs?|exits?)": -1.5,
}
_POS_RE = [(re.compile(p, re.I), w) for p, w in _POS.items()]
_NEG_RE = [(re.compile(p, re.I), w) for p, w in _NEG.items()]
_TAGS = [("Upgrade", r"upgrad"), ("Guidance raised", r"rais\w+ .{0,20}(guidance|outlook|forecast)"),
         ("Beat", r"\b(beat|beats|tops|exceeds|crush)"), ("Contract / deal", r"contract|deal|order|partnership"),
         ("Approval", r"approv|clearance"), ("Buyback", r"buy ?back|repurchase"), ("Takeover", r"takeover|acqui|buyout"),
         ("Downgrade", r"downgrad"), ("Guidance cut", r"(cuts?|lowers?|slashes?).{0,20}(guidance|outlook|forecast)"),
         ("Miss", r"\b(miss|misses|falls short)"), ("Probe / lawsuit", r"probe|investigat|lawsuit|sued|fraud"),
         ("Offering", r"offering|dilution|share sale")]


def score_headline(text: str) -> tuple[float, list[str]]:
    """Positive/negative tone of one headline (+ summary) with a few catalyst tags. Simple keyword rules,
    so treat it as a pointer to read the article, not as a verdict."""
    text = text or ""
    s = sum(w for rx, w in _POS_RE if rx.search(text)) + sum(w for rx, w in _NEG_RE if rx.search(text))
    tags = [name for name, rx in _TAGS if re.search(rx, text, re.I)]
    return max(-4.0, min(4.0, s)), tags


def parse_news(raw: list, now: float | None = None) -> list[dict]:
    """Normalise yfinance's news (schema differs between versions) into plain dicts."""
    now = now or time.time()
    out = []
    for n in raw or []:
        c = n.get("content", n) if isinstance(n, dict) else {}
        title = c.get("title") or ""
        if not title:
            continue
        summary = c.get("summary") or c.get("description") or ""
        link = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                or n.get("link") or "")
        src = (c.get("provider") or {}).get("displayName") or n.get("publisher") or ""
        ts = None
        pub = c.get("pubDate") or c.get("displayTime")
        if pub:
            try:
                ts = pd.Timestamp(pub).timestamp()
            except Exception:
                ts = None
        if ts is None and n.get("providerPublishTime"):
            ts = float(n["providerPublishTime"])
        age = max(0.0, (now - ts) / 86400) if ts else 7.0
        s, tags = score_headline(f"{title}. {summary}")
        out.append({"title": title, "link": link, "source": src, "age_days": age, "tone": s, "tags": tags})
    return out


def news_signal(items: list[dict]) -> dict:
    """0..100 (50 = neutral) from recency-weighted headline tone, plus buzz (fresh article count)."""
    if not items:
        return {"score": None, "reasons": [], "risks": [], "n": 0, "buzz": 0}
    tot = sum(i["tone"] * 0.5 ** (i["age_days"] / 3) for i in items)
    score = 50 + 50 * math.tanh(tot / 4)
    fresh = sum(1 for i in items if i["age_days"] <= 3)
    good, bad = [], []
    pos = sorted([i for i in items if i["tone"] >= 1.5], key=lambda i: i["age_days"])[:2]
    neg = sorted([i for i in items if i["tone"] <= -1.5], key=lambda i: i["age_days"])[:2]
    for i in pos:
        good.append(f"Positive news ({i['age_days']:.0f}d ago): {i['title']}")
    for i in neg:
        bad.append(f"Negative news ({i['age_days']:.0f}d ago): {i['title']}")
    if fresh >= 6 and not bad:
        good.append(f"News buzz: {fresh} articles in the last 3 days. Attention often arrives before a move.")
    return {"score": float(score), "reasons": good, "risks": bad, "n": len(items), "buzz": fresh}


# --------------------------------------------------------------------------------------
# 3. Earnings, analysts, insiders, short interest
# --------------------------------------------------------------------------------------
def _num(x):
    try:
        f = float(x)
        return f if f == f else None
    except Exception:
        return None


def earnings_signal(edates: pd.DataFrame | None, info: dict, now: pd.Timestamp | None = None) -> dict:
    """Beat streak and growth from past reports; flags a report coming up soon."""
    now = now or pd.Timestamp.now()
    reasons, risks, parts = [], [], []
    next_date, beats, surprises = None, 0, []
    if edates is not None and len(edates):
        e = edates.copy()
        e.index = pd.to_datetime(e.index, errors="coerce")
        try:
            e.index = e.index.tz_localize(None)
        except Exception:
            pass
        e = e[~e.index.isna()].sort_index()
        fut = e[e.index > now]
        if len(fut):
            next_date = fut.index[0]
        col = next((c for c in e.columns if "surprise" in str(c).lower()), None)
        rep = e[e.index <= now]
        if col is not None:
            s = pd.to_numeric(rep[col], errors="coerce").dropna().tail(4)
            surprises = list(s.values)
            beats = int((s > 0).sum())
            if len(s) >= 2:
                avg = float(s.mean())
                score = 50 + (beats - len(s) / 2) * 14 + max(-10, min(20, avg)) * 1.2
                parts.append(score)
                if beats == len(s) and avg > 0:
                    reasons.append(f"Beat earnings estimates in each of the last {len(s)} quarters "
                                   f"(average surprise {avg:+.1f}%).")
                elif beats >= len(s) - 1 and avg > 3:
                    reasons.append(f"Beat estimates in {beats} of the last {len(s)} quarters (average surprise {avg:+.1f}%).")
                elif beats <= 1:
                    risks.append(f"Missed estimates in {len(s) - beats} of the last {len(s)} quarters.")
    eg = _num(info.get("earningsQuarterlyGrowth"))
    rg = _num(info.get("revenueGrowth"))
    g = [x for x in (eg, rg) if x is not None]
    if g:
        gs = 50 + max(-30, min(40, 100 * sum(g) / len(g))) * 1.0
        parts.append(gs)
        if rg is not None and rg > 0.15:
            reasons.append(f"Revenue growing {rg:.0%} year over year.")
        if eg is not None and eg > 0.2:
            reasons.append(f"Quarterly earnings up {eg:.0%} year over year.")
        if rg is not None and rg < -0.05:
            risks.append(f"Revenue shrinking {rg:.0%} year over year.")
    days = None
    if next_date is not None:
        days = (next_date.normalize() - now.normalize()).days
        if 0 <= days <= 14:
            risks.append(f"Earnings report in {days} day(s) ({next_date:%b %d}). That is a big catalyst, but it "
                         f"cuts both ways: the stock can gap down as easily as up.")
    score = float(np.clip(np.mean(parts), 0, 100)) if parts else None
    return {"score": score, "reasons": reasons, "risks": risks, "next_date": next_date, "days_to": days,
            "beats": beats, "quarters": len(surprises)}


def analyst_signal(updown: pd.DataFrame | None, info: dict, price: float, now: pd.Timestamp | None = None) -> dict:
    """Recent upgrades vs downgrades, plus price-target upside and consensus."""
    now = now or pd.Timestamp.now()
    reasons, risks, parts = [], [], []
    ups = downs = 0
    if updown is not None and len(updown):
        u = updown.copy()
        u.index = pd.to_datetime(u.index, errors="coerce")
        try:
            u.index = u.index.tz_localize(None)
        except Exception:
            pass
        u = u[u.index >= now - pd.Timedelta(days=45)]
        act = u["Action"].astype(str).str.lower() if "Action" in u else pd.Series([], dtype=str)
        ups = int(act.isin(["up", "upgrade"]).sum())
        downs = int(act.isin(["down", "downgrade"]).sum())
        if "ToGrade" in u:  # new coverage with a buy-type rating counts as a mild positive
            ini = u[act.isin(["init", "initiated"])]
            ups += int(ini["ToGrade"].astype(str).str.lower().str.contains("buy|outperform|overweight").sum())
        parts.append(50 + max(-3, min(3, ups - downs)) * 15)
        if ups - downs >= 1:
            reasons.append(f"{ups} analyst upgrade(s) vs {downs} downgrade(s) in the last 45 days.")
        elif downs - ups >= 1:
            risks.append(f"{downs} analyst downgrade(s) vs {ups} upgrade(s) in the last 45 days.")
    tgt = _num(info.get("targetMeanPrice"))
    if tgt and price:
        up = tgt / price - 1
        parts.append(50 + max(-25, min(35, up * 100)) * 1.2)
        n = int(_num(info.get("numberOfAnalystOpinions")) or 0)
        if up >= 0.20 and n >= 5:
            reasons.append(f"Average analyst price target is {up:+.0%} above the current price ({n} analysts).")
        elif up <= -0.05:
            risks.append(f"Price is already above the average analyst target ({up:+.0%}).")
    rec = _num(info.get("recommendationMean"))
    if rec:
        parts.append(50 + (3 - rec) / 2 * 40)
    score = float(np.clip(np.mean(parts), 0, 100)) if parts else None
    return {"score": score, "reasons": reasons, "risks": risks, "ups": ups, "downs": downs, "target": tgt}


def insider_signal(tx: pd.DataFrame | None, now: pd.Timestamp | None = None) -> dict:
    """Open-market insider buying in the last 90 days. Sales are weak evidence (people sell for many reasons)."""
    now = now or pd.Timestamp.now()
    if tx is None or not len(tx):
        return {"score": None, "reasons": [], "risks": [], "buyers": 0, "buy_value": 0.0}
    t = tx.copy()
    dcol = next((c for c in t.columns if "start" in str(c).lower() or "date" in str(c).lower()), None)
    if dcol is not None:
        t[dcol] = pd.to_datetime(t[dcol], errors="coerce")
        try:
            t[dcol] = t[dcol].dt.tz_localize(None)
        except Exception:
            pass
        t = t[t[dcol] >= now - pd.Timedelta(days=90)]
    if not len(t):
        return {"score": None, "reasons": [], "risks": [], "buyers": 0, "buy_value": 0.0}
    text = (t["Text"].astype(str) if "Text" in t else pd.Series([""] * len(t), index=t.index)).str.lower()
    if "Transaction" in t:
        text = text + " " + t["Transaction"].astype(str).str.lower()
    skip = text.str.contains("gift|award|grant|exercise|conversion|option")
    buy = text.str.contains("purchase|buy|bought") & ~skip
    sell = text.str.contains("sale|sold|sell") & ~skip
    val = pd.to_numeric(t["Value"], errors="coerce").fillna(0) if "Value" in t else pd.Series(0.0, index=t.index)
    who = t["Insider"].astype(str) if "Insider" in t else pd.Series(range(len(t)), index=t.index).astype(str)
    buyers = int(who[buy].nunique())
    buy_val, sell_val = float(val[buy].sum()), float(val[sell].sum())
    reasons, risks = [], []
    score = None
    if buyers:
        score = float(min(100, 60 + 12 * buyers + (8 if buy_val >= 250_000 else 0)))
        reasons.append(f"{buyers} insider(s) bought shares on the open market in the last 90 days"
                       + (f" (about ${buy_val:,.0f})" if buy_val else "") + ". Insiders buy for one reason: "
                       "they expect the price to rise."
                       + (" Several buying together is a stronger signal." if buyers >= 2 else ""))
    elif sell.any():
        score = 42.0
        if sell_val >= 5_000_000:
            risks.append(f"Insiders sold about ${sell_val:,.0f} in the last 90 days (often routine, but worth a look).")
    return {"score": score, "reasons": reasons, "risks": risks, "buyers": buyers, "buy_value": buy_val}


def short_signal(info: dict, setup_ok: bool) -> dict:
    """High short interest is fuel for a squeeze, but only matters if the chart is already turning up."""
    spf = _num(info.get("shortPercentOfFloat"))
    if spf is None:
        return {"score": None, "reasons": [], "risks": [], "pct": None}
    spf = spf * 100 if spf < 1.5 else spf  # fraction vs already-percent
    dtc = _num(info.get("shortRatio"))
    reasons, risks = [], []
    if spf >= 12 and setup_ok:
        score = float(min(85, 55 + (spf - 12) * 1.5))
        reasons.append(f"{spf:.0f}% of shares are sold short"
                       + (f" ({dtc:.1f} days to cover)" if dtc else "")
                       + " while the chart is firming. If it breaks out, shorts buying back can speed the move up.")
    elif spf >= 12:
        score = 50.0
        risks.append(f"{spf:.0f}% of the float is sold short: many traders are betting against it.")
    else:
        score = 50.0
    return {"score": score, "reasons": reasons, "risks": risks, "pct": spf}


# --------------------------------------------------------------------------------------
# 4. Fetching (network) and combining
# --------------------------------------------------------------------------------------
def fetch_catalysts(sym: str) -> dict:
    """All the slow per-symbol lookups. Each one is optional: a failure just means that signal is skipped."""
    out: dict = {"info": {}, "news": [], "edates": None, "updown": None, "insiders": None}
    t = yf.Ticker(sym)
    kind = asset_type(sym)
    for key, fn in (("info", lambda: t.info or {}), ("news", lambda: t.news or [])):
        try:
            out[key] = fn()
        except Exception:
            pass
    if kind == "Stock":
        for key, fn in (("edates", lambda: t.get_earnings_dates(limit=8)),
                        ("updown", lambda: t.upgrades_downgrades),
                        ("insiders", lambda: t.insider_transactions)):
            try:
                out[key] = fn()
            except Exception:
                pass
    return out


def fetch_many(symbols: list[str], workers: int = 8) -> dict[str, dict]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(zip(symbols, ex.map(fetch_catalysts, symbols)))


CAT_WEIGHTS = {"Setup": 0.45, "News": 0.15, "Earnings": 0.12, "Analysts": 0.13, "Insiders": 0.10, "Short interest": 0.05}


def combine(sym: str, row: pd.Series, cat: dict | None, price: float) -> dict:
    """One symbol's final breakout score, stage, reasons and risks."""
    good, bad = setup_reasons(row)
    comps = {"Setup": float(row["score"])}
    detail: dict = {}
    if cat is not None:
        info = cat.get("info") or {}
        news = parse_news(cat.get("news"))
        detail["news_items"] = news
        sigs = {
            "News": news_signal(news),
            "Earnings": earnings_signal(cat.get("edates"), info),
            "Analysts": analyst_signal(cat.get("updown"), info, price),
            "Insiders": insider_signal(cat.get("insiders")),
            "Short interest": short_signal(info, bool(row["score"] >= 55)),
        }
        for name, s in sigs.items():
            if s.get("score") is not None:
                comps[name] = s["score"]
            good += s.get("reasons", [])
            bad += s.get("risks", [])
        detail.update({"signals": sigs, "info": info})
    w = {k: CAT_WEIGHTS[k] for k in comps}
    total = sum(comps[k] * w[k] for k in comps) / sum(w.values())
    # the price setup gates everything: great news on a broken chart should not rank high
    if row["score"] < 40:
        total = min(total, 59.0)
    stage = stage_of(row)
    if stage == "Extended (may have already run)":
        total = min(total, 69.0)
    catalysts = []
    sg = detail.get("signals", {})
    if sg.get("News", {}).get("score") is not None and sg["News"]["score"] >= 62:
        catalysts.append("News +")
    if sg.get("News", {}).get("score") is not None and sg["News"]["score"] <= 38:
        catalysts.append("News −")
    ed = sg.get("Earnings", {})
    if ed.get("quarters") and ed["beats"] == ed["quarters"]:
        catalysts.append(f"Beat ×{ed['beats']}")
    if sg.get("Analysts", {}).get("ups", 0) - sg.get("Analysts", {}).get("downs", 0) >= 1:
        catalysts.append(f"Upgrades {sg['Analysts']['ups']}")
    if sg.get("Insiders", {}).get("buyers"):
        catalysts.append(f"Insider buy ×{sg['Insiders']['buyers']}")
    if sg.get("Short interest", {}).get("reasons"):
        catalysts.append("Squeeze fuel")
    if ed.get("days_to") is not None and 0 <= ed["days_to"] <= 14:
        catalysts.append(f"Earnings {ed['days_to']}d")
    return {"score": float(total), "stage": stage, "comps": comps, "good": good, "bad": bad,
            "catalysts": ", ".join(catalysts) or "none found", "detail": detail,
            "signals_used": len(comps), "next_earnings": ed.get("next_date")}


def rating(score: float) -> str:
    return "Strong candidate" if score >= 75 else "Watch" if score >= 62 else "Weak"


def scan(data: dict[str, pd.DataFrame], bench: pd.Series | None, types: list | None, min_price: float,
         min_avg_volume: float, research_n: int, fetcher=fetch_many) -> tuple[pd.DataFrame, dict, dict]:
    """Stage 1 on everything, stage 2 on the best `research_n`. Returns (table, details, stats)."""
    rows, last = {}, {}
    skipped = 0
    for sym, raw in data.items():
        d = raw.dropna(subset=["Close"])
        kind = asset_type(sym)
        if len(d) < 220 or (types and kind not in types) or kind in ("Index", "Future", "Currency"):
            skipped += 1
            continue
        avg_vol = float(d["Volume"].iloc[-21:-1].mean())
        if kind in ("Stock", "ETF") and (float(d["Close"].iloc[-1]) < min_price or not avg_vol >= min_avg_volume):
            skipped += 1
            continue
        f = setup_frame(d, bench if kind in ("Stock", "ETF") else None)
        if f["score"].iloc[-1] != f["score"].iloc[-1]:
            skipped += 1
            continue
        last[sym] = f.iloc[-1]
        rows[sym] = (kind, float(d["Close"].iloc[-1]))
    stats = {"screened": len(data), "skipped": skipped, "setup_scored": len(last), "researched": 0}
    if not last:
        return pd.DataFrame(), {}, stats

    pre = sorted(last, key=lambda s: last[s]["score"], reverse=True)[:research_n]
    cats = fetcher(pre) if pre else {}
    stats["researched"] = len(pre)

    out, details = [], {}
    for sym in pre:
        row = last[sym]
        kind, price = rows[sym]
        r = combine(sym, row, cats.get(sym), price)
        details[sym] = r
        nxt = r["next_earnings"]
        out.append({"Ticker": sym, "Type": kind, "Rating": rating(r["score"]), "Breakout score": r["score"],
                    "Stage": r["stage"], "Price": price, "1d %": row["ret1"] * 100, "20d %": row["ret20"] * 100,
                    "From 52w high %": row["from_high"] * 100, "Rel. volume": row["relvol"],
                    "Catalysts": r["catalysts"], "Next earnings": (f"{nxt:%b %d}" if nxt is not None else ""),
                    "Signals": f"{r['signals_used']}/6"})
    table = pd.DataFrame(out).sort_values("Breakout score", ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", table.index + 1)
    return table, details, stats


# --------------------------------------------------------------------------------------
# 5. Historical check of the price setup
# --------------------------------------------------------------------------------------
def setup_history(data: dict[str, pd.DataFrame], bench: pd.Series | None, horizon: int = 20, run_pct: float = 10.0,
                  step: int = 5) -> dict:
    """For every symbol and every `step`-th day: setup score then, versus what happened over the next `horizon`
    trading days. Answers "when the setup score was high, did prices actually run more often?" """
    recs = []
    for sym, raw in data.items():
        d = raw.dropna(subset=["Close"])
        kind = asset_type(sym)
        if len(d) < 320 or kind in ("Index", "Future", "Currency"):
            continue
        f = setup_frame(d, bench if kind in ("Stock", "ETF") else None)
        c = d["Close"]
        fwd_max = pd.concat([c.shift(-k) for k in range(1, horizon + 1)], axis=1).max(axis=1) / c - 1
        fwd_end = c.shift(-horizon) / c - 1
        m = pd.DataFrame({"score": f["score"], "run": fwd_max * 100, "ret": fwd_end * 100}).iloc[260:-horizon:step]
        m = m.dropna()
        if len(m):
            recs.append(m)
    if not recs:
        return {"ok": False}
    allm = pd.concat(recs)
    bins = [(0, 40, "Below 40"), (40, 55, "40 – 55"), (55, 70, "55 – 70"), (70, 101, "70 and above")]
    rows = []
    for lo, hi, name in bins:
        s = allm[(allm["score"] >= lo) & (allm["score"] < hi)]
        if len(s) >= 30:
            rows.append({"Setup score": name, "Cases": len(s), f"Ran +{run_pct:.0f}% within {horizon} days":
                         float((s["run"] >= run_pct).mean() * 100),
                         f"Avg return after {horizon} days": float(s["ret"].mean()),
                         "Fell more than 10%": float((s["ret"] <= -10).mean() * 100)})
    base = {"cases": len(allm), "ran": float((allm["run"] >= run_pct).mean() * 100), "ret": float(allm["ret"].mean())}
    return {"ok": bool(rows), "table": pd.DataFrame(rows), "base": base, "horizon": horizon, "run_pct": run_pct}
