"""Crypto market scan: coins by category, a bullish/bearish rating, volatility warnings, volume breakdown.
Pure logic (no Streamlit). Coins trade 24/7, so a "day" here is the UTC day Yahoo uses for its daily bars,
and Yahoo reports crypto volume directly in US dollars."""
from __future__ import annotations

import numpy as np
import pandas as pd

from market import _closes, _map, _trend_series, regime_label, symbol_stats

# symbol -> (name, category). Curated: edit freely. Coins Yahoo has no data for are skipped automatically.
COINS: dict[str, tuple[str, str]] = {
    "BTC-USD": ("Bitcoin", "Layer 1"), "ETH-USD": ("Ethereum", "Layer 1"), "SOL-USD": ("Solana", "Layer 1"),
    "ADA-USD": ("Cardano", "Layer 1"), "AVAX-USD": ("Avalanche", "Layer 1"), "DOT-USD": ("Polkadot", "Layer 1"),
    "NEAR-USD": ("NEAR Protocol", "Layer 1"), "ATOM-USD": ("Cosmos", "Layer 1"), "ALGO-USD": ("Algorand", "Layer 1"),
    "TRX-USD": ("TRON", "Layer 1"), "HBAR-USD": ("Hedera", "Layer 1"), "ICP-USD": ("Internet Computer", "Layer 1"),
    "XRP-USD": ("XRP", "Payments & legacy"), "XLM-USD": ("Stellar", "Payments & legacy"),
    "LTC-USD": ("Litecoin", "Payments & legacy"), "BCH-USD": ("Bitcoin Cash", "Payments & legacy"),
    "ETC-USD": ("Ethereum Classic", "Payments & legacy"), "XMR-USD": ("Monero", "Payments & legacy"),
    "LINK-USD": ("Chainlink", "DeFi & oracles"), "UNI7083-USD": ("Uniswap", "DeFi & oracles"),
    "AAVE-USD": ("Aave", "DeFi & oracles"), "MKR-USD": ("Maker", "DeFi & oracles"),
    "DOGE-USD": ("Dogecoin", "Meme coins"), "SHIB-USD": ("Shiba Inu", "Meme coins"),
    "FIL-USD": ("Filecoin", "Infrastructure"), "VET-USD": ("VeChain", "Infrastructure"),
    "THETA-USD": ("Theta Network", "Infrastructure"),
    "BNB-USD": ("BNB", "Exchange tokens"), "CRO-USD": ("Cronos", "Exchange tokens"),
}
CATEGORIES = list(dict.fromkeys(cat for _, cat in COINS.values()))
CATEGORY_OF = {sym: cat for sym, (_, cat) in COINS.items()}
NAME_OF = {sym: name for sym, (name, _) in COINS.items()}
TOP_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD"]

CRYPTO_REGIME_WEIGHTS = {
    "Bitcoin & Ethereum trend": 0.30, "Short-term momentum": 0.15, "Breadth: above 50-day": 0.15,
    "Breadth: above 200-day": 0.10, "Participation (up today)": 0.10, "Volatility (Bitcoin)": 0.10,
    "Alt strength vs Bitcoin": 0.10,
}
LEVELS = ["LOW", "NORMAL", "ELEVATED", "HIGH", "EXTREME"]
DAYS = 365


def crypto_symbols() -> list[str]:
    return list(COINS)


def day_fraction_utc(now_utc: pd.Timestamp) -> float:
    """Share of the current UTC day that has passed (crypto volume has no open or close, so this is linear)."""
    return max(0.05, (now_utc.hour * 60 + now_utc.minute) / 1440)


# ---------------------------------------------------------------- rating
def crypto_regime(data: dict[str, pd.DataFrame]) -> dict:
    F, raw = pd.DataFrame(), {}
    big = [s for s in ("BTC-USD", "ETH-USD") if s in data]
    if big:
        T = pd.DataFrame({s: _trend_series(data[s]["Close"].dropna()) for s in big})
        F["Bitcoin & Ethereum trend"] = T.mean(axis=1, skipna=True)
        raw["trends"] = T.dropna(how="all").iloc[-1].to_dict() if T.notna().any().any() else {}
    if "BTC-USD" in data:
        c = data["BTC-USD"]["Close"].dropna()
        s9, s20 = c.rolling(9).mean(), c.rolling(20).mean()
        F["Short-term momentum"] = pd.Series(np.where(c > s20, 1.0, -1.0) + np.where(s9 > s20, 1.0, -1.0),
                                             index=c.index).where(s20.notna())
        raw["btc_short"] = (bool(c.iloc[-1] > s20.iloc[-1]), bool(s9.iloc[-1] > s20.iloc[-1]))
        r = c.pct_change()
        rv = r.rolling(30).std() * np.sqrt(DAYS) * 100
        chg5 = rv / rv.shift(5) - 1
        lvl = np.select([rv < 35, rv < 55, rv < 75, rv < 100], [1.5, 0.5, -0.5, -1.5], default=-2.0)
        adj = np.select([chg5 < -0.15, chg5 > 0.25], [0.5, -0.5], default=0.0)
        F["Volatility (Bitcoin)"] = pd.Series(np.clip(lvl + adj, -2, 2), index=c.index).where(chg5.notna())
        raw["rv"] = (float(rv.iloc[-1]), float(chg5.iloc[-1]) if not np.isnan(chg5.iloc[-1]) else np.nan)

    C = _closes(data, list(data), tail=420)
    if not C.empty:
        s50, s200 = C.rolling(50).mean(), C.rolling(200).mean()
        ret = C.pct_change()
        v50, v200, vr = s50.notna(), s200.notna(), ret.notna()
        n50, n200, nr = v50.sum(axis=1), v200.sum(axis=1), vr.sum(axis=1)
        p50 = ((C > s50) & v50).sum(axis=1) / n50.where(n50 >= 10) * 100
        p200 = ((C > s200) & v200).sum(axis=1) / n200.where(n200 >= 10) * 100
        padv = ((ret > 0) & vr).sum(axis=1) / nr.where(nr >= 10) * 100
        F["Breadth: above 50-day"] = _map(p50, [75, 60, 40, 25], [2.0, 1.0, 0.0, -1.0], -2.0)
        F["Breadth: above 200-day"] = _map(p200, [75, 60, 45, 30], [2.0, 1.0, 0.0, -1.0], -2.0)
        F["Participation (up today)"] = _map(padv, [70, 58, 42, 30], [2.0, 1.0, 0.0, -1.0], -2.0)
        raw.update(p50=p50.dropna().iloc[-1] if p50.notna().any() else np.nan,
                   p200=p200.dropna().iloc[-1] if p200.notna().any() else np.nan,
                   padv=padv.dropna().iloc[-1] if padv.notna().any() else np.nan, n=int(nr.iloc[-1]))
        if "BTC-USD" in C and C.shape[1] > 4:
            r21 = C / C.shift(21) - 1
            alts = r21.drop(columns=["BTC-USD"]).median(axis=1)
            d = alts - r21["BTC-USD"]
            F["Alt strength vs Bitcoin"] = pd.Series(
                np.select([d > .05, d > .01, d < -.05, d < -.01], [1.0, .5, -1.0, -.5], default=0.0),
                index=d.index).where(d.notna())
            raw["alt"] = float(d.dropna().iloc[-1]) if d.notna().any() else np.nan

    if F.empty:
        return {"ok": False}
    W = pd.Series({k: CRYPTO_REGIME_WEIGHTS.get(k, 0.0) for k in F.columns})
    den = (F.notna() * W).sum(axis=1)
    score = ((F.fillna(0) * W).sum(axis=1) / den.where(den > 0)).where(den >= 0.5)
    last = score.last_valid_index()
    if last is None:
        return {"ok": False}
    s_now, clean = float(score.loc[last]), score.dropna()
    prev = float(clean.iloc[-8]) if len(clean) >= 8 else np.nan      # a week ago (coins trade every day)
    delta = s_now - prev if not np.isnan(prev) else np.nan
    direction = ("Improving" if delta > 0.25 else "Deteriorating" if delta < -0.25 else "Stable") \
        if not np.isnan(delta) else "n/a"
    comps = []
    for name, wgt in CRYPTO_REGIME_WEIGHTS.items():
        val = F[name].loc[last] if name in F and last in F.index else np.nan
        comps.append({"Factor": name, "available": not np.isnan(val), "score": None if np.isnan(val) else float(val),
                      "weight": wgt, "reason": _reason(name, raw, val)})
    return {"ok": True, "score": s_now, "pct": round((s_now + 2) / 4 * 100), "label": regime_label(s_now),
            "direction": direction, "delta": delta, "components": comps,
            "history": ((clean + 2) / 4 * 100).iloc[-120:], "breadth": raw, "asof": last}


def _reason(name: str, raw: dict, val) -> str:
    if np.isnan(val):
        return "Not enough data."
    if name == "Bitcoin & Ethereum trend":
        parts = ", ".join(f"{NAME_OF.get(k, k)} {v:+.0f}" for k, v in raw.get("trends", {}).items() if not np.isnan(v))
        return f"Trend scores (-2 to +2, from 50/200-day averages): {parts}."
    if name == "Short-term momentum":
        a, b = raw["btc_short"]
        return f"Bitcoin is {'above' if a else 'below'} its 20-day average; 9-day is {'above' if b else 'below'} 20-day."
    if name == "Breadth: above 50-day":
        return f"{raw['p50']:.0f}% of {raw['n']} tracked coins are above their 50-day average."
    if name == "Breadth: above 200-day":
        return f"{raw['p200']:.0f}% of tracked coins are above their 200-day average."
    if name == "Participation (up today)":
        return f"{raw['padv']:.0f}% of tracked coins are up today (UTC day)."
    if name == "Volatility (Bitcoin)":
        v, c5 = raw["rv"]
        return f"Bitcoin's 30-day volatility is {v:.0f}% a year" + (f" ({c5:+.0%} vs 5 days ago)." if not np.isnan(c5) else ".")
    if name == "Alt strength vs Bitcoin":
        return f"The typical altcoin returned {raw['alt']:+.1%} versus Bitcoin over the last month."
    return ""


# ---------------------------------------------------------------- volatility
def crypto_volatility(data: dict[str, pd.DataFrame], coins: pd.DataFrame) -> dict:
    out = {"ok": False, "warnings": []}
    if "BTC-USD" not in data:
        return out
    r = data["BTC-USD"]["Close"].dropna().pct_change()
    if len(r) < 40:
        return out
    rv = float(r.iloc[-30:].std() * np.sqrt(DAYS) * 100)
    today = float(r.iloc[-1] * 100)
    avg_abs = float(r.iloc[-31:-1].abs().mean() * 100)
    z = abs(today) / avg_abs if avg_abs > 0 else None
    lvl = 0 if rv < 35 else 1 if rv < 55 else 2 if rv < 75 else 3 if rv < 100 else 4
    down5 = float((coins["1d %"] <= -5).mean() * 100) if not coins.empty else 0.0
    if (z is not None and z >= 2.5) or down5 >= 30:
        lvl = min(4, lvl + 1)
    w = []
    if z is not None and z >= 2:
        w.append(f"Bitcoin moved {today:+.1f}% today, {z:.1f}x its normal daily move.")
    if rv >= 75:
        w.append(f"Bitcoin's 30-day volatility is {rv:.0f}% a year, which is very high.")
    if down5 >= 30:
        w.append(f"{down5:.0f}% of tracked coins are down more than 5% today: broad selling.")
    if not coins.empty:
        wild = float((coins["Volatility %"] >= 120).mean() * 100)
        if wild >= 40:
            w.append(f"{wild:.0f}% of tracked coins have 20-day volatility above 120% a year.")
    if rv < 30 and not w:
        w.append(f"Bitcoin volatility is unusually low ({rv:.0f}% a year). Crypto can turn sharply from calm periods.")
    return {"ok": True, "level": LEVELS[lvl], "idx": lvl, "realized": rv, "btc_today": today, "z": z, "warnings": w}


# ---------------------------------------------------------------- categories + volume
def crypto_group_table(coins: pd.DataFrame) -> pd.DataFrame:
    from market import signal_label
    rows = []
    for cat in CATEGORIES:
        syms = [s for s, c in CATEGORY_OF.items() if c == cat]
        s = coins[coins.index.isin(syms)]
        if s.empty:
            continue
        sc = s["Score"].mean()
        rows.append({
            "Business type": cat, "Coins": len(s), "Signal": signal_label(sc),
            "Rating": (sc + 2) / 4 * 100 if not np.isnan(sc) else np.nan,
            "1D": s["1d %"].mean(), "5D": s["5d %"].mean(), "1M": s["1m %"].mean(), "3M": s["3m %"].mean(),
            "% above 50-day": s["above50"].astype(float).mean() * 100, "% up today": s["up_today"].mean() * 100,
            "Volatility %": s["Volatility %"].mean(),
            "Leader today": s.loc[s["1d %"].idxmax(), "Ticker"], "Laggard today": s.loc[s["1d %"].idxmin(), "Ticker"],
        })
    return pd.DataFrame(rows)


def crypto_volume(coins: pd.DataFrame, frac: float) -> dict:
    """Dollars traded today vs the 20-day average, by category and by coin (each coin counted once)."""
    empty = pd.DataFrame(columns=["name", "n", "dollars", "avg_dollars", "shares", "avg_shares"])
    if coins.empty:
        return {"industries": empty, "indexes": empty, "frac": frac}
    c = coins.copy()
    c["group"] = c["Ticker"].map(lambda t: CATEGORY_OF.get(t, "Other"))
    agg = c.groupby("group").agg(n=("Ticker", "count"), dollars=("dollar_vol", "sum"),
                                 avg_dollars=("avg_dollar_vol", "sum")).reset_index().rename(columns={"group": "name"})
    agg["shares"], agg["avg_shares"] = agg["dollars"], agg["avg_dollars"]
    top = c.sort_values("dollar_vol", ascending=False).head(10)
    rest = c.drop(top.index)
    rows = [{"name": NAME_OF.get(t, t), "n": 1, "dollars": r["dollar_vol"], "avg_dollars": r["avg_dollar_vol"]}
            for t, r in top.iterrows()]
    if not rest.empty:
        rows.append({"name": "Other coins", "n": len(rest), "dollars": rest["dollar_vol"].sum(),
                     "avg_dollars": rest["avg_dollar_vol"].sum()})
    by_coin = pd.DataFrame(rows)
    by_coin["shares"], by_coin["avg_shares"] = by_coin["dollars"], by_coin["avg_dollars"]
    return {"industries": agg, "indexes": by_coin, "frac": frac}


def compute_crypto_pulse(data: dict[str, pd.DataFrame], weights: dict | None = None,
                         now: pd.Timestamp | None = None) -> dict:
    coins = symbol_stats(data, weights, periods_per_year=DAYS, volume_is_dollars=True)
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    return {"coins": coins, "loaded": len(data), "requested": len(COINS), "regime": crypto_regime(data),
            "vol": crypto_volatility(data, coins) if not coins.empty else {"ok": False, "warnings": []},
            "groups": crypto_group_table(coins) if not coins.empty else pd.DataFrame(),
            "volume": crypto_volume(coins, day_fraction_utc(now))}
