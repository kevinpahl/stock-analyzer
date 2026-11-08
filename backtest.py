"""Backtester for the technical score.

Rules: go long when the score reaches `buy_at`, go to cash when it falls to
`sell_at` (the gap between them avoids whipsaw). A signal is read at the close
of day t and filled at the close of day t+1, so there is no same-bar lookahead.
Long-only, fully invested or in cash, no shorting or leverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer import technical_score_series

TRADING_DAYS = 252


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("total_return", "cagr", "max_drawdown", "volatility", "sharpe")}
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    sd = r.std()
    return {
        "total_return": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "volatility": sd * np.sqrt(TRADING_DAYS),
        "sharpe": r.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
    }


def extract_trades(pos: pd.Series, close: pd.Series, cost_bps: float) -> pd.DataFrame:
    p, c, idx = pos.to_numpy(), close.to_numpy(), pos.index
    rt_cost = 2 * cost_bps / 1e4
    rows, entry = [], None
    for i in range(1, len(p)):
        if p[i] == 1 and p[i - 1] == 0:
            entry = i
        elif p[i] == 0 and p[i - 1] == 1 and entry is not None:
            rows.append((idx[entry - 1], c[entry - 1], idx[i - 1], c[i - 1], i - entry, "closed"))
            entry = None
    if entry is not None:  # still holding at the end
        rows.append((idx[entry - 1], c[entry - 1], idx[-1], c[-1], len(p) - entry, "open"))
    t = pd.DataFrame(rows, columns=["Entry date", "Entry price", "Exit date", "Exit price", "Days held", "Status"])
    if not t.empty:
        t["Return"] = t["Exit price"] / t["Entry price"] - 1 - rt_cost
    else:
        t["Return"] = []
    return t


def run_backtest(df: pd.DataFrame, bench: pd.Series | None = None, buy_at: float = 0.4,
                 sell_at: float = -0.4, cost_bps: float = 5.0, weights: dict | None = None) -> dict:
    if sell_at >= buy_at:
        raise ValueError("sell_at must be below buy_at")
    close = df["Close"]
    score = technical_score_series(df, bench, weights)

    holding, state = False, np.zeros(len(df))
    for i, s in enumerate(score.to_numpy()):
        if not np.isnan(s):
            if not holding and s >= buy_at:
                holding = True
            elif holding and s <= sell_at:
                holding = False
        state[i] = holding
    state = pd.Series(state, index=df.index)

    pos = state.shift(2).fillna(0)                 # decide at close t, fill at close t+1, earn from t+2
    ret = close.pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(0)
    strat = pos * ret - turnover * cost_bps / 1e4

    start = score.first_valid_index()
    if start is None:
        raise ValueError("Not enough history: need at least ~200 trading days.")
    live = df.index >= start
    strat, bh, pos_l = strat[live].copy(), ret[live].copy(), pos[live]
    strat.iloc[0] = 0.0
    bh.iloc[0] = 0.0

    trades = extract_trades(pos, close, cost_bps)
    trades = trades[trades["Entry date"] >= start] if not trades.empty else trades
    closed = trades[trades["Status"] == "closed"] if not trades.empty else trades
    return {
        "score": score[live], "close": close[live], "pos": pos_l,
        "strat_ret": strat, "bh_ret": bh,
        "equity": pd.DataFrame({"Strategy": (1 + strat).cumprod(), "Buy & hold": (1 + bh).cumprod()}),
        "strategy": metrics(strat), "buyhold": metrics(bh),
        "exposure": float(pos_l.mean()),
        "trades": trades,
        "n_trades": int(len(trades)),
        "win_rate": float((closed["Return"] > 0).mean()) if len(closed) else np.nan,
        "avg_trade": float(closed["Return"].mean()) if len(closed) else np.nan,
        "start": start,
    }


def split_metrics(strat_ret: pd.Series, bh_ret: pd.Series, frac: float = 0.7) -> dict:
    """In-sample / out-of-sample comparison. If you tuned the thresholds on the first
    part, only the second part is an honest test."""
    k = int(len(strat_ret) * frac)
    return {
        "boundary": strat_ret.index[k] if 0 < k < len(strat_ret) else None,
        "in": (metrics(strat_ret.iloc[:k]), metrics(bh_ret.iloc[:k])),
        "out": (metrics(strat_ret.iloc[k:]), metrics(bh_ret.iloc[k:])),
    }
