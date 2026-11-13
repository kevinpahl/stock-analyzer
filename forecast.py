"""Market forecasting for the S&P 500 and Nasdaq. Pure numpy/pandas, no Streamlit.

Three honest ingredients:
  1. A small logistic-regression model: probability the index closes higher over the next 1, 5 or 21 sessions.
  2. Volatility-based price ranges (blend of implied volatility from the VIX/VXN and recent realized volatility).
  3. A live read from futures, Asia, the VIX and today's trend (rule-based, plus how Asia has actually predicted
     the US close historically).
Every model is tested walk-forward (trained only on the past, scored on the future) so the app can show how
little edge a model like this really has.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer import rsi

HORIZONS = (1, 5, 21)
L2 = 50.0                  # strong shrinkage keeps probabilities near the base rate
PROB_CLIP = (0.30, 0.70)   # never claim more certainty than a model like this can justify
TRAIN_ROWS = 3780          # about 15 years of daily bars
Z50, Z90 = 0.6745, 1.645
FEATURE_NAMES = {
    "ret_1d": "yesterday's move", "ret_5d": "5-day move", "ret_21d": "1-month move",
    "dist_sma50": "distance from 50-day average", "dist_sma200": "distance from 200-day average",
    "rsi": "RSI", "vol_ratio": "short vs long volatility", "dd_252": "drawdown from 1-year high",
    "vix_level": "VIX level", "vix_chg5": "VIX 5-day change",
}


# ------------------------------------------------------------------ features + model
def build_features(close: pd.Series, vix: pd.Series | None = None) -> pd.DataFrame:
    """Everything here uses data up to and including each day's close (no look-ahead)."""
    c = close.dropna()
    r = c.pct_change()
    X = pd.DataFrame(index=c.index)
    X["ret_1d"] = r
    X["ret_5d"] = c.pct_change(5)
    X["ret_21d"] = c.pct_change(21)
    X["dist_sma50"] = c / c.rolling(50).mean() - 1
    X["dist_sma200"] = c / c.rolling(200).mean() - 1
    X["rsi"] = (rsi(c) - 50) / 50
    X["vol_ratio"] = np.log(r.rolling(10).std() / r.rolling(60).std())
    X["dd_252"] = c / c.rolling(252, min_periods=60).max() - 1
    if vix is not None and len(vix.dropna()):
        v = vix.reindex(c.index).ffill()
        X["vix_level"] = np.log(v)
        X["vix_chg5"] = v.pct_change(5)
    return X.replace([np.inf, -np.inf], np.nan)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = L2, iters: int = 60) -> dict:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    P = np.eye(Z.shape[1]) * l2
    P[0, 0] = 0.0  # don't penalise the intercept
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + P @ w
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + P + 1e-8 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"w": w, "mu": mu, "sd": sd}


def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    z = model["w"][0] + ((X - model["mu"]) / model["sd"]) @ model["w"][1:]
    return np.clip(1 / (1 + np.exp(-np.clip(z, -30, 30))), *PROB_CLIP)


def _targets(close: pd.Series, h: int) -> pd.Series:
    fut = close.shift(-h)
    return (fut > close).astype(float).where(fut.notna())


def model_forecast(close: pd.Series, vix: pd.Series | None, h: int) -> dict:
    """Probability that the index is higher h sessions after the last close in `close`."""
    X, y = build_features(close, vix), _targets(close, h)
    ok = X.notna().all(axis=1)
    if not ok.iloc[-1]:
        return {"ok": False}
    train = ok & y.notna()
    Xt, yt = X[train].iloc[-TRAIN_ROWS:], y[train].iloc[-TRAIN_ROWS:]
    if len(Xt) < 750:
        return {"ok": False}
    m = fit_logistic(Xt.to_numpy(), yt.to_numpy())
    x = X.iloc[-1]
    p = float(predict_proba(m, x.to_numpy()[None, :])[0])
    z = (x.to_numpy() - m["mu"]) / m["sd"]
    contrib = sorted(zip(X.columns, m["w"][1:] * z), key=lambda t: -abs(t[1]))[:3]
    return {"ok": True, "p_up": p, "base_rate": float(yt.mean()), "train_rows": len(Xt),
            "drivers": [{"feature": FEATURE_NAMES.get(n, n), "pushes": "up" if v > 0 else "down",
                         "strength": float(abs(v))} for n, v in contrib]}


def walk_forward(close: pd.Series, vix: pd.Series | None, h: int, test_rows: int = 756, step: int = 21) -> dict:
    """Retrain every `step` days using only outcomes that were already known, then score the next block."""
    X, y = build_features(close, vix), _targets(close, h)
    ok = (X.notna().all(axis=1) & y.notna()).to_numpy()
    idx = np.flatnonzero(ok)
    n = len(idx)
    if n < 1500:
        return {"n": 0}
    Xv, yv = X.to_numpy(), y.to_numpy()
    P, Y, B = [], [], []
    for s in range(max(1000, n - test_rows), n, step):
        cutoff = idx[s]
        tr = idx[(idx + h) <= cutoff][-TRAIN_ROWS:]   # only rows whose outcome had already happened
        te = idx[s:s + step]
        m = fit_logistic(Xv[tr], yv[tr])
        P.append(predict_proba(m, Xv[te]))
        Y.append(yv[te])
        B.append(np.full(len(te), yv[tr].mean()))
    P, Y, B = np.concatenate(P), np.concatenate(Y), np.concatenate(B)
    acc = float(((P > 0.5) == (Y == 1)).mean())
    base_acc = float(((B > 0.5) == (Y == 1)).mean())
    brier, base_brier = float(((P - Y) ** 2).mean()), float(((B - Y) ** 2).mean())
    skill = 1 - brier / base_brier
    # Outcomes over h days overlap, so there are only about n/h independent samples. Judge the accuracy gain
    # against that, otherwise noise gets mistaken for skill.
    z = (acc - base_acc) / np.sqrt(0.25 / max(len(P) / h, 1))
    edge = ("slight edge" if skill > 0.005 and z > 2 else
            "marginal" if skill > 0 and z > 1 else "no reliable edge")
    cal = []
    for lo, hi in ((0, .45), (.45, .5), (.5, .55), (.55, 1.01)):
        m_ = (P >= lo) & (P < hi)
        if m_.sum() >= 20:
            cal.append({"predicted": float(P[m_].mean()), "actual": float(Y[m_].mean()), "n": int(m_.sum())})
    return {"n": int(len(P)), "accuracy": acc, "baseline_accuracy": base_acc, "brier": brier,
            "baseline_brier": base_brier, "skill": float(skill), "z": float(z), "edge": edge, "calibration": cal}


# ------------------------------------------------------------------ ranges
def vol_estimates(close: pd.Series, implied_ann_pct: float | None = None) -> dict:
    r = close.pct_change().dropna()
    ewma = float(np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean().iloc[-1]))   # RiskMetrics lambda 0.94
    imp = implied_ann_pct / 100 / np.sqrt(252) if implied_ann_pct else None
    return {"ewma_daily": ewma, "implied_daily": imp, "sigma_daily": 0.5 * ewma + 0.5 * imp if imp else ewma}


def range_for(last: float, sigma_d: float, h: int) -> dict:
    s = sigma_d * np.sqrt(h)
    return {"horizon": h, "sigma_pct": s * 100, "lo68": last * np.exp(-s), "hi68": last * np.exp(s),
            "lo90": last * np.exp(-Z90 * s), "hi90": last * np.exp(Z90 * s)}


def cone(last: float, sigma_d: float, anchor: pd.Timestamp, days: int = 10) -> pd.DataFrame:
    dates = [anchor] + list(pd.bdate_range(anchor + pd.Timedelta(days=1), periods=days))
    rows = []
    for d in range(days + 1):
        s = sigma_d * np.sqrt(d)
        rows.append({"lo50": last * np.exp(-Z50 * s), "hi50": last * np.exp(Z50 * s),
                     "lo90": last * np.exp(-Z90 * s), "hi90": last * np.exp(Z90 * s)})
    return pd.DataFrame(rows, index=dates)


# ------------------------------------------------------------------ Asia lead + live read
ASIA_BUCKETS = ["down more than 1%", "down 0.25% to 1%", "flat (within 0.25%)", "up 0.25% to 1%", "up more than 1%"]


def asia_bucket(x: float) -> str:
    return ASIA_BUCKETS[0] if x <= -1 else ASIA_BUCKETS[1] if x <= -0.25 else ASIA_BUCKETS[2] if x < 0.25 \
        else ASIA_BUCKETS[3] if x < 1 else ASIA_BUCKETS[4]


def asia_composite(asia: dict[str, pd.DataFrame]) -> pd.Series:
    """Average daily % change of the Asian indexes, by calendar date (needs at least 3 markets that day).
    An Asian session on date D finishes before the US session on date D opens, so it is known in advance."""
    rets = pd.DataFrame({k: d["Close"].pct_change() for k, d in asia.items() if len(d) > 30})
    if rets.empty:
        return pd.Series(dtype=float)
    comp = rets.mean(axis=1, skipna=True) * 100
    return comp[rets.notna().sum(axis=1) >= 3]


def asia_lead_table(us_close: pd.Series, comp: pd.Series, years: int = 10) -> tuple[pd.DataFrame, float]:
    """How often the US index closed higher that same day, split by how Asia did earlier that day."""
    us = us_close.pct_change() * 100
    df = pd.concat([comp.rename("asia"), us.rename("us")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(), float("nan")
    df = df[df.index >= df.index.max() - pd.DateOffset(years=years)]
    df["bucket"] = df["asia"].map(asia_bucket)
    rows = []
    for b in ASIA_BUCKETS:
        s = df[df["bucket"] == b]["us"]
        rows.append({"bucket": b, "sessions": int(len(s)),
                     "us_up_pct": float((s > 0).mean() * 100) if len(s) else np.nan,
                     "avg_us_move_pct": float(s.mean()) if len(s) else np.nan})
    return pd.DataFrame(rows), float(df["asia"].corr(df["us"]))


def _step(val: float, mild: float, strong: float) -> int:
    return 2 if val >= strong else 1 if val >= mild else -2 if val <= -strong else -1 if val <= -mild else 0


def live_lean(fut_pct=None, asia_pct=None, vix_pct=None, idx_pct=None) -> dict | None:
    """Which way do the signals available right now point? Rule-based and transparent, not a trained model."""
    comps = []
    if fut_pct is not None:
        comps.append(("Futures vs prior close", _step(fut_pct, .15, .5), .40, f"{fut_pct:+.2f}%"))
    if asia_pct is not None:
        comps.append(("Asia sessions (average)", _step(asia_pct, .25, .75), .25, f"{asia_pct:+.2f}%"))
    if vix_pct is not None:
        comps.append(("VIX move today", -_step(vix_pct, 5, 12), .15, f"{vix_pct:+.1f}%"))
    if idx_pct is not None:
        comps.append(("Index so far today", _step(idx_pct, .15, .5), .20, f"{idx_pct:+.2f}%"))
    if not comps:
        return None
    score = sum(s * w for _, s, w, _ in comps) / sum(w for _, _, w, _ in comps)
    label = "LEANING HIGHER" if score >= 0.6 else "LEANING LOWER" if score <= -0.6 else "MIXED"
    return {"label": label, "score": float(score),
            "components": [{"name": n, "score": s, "detail": d} for n, s, _, d in comps]}


def edge_text(edge: str) -> str:
    return {"slight edge": "slight edge in testing", "marginal": "marginal edge in testing"}.get(
        edge, "no reliable edge in testing")
