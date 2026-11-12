"""Plotly figures with one consistent, clean style (works in light and dark themes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN, BLUE, GRAY = "#16a34a", "#dc2626", "#2563eb", "#94a3b8"
GRID = "rgba(128,128,128,0.16)"
MA = {"SMA 9": ("SMA9", "#f59e0b"), "SMA 20": ("SMA20", "#2563eb"),
      "SMA 50": ("SMA50", "#7c3aed"), "SMA 200": ("SMA200", "#64748b")}


def _style(fig: go.Figure, height: int, rangebreaks: list | None = None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=30, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                      xaxis_rangeslider_visible=False, bargap=0.15)
    fig.update_xaxes(showgrid=False, showline=True, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    if rangebreaks:
        fig.update_xaxes(rangebreaks=rangebreaks)
    return fig


def price_chart(df: pd.DataFrame, start, end, chart_type: str, overlays: list[str],
                panels: list[str], interval: str, intraday: bool = False, us_hours: bool = True,
                prepost: bool = False) -> go.Figure:
    """df must already contain indicator columns (see analyzer.add_indicators)."""
    d = df.loc[start:end]
    n = len(panels)
    heights = [1.0] if n == 0 else [0.64] + [0.36 / n] * n
    fig = make_subplots(rows=1 + n, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.025)

    if chart_type == "Candles":
        fig.add_trace(go.Candlestick(
            x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price",
            increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                                 line=dict(color=BLUE, width=2)), row=1, col=1)

    for name in overlays:
        if name in MA:
            col, color = MA[name]
            if d[col].notna().any():
                fig.add_trace(go.Scatter(x=d.index, y=d[col], name=name,
                                         line=dict(color=color, width=1.4)), row=1, col=1)
        elif name == "Bollinger bands":
            edge = dict(color="rgba(100,116,139,0.6)", width=1, dash="dot")
            fig.add_trace(go.Scatter(x=d.index, y=d["BB_high"], name="Upper band", line=edge), row=1, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=d["BB_low"], name="Lower band", line=edge,
                                     fill="tonexty", fillcolor="rgba(100,116,139,0.08)"), row=1, col=1)

    r = 2
    for p in panels:
        if p == "Volume":
            colors = np.where(d["Close"] >= d["Open"], UP, DOWN).tolist()
            fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=colors, marker_opacity=0.55,
                                 name="Volume", showlegend=False), row=r, col=1)
            fig.update_yaxes(title_text="Volume", row=r, col=1)
        elif p == "RSI":
            fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI",
                                     line=dict(color="#0d9488", width=1.5)), row=r, col=1)
            for lvl in (30, 70):
                fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(128,128,128,0.6)", row=r, col=1)
            fig.update_yaxes(range=[0, 100], title_text="RSI", row=r, col=1)
        elif p == "MACD":
            hist = d["MACD_hist"]
            fig.add_trace(go.Bar(x=d.index, y=hist, marker_color=np.where(hist >= 0, UP, DOWN).tolist(),
                                 marker_opacity=0.5, name="MACD hist", showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD",
                                     line=dict(color=BLUE, width=1.3)), row=r, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=d["MACD_signal"], name="Signal",
                                     line=dict(color="#f59e0b", width=1.3)), row=r, col=1)
            fig.update_yaxes(title_text="MACD", row=r, col=1)
        r += 1
    if intraday:  # hide nights and weekends so intraday bars sit side by side
        breaks = [dict(bounds=["sat", "mon"])]
        if us_hours:
            breaks.append(dict(bounds=[20, 4] if prepost else [16, 9.5], pattern="hour"))
    else:
        breaks = [dict(bounds=["sat", "mon"])] if interval == "Daily" else None
    return _style(fig, 460 + 150 * n, breaks)


def equity_chart(eq: pd.DataFrame) -> go.Figure:
    dd = eq / eq.cummax() - 1
    colors = {"Strategy": (BLUE, "rgba(37,99,235,0.15)"), "Buy & hold": (GRAY, "rgba(148,163,184,0.15)")}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=("Growth of $1", "Drawdown"))
    for c in eq.columns:
        line, fill = colors.get(c, (BLUE, "rgba(37,99,235,0.15)"))
        fig.add_trace(go.Scatter(x=eq.index, y=eq[c], name=c, line=dict(color=line, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd[c], name=f"{c} drawdown", showlegend=False,
                                 line=dict(color=line, width=1), fill="tozeroy", fillcolor=fill), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    return _style(fig, 520)


def trades_chart(close: pd.Series, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=close.index, y=close, name="Close", line=dict(color=GRAY, width=1.3)))
    if not trades.empty:
        fig.add_trace(go.Scatter(x=trades["Entry date"], y=trades["Entry price"], mode="markers", name="Buy",
                                 marker=dict(symbol="triangle-up", size=11, color=UP)))
        ex = trades[trades["Status"] == "closed"]
        fig.add_trace(go.Scatter(x=ex["Exit date"], y=ex["Exit price"], mode="markers", name="Sell",
                                 marker=dict(symbol="triangle-down", size=11, color=DOWN)))
    return _style(fig, 340)


def regime_chart(hist: pd.Series) -> go.Figure:
    """Market rating (0 = strongly bearish, 100 = strongly bullish) over time, with shaded zones."""
    fig = go.Figure()
    for lo, hi, col in [(0, 20, "rgba(153,27,27,0.14)"), (20, 40, "rgba(220,38,38,0.08)"),
                        (40, 60, "rgba(245,158,11,0.08)"), (60, 80, "rgba(22,163,74,0.08)"),
                        (80, 100, "rgba(10,125,51,0.14)")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0)
    fig.add_trace(go.Scatter(x=hist.index, y=hist, mode="lines", name="Market rating",
                             line=dict(color=BLUE, width=2.5)))
    fig.update_yaxes(range=[0, 100])
    return _style(fig, 300)


def group_bars(df: pd.DataFrame, col: str) -> go.Figure:
    d = df.dropna(subset=[col]).sort_values(col)
    fig = go.Figure(go.Bar(x=d[col], y=d["Business type"], orientation="h",
                           marker_color=[UP if v >= 0 else DOWN for v in d[col]],
                           text=[f"{v:+.1f}%" for v in d[col]], textposition="outside", cliponaxis=False))
    fig.update_xaxes(ticksuffix="%")
    fig.update_layout(showlegend=False)
    return _style(fig, max(320, 28 * len(d) + 80))


def intraday_chart(df: pd.DataFrame, prev_close: float | None = None, name: str = "SPY") -> go.Figure:
    fig = go.Figure(go.Scatter(x=df.index, y=df["Close"], name=name, line=dict(color=BLUE, width=2)))
    if float(df["Volume"].sum()) > 0:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
        fig.add_trace(go.Scatter(x=df.index, y=vwap, name="VWAP", line=dict(color="#f59e0b", width=1.3, dash="dot")))
    if prev_close:
        fig.add_hline(y=prev_close, line_dash="dash", line_color="rgba(128,128,128,0.6)",
                      annotation_text="Prev close", annotation_position="top left")
    return _style(fig, 300, [dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])


def _fmt_vol(v: float, measure: str) -> str:
    sign = "+" if v > 0 else "-" if v < 0 else ""
    a, pre = abs(v), ("$" if measure == "dollars" else "")
    return f"{sign}{pre}{a / 1e9:.1f}B" if a >= 1e9 else f"{sign}{pre}{a / 1e6:.0f}M"


def volume_waterfall(df: pd.DataFrame, measure: str = "dollars", mode: str = "total", frac: float = 1.0) -> go.Figure:
    """Waterfall of traded volume. mode='total': each step adds that group's volume today, ending in the total.
    mode='change': starts at the 20-day average pace and each step is the group's change vs that average."""
    col, avg = ("dollars", "avg_dollars") if measure == "dollars" else ("shares", "avg_shares")
    d = df.fillna(0).copy()
    if mode == "change":
        d["delta"] = d[col] - d[avg] * frac
        d = d.sort_values("delta", ascending=False)
        base = float((d[avg] * frac).sum())
        x = ["20-day average pace"] + list(d["name"]) + ["Today"]
        y = [base] + list(d["delta"]) + [0]
        kinds = ["absolute"] + ["relative"] * len(d) + ["total"]
        text = [_fmt_vol(base, measure).lstrip("+")] + [_fmt_vol(v, measure) for v in d["delta"]] + \
               [_fmt_vol(float(d[col].sum()), measure).lstrip("+")]
        inc, dec = UP, DOWN
    else:
        d = d.sort_values(col, ascending=False)
        x = list(d["name"]) + ["Total"]
        y = list(d[col]) + [0]
        kinds = ["relative"] * len(d) + ["total"]
        text = [_fmt_vol(v, measure).lstrip("+") for v in d[col]] + [_fmt_vol(float(d[col].sum()), measure).lstrip("+")]
        inc, dec = BLUE, DOWN
    scale, unit = (1e9, "$ billions") if measure == "dollars" else (1e6, "shares, millions")
    fig = go.Figure(go.Waterfall(
        x=x, y=[v / scale for v in y], measure=kinds, text=text, textposition="outside",
        connector=dict(line=dict(color=GRID)), increasing=dict(marker=dict(color=inc)),
        decreasing=dict(marker=dict(color=dec)), totals=dict(marker=dict(color="#475569")), cliponaxis=False))
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(title_text=unit)
    fig.update_layout(showlegend=False)
    return _style(fig, 440)
