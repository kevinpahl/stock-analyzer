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


def _style(fig: go.Figure, height: int, weekend_gaps: bool = False) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=30, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                      xaxis_rangeslider_visible=False, bargap=0.15)
    fig.update_xaxes(showgrid=False, showline=True, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    if weekend_gaps:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def price_chart(df: pd.DataFrame, start, end, chart_type: str, overlays: list[str],
                panels: list[str], interval: str) -> go.Figure:
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
    return _style(fig, 460 + 150 * n, weekend_gaps=(interval == "Daily"))


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
