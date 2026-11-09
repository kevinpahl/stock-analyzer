# Stock Analyzer

    pip install -r requirements.txt
    streamlit run app.py

Tabs:
- **Analyze**: Buy/Hold/Sell score for a stock, ETF or index (e.g. ^GSPC = S&P 500) with a chart you can set to
  any range (1M to Max, or custom dates) and interval (daily, weekly, monthly, quarterly), plus 9/20/50/200-day
  moving averages, Bollinger bands, volume, RSI and MACD.
- **Backtest**: replays the technical score on past data vs buy & hold (next-close fills, trading costs,
  in-sample / out-of-sample split, custom test period).
- **Top movers**: scans gainers/losers/most-active (or your own list), keeps liquid names, ranks movers by score.

Files: app.py (UI), analyzer.py (scoring), backtest.py, screener.py, charts.py, utils.py,
.streamlit/config.toml (theme, optional).
