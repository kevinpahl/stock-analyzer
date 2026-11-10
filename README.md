# Stock Analyzer

    pip install -r requirements.txt
    streamlit run app.py

Sections (switch at the top of the page):
- **Analyze**: live price, day range and Buy/Hold/Sell signal that auto-refresh (sidebar: Off / 10 / 30 / 60 sec).
  Chart view **Live** = intraday bars (1m to 1h, 1D to 1M, optional extended hours);
  **History** = daily / weekly / monthly / quarterly bars over any range or custom dates.
  Works for stocks, ETFs and indexes (e.g. ^GSPC). 9/20/50/200-day moving averages, Bollinger, volume, RSI, MACD.
- **Favorites**: save symbols (sidebar ☆ button, or type them) and watch them in one live table. The list is
  stored in the page address (?fav=...), so bookmark the page to keep it.
- **Market pulse**: scans ~200 stocks and ETFs, rates the market bullish/bearish (with trend over time), warns on
  volatility, and breaks results out by business type (energy, AI, semiconductors, ...) with stock-level filters.
- **Backtest**: replays the technical score on past data vs buy & hold (custom test period, costs, in/out-of-sample).
- **Top movers**: scans gainers/losers/most-active (or your own list), ranks movers by score; optional auto-refresh.

Files: app.py, analyzer.py, backtest.py, screener.py, charts.py, utils.py, market.py, requirements.txt,
.streamlit/config.toml (optional theme).
