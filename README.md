# Stock Analyzer

    pip install -r requirements.txt
    streamlit run app.py

Sections (switch at the top of the page):
- **Analyze**: live price, day range and Buy/Hold/Sell signal that auto-refresh (sidebar: Off / 10 / 30 / 60 sec).
  Chart view **Live** = intraday bars (1m to 1h, 1D to 1M, optional extended hours);
  **History** = daily / weekly / monthly / quarterly bars over any range or custom dates.
  Works for stocks, ETFs and indexes (e.g. ^GSPC). 9/20/50/200-day moving averages, Bollinger, volume, RSI, MACD.
- **Trading**: scrolling live ticker tape (your favorites first), US index / futures / rates / commodities boards,
  intraday S&P and Nasdaq charts, most-active stocks, the Asian sessions (open/closed status, live moves, how Asia has
  historically led the US close), and a tested statistical forecast for the S&P 500 and Nasdaq with live signals.
- **Favorites**: save symbols (sidebar ☆ button, or type them) and watch them in one live table. Create an account
  (sidebar) and your favorites are saved to it and come back when you log in.
- **Market pulse**: scans ~200 stocks and ETFs, rates the market bullish/bearish (with trend over time), warns on
  volatility, shows where volume is trading (waterfall by business type and index), and breaks results out by business type (energy, AI, semiconductors, ...) with stock-level filters.
- **Backtest**: replays the technical score on past data vs buy & hold (custom test period, costs, in/out-of-sample).
- **Top movers**: scans gainers/losers/most-active (or your own list), ranks movers by score; optional auto-refresh.

Files: app.py, analyzer.py, backtest.py, screener.py, charts.py, utils.py, market.py, accounts.py, forecast.py, sessions.py, requirements.txt,
.streamlit/config.toml (optional theme).

## Accounts setup (so favorites survive restarts)
Accounts need a small database. The free Upstash Redis tier works well:
1. Sign up at upstash.com, create a Redis database, and copy its **REST URL** and **REST token** (REST API section).
2. In Streamlit Cloud: your app, Settings, Secrets, then add:

       UPSTASH_REDIS_REST_URL = "https://your-database.upstash.io"
       UPSTASH_REDIS_REST_TOKEN = "your-token"

   Never put these in GitHub. Running locally, set the same names as environment variables instead.
Without them, accounts are kept in a local file (`accounts_local.json`), which works on your own PC but is erased
when Streamlit Cloud restarts the app. Passwords are stored only as salted hashes; five wrong tries lock an account
for 15 minutes. There is no password reset, so you (the owner) can delete a user's key in the Upstash console.
