# Stock Analyzer

    pip install -r requirements.txt
    streamlit run app.py

Sections (switch at the top of the page):
- **Analyze**: live price, day range and Buy/Hold/Sell signal that auto-refresh (sidebar: Off / 10 / 30 / 60 sec).
  Chart view **Live** = bars from 1 min to 1 day over ranges 1D, 5D, 1M, 2M, 3M, 6M or 1Y (optional extended hours);
  **History** = daily / weekly / monthly / quarterly bars over any range or custom dates.
  Works for stocks, ETFs and indexes (e.g. ^GSPC). 9/20/50/200-day moving averages, Bollinger, volume, RSI, MACD.
  Green up / red down arrows on the chart mark where the technical score moved into Buy / Sell territory (toggle "Buy/sell arrows").
- **Trading**: scrolling live ticker tape (your favorites first), US index / futures / rates / commodities boards,
  intraday S&P and Nasdaq charts, most-active stocks, the Asian sessions (open/closed status, live moves, how Asia has
  historically led the US close), and a tested statistical forecast for the S&P 500 and Nasdaq with live signals.
- **Forecast**: pick any single stock, ETF, index or coin and get the tested up/down probability and price ranges.
- **Crypto**: the same tools for coins (BTC, ETH, SOL and ~25 more, 24/7): live ticker tape and board, Analyze a coin
  (live ranges 1D to 1Y, bar sizes 1 min to 1 day, Buy/Hold/Sell score vs Bitcoin), a coin market pulse (bullish/bearish rating,
  volatility, volume waterfall by category such as Layer 1, DeFi, Meme) and a tested daily forecast (UTC close).
- **Favorites**: save symbols (sidebar ☆ button, or type them) and watch them in one live table. Create an account
  (sidebar) and your favorites are saved to it and come back when you log in.
- **Favorites** also charts your saved symbols together: a separate chart for each symbol (or one comparison chart with all lines starting at the same point), with the same ranges, bar sizes, candles, indicators and buy/sell arrows as Analyze.
- **Market pulse**: scans ~200 stocks and ETFs, rates the market bullish/bearish (with trend over time), warns on
  volatility, shows where volume is trading (waterfall by business type and index), and breaks results out by business type (energy, AI, semiconductors, ...) with stock-level filters.
- **Backtest**: replays the technical score on past data vs buy & hold (custom test period, costs, in/out-of-sample).
- **Top movers**: scans gainers/losers/most-active, built-in lists of ETFs, indexes, futures, currencies or crypto, or
  your own list (any Yahoo symbol), and ranks movers by score. Filter by type (stock, ETF, index, future, currency, crypto).

Tablets and phones: a layout toggle ("Compact layout") stacks columns and moves the symbol box onto the page; it turns on automatically on most phones and Android tablets.

Files: app.py, crypto.py, analyzer.py, backtest.py, screener.py, charts.py, utils.py, market.py, accounts.py, forecast.py, sessions.py, requirements.txt,
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

## Stay signed in
After you log in (with "Keep me signed in" ticked) a cookie keeps you signed in for 30 days on that device, even if you
close the app. Only a hash of the token is stored on the server. Logging out, or changing your password, ends it.
This needs the Upstash database (above); with the local-file fallback it resets when Streamlit Cloud restarts.

## Install on an iPad (home-screen app)
No extra files are needed; the icon is built into `app.py`.
On the iPad open your app's address in **Safari**, tap Share, **Add to Home Screen**, then Add. It opens full screen with
its own icon. Sign in once inside the home-screen app (it keeps its own storage, separate from Safari).
