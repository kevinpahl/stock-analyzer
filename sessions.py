"""Trading sessions for Asia (plus the US) with live open/closed status.
Scheduled hours only: exchange holidays and special closures are not tracked."""
from __future__ import annotations

import pandas as pd

ASIA = [
    {"name": "Nikkei 225", "sym": "^N225", "country": "Japan", "tz": "Asia/Tokyo",
     "sessions": [("09:00", "11:30"), ("12:30", "15:30")]},
    {"name": "Hang Seng", "sym": "^HSI", "country": "Hong Kong", "tz": "Asia/Hong_Kong",
     "sessions": [("09:30", "12:00"), ("13:00", "16:00")]},
    {"name": "Shanghai Composite", "sym": "000001.SS", "country": "China", "tz": "Asia/Shanghai",
     "sessions": [("09:30", "11:30"), ("13:00", "15:00")]},
    {"name": "KOSPI", "sym": "^KS11", "country": "South Korea", "tz": "Asia/Seoul",
     "sessions": [("09:00", "15:30")]},
    {"name": "Taiwan Weighted", "sym": "^TWII", "country": "Taiwan", "tz": "Asia/Taipei",
     "sessions": [("09:00", "13:30")]},
    {"name": "ASX 200", "sym": "^AXJO", "country": "Australia", "tz": "Australia/Sydney",
     "sessions": [("10:00", "16:00")]},
    {"name": "Nifty 50", "sym": "^NSEI", "country": "India", "tz": "Asia/Kolkata",
     "sessions": [("09:15", "15:30")]},
    {"name": "Straits Times", "sym": "^STI", "country": "Singapore", "tz": "Asia/Singapore",
     "sessions": [("09:00", "17:00")]},
]
US = {"name": "US (NYSE, Nasdaq)", "sym": "^GSPC", "country": "United States", "tz": "America/New_York",
      "sessions": [("09:30", "16:00")]}


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def fmt_minutes(mins: float) -> str:
    mins = int(round(mins))
    d, rem = divmod(mins, 1440)
    h, m = divmod(rem, 60)
    return f"{d}d {h}h" if d else f"{h}h {m:02d}m" if h else f"{m}m"


def session_status(market: dict, now_utc: pd.Timestamp) -> tuple[str, str]:
    """('Open' | 'Lunch break' | 'Pre-open' | 'Closed', plain-English detail)."""
    loc = now_utc.tz_convert(market["tz"])
    mins = loc.hour * 60 + loc.minute
    spans = [(_hm(a), _hm(b)) for a, b in market["sessions"]]
    if loc.weekday() < 5:
        for i, (a, b) in enumerate(spans):
            if a <= mins < b:
                return "Open", f"closes in {fmt_minutes(b - mins)}"
            if mins < a:
                return ("Pre-open", f"opens in {fmt_minutes(a - mins)}") if i == 0 else \
                       ("Lunch break", f"reopens in {fmt_minutes(a - mins)}")
    first_h, first_m = divmod(spans[0][0], 60)
    for d in range(0, 8):
        cand = (loc + pd.Timedelta(days=d)).replace(hour=first_h, minute=first_m, second=0, microsecond=0)
        if cand > loc and cand.weekday() < 5:
            return "Closed", f"opens in {fmt_minutes((cand - loc).total_seconds() / 60)}"
    return "Closed", ""


def timeline_intervals(markets: list[dict], now_utc: pd.Timestamp, back_h: int = 10, fwd_h: int = 14) -> list[dict]:
    """Open periods in US Eastern time that overlap [now - back_h, now + fwd_h]."""
    lo, hi = now_utc - pd.Timedelta(hours=back_h), now_utc + pd.Timedelta(hours=fwd_h)
    out = []
    for m in markets:
        loc = now_utc.tz_convert(m["tz"])
        for off in (-1, 0, 1):
            day = (loc + pd.Timedelta(days=off)).date()
            if pd.Timestamp(day).weekday() >= 5:
                continue
            for a, b in m["sessions"]:
                s = pd.Timestamp(day.year, day.month, day.day, *map(int, a.split(":")), tz=m["tz"])
                e = pd.Timestamp(day.year, day.month, day.day, *map(int, b.split(":")), tz=m["tz"])
                if e > lo and s < hi:
                    out.append({"name": m["name"], "start": s.tz_convert("America/New_York").tz_localize(None),
                                "end": e.tz_convert("America/New_York").tz_localize(None)})
    return out


def futures_weekend(now_et: pd.Timestamp) -> bool:
    """True while US index futures are closed for the weekend (Friday 5pm to Sunday 6pm Eastern)."""
    wd, hr = now_et.weekday(), now_et.hour
    return wd == 5 or (wd == 6 and hr < 18) or (wd == 4 and hr >= 17)
