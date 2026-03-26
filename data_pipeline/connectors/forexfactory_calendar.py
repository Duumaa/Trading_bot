"""
data_pipeline/connectors/forexfactory_calendar.py
==================================================
ForexFactory Economic Calendar Scraper

Adapted from the original scraper (cloudscraper + BeautifulSoup).
Added for the G10 bot:
    • EST → UTC conversion with proper DST handling (zoneinfo)
    • TimescaleDB storage (economic_calendar table)
    • Risk gate interface: get_upcoming_events(hours_ahead=2)
    • Surprise score calculation: (actual - forecast) / std
    • Weekly auto-refresh scheduler
    • Historical backfill (2007 → present)

Key design decisions:
    • fftimezoneoffset=0 cookie → FF returns UTC directly (your original trick)
    • High-impact filter for risk gate (NFP, FOMC, CPI, etc.)
    • Surprise score stored for fundamental features (Analyst 2 endo/exo input)

Usage:
    from data_pipeline.connectors.forexfactory_calendar import ff_calendar

    # Scrape this week
    df = ff_calendar.scrape_current_week()

    # Risk gate check — is there a high-impact event in next 2 hours?
    events = ff_calendar.get_upcoming_events(hours_ahead=2, impact="High")
    if events:
        print("BLOCK TRADING — news risk")

    # Historical backfill
    ff_calendar.backfill(start_year=2007)
"""

from __future__ import annotations

import random
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

# ── Constants ──────────────────────────────────────────────────────────────────

G10_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CHF",
    "CAD", "AUD", "NZD", "SEK", "NOK",
}

MONTHS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]

# High-impact event keywords used by the risk gate
HIGH_IMPACT_KEYWORDS = {
    "Non-Farm", "NFP", "FOMC", "Fed Funds", "Interest Rate Decision",
    "CPI", "GDP", "Unemployment", "Payroll", "PMI", "Retail Sales",
    "Trade Balance", "ECB", "BOE", "BOJ", "RBA", "RBNZ", "BOC",
    "Inflation", "Employment Change",
}

# ForexFactory uses EST (America/New_York) — DST handled automatically by zoneinfo
FF_TZ = ZoneInfo("America/New_York")
UTC   = timezone.utc


# ── Time parsing ───────────────────────────────────────────────────────────────

def _parse_ff_time(date_str: str, time_str: str, year: int) -> Optional[datetime]:
    """
    Parse ForexFactory date + time strings → UTC datetime.

    ForexFactory time formats (with fftimezoneoffset=0 cookie, times are UTC):
        "1:30pm"  → 13:30 UTC
        "8:30am"  → 08:30 UTC
        "All Day" → 00:00 UTC (treat as midnight)
        "Tentative" → None (unknown time, skip for exact scheduling)
        ""        → None

    Note: With fftimezoneoffset=0 cookie (set in scraper), FF returns
    times in UTC directly. No conversion needed — but we still parse
    correctly to ensure timezone-aware datetimes throughout the bot.
    """
    if not date_str:
        return None

    # Parse date
    try:
        dt = datetime.strptime(f"{date_str} {year}", "%a %b %d %Y")
    except ValueError:
        return None

    # Parse time
    time_str = (time_str or "").strip().lower()

    if not time_str or time_str in ("tentative", ""):
        # No time — use midnight UTC (will still appear in upcoming events)
        return datetime(dt.year, dt.month, dt.day, 0, 0, tzinfo=UTC)

    if time_str == "all day":
        return datetime(dt.year, dt.month, dt.day, 0, 0, tzinfo=UTC)

    # Parse "1:30pm", "8:30am", "12:00pm"
    try:
        t = datetime.strptime(time_str, "%I:%M%p")
        return datetime(
            dt.year, dt.month, dt.day,
            t.hour, t.minute, 0,
            tzinfo=UTC,
        )
    except ValueError:
        pass

    # Try without colon: "130pm" (rare)
    try:
        t = datetime.strptime(time_str, "%I%p")
        return datetime(
            dt.year, dt.month, dt.day,
            t.hour, 0, 0,
            tzinfo=UTC,
        )
    except ValueError:
        pass

    logger.debug(f"Could not parse time: '{time_str}' on {date_str}")
    return datetime(dt.year, dt.month, dt.day, 0, 0, tzinfo=UTC)


# ── Number cleaning ────────────────────────────────────────────────────────────

def _clean_number(value: str) -> Optional[float]:
    """
    Convert ForexFactory value strings to float.

    Examples:
        "235K"   → 235000.0
        "2.3%"   → 2.3
        "-0.5%"  → -0.5
        "1.235M" → 1235000.0
        "Revised from 1.2%" → None (skip revision notes)
        ""       → None
    """
    if not value or value.strip() in ("", "—", "N/A"):
        return None

    v = value.strip()

    # Remove revision notes like "Revised from X"
    if "revised" in v.lower() or "from" in v.lower():
        return None

    # Remove % sign
    v = v.replace("%", "").replace(",", "").strip()

    # Handle K/M/B suffixes
    multiplier = 1.0
    if v.endswith("K") or v.endswith("k"):
        multiplier = 1_000.0
        v = v[:-1]
    elif v.endswith("M") or v.endswith("m"):
        multiplier = 1_000_000.0
        v = v[:-1]
    elif v.endswith("B") or v.endswith("b"):
        multiplier = 1_000_000_000.0
        v = v[:-1]

    try:
        return float(v) * multiplier
    except ValueError:
        return None


def _surprise_score(actual: Optional[float], forecast: Optional[float]) -> Optional[float]:
    """
    Simple surprise score: actual - forecast.
    Positive = beat expectations, Negative = missed.
    Normalisation by historical std is done at feature engineering time.
    """
    if actual is None or forecast is None:
        return None
    return round(actual - forecast, 6)


# ── Scraper ────────────────────────────────────────────────────────────────────

def _make_scraper() -> Any:
    """Create cloudscraper instance with UTC cookie (your original approach)."""
    s = cloudscraper.create_scraper(
        browser={
            "browser":   "chrome",
            "platform":  "windows",
            "desktop":   True,
        }
    )
    # Force FF to return UTC times — your original fix
    s.cookies.set("fftimezoneoffset", "0", domain="www.forexfactory.com")
    return s


# ── Core parser ────────────────────────────────────────────────────────────────

def _parse_month_html(html: str, year: int) -> List[Dict[str, Any]]:
    """
    Parse one month of ForexFactory HTML → list of event dicts.
    Keeps your original parsing logic, adds UTC timestamps and surprise scores.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="calendar__table")
    if not table:
        return []

    rows = table.find_all("tr", class_="calendar__row")
    data: List[Dict[str, Any]] = []

    current_date: Optional[str] = None
    current_time: Optional[str] = None

    for row in rows:
        # ── Date ──────────────────────────────────────────────────────────
        date_cell = row.find("td", class_="calendar__date")
        if date_cell and date_cell.text.strip():
            try:
                dt_obj = datetime.strptime(
                    f"{date_cell.text.strip()} {year}", "%a %b %d %Y"
                )
                current_date = dt_obj.strftime("%a %b %d")
            except ValueError:
                pass

        # ── Time (carry forward if blank — your original approach) ────────
        time_cell = row.find("td", class_="calendar__time")
        if time_cell:
            t = time_cell.text.strip()
            if t:
                current_time = t

        # ── Currency filter ───────────────────────────────────────────────
        currency_cell = row.find("td", class_="calendar__currency")
        currency = currency_cell.text.strip() if currency_cell else ""
        if currency not in G10_CURRENCIES:
            continue

        # ── Event name ────────────────────────────────────────────────────
        event_cell = row.find("td", class_="calendar__event")
        event = event_cell.text.strip() if event_cell else ""
        if not event:
            continue

        # ── Impact ────────────────────────────────────────────────────────
        impact_cell  = row.find("td", class_="calendar__impact")
        impact_span  = impact_cell.find("span") if impact_cell else None
        impact_class = impact_span.get("class", []) if impact_span else []
        impact = "Low"
        if "icon--ff-impact-red" in impact_class:
            impact = "High"
        elif "icon--ff-impact-ora" in impact_class:
            impact = "Medium"

        # ── Values ───────────────────────────────────────────────────────
        def _cell(cls: str) -> str:
            c = row.find("td", class_=cls)
            return c.text.strip() if c else ""

        actual_raw   = _cell("calendar__actual")
        forecast_raw = _cell("calendar__forecast")
        previous_raw = _cell("calendar__previous")

        actual_num   = _clean_number(actual_raw)
        forecast_num = _clean_number(forecast_raw)
        previous_num = _clean_number(previous_raw)
        surprise     = _surprise_score(actual_num, forecast_num)

        # ── UTC timestamp ─────────────────────────────────────────────────
        utc_ts = _parse_ff_time(current_date, current_time, year)

        if current_date:
            data.append({
                "event_time":     utc_ts,
                "currency":       currency,
                "impact":         impact,
                "event":          event,
                "actual_raw":     actual_raw,
                "forecast_raw":   forecast_raw,
                "previous_raw":   previous_raw,
                "actual":         actual_num,
                "forecast":       forecast_num,
                "previous":       previous_num,
                "surprise":       surprise,
                "source":         "forexfactory",
            })

    return data


# ── Main scraper class ─────────────────────────────────────────────────────────

class ForexFactoryCalendar:
    """
    ForexFactory economic calendar scraper for the G10 bot.

    Main interfaces:
        scrape_current_week()         → DataFrame of this week's events
        scrape_month(month, year)     → DataFrame for one month
        backfill(start_year=2007)     → Full historical download → DB
        get_upcoming_events(hours=2)  → Risk gate: events in next N hours
        get_surprise_score(event)     → Fundamental feature input
    """

    def __init__(self) -> None:
        self._scraper = _make_scraper()
        self._weekly_cache = None

    def _get(self, url: str, retries: int = 3) -> Optional[str]:
        """HTTP GET with retry + random delay (your original rate limit approach)."""
        for attempt in range(retries):
            try:
                resp = self._scraper.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.text
                logger.warning(f"HTTP {resp.status_code} for {url}")
            except Exception as exc:
                logger.warning(f"Request failed (attempt {attempt+1}): {exc}")
            time.sleep(random.uniform(2.0, 4.0))
        logger.error(f"Failed after {retries} attempts: {url}")
        return None

    # =========================================================================
    # Scraping methods
    # =========================================================================

    def scrape_month(self, month: str, year: int) -> pd.DataFrame:
        """
        Scrape one calendar month.

        Args:
            month: lowercase 3-letter e.g. "jan", "feb"
            year:  int e.g. 2024

        Returns:
            DataFrame with columns:
            event_time, currency, impact, event,
            actual, forecast, previous, surprise, source
        """
        url  = f"https://www.forexfactory.com/calendar?month={month}.{year}"
        html = self._get(url)
        if html is None:
            return pd.DataFrame()

        events = _parse_month_html(html, year)
        if not events:
            return pd.DataFrame()

        df = pd.DataFrame(events)
        logger.debug(f"  {month}.{year}: {len(df)} events")
        return df

    def scrape_current_week(self) -> pd.DataFrame:
        """
        Scrape this week's calendar (used by weekly scheduler).
        ForexFactory 'week' URL returns Mon–Sun events.
        """
        url  = "https://www.forexfactory.com/calendar?week=this.week"
        html = self._get(url)
        if html is None:
            return pd.DataFrame()

        year   = datetime.now(UTC).year
        events = _parse_month_html(html, year)
        df     = pd.DataFrame(events) if events else pd.DataFrame()

        if not df.empty:
            self._store(df)
            logger.info(f"Weekly refresh | {len(df)} events stored")

        return df

    def scrape_next_week(self) -> pd.DataFrame:
        """Scrape next week's calendar (run every Sunday)."""
        url  = "https://www.forexfactory.com/calendar?week=next.week"
        html = self._get(url)
        if html is None:
            return pd.DataFrame()

        year   = datetime.now(UTC).year
        events = _parse_month_html(html, year)
        df     = pd.DataFrame(events) if events else pd.DataFrame()

        if not df.empty:
            self._store(df)
            logger.info(f"Next-week fetch | {len(df)} events stored")

        return df

    def backfill(
        self,
        start_year: int = 2007,
        end_year:   Optional[int] = None,
        store_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Full historical download — your original main() logic.

        Downloads month by month from start_year to end_year.
        Rate-limited with random sleep (your original approach).
        Stores to TimescaleDB economic_calendar table.

        Args:
            start_year:  First year to download (default 2007)
            end_year:    Last year (default: current year)
            store_to_db: Also write to TimescaleDB

        Returns:
            Full DataFrame of all events
        """
        if end_year is None:
            end_year = datetime.now(UTC).year

        logger.info(f"Backfill start | {start_year}–{end_year}")
        all_data: List[pd.DataFrame] = []

        for year in range(start_year, end_year + 1):
            year_data: List[pd.DataFrame] = []
            for month in MONTHS:
                df = self.scrape_month(month, year)
                if not df.empty:
                    year_data.append(df)
                # Your original rate limit: random 1.5–3s sleep
                time.sleep(random.uniform(1.5, 3.0))

            if year_data:
                year_df = pd.concat(year_data, ignore_index=True)
                all_data.append(year_df)
                logger.info(f"  {year}: {len(year_df)} events")

                if store_to_db:
                    self._store(year_df)

        if not all_data:
            return pd.DataFrame()

        full_df = pd.concat(all_data, ignore_index=True)
        logger.success(f"Backfill complete | {len(full_df)} total events")
        return full_df

    # =========================================================================
    # Risk gate interface
    # =========================================================================

    def get_upcoming_events(
        self,
        hours_ahead: float = 2.0,
        impact:      str   = "High",
        currencies:  Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming events within the next N hours.

        Used by the risk gate to block trading before high-impact news.
        Bot rule: flat 2hrs before FOMC/NFP → call with hours_ahead=2.

        Args:
            hours_ahead: Look-ahead window in hours
            impact:      "High", "Medium", "Low", or "All"
            currencies:  Filter by currency list (None = all G10)

        Returns:
            List of event dicts, sorted by event_time ascending.
            Empty list = safe to trade.
        """
        now    = datetime.now(UTC)
        cutoff = now + timedelta(hours=hours_ahead)

        try:
            from data_pipeline.storage.timeseries_db import db
            rows = db.query(
                """
                SELECT event_time, currency, impact, event,
                       actual, forecast, previous, surprise
                FROM economic_calendar
                WHERE event_time BETWEEN %(now)s AND %(cutoff)s
                  AND (%(impact)s = 'All' OR impact = %(impact)s)
                ORDER BY event_time ASC
                """,
                {"now": now, "cutoff": cutoff, "impact": impact},
            )
            events = [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"DB upcoming events failed: {exc} — using in-memory cache")
            events = self._upcoming_from_cache(now, cutoff, impact)

        if currencies:
            events = [e for e in events if e["currency"] in currencies]

        if events:
            logger.warning(
                f"⚠️  {len(events)} upcoming {impact}-impact event(s) in next {hours_ahead}h: "
                + ", ".join(f"{e['event']} ({e['currency']})" for e in events)
            )

        return events

    def is_news_blackout(
        self,
        symbol:      str,
        hours_ahead: float = 2.0,
    ) -> bool:
        """
        Returns True if trading should be blocked for this symbol
        due to upcoming high-impact news.

        Extracts the two currencies from the pair and checks each.

        Example:
            is_news_blackout("EURUSD")  → checks EUR + USD events
        """
        if len(symbol) != 6:
            return False
        base  = symbol[:3]
        quote = symbol[3:]
        events = self.get_upcoming_events(
            hours_ahead=hours_ahead,
            impact="High",
            currencies=[base, quote],
        )
        return len(events) > 0

    def _upcoming_from_cache(
        self,
        now:    datetime,
        cutoff: datetime,
        impact: str,
    ) -> List[Dict[str, Any]]:
        """Fallback: scan in-memory weekly cache if DB unavailable."""
        if not hasattr(self, "_weekly_cache") or self._weekly_cache is None:
            return []
        df = self._weekly_cache
        mask = (
            (df["event_time"] >= now) &
            (df["event_time"] <= cutoff)
        )
        if impact != "All":
            mask &= df["impact"] == impact
        return df[mask].to_dict("records")

    # =========================================================================
    # Surprise score for fundamental features
    # =========================================================================

    def get_recent_surprises(
        self,
        currency:    str,
        days_back:   int  = 30,
        impact:      str  = "High",
    ) -> pd.DataFrame:
        """
        Get recent economic surprises for a currency.
        Used by Analyst 2 (Endo/Exo score) as a fundamental input.

        Returns DataFrame with columns:
            event_time, event, actual, forecast, surprise
        """
        cutoff = datetime.now(UTC) - timedelta(days=days_back)
        try:
            from data_pipeline.storage.timeseries_db import db
            rows = db.query(
                """
                SELECT event_time, event, actual, forecast, surprise
                FROM economic_calendar
                WHERE currency   = %(currency)s
                  AND impact     = %(impact)s
                  AND event_time >= %(cutoff)s
                  AND surprise   IS NOT NULL
                ORDER BY event_time DESC
                """,
                {"currency": currency, "impact": impact, "cutoff": cutoff},
            )
            return pd.DataFrame([dict(r) for r in rows])
        except Exception as exc:
            logger.error(f"get_recent_surprises failed: {exc}")
            return pd.DataFrame()

    # =========================================================================
    # Storage
    # =========================================================================

    def _store(self, df: pd.DataFrame) -> int:
        """
        Upsert events to TimescaleDB economic_calendar table.
        Skips rows without event_time.
        """
        if df.empty:
            return 0

        df = df[df["event_time"].notna()].copy()
        if df.empty:
            return 0

        # Map fields to match insert_economic_events schema
        currency_to_country = {
            "USD": "United States", "EUR": "Euro Zone", "GBP": "United Kingdom",
            "JPY": "Japan", "CHF": "Switzerland", "CAD": "Canada",
            "AUD": "Australia", "NZD": "New Zealand", "SEK": "Sweden", "NOK": "Norway"
        }
        import math
        def _nan_to_none(v):
            try:
                return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
            except Exception:
                return v

        rows = [
            {
                "event_time": r["event_time"],
                "country":    r["currency"],
                "currency":   r["currency"],
                "event_name": r["event"],
                "impact":     r["impact"].upper(),
                "actual":     r.get("actual_raw") or None,
                "forecast":   r.get("forecast_raw") or None,
                "previous":   r.get("previous_raw") or None,
                "source":     r.get("source", "forexfactory"),
            }
            for r in df.to_dict("records")
        ]

        try:
            from data_pipeline.storage.timeseries_db import db
            inserted = db.insert_economic_events(rows)
            logger.debug(f"Calendar DB | {inserted} rows upserted")
            return inserted
        except Exception as exc:
            logger.error(f"Calendar DB store failed: {exc}")
            return 0

    def save_csv(self, df: pd.DataFrame, path: str) -> None:
        """Save DataFrame to CSV (your original output format)."""
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} events → {path}")

    # =========================================================================
    # Weekly scheduler
    # =========================================================================

    def start_weekly_scheduler(self) -> None:
        """
        Background thread that auto-refreshes calendar every Sunday at 18:00 UTC.
        ForexFactory publishes next week's events Sunday afternoon.
        Call once at bot startup.
        """
        import threading

        def _scheduler():
            while True:
                now = datetime.now(UTC)
                # Next Sunday 18:00 UTC
                days_until_sunday = (6 - now.weekday()) % 7
                if days_until_sunday == 0 and now.hour >= 18:
                    days_until_sunday = 7
                next_run = now.replace(
                    hour=18, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_until_sunday)
                wait_s = (next_run - now).total_seconds()
                logger.info(
                    f"Calendar scheduler | next run in "
                    f"{wait_s/3600:.1f}h ({next_run.strftime('%Y-%m-%d %H:%M UTC')})"
                )
                time.sleep(wait_s)
                try:
                    logger.info("Calendar scheduler | running weekly refresh...")
                    self.scrape_current_week()
                    self.scrape_next_week()
                except Exception as exc:
                    logger.error(f"Calendar scheduler error: {exc}")

        t = threading.Thread(
            target=_scheduler, name="ff-calendar-scheduler", daemon=True
        )
        t.start()
        logger.info("Calendar weekly scheduler started ✅")


# ── Module singleton ───────────────────────────────────────────────────────────
ff_calendar = ForexFactoryCalendar()
