"""
main_loop.py
=============
G10 Autonomous Dominance Architecture — Main Trading Loop

The central orchestrator. Runs continuously, waking on every M15
candle close to evaluate signals and manage positions.

Thread architecture:
  Main thread    — M15 bar loop (signal eval + order execution)
  Thread 2       — Live data stream (cTrader ticks + order book)
  Thread 3       — Feature schedulers (stat/vwap/flow/liq every N min)
  Thread 4       — Data pipeline (news/COT/yields every 4h)
  Thread 5       — Sentiment scheduler (Groq every 4h)
  Thread 6       — Opus scheduler (weekly Sunday 20:00 UTC)
  Thread 7       — Signal aggregator scheduler (every 15min)
  Thread 8       — Daily snapshot + reset (23:50 UTC)

M15 bar loop sequence (every 15 minutes):
  1.  Fetch latest M15 candles from cTrader → store DB
  2.  Refresh stat + vwap features (if cache stale)
  3.  Run regime classifier (Analyst 1)
  4.  Run exogenous score (Analyst 2)
  5.  Evaluate all 12 pairs through signal aggregator
  6.  For each tradeable signal:
        a. Risk manager pre-trade check
        b. Calculate position sizing
        c. Place market order via cTrader
  7.  Monitor open positions (check SL/TP hit via reconcile)
  8.  Log performance metrics

Startup sequence:
  1.  Load .env
  2.  Connect TimescaleDB + Redis
  3.  Connect cTrader (authenticate)
  4.  Fetch last 3 days M15 candles (warm up features)
  5.  Compute all features (cold start)
  6.  Run Analyst 2 (exogenous scores)
  7.  Check Opus weekly analysis (run if stale)
  8.  Run Groq sentiment (if stale)
  9.  Start all background threads
  10. Enter M15 main loop

Graceful shutdown:
  SIGINT / SIGTERM → close all positions → stop threads → exit

Usage:
  python3 main_loop.py              # start bot
  python3 main_loop.py --dry-run    # evaluate signals, no orders
  python3 main_loop.py --status     # print current state and exit
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from dotenv import load_dotenv
from intelligence.monitoring.telegram_alerts import alerts

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))
load_dotenv(str(ROOT / ".env"))

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

M15_SECONDS          = 900     # 15 minutes
CANDLE_WARMUP_DAYS   = 3       # fetch last N days on startup
FEATURE_REFRESH_MIN  = 14      # refresh stat/vwap if > N min old
DATA_PIPELINE_HOURS  = 4       # news/COT/yields refresh interval
CALENDAR_HOURS       = 1       # ForexFactory calendar refresh interval
SENTIMENT_HOURS      = 4       # Groq sentiment refresh interval
POSITION_SYNC_BARS   = 4       # sync positions every N bars (1h)
DAILY_SNAPSHOT_HOUR  = 23      # UTC hour for daily snapshot
DAILY_SNAPSHOT_MIN   = 50

# Market session hours (UTC)
SYDNEY_OPEN   = 21   # Sunday 21:00 UTC
TOKYO_OPEN    = 0    # 00:00 UTC
LONDON_OPEN   = 7    # 07:00 UTC
NY_OPEN       = 12   # 12:00 UTC
NY_CLOSE      = 21   # 21:00 UTC (weekdays)

# ══════════════════════════════════════════════════════════════
# MARKET HOURS
# ══════════════════════════════════════════════════════════════

def _is_market_open() -> bool:
    """
    Check if forex market is currently open for trading.

    Closed windows:
      - Saturday all day
      - Sunday before 21:00 UTC (market closed)
      - Sunday 21:00 → Monday 23:30 UTC (thin liquidity blackout)
        → Asian open gap window: wide spreads, FIX drops, SL hunts
        → In Mongolia time: market opens 06:00 Mon but bot waits until 07:30
      - Any day 21:00 → 22:00 UTC (daily rollover / session transition)
        → Spreads widen briefly during broker rollover
    """
    now     = datetime.now(timezone.utc)
    weekday = now.weekday()   # 0=Mon, 5=Sat, 6=Sun
    hour    = now.hour
    minute  = now.minute

    # Saturday — always closed
    if weekday == 5:
        return False

    # Sunday before 22:00 UTC — market closed + early Asian gap window
    if weekday == 6:
        return False

    # Monday before 23:30 UTC — thin liquidity blackout
    # FIX collector drops connections, spreads wide, VWAP stale
    # Mongolia time: 06:00-07:30 Mon = 22:00 Sun - 23:30 Mon UTC
    if weekday == 0 and hour == 7 and minute < 30:
        return False
    if weekday == 0 and hour < 7:
        return False

    # Daily rollover blackout — 21:00-22:00 UTC every day
    # Broker widens spreads during session transition
    if hour == 21:
        return False

    return True


def _seconds_until_next_m15() -> float:
    """
    Calculate seconds until the next M15 candle close.
    M15 candles close at :00, :15, :30, :45 each hour.
    Add 2 seconds buffer for candle to be written to DB.
    """
    now     = datetime.now(timezone.utc)
    minutes = now.minute
    seconds = now.second
    micros  = now.microsecond / 1_000_000

    # Minutes until next :00, :15, :30, :45
    next_bar_min = ((minutes // 15) + 1) * 15
    wait_min     = next_bar_min - minutes
    wait_sec     = (wait_min * 60) - seconds - micros + 2.0  # 2s buffer

    return max(wait_sec, 1.0)


def _next_m15_time() -> datetime:
    """Return the datetime of the next M15 candle close."""
    now         = datetime.now(timezone.utc)
    wait        = _seconds_until_next_m15()
    return now + timedelta(seconds=wait)


# ══════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════

def _startup_data_fetch(connector) -> bool:
    """
    Fetch last N days of M15 candles on startup.
    Warms up the feature engines with recent data.
    """
    logger.info(f"MainLoop | Fetching last {CANDLE_WARMUP_DAYS} days M15 candles...")
    try:
        connector.get_all_pairs_historical(
            timeframe="M15",
            days_back=CANDLE_WARMUP_DAYS,
            store_to_db=True,
        )
        logger.success("MainLoop | Candle warmup complete")
        return True
    except Exception as exc:
        logger.error(f"MainLoop | Candle fetch failed: {exc}")
        return False


def _startup_features() -> bool:
    """
    Compute all feature engines on startup (cold start).
    Populates Redis with fresh features before first bar evaluation.
    """
    logger.info("MainLoop | Computing features (cold start)...")

    engines = [
        ("yield",       "features.yield_features",       "yield_feature_engine"),
        ("statistical", "features.statistical_features", "stat_engine"),
        ("vwap",        "features.vwap_features",        "vwap_engine"),
        ("event",       "features.event_features",       "event_engine"),
        ("orderflow",   "features.orderflow_features",   "flow_engine"),
        ("liquidity",   "features.liquidity_features",   "liq_engine"),
    ]

    for name, module_path, singleton_name in engines:
        try:
            import importlib
            mod    = importlib.import_module(module_path)
            engine = getattr(mod, singleton_name)
            engine.compute_all()
            logger.success(f"MainLoop | {name} features ready")
        except Exception as exc:
            logger.warning(f"MainLoop | {name} features failed: {exc}")

    return True


def _startup_analysts() -> bool:
    """
    Run Analysts 1 + 2 on startup.
    """
    logger.info("MainLoop | Running startup analysis...")
    from data_pipeline.connectors.ctrader_rest_api import connector

    # Analyst 2 — exogenous scores
    try:
        from models.analyst2_endo_exo.score_calculator import analyst2
        analyst2.compute_all()
        logger.success("MainLoop | Analyst 2 (Endo/Exo) ready")
    except Exception as exc:
        logger.warning(f"MainLoop | Analyst 2 startup failed: {exc}")

    # Analyst 1 — regime classifier
    try:
        from models.analyst1_regime.regime_classifier import regime_classifier
        regime_classifier.compute_all()
        logger.success("MainLoop | Analyst 1 (Regime) ready")
    except Exception as exc:
        logger.warning(f"MainLoop | Analyst 1 startup failed: {exc}")

    # Analyst 6 — Opus weekly (run if no cached analysis)
    try:
        from models.analyst6_opus.opus_deep_analysis import opus_analyst
        existing = opus_analyst.get_weekly_analysis()
        if not existing:
            logger.info("MainLoop | No Opus analysis found — running now...")
            opus_analyst.run_weekly()
        else:
            week_of = existing.get("week_of", "unknown")
            logger.success(f"MainLoop | Opus analysis loaded (week_of={week_of})")
    except Exception as exc:
        logger.warning(f"MainLoop | Opus startup failed: {exc}")

    # Analyst 5 — Groq sentiment (run if stale)
    try:
        from models.analyst5_sentiment.groq_sentiment import groq_sentiment
        import redis as redis_lib, os, json
        r   = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
        )
        has_sentiment = r.get("features:sentiment:EURUSD")
        if not has_sentiment:
            logger.info("MainLoop | No sentiment found — running Groq...")
            groq_sentiment.run_daily()
        else:
            logger.success("MainLoop | Groq sentiment loaded from cache")
    except Exception as exc:
        logger.warning(f"MainLoop | Groq startup failed: {exc}")
    # Start Telegram command bot
    # Start Telegram command bot
    alerts.start_command_bot()

    # Subscribe to real-time spot prices via REST API connection
    try:
        pairs = [
            "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","NZDUSD",
            "USDCAD","EURGBP","EURJPY","GBPJPY","AUDJPY","NZDJPY"
        ]
        connector.subscribe_spots(pairs)
        logger.success("MainLoop | Spot price subscriptions active ✅")
    except Exception as exc:
        logger.warning(f"MainLoop | Spot subscription failed: {exc}")

    return True


# ══════════════════════════════════════════════════════════════
# BAR PROCESSING
# ══════════════════════════════════════════════════════════════

def _fetch_latest_candles(connector) -> bool:
    """Fetch last 1 day of M15 candles to update DB."""
    try:
        connector.get_all_pairs_historical(
            timeframe="M15",
            days_back=1,
            store_to_db=True,
        )
        return True
    except Exception as exc:
        logger.warning(f"MainLoop | Candle refresh failed: {exc}")
        return False


def _refresh_fast_features() -> None:
    """
    Refresh stat + vwap features (runs every M15 bar).
    These are the fastest-moving features that need fresh data
    each bar for accurate regime and signal computation.
    """
    fast_engines = [
        ("features.statistical_features", "stat_engine"),
        ("features.vwap_features",        "vwap_engine"),
        ("features.event_features",       "event_engine"),
    ]
    for module_path, singleton_name in fast_engines:
        try:
            import importlib
            mod    = importlib.import_module(module_path)
            engine = getattr(mod, singleton_name)
            engine.compute_all()
        except Exception as exc:
            logger.debug(f"MainLoop | Feature refresh warning: {exc}")


def _process_bar(
    bar_number:   int,
    dry_run:      bool,
    connector,
) -> List[Dict]:
    """
    Process a single M15 bar.
    Returns list of order results from this bar.
    """
    now = datetime.now(timezone.utc)
    logger.info(
        f"MainLoop | ── BAR {bar_number} "
        f"{now.strftime('%Y-%m-%d %H:%M UTC')} ──"
    )
    results = []

    # Respect Telegram /pause command
    if alerts.is_paused:
        logger.info("MainLoop | Trading paused via Telegram -- skipping bar")
        return result
    # 1. Fetch latest candles
    _fetch_latest_candles(connector)

    # Sync live balance every M15 cycle so kill switch uses real numbers
    try:
        account_info = connector.get_account_balance()
        if account_info:
            balance = account_info.get("balance", 0.0)
            equity  = account_info.get("equity",  balance)
            if balance > 0:
                from execution.risk_manager import risk_manager
                risk_manager.account.update(balance, equity)
                logger.debug(f"MainLoop | Balance synced: ${balance:,.2f}")
    except Exception as exc:
        logger.warning(f"MainLoop | Balance sync skipped: {exc}")

    # 2. Refresh fast features
    _refresh_fast_features()

    # 3. Run regime classifier
    try:
        from models.analyst1_regime.regime_classifier import regime_classifier
        regime_classifier.compute_all()
    except Exception as exc:
        logger.warning(f"MainLoop | Regime compute failed: {exc}")

    # 4. Evaluate signals
    try:
        from execution.signal_aggregator import signal_aggregator
        decisions = signal_aggregator.evaluate_all()
        tradeable = sorted(
            [d for d in decisions.values() if d.get("action") in ("LONG", "SHORT")],
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )
        logger.info(
            f"MainLoop | Signals: "
            f"{sum(1 for d in decisions.values() if d['action']=='FLAT')} FLAT | "
            f"{len(tradeable)} tradeable: "
            f"{[d['pair'] for d in tradeable]}"
        )
    except Exception as exc:
        logger.error(f"MainLoop | Signal evaluation failed: {exc}")
        return results

    # 4b. Technical invalidation check on open positions
    try:
        from execution.technical_exit_monitor import tech_exit_monitor
        exit_actions = tech_exit_monitor.run_checks()
        closed = [p for p, a in exit_actions.items() if a in ("CLOSED_FLIP", "CLOSED_DECAY")]
        if closed:
            logger.warning(f"MainLoop | TechExit closed positions: {closed}")
    except Exception as exc:
        logger.error(f"MainLoop | Technical exit monitor failed: {exc}")

    # 5. Execute signals
    if not dry_run and tradeable:
        from execution.order_executor import order_executor
        from execution.risk_manager   import risk_manager

        # Reconcile Redis positions against cTrader reality
        # Clears stale keys for positions closed via SL/TP without bot noticing
        order_executor.reconcile_positions()

        if risk_manager.is_halted():
            logger.warning("MainLoop | Trading halted — skipping execution")
            return results

        for decision in tradeable:
            try:
                result = order_executor.execute_signal(decision)
                if result and result.get("success"):
                    results.append(result)
                    logger.success(
                        f"MainLoop | ✅ Executed: "
                        f"{decision['pair']} {decision['action']} "
                        f"@ {result.get('fill_price', '?')}"
                    )
            except Exception as exc:
                logger.error(
                    f"MainLoop | Execution failed "
                    f"{decision['pair']}: {exc}"
                )

    elif dry_run and tradeable:
        logger.info(
            f"MainLoop | DRY RUN — would execute: "
            f"{[(d['pair'], d['action']) for d in tradeable]}"
        )

    # 6. Sync positions every POSITION_SYNC_BARS bars
    if bar_number % POSITION_SYNC_BARS == 0:
        try:
            from execution.order_executor import order_executor
            open_pos = order_executor.get_open_positions_local()
            if open_pos:
                logger.info(
                    f"MainLoop | Open positions: "
                    f"{[p['pair'] for p in open_pos]}"
                )
        except Exception as exc:
            logger.debug(f"MainLoop | Position sync: {exc}")

    return results


# ══════════════════════════════════════════════════════════════
# BACKGROUND THREADS
# ══════════════════════════════════════════════════════════════

def _start_background_threads(dry_run: bool) -> None:
    """Start all background scheduler threads."""

    # ── Live data stream ──────────────────────────────────────
    def _stream_thread():
        from data_pipeline.connectors.ctrader_live_stream import live_stream
        while True:
            try:
                live_stream.connect()
                logger.info("MainLoop | Live stream started")
            except Exception as exc:
                logger.warning(
                    f"MainLoop | Live stream failed: {exc} "
                    f"— retry in 30s"
                )
            time.sleep(30)

    def _tick_watchdog_thread():
        """
        Watchdog: checks every 5 min if ticks are arriving.
        If no tick in DB for 10+ minutes → force live stream reconnect.
        Sends Telegram alert on reconnect.
        """
        import time as _time
        from data_pipeline.storage.timeseries_db import db
        from sqlalchemy import text as sa_text

        logger.info("MainLoop | Tick watchdog started")
        _time.sleep(120)  # wait 2 min after startup before first check

        while True:
            try:
                with db.connection() as conn:
                    row = conn.execute(sa_text("""
                        SELECT NOW() - MAX(time) as age
                        FROM ticks
                        WHERE time > NOW() - INTERVAL '30 minutes'
                    """)).fetchone()

                age_seconds = None
                if row and row[0]:
                    age_seconds = row[0].total_seconds()

                if age_seconds is None or age_seconds > 600:
                    # No ticks for 10+ minutes — reconnect
                    age_str = f"{int(age_seconds)}s" if age_seconds else "unknown"
                    logger.warning(
                        f"MainLoop | Tick watchdog: no ticks for {age_str} "
                        f"— forcing live stream reconnect"
                    )
                    try:
                        from data_pipeline.connectors.ctrader_live_stream import live_stream
                        live_stream.disconnect()
                        _time.sleep(3)
                        live_stream.connect()
                        logger.success("MainLoop | Tick watchdog: FIX collector restarted")
                        try:
                            alerts.send_message(
                                f"⚠️ Tick watchdog: no ticks for {age_str}\n"
                                f"Live stream reconnected automatically."
                            )
                        except Exception:
                            pass
                    except Exception as exc:
                        logger.error(f"MainLoop | Tick watchdog reconnect failed: {exc}")
                else:
                    logger.debug(
                        f"MainLoop | Tick watchdog: last tick {int(age_seconds)}s ago ✅"
                    )

            except Exception as exc:
                logger.warning(f"MainLoop | Tick watchdog error: {exc}")

            _time.sleep(300)  # check every 5 minutes

    # ── Data pipeline (news/COT/yields every 4h) ──────────────
    def _data_pipeline_thread():
        logger.info("MainLoop | Data pipeline thread started")
        while True:
            try:
                # News scraper
                from data_pipeline.connectors.rss_news_scraper import news_scraper
                news_scraper.fetch_all()
            except Exception as exc:
                logger.debug(f"MainLoop | News scrape: {exc}")

            try:
                # ForexFactory calendar
                from data_pipeline.connectors.forexfactory_calendar import ff_calendar
                ff_calendar.scrape_current_week()
                ff_calendar.scrape_next_week()
            except Exception as exc:
                logger.debug(f"MainLoop | Calendar scrape: {exc}")

            try:
                # Yield scraper
                from data_pipeline.connectors.yield_scraper import yield_scraper
                yield_scraper.fetch_latest()
            except Exception as exc:
                logger.debug(f"MainLoop | Yield scrape: {exc}")
            try:
                # Commodity prices (Oil + Gold) — FRED + yfinance
                import os, requests, yfinance as yf, pandas as pd
                from sqlalchemy import text
                from data_pipeline.storage.timeseries_db import db
                # WTI Oil via FRED
                fred_key = os.getenv("FRED_API_KEY")
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": "DCOILWTICO", "api_key": fred_key,
                            "observation_start": "2024-01-01",
                            "file_type": "json", "sort_order": "asc"},
                    timeout=30
                )
                oil_rows = [(o["date"], float(o["value"]))
                            for o in resp.json().get("observations", [])
                            if o["value"] not in (".", "")]
                # Gold via yfinance
                gold_df = yf.download("GC=F", start="2024-01-01",
                                      progress=False, auto_adjust=True)
                gold_rows = [(d.strftime("%Y-%m-%d"), float(r["Close"].iloc[0])
                              if hasattr(r["Close"], "iloc") else float(r["Close"]))
                             for d, r in gold_df.iterrows()
                             if not pd.isna(r["Close"].iloc[0]
                             if hasattr(r["Close"], "iloc") else r["Close"])]
                with db.connection() as conn:
                    for date, price in oil_rows:
                        conn.execute(text("""
                            INSERT INTO macro_indicators
                                (time, country, indicator, actual, source)
                            VALUES (:t, 'OIL', 'WTI_CLOSE', :p, 'fred')
                            ON CONFLICT DO NOTHING
                        """), {"t": f"{date}T00:00:00+00:00", "p": price})
                    for date, price in gold_rows:
                        conn.execute(text("""
                            INSERT INTO macro_indicators
                                (time, country, indicator, actual, source)
                            VALUES (:t, 'XAU', 'GOLD_CLOSE', :p, 'yfinance')
                            ON CONFLICT DO NOTHING
                        """), {"t": f"{date}T00:00:00+00:00", "p": price})
                    conn.commit()
                logger.info(f"MainLoop | Commodities refreshed: {len(oil_rows)} oil, {len(gold_rows)} gold rows")
            except Exception as exc:
                logger.debug(f"MainLoop | Commodity refresh: {exc}")
            time.sleep(CALENDAR_HOURS * 3600)

    # ── Groq sentiment (every 4h) ─────────────────────────────
    def _sentiment_thread():
        logger.info("MainLoop | Sentiment thread started")
        while True:
            try:
                from models.analyst5_sentiment.groq_sentiment import groq_sentiment
                groq_sentiment.run_daily()
            except Exception as exc:
                logger.warning(f"MainLoop | Sentiment thread: {exc}")
            time.sleep(SENTIMENT_HOURS * 3600)

    # ── Analyst 2 refresh (every 4h) ─────────────────────────
    def _analyst2_thread():
        logger.info("MainLoop | Analyst 2 thread started")
        while True:
            time.sleep(4 * 3600)
            try:
                from models.analyst2_endo_exo.score_calculator import analyst2
                analyst2.compute_all()
            except Exception as exc:
                logger.warning(f"MainLoop | Analyst 2 refresh: {exc}")

    # ── Opus weekly (Sunday 20:00 UTC) ────────────────────────
    def _opus_thread():
        try:
            from models.analyst6_opus.opus_deep_analysis import opus_analyst
            opus_analyst.start_scheduler()
        except Exception as exc:
            logger.warning(f"MainLoop | Opus scheduler: {exc}")

    # ── Daily snapshot + reset (23:50 UTC) ───────────────────
    def _daily_thread():
        logger.info("MainLoop | Daily thread started")
        while True:
            now = datetime.now(timezone.utc)
            # Calculate seconds until 23:50 UTC
            target = now.replace(hour=DAILY_SNAPSHOT_HOUR,
                                 minute=DAILY_SNAPSHOT_MIN,
                                 second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            wait = (target - now).total_seconds()
            time.sleep(wait)

            try:
                from execution.risk_manager import risk_manager
                risk_manager.take_daily_snapshot()
                risk_manager.account.reset_daily()
                logger.success("MainLoop | Daily snapshot taken + P&L reset")

                # Monthly portfolio stop baseline reset (1st of each month)
                if datetime.now(timezone.utc).day == 1:
                    import redis, os
                    r = redis.Redis(
                        host=os.getenv("REDIS_HOST", "localhost"),
                        port=int(os.getenv("REDIS_PORT", "6379")),
                        password=os.getenv("REDIS_PASSWORD") or None,
                        decode_responses=True
                    )
                    balance = risk_manager.account.get().get("balance", 0.0)
                    r.set("risk:initial_balance", str(balance))
                    logger.info(f"MainLoop | Monthly portfolio stop reset: ${balance:,.2f}")

            except Exception as exc:
                logger.error(f"MainLoop | Daily snapshot failed: {exc}")
            # Wait until 00:05 UTC next day then do a second reset
            # to ensure kill switch is cleared before Asian session opens
            try:
                now2 = datetime.now(timezone.utc)
                next_day = (now2 + timedelta(days=1)).replace(
                    hour=0, minute=5, second=0, microsecond=0)
                time.sleep(max(0, (next_day - now2).total_seconds()))
                risk_manager.account.reset_daily()
                logger.success("MainLoop | 00:05 UTC reset complete — kill switch cleared")
            except Exception as exc:
                logger.error(f"MainLoop | 00:05 reset failed: {exc}")

    # ── COT downloader (weekly Friday) ────────────────────────
    def _cot_thread():
        logger.info("MainLoop | COT thread started")
        while True:
            now = datetime.now(timezone.utc)
            # Run every Friday at 22:00 UTC (after CFTC release)
            days_until_friday = (4 - now.weekday()) % 7
            if days_until_friday == 0 and now.hour >= 22:
                days_until_friday = 7
            next_friday = now.replace(hour=22, minute=0, second=0) + \
                          timedelta(days=days_until_friday)
            wait = (next_friday - now).total_seconds()
            time.sleep(wait)
            try:
                from data_pipeline.connectors.cot_downloader import cot_downloader
                df = cot_downloader.fetch_latest()
                if len(df) > 0:
                    cot_downloader._store(df)
                logger.success("MainLoop | COT data updated")
            except Exception as exc:
                logger.warning(f"MainLoop | COT update failed: {exc}")

    # Start all threads as daemons
    threads = [
        ("data-pipeline",  _data_pipeline_thread),
        ("sentiment",      _sentiment_thread),
        ("analyst2",       _analyst2_thread),
        ("opus",           _opus_thread),
        ("daily-snapshot", _daily_thread),
        ("cot-weekly",     _cot_thread),
    ]

    # ── Feature schedulers (every 5–15 min) ──────────────────
    def _feature_scheduler_thread():
        logger.info("MainLoop | Feature scheduler thread started")
        try:
            from features.statistical_features import stat_engine
            stat_engine.start_scheduler()
            logger.info("MainLoop | stat_engine scheduler started")
        except Exception as exc:
            logger.warning(f"MainLoop | stat scheduler: {exc}")
        try:
            from features.vwap_features import vwap_engine
            vwap_engine.start_scheduler()
            logger.info("MainLoop | vwap_engine scheduler started")
        except Exception as exc:
            logger.warning(f"MainLoop | vwap scheduler: {exc}")
        try:
            from features.orderflow_features import flow_engine
            flow_engine.start_scheduler()
            logger.info("MainLoop | flow_engine scheduler started")
        except Exception as exc:
            logger.warning(f"MainLoop | flow scheduler: {exc}")
        try:
            from features.entry_engine import entry_engine
            entry_engine.start_scheduler()
            logger.info("MainLoop | entry_engine scheduler started")
        except Exception as exc:
            logger.warning(f"MainLoop | entry scheduler: {exc}")
        try:
            from execution.trailing_stop import trailing_stop_manager
            trailing_stop_manager.start()
            logger.info("MainLoop | trailing_stop_manager started")
        except Exception as exc:
            logger.warning(f"MainLoop | trailing stop: {exc}")
        try:
            from features.liquidity_features import liq_engine
            liq_engine.start_scheduler()
            logger.info("MainLoop | liq_engine scheduler started")
        except Exception as exc:
            logger.warning(f"MainLoop | liq scheduler: {exc}")
        try:
            from features.analyst4_inference import analyst4
            analyst4.start_scheduler()
            logger.info("MainLoop | analyst4 scheduler started")
        except Exception as exc:
            logger.warning(f"MainLoop | analyst4 scheduler: {exc}")
        # Thread stays alive so daemon doesn't get GC'd
        while True:
            time.sleep(3600)

    def _tick_watchdog_thread():
        """
        Watchdog: checks every 5 min if FIX ticks are arriving.
        If no tick in DB for 10+ min → restart fix_collector.
        Sends Telegram alert when triggered.
        """
        from data_pipeline.storage.timeseries_db import db
        from sqlalchemy import text as sa_text

        logger.info("MainLoop | Tick watchdog started")
        time.sleep(120)  # wait 2 min after startup before first check

        while True:
            try:
                with db.connection() as conn:
                    row = conn.execute(sa_text("""
                        SELECT EXTRACT(EPOCH FROM (NOW() - MAX(time)))
                        FROM ticks
                        WHERE time > NOW() - INTERVAL '30 minutes'
                    """)).fetchone()

                age_seconds = float(row[0]) if row and row[0] else None

                if age_seconds is None or age_seconds > 600:
                    # No ticks for 10+ min — restart FIX collector
                    age_str = f"{int(age_seconds)}s" if age_seconds else "unknown"
                    logger.warning(
                        f"MainLoop | Tick watchdog: no ticks for {age_str} "
                        f"— restarting FIX collector"
                    )
                    try:
                        from data_pipeline.connectors.ctrader_fix import fix_collector
                        fix_collector.stop()
                        time.sleep(5)
                        fix_collector.start()
                        logger.success(
                            "MainLoop | Tick watchdog: FIX collector restarted"
                        )
                        try:
                            alerts.send_message(
                                f"⚠️ <b>Tick Watchdog</b>\n"
                                f"No ticks for {age_str}\n"
                                f"FIX collector restarted automatically."
                            )
                        except Exception:
                            pass
                    except Exception as exc:
                        logger.error(
                            f"MainLoop | Tick watchdog restart failed: {exc}"
                        )
                else:
                    logger.debug(
                        f"MainLoop | Tick watchdog: last tick "
                        f"{int(age_seconds)}s ago ✅"
                    )

            except Exception as exc:
                logger.warning(f"MainLoop | Tick watchdog error: {exc}")

            time.sleep(300)  # check every 5 minutes

    threads = [
        ("data-pipeline",      _data_pipeline_thread),
        ("sentiment",          _sentiment_thread),
        ("analyst2",           _analyst2_thread),
        ("opus",               _opus_thread),
        ("daily-snapshot",     _daily_thread),
        ("cot-weekly",         _cot_thread),
        ("feature-schedulers", _feature_scheduler_thread),
        #("tick-watchdog",      _tick_watchdog_thread),   # ← NEW
    ]
    # Stream thread only when market open
    # Stream thread only when market open
    if _is_market_open():
        # OpenAPI WebSocket live stream — replaces FIX collector
        # Uses same auth token as REST API — no separate credentials needed
        # Provides: real-time bid/ask prices + L2 order book depth (20 levels)
        pass  # spot subscriptions handled in _startup_analysts() after REST API auth

    for name, target in threads:
        t = threading.Thread(target=target, daemon=True, name=name)
        t.start()
        logger.info(f"MainLoop | Thread started: {name}")


# ══════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN
# ══════════════════════════════════════════════════════════════

_shutdown_event = threading.Event()

def _handle_shutdown(signum, frame) -> None:
    """Handle SIGINT/SIGTERM gracefully."""
    logger.warning(f"MainLoop | Shutdown signal received ({signum})")
    _shutdown_event.set()


def _graceful_shutdown() -> None:
    """Close all open positions and stop cleanly."""
    logger.warning("MainLoop | Initiating graceful shutdown...")

    try:
        from execution.order_executor import order_executor
        from execution.risk_manager   import risk_manager

        open_positions = order_executor.get_open_positions_local()
        if open_positions:
            logger.warning(
                f"MainLoop | Closing {len(open_positions)} "
                f"open positions..."
            )
            for pos in open_positions:
                try:
                    order_executor.close_position(
                        pair        = pos["pair"],
                        position_id = pos["position_id"],
                        reason      = "CLOSED_MANUAL",
                    )
                except Exception as exc:
                    logger.error(
                        f"MainLoop | Failed to close "
                        f"{pos['pair']}: {exc}"
                    )

        # Take final snapshot
        risk_manager.take_daily_snapshot()

    except Exception as exc:
        logger.error(f"MainLoop | Shutdown error: {exc}")

    alerts.send_shutdown(reason="Graceful shutdown")
    logger.success("MainLoop | Shutdown complete")


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main(dry_run: bool = False, status_only: bool = False) -> None:
    """
    Main trading loop entry point.

    Args:
        dry_run:     Evaluate signals but never place orders
        status_only: Print current state and exit
    """
    # Register shutdown handlers
    signal.signal(signal.SIGINT,  _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    logger.info("=" * 65)
    logger.info("G10 AUTONOMOUS DOMINANCE ARCHITECTURE")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE PAPER'}")
    logger.info(f"Start: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 65)

    # ── Status only ───────────────────────────────────────────
    if status_only:
        from execution.risk_manager   import risk_manager
        from execution.signal_aggregator import signal_aggregator
        from execution.order_executor import order_executor
        risk_manager.print_status()
        signal_aggregator.print_summary()
        order_executor.print_status()
        return

    # ── Connect cTrader ───────────────────────────────────────
    logger.info("MainLoop | Connecting to cTrader...")
    try:
        from data_pipeline.connectors.ctrader_rest_api import connector
        connector.connect()
        logger.success("MainLoop | cTrader connected ✅")
    except Exception as exc:
        logger.critical(f"MainLoop | cTrader connection failed: {exc}")
        sys.exit(1)

    # ── Startup sequence ──────────────────────────────────────
    logger.info("MainLoop | Running startup sequence...")

    # ── Startup reconciliation — ALWAYS first ────────────────────────────
    # Compares DB open trades against cTrader before doing anything else
    # Catches any SL/TP hits that occurred while bot was offline
    try:
        logger.info("MainLoop | Startup reconciliation — DB vs cTrader...")
        from data_pipeline.connectors.ctrader_rest_api import connector as ct_conn
        from data_pipeline.storage.timeseries_db import db
        from sqlalchemy import text as sa_text
        from execution.risk_manager import risk_manager, _get_current_price

        live_positions = ct_conn.get_open_positions()
        live_pos_ids   = {str(p["position_id"]) for p in live_positions}

        with db.connection() as conn:
            db_open = conn.execute(sa_text("""
                SELECT id, symbol, direction, entry_price,
                       stop_loss, take_profit, ctrader_position_id
                FROM trades WHERE status = 'OPEN'
            """)).fetchall()

        zombies = 0
        for row in db_open:
            tid, sym, dirn, entry, sl, tp, ct_pos_id = row
            if ct_pos_id and str(ct_pos_id) not in live_pos_ids:
                current = _get_current_price(sym)
                if dirn == "BUY":
                    if tp and current >= tp:
                        exit_price, close_status = tp, "CLOSED_TP"
                    else:
                        exit_price, close_status = sl, "CLOSED_SL"
                else:
                    if tp and current <= tp:
                        exit_price, close_status = tp, "CLOSED_TP"
                    else:
                        exit_price, close_status = sl, "CLOSED_SL"

                risk_manager.record_trade_close(tid, exit_price, close_status)
                logger.warning(
                    f"MainLoop | Startup reconcile: zombie trade {tid} "
                    f"{sym} → {close_status} @ {exit_price}"
                )
                zombies += 1

        if zombies:
            logger.warning(
                f"MainLoop | Startup reconcile: fixed {zombies} zombie "
                f"DB record(s) — positions were closed while bot was offline"
            )
        else:
            logger.success("MainLoop | Startup reconcile: DB in sync with cTrader ✅")

    except Exception as exc:
        logger.warning(f"MainLoop | Startup reconcile failed: {exc}")

    _startup_data_fetch(connector)
    _startup_features()
    _startup_analysts()

    # ── Initial account state ─────────────────────────────────
    try:
        from execution.risk_manager import risk_manager
        account_info = connector.get_account_balance()
        if account_info:
            balance = account_info.get("balance", 10_000.0)
            equity  = account_info.get("equity",  balance)
            risk_manager.account.update(balance, equity)
            risk_manager.account.reset_daily()  # set daily_start to live balance
            alerts.send_startup(mode="PAPER", balance=balance)
            logger.success(
                f"MainLoop | Account: ${balance:,.2f} "
                f"equity=${equity:,.2f}"
            )
    except Exception as exc:
        logger.warning(f"MainLoop | Account balance fetch failed: {exc}")

    # ── Start background threads ──────────────────────────────
    _start_background_threads(dry_run)

    # ── Initial signal evaluation ─────────────────────────────
    logger.info("MainLoop | Running initial signal evaluation...")
    from execution.signal_aggregator import signal_aggregator
    signal_aggregator.print_summary()

    # ── M15 main loop ─────────────────────────────────────────
    bar_number = 0
    logger.info("MainLoop | Entering M15 bar loop...")

    while not _shutdown_event.is_set():

        # Wait for market to open
        if not _is_market_open():
            now = datetime.now(timezone.utc)
            logger.info(
                f"MainLoop | Market closed "
                f"({now.strftime('%A %H:%M UTC')}) — "
                f"waiting for Sydney open (Sun 21:00 UTC)"
            )
            # Check every 30 minutes while closed
            for _ in range(180):
                if _shutdown_event.is_set():
                    break
                if _is_market_open():
                    break
                time.sleep(10)
            continue

        # Wait for next M15 bar close
        wait_sec  = _seconds_until_next_m15()
        next_time = _next_m15_time()
        logger.info(
            f"MainLoop | Next bar: "
            f"{next_time.strftime('%H:%M:%S UTC')} "
            f"(in {wait_sec:.0f}s)"
        )

        # Sleep in small chunks to allow clean shutdown
        slept = 0.0
        while slept < wait_sec and not _shutdown_event.is_set():
            chunk = min(10.0, wait_sec - slept)
            time.sleep(chunk)
            slept += chunk

        if _shutdown_event.is_set():
            break

        # Process this bar
        bar_number += 1
        try:
            _process_bar(bar_number, dry_run, connector)
        except Exception as exc:
            logger.error(f"MainLoop | Bar {bar_number} failed: {exc}")
            # Never crash the loop — log and continue
            time.sleep(5)

    # ── Shutdown ──────────────────────────────────────────────
    _graceful_shutdown()


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G10 Autonomous Dominance Architecture"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate signals but do not place orders",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current system state and exit",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, status_only=args.status)
