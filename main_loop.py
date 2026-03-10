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
    Check if forex market is currently open.
    Forex is closed Saturday 21:00 UTC to Sunday 21:00 UTC.
    """
    now     = datetime.now(timezone.utc)
    weekday = now.weekday()   # 0=Mon, 5=Sat, 6=Sun
    hour    = now.hour

    if weekday == 5:   # Saturday — always closed
        return False
    if weekday == 6 and hour < 21:   # Sunday before 21:00
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
    Ensures regime + exogenous scores are cached before first bar.
    """
    logger.info("MainLoop | Running startup analysis...")

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

    # 1. Fetch latest candles
    _fetch_latest_candles(connector)

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
        tradeable = [
            d for d in decisions.values()
            if d.get("action") in ("LONG", "SHORT")
        ]
        logger.info(
            f"MainLoop | Signals: "
            f"{sum(1 for d in decisions.values() if d['action']=='FLAT')} FLAT | "
            f"{len(tradeable)} tradeable: "
            f"{[d['pair'] for d in tradeable]}"
        )
    except Exception as exc:
        logger.error(f"MainLoop | Signal evaluation failed: {exc}")
        return results

    # 5. Execute signals
    if not dry_run and tradeable:
        from execution.order_executor import order_executor
        from execution.risk_manager   import risk_manager

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
        try:
            from data_pipeline.connectors.ctrader_live_stream import live_stream
            live_stream.connect()
            logger.info("MainLoop | Live stream started")
        except Exception as exc:
            logger.warning(f"MainLoop | Live stream failed: {exc}")

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

            time.sleep(DATA_PIPELINE_HOURS * 3600)

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
            except Exception as exc:
                logger.error(f"MainLoop | Daily snapshot failed: {exc}")

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
                cot_downloader.download_and_store()
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

    # Stream thread only when market open
    if _is_market_open():
        threads.insert(0, ("live-stream", _stream_thread))
        # FIX tick collector — runs alongside live stream for L2 data
        try:
            from data_pipeline.connectors.ctrader_fix import fix_collector
            threads.insert(1, ("fix-collector", fix_collector.start))
        except Exception as exc:
            logger.warning(f"MainLoop | FIX collector unavailable: {exc}")

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