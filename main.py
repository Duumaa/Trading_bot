"""
main.py
=======
G10 Dominance Architecture — System Orchestrator

This is the single entry point for the entire trading system.

Thread architecture:
    Thread 1 (FIX Stream):     Real-time market data ingestion via cTrader FIX
    Thread 2 (Feature Engine): M15 feature calculation on each candle close
    Thread 3 (Analyst Engine): All 6 analyst signals (independent, scheduled)
    Thread 4 (Execution):      Signal aggregation → risk check → FIX order send
    Thread 5 (Monitoring):     Performance tracking, drift detection, alerts
    Thread 6 (Schedulers):     Scrapers, COT downloads, weekly Opus analysis

Startup sequence:
    1. Load config & validate .env
    2. Setup logging
    3. Health-check all infrastructure (DB, Redis, InfluxDB)
    4. Start data ingestion threads
    5. Start analyst scheduled jobs
    6. Enter main M15 loop

Usage:
    python main.py
    python main.py --mode PAPER
    python main.py --mode LIVE  # requires explicit flag
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger

# ── Config (must be first after stdlib) ──────────────────────────────────────
from config.logging_setup import setup_logging
from config.settings import TradingMode, settings


# ── Globals ───────────────────────────────────────────────────────────────────
_SHUTDOWN_EVENT = threading.Event()
_THREADS: Dict[str, threading.Thread] = {}


# ── Signal handlers ───────────────────────────────────────────────────────────

def _handle_shutdown(signum: int, frame) -> None:  # type: ignore[type-arg]
    """Graceful shutdown on SIGINT / SIGTERM."""
    logger.warning(f"Shutdown signal received (signal={signum}). Initiating graceful stop...")
    _SHUTDOWN_EVENT.set()


# ── Infrastructure health checks ──────────────────────────────────────────────

def _check_infrastructure() -> bool:
    """
    Verify all required services are reachable before starting.
    Returns True if all checks pass, False otherwise.
    """
    all_ok = True

    # TimescaleDB
    try:
        from data_pipeline.storage.timeseries_db import db
        if db.health_check():
            logger.info("✅ TimescaleDB: healthy")
        else:
            logger.error("❌ TimescaleDB: unreachable")
            all_ok = False
    except Exception as e:
        logger.error(f"❌ TimescaleDB: {e}")
        all_ok = False

    # Redis
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=3)
        r.ping()
        logger.info("✅ Redis: healthy")
    except Exception as e:
        logger.error(f"❌ Redis: {e}")
        all_ok = False

    # InfluxDB
    try:
        from influxdb_client import InfluxDBClient
        client = InfluxDBClient(
            url=settings.INFLUX_URL,
            token=settings.INFLUX_TOKEN,
            org=settings.INFLUX_ORG,
            timeout=3_000
        )
        health = client.health()
        if health.status == "pass":
            logger.info("✅ InfluxDB: healthy")
        else:
            logger.warning(f"⚠️  InfluxDB: status={health.status}")
        client.close()
    except Exception as e:
        logger.warning(f"⚠️  InfluxDB: {e} (non-critical, metrics only)")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        experiments = mlflow.search_experiments()
        logger.info(f"✅ MLflow: healthy ({len(experiments)} experiments)")
    except Exception as e:
        logger.warning(f"⚠️  MLflow: {e} (non-critical, model registry)")

    return all_ok


# ── Thread targets ────────────────────────────────────────────────────────────

def _thread_fix_stream() -> None:
    """
    Thread 1: cTrader FIX API real-time market data stream.

    Subscribes to L2 depth and ticks for all G10 pairs.
    Feeds data into TimescaleDB and Redis pub/sub.
    Runs continuously until shutdown.
    """
    logger.info("[Thread-FIX] Starting FIX market data stream...")

    # Import deferred to allow other threads to start
    try:
        from data_pipeline.connectors.ctrader_fix import CTraderFIXConnector
        connector = CTraderFIXConnector()

        if not connector.connect():
            logger.error("[Thread-FIX] Failed to connect to cTrader FIX. Aborting thread.")
            return

        connector.subscribe_market_data(settings.G10_PAIRS)
        logger.info(f"[Thread-FIX] Subscribed to {len(settings.G10_PAIRS)} pairs")

        while not _SHUTDOWN_EVENT.is_set():
            time.sleep(1)   # FIX callbacks are async; loop keeps thread alive

        connector.disconnect()
        logger.info("[Thread-FIX] Disconnected cleanly.")

    except ImportError:
        logger.warning("[Thread-FIX] CTraderFIXConnector not yet implemented — stub running.")
        while not _SHUTDOWN_EVENT.is_set():
            time.sleep(5)

    except Exception as e:
        logger.exception(f"[Thread-FIX] Fatal error: {e}")


def _thread_feature_engine() -> None:
    """
    Thread 2: Feature calculation on each M15 candle close.

    Waits for candle close signal (from FIX stream via Redis),
    then computes all feature vectors for all 12 pairs.
    """
    logger.info("[Thread-Features] Starting feature engine...")

    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        pubsub = r.pubsub()
        pubsub.subscribe("candle_close_m15")
        logger.info("[Thread-Features] Subscribed to candle_close_m15 channel")
    except Exception as e:
        logger.warning(f"[Thread-Features] Redis unavailable: {e}. Running in polling mode.")
        pubsub = None

    def _compute_features_for_all_pairs() -> None:
        """Compute and store feature vectors for all G10 pairs."""
        try:
            # Will be implemented in features/ modules
            logger.debug("[Thread-Features] Feature calculation cycle complete.")
        except Exception as e:
            logger.error(f"[Thread-Features] Feature computation error: {e}")

    while not _SHUTDOWN_EVENT.is_set():
        try:
            if pubsub:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    _compute_features_for_all_pairs()
            else:
                # Polling fallback: compute every 15 minutes aligned to clock
                now = datetime.now(tz=timezone.utc)
                if now.minute % 15 == 0 and now.second < 5:
                    _compute_features_for_all_pairs()
                    time.sleep(10)  # prevent double-fire within same minute
                else:
                    time.sleep(1)
        except Exception as e:
            logger.error(f"[Thread-Features] Loop error: {e}")
            time.sleep(5)

    logger.info("[Thread-Features] Stopped.")


def _thread_analyst_engine() -> None:
    """
    Thread 3: Manages all 6 analyst scheduled inference jobs.

    - Analysts 1-4: run on every M15 close (triggered via Redis)
    - Analyst 5: runs every 15 minutes (Groq news sentiment)
    - Analyst 6: runs weekly Sunday 20:00 UTC (Claude Opus)
    """
    logger.info("[Thread-Analysts] Starting analyst engine...")

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        scheduler = BackgroundScheduler(timezone="UTC")

        # Analyst 5: Groq sentiment every 15 minutes
        scheduler.add_job(
            func=_run_analyst5_groq,
            trigger=IntervalTrigger(minutes=15),
            id="analyst5_groq",
            name="Analyst 5: Groq Daily Sentiment",
            max_instances=1,
            coalesce=True,
        )

        # Analyst 6: Claude Opus weekly on Sunday at 20:00 UTC
        scheduler.add_job(
            func=_run_analyst6_opus,
            trigger=CronTrigger(day_of_week="sun", hour=20, minute=0),
            id="analyst6_opus",
            name="Analyst 6: Claude Opus Weekly Analysis",
            max_instances=1,
            coalesce=True,
        )

        scheduler.start()
        logger.info("[Thread-Analysts] APScheduler running. Analysts 5 & 6 scheduled.")

        while not _SHUTDOWN_EVENT.is_set():
            time.sleep(1)

        scheduler.shutdown(wait=False)
        logger.info("[Thread-Analysts] Scheduler stopped.")

    except ImportError:
        logger.warning("[Thread-Analysts] APScheduler not available — stub running.")
        while not _SHUTDOWN_EVENT.is_set():
            time.sleep(60)


def _run_analyst5_groq() -> None:
    """Run Analyst 5 (Groq daily sentiment) — called by scheduler."""
    try:
        logger.debug("[Analyst-5] Groq sentiment run starting...")
        # from models.analyst5_daily_groq.groq_sentiment import GroqSentimentAnalyst
        # analyst5 = GroqSentimentAnalyst()
        # analyst5.run()
    except Exception as e:
        logger.error(f"[Analyst-5] Error: {e}")


def _run_analyst6_opus() -> None:
    """Run Analyst 6 (Claude Opus weekly) — called by scheduler."""
    try:
        logger.info("[Analyst-6] Claude Opus weekly analysis starting...")
        # from models.analyst6_weekly_opus.opus_deep_analysis import OpusWeeklyAnalyst
        # analyst6 = OpusWeeklyAnalyst()
        # analyst6.run()
    except Exception as e:
        logger.error(f"[Analyst-6] Error: {e}")


def _thread_execution() -> None:
    """
    Thread 4: Signal aggregation → risk management → order execution.

    Triggered on each M15 close (via Redis pub/sub).
    Gate logic: Fundamental confirms → Technical times → Sentiment validates.
    """
    logger.info("[Thread-Execution] Starting execution engine...")

    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        pubsub = r.pubsub()
        pubsub.subscribe("candle_close_m15")
    except Exception as e:
        logger.warning(f"[Thread-Execution] Redis unavailable: {e}. Running in poll mode.")
        pubsub = None

    last_cycle_minute = -1

    while not _SHUTDOWN_EVENT.is_set():
        try:
            should_run = False

            if pubsub:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    should_run = True
            else:
                now = datetime.now(tz=timezone.utc)
                if now.minute % 15 == 0 and now.minute != last_cycle_minute:
                    should_run = True
                    last_cycle_minute = now.minute

            if should_run:
                _run_execution_cycle()

        except Exception as e:
            logger.error(f"[Thread-Execution] Loop error: {e}")
            time.sleep(5)

    logger.info("[Thread-Execution] Stopped.")


def _run_execution_cycle() -> None:
    """
    Single M15 execution cycle — the core trading loop.

    1. Fetch all 6 analyst signals from DB/cache
    2. Run gate logic (Fund → Tech → Sent)
    3. Pre-trade checks (event filter, daily loss limit)
    4. Risk size → send order via FIX
    5. Update monitoring
    """
    cycle_start = datetime.now(tz=timezone.utc)
    logger.info(f"[Execution] M15 cycle | {cycle_start.strftime('%Y-%m-%d %H:%M')} UTC")

    try:
        # Step 1-6 will be filled in as each module is built (Phases 4 & 5)
        # Placeholder logging for now
        logger.debug("[Execution] Cycle complete (placeholder — awaiting analyst modules)")

    except Exception as e:
        logger.error(f"[Execution] Cycle error: {e}")


def _thread_monitoring() -> None:
    """
    Thread 5: Performance tracking, drift detection, Prometheus metrics, alerts.
    Runs on a slow loop (every 60 seconds).
    """
    logger.info("[Thread-Monitoring] Starting monitoring...")

    while not _SHUTDOWN_EVENT.is_set():
        try:
            # Will be filled in by intelligence/monitoring modules (Phase 6)
            time.sleep(60)
        except Exception as e:
            logger.error(f"[Thread-Monitoring] Error: {e}")
            time.sleep(60)

    logger.info("[Thread-Monitoring] Stopped.")


def _thread_schedulers() -> None:
    """
    Thread 6: Background schedulers for scrapers, COT downloads, reminders.

    Schedule:
    - ForexFactory calendar: daily at 00:05 UTC
    - COT report download: weekly Friday at 21:30 UTC
    - Investing.com yield scrape: daily at 17:05 EST
    - Data freshness alerts: daily at 08:00 UTC
    """
    logger.info("[Thread-Schedulers] Starting background schedulers...")

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BackgroundScheduler(timezone="UTC")

        # ForexFactory calendar scrape
        scheduler.add_job(
            func=lambda: logger.info("[Scraper] ForexFactory calendar — not yet implemented"),
            trigger=CronTrigger(hour=0, minute=5),
            id="forexfactory_scrape",
            max_instances=1,
        )

        # COT download (CFTC releases Fridays ~21:30 UTC)
        scheduler.add_job(
            func=lambda: logger.info("[Scraper] COT download — not yet implemented"),
            trigger=CronTrigger(day_of_week="fri", hour=22, minute=0),
            id="cot_download",
            max_instances=1,
        )

        # Yield scrape backup (Investing.com)
        scheduler.add_job(
            func=lambda: logger.info("[Scraper] Yield scrape — not yet implemented"),
            trigger=CronTrigger(hour=22, minute=30),  # 17:30 EST
            id="yield_scrape",
            max_instances=1,
        )

        scheduler.start()
        logger.info("[Thread-Schedulers] All background jobs scheduled.")

        while not _SHUTDOWN_EVENT.is_set():
            time.sleep(1)

        scheduler.shutdown(wait=False)

    except ImportError:
        logger.warning("[Thread-Schedulers] APScheduler not available — stub running.")
        while not _SHUTDOWN_EVENT.is_set():
            time.sleep(60)

    logger.info("[Thread-Schedulers] Stopped.")


# ── Main entry point ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G10 Dominance Architecture — Autonomous Forex Trading System"
    )
    parser.add_argument(
        "--mode",
        choices=["PAPER", "LIVE"],
        default=None,
        help="Override trading mode from .env (default: use TRADING_MODE env var)"
    )
    parser.add_argument(
        "--skip-infra-check",
        action="store_true",
        help="Skip infrastructure health checks (useful for testing)"
    )
    return parser.parse_args()


def main() -> int:
    """
    System entry point.

    Returns:
        0 on clean shutdown, 1 on startup failure.
    """
    args = parse_args()

    # ── Setup logging first ───────────────────────────────────────────────────
    setup_logging()

    logger.info("=" * 70)
    logger.info("  G10 DOMINANCE ARCHITECTURE — Starting")
    logger.info(f"  Mode: {settings.TRADING_MODE.value}")
    logger.info(f"  Pairs: {', '.join(settings.G10_PAIRS)}")
    logger.info(f"  Timeframe: {settings.PRIMARY_TIMEFRAME}")
    logger.info("=" * 70)

    # ── Live mode safety gate ─────────────────────────────────────────────────
    if settings.TRADING_MODE == TradingMode.LIVE or args.mode == "LIVE":
        logger.warning("⚠️  LIVE TRADING MODE — Real money at risk!")
        response = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if response.strip() != "CONFIRM LIVE TRADING":
            logger.info("Live mode not confirmed. Exiting.")
            return 0

    # ── Infrastructure checks ─────────────────────────────────────────────────
    if not args.skip_infra_check:
        logger.info("Running infrastructure health checks...")
        if not _check_infrastructure():
            logger.error("Infrastructure check failed. Fix issues above before starting.")
            logger.error("Hint: run 'docker-compose up -d' and wait for all containers to be healthy.")
            return 1
        logger.info("All infrastructure checks passed ✅")
    else:
        logger.warning("Infrastructure checks skipped (--skip-infra-check)")

    # ── Register shutdown handlers ────────────────────────────────────────────
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # ── Launch threads ────────────────────────────────────────────────────────
    thread_configs = [
        ("FIX-Stream",     _thread_fix_stream,      True),   # daemon=True
        ("Feature-Engine", _thread_feature_engine,  True),
        ("Analyst-Engine", _thread_analyst_engine,  True),
        ("Execution",      _thread_execution,        True),
        ("Monitoring",     _thread_monitoring,       True),
        ("Schedulers",     _thread_schedulers,       True),
    ]

    for name, target, daemon in thread_configs:
        t = threading.Thread(target=target, name=name, daemon=daemon)
        t.start()
        _THREADS[name] = t
        logger.info(f"Started thread: {name}")

    logger.info(f"All {len(_THREADS)} threads running. System is live.")

    # ── Main thread: wait for shutdown ────────────────────────────────────────
    try:
        while not _SHUTDOWN_EVENT.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        _SHUTDOWN_EVENT.set()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    logger.info("Waiting for threads to stop (timeout=10s)...")
    for name, thread in _THREADS.items():
        thread.join(timeout=10)
        if thread.is_alive():
            logger.warning(f"Thread '{name}' did not stop cleanly within timeout.")
        else:
            logger.info(f"Thread '{name}' stopped.")

    logger.info("G10 Dominance Architecture — Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())