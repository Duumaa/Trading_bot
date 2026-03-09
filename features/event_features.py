"""
features/event_features.py
===========================
Economic calendar proximity and event surprise features for all 12 G10 pairs.

This module is the PRIMARY input for the PNS (Pre-News Stagnation) regime
in Analyst 1. When a HIGH impact event is approaching, the bot must:
  1. Detect it early (hours_until_high_impact)
  2. Enter pre-event blackout 2 hours before (blackout_flag)
  3. Go flat and wait

It also scores historical event surprises — how much did the actual print
deviate from consensus forecast? Large surprises = large moves. This feeds
the directional bias for Analyst 2 (Endo/Exo score).

Trading rationale:
  - Pre-News Stagnation (PNS) is a distinct regime — price coils before
    major events. Trading into a coil means being caught in the release spike.
  - Event surprise scoring tells us whether the market was right or wrong.
    A hawkish surprise on CPI = USD bullish. A dovish surprise = USD bearish.
  - The 2-hour blackout before FOMC/NFP/ECB is a hard risk rule, not a
    suggestion. This feature is the gate that enforces it.
  - Impact weighting: HIGH events dominate. MEDIUM events add context.
    LOW events are ignored for regime and blackout purposes.

Schema note (confirmed 2026-03-09):
  Table has BOTH country VARCHAR(3) AND currency VARCHAR(3) columns.
  Use currency column for pair matching (more reliable than country).
  actual/forecast/previous are VARCHAR — must parse to float for surprise.
  impact values: 'HIGH', 'MEDIUM', 'LOW' (uppercase, enforced by CHECK).

Features produced per pair (8 total):
  hours_until_high_impact   Hours until next HIGH impact event for either
                            currency in the pair. 999.0 if none in 7 days.
  hours_until_any_impact    Hours until next MEDIUM or HIGH event.
  high_impact_count_24h     Number of HIGH impact events in next 24 hours.
  blackout_flag             1.0 if HIGH impact event within 2 hours else 0.0
  post_event_flag           1.0 if HIGH impact event in last 30 minutes
  surprise_last             Surprise score of most recent event [-1, +1]
  surprise_avg_5            Avg surprise score of last 5 events [-1, +1]
  event_density_24h         Total events (any impact) in next 24h (normalised)

Redis keys:  features:event:{PAIR}  (e.g. features:event:EURUSD)
TTL:         3600 seconds (1 hour — calendar changes infrequently)
Scheduler:   background thread, runs compute_all() every hour

Usage:
    from features.event_features import event_engine
    feats = event_engine.get_features("EURUSD")
    # {"hours_until_high_impact": 4.5, "blackout_flag": 0.0, ...}
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text

load_dotenv("/workspaces/Trading_bot/.env")

from data_pipeline.storage.timeseries_db import db  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

# Map each pair to its two component currencies
PAIR_CURRENCIES: Dict[str, Tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"), "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
    "AUDJPY": ("AUD", "JPY"), "NZDJPY": ("NZD", "JPY"),
}

# Currencies tracked in the system
ALL_CURRENCIES: List[str] = [
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"
]

# Pre-event blackout window (hard risk rule from course)
BLACKOUT_HOURS: float = 2.0

# Post-event window for detecting market reaction
POST_EVENT_MINUTES: int = 30

# Lookahead window for calendar queries
LOOKAHEAD_DAYS: int = 7

# Lookback window for surprise scoring
LOOKBACK_DAYS: int = 30

# No event sentinel value
NO_EVENT_HOURS: float = 999.0

# Max events in 24h for normalisation (realistic upper bound)
MAX_EVENTS_24H: int = 20

REDIS_TTL: int = 3600
REDIS_PREFIX: str = "features:event"
SCHEDULER_INTERVAL: int = 3600

# Neutral fallback
NEUTRAL: Dict[str, float] = {
    "hours_until_high_impact": NO_EVENT_HOURS,
    "hours_until_any_impact":  NO_EVENT_HOURS,
    "high_impact_count_24h":   0.0,
    "blackout_flag":           0.0,
    "post_event_flag":         0.0,
    "surprise_last":           0.0,
    "surprise_avg_5":          0.0,
    "event_density_24h":       0.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Surprise parser
# ──────────────────────────────────────────────────────────────────────────────

def _parse_value(s: Optional[str]) -> Optional[float]:
    """
    Parse ForexFactory actual/forecast/previous strings to float.

    Handles formats like: "4.2%", "12.4B", "-0.3K", "1.23M", "48.2"
    Returns None if unparseable (event has no numeric value e.g. speeches).
    """
    if not s or not s.strip():
        return None
    s = s.strip()

    # Strip trailing % sign
    s = s.replace("%", "")

    # Handle multiplier suffixes
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    for suffix, mult in multipliers.items():
        if s.upper().endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                return None

    # Plain number
    try:
        return float(s)
    except ValueError:
        return None


def _surprise_score(actual: Optional[str], forecast: Optional[str]) -> float:
    """
    Compute normalised surprise score from actual vs forecast.

    Score = (actual - forecast) / |forecast| clamped to [-1, +1].
    Positive = beat expectations (bullish for the currency).
    Negative = missed expectations (bearish for the currency).
    Returns 0.0 if either value is unparseable.
    """
    a = _parse_value(actual)
    f = _parse_value(forecast)

    if a is None or f is None:
        return 0.0

    denom = abs(f) if abs(f) > 1e-8 else 1.0
    raw   = (a - f) / denom
    return float(np.clip(raw, -1.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────────────

class EventFeatureEngine:
    """
    Computes economic calendar proximity and event surprise features.

    Loads upcoming events from the economic_events table, then for each
    pair derives time-to-event features, blackout flags, and historical
    surprise scores.

    The blackout_flag is the most critical output — when it is 1.0,
    the signal aggregator must NOT open new positions regardless of
    what other analysts say.
    """

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────

    @property
    def redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )
        return self._redis

    def _redis_get(self, key: str) -> Optional[Dict[str, float]]:
        try:
            raw = self.redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning(f"EventEngine | Redis GET failed [{key}]: {exc}")
        return None

    def _redis_set(self, key: str, data: Dict[str, float], ttl: int = REDIS_TTL) -> None:
        try:
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning(f"EventEngine | Redis SET failed [{key}]: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_features(self, pair: str) -> Dict[str, float]:
        """
        Return event features for a single pair.

        Args:
            pair: Currency pair e.g. "EURUSD"

        Returns:
            Dict of 8 feature_name → float.
        """
        key = f"{REDIS_PREFIX}:{pair}"
        cached = self._redis_get(key)
        if cached:
            logger.debug(f"EventEngine | Cache HIT: {pair}")
            return cached

        logger.info(f"EventEngine | Cache MISS for {pair} — computing all pairs")
        return self.compute_all().get(pair, dict(NEUTRAL))

    def get_blackout_pairs(self) -> List[str]:
        """
        Return list of pairs currently in pre-event blackout.

        Used by the signal aggregator as a hard gate — no new positions
        for any pair in this list.

        Returns:
            List of pair strings e.g. ["EURUSD", "GBPUSD"]
        """
        blackout = []
        for pair in PAIRS:
            feats = self.get_features(pair)
            if feats.get("blackout_flag", 0.0) == 1.0:
                blackout.append(pair)
        return blackout

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """
        Compute event features for all 12 pairs and cache in Redis.

        Loads all upcoming and recent events once, then derives
        pair-specific features without additional DB queries.

        Returns:
            Dict of pair → feature dict
        """
        now_utc = datetime.now(timezone.utc)
        logger.info(f"EventEngine | Computing event features | UTC {now_utc.strftime('%H:%M')}")

        # Load all events once — shared across all pairs
        upcoming_df = self._load_upcoming_events(now_utc)
        recent_df   = self._load_recent_events(now_utc)

        results: Dict[str, Dict[str, float]] = {}

        for pair in PAIRS:
            try:
                currencies = PAIR_CURRENCIES[pair]
                feats = self._compute_features(
                    pair, currencies, upcoming_df, recent_df, now_utc
                )
                key = f"{REDIS_PREFIX}:{pair}"
                self._redis_set(key, feats)
                results[pair] = feats
                logger.debug(
                    f"EventEngine | {pair}: blackout={feats['blackout_flag']:.0f} "
                    f"next_high={feats['hours_until_high_impact']:.1f}h"
                )

            except Exception as exc:
                logger.error(f"EventEngine | {pair} failed: {exc}")
                results[pair] = dict(NEUTRAL)
                self._redis_set(f"{REDIS_PREFIX}:{pair}", dict(NEUTRAL))

        # Log any active blackouts prominently
        blackouts = [p for p, f in results.items() if f.get("blackout_flag") == 1.0]
        if blackouts:
            logger.warning(f"EventEngine | ⚠️  BLACKOUT ACTIVE: {blackouts}")

        logger.success(f"EventEngine | compute_all done: {len(results)}/{len(PAIRS)} pairs")
        return results

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_upcoming_events(self, now_utc: datetime) -> pd.DataFrame:
        """
        Load upcoming economic events for all tracked currencies.

        Fetches events from now → now + LOOKAHEAD_DAYS.
        Uses the currency column (confirmed present in real schema).

        Returns:
            DataFrame with upcoming events, empty DataFrame if none.
        """
        query = text("""
            SELECT
                event_time,
                currency,
                event_name,
                impact,
                forecast,
                actual
            FROM economic_events
            WHERE currency  = ANY(:currencies)
              AND event_time >= :now
              AND event_time <= :horizon
            ORDER BY event_time ASC
        """)

        try:
            horizon = now_utc + timedelta(days=LOOKAHEAD_DAYS)
            with db.connection() as conn:
                result = conn.execute(query, {
                    "currencies": ALL_CURRENCIES,
                    "now":        now_utc,
                    "horizon":    horizon,
                })
                rows = result.fetchall()

            if not rows:
                logger.debug("EventEngine | No upcoming events found")
                return pd.DataFrame(
                    columns=["event_time", "currency", "event_name",
                             "impact", "forecast", "actual"]
                )

            df = pd.DataFrame(
                rows,
                columns=["event_time", "currency", "event_name",
                         "impact", "forecast", "actual"],
            )
            df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
            return df

        except Exception as exc:
            logger.error(f"EventEngine | Upcoming events load failed: {exc}")
            return pd.DataFrame()

    def _load_recent_events(self, now_utc: datetime) -> pd.DataFrame:
        """
        Load recent past events for surprise scoring.

        Fetches events from now - LOOKBACK_DAYS → now.
        Only loads events with actual values (needed for surprise calc).

        Returns:
            DataFrame with recent events, empty DataFrame if none.
        """
        query = text("""
            SELECT
                event_time,
                currency,
                event_name,
                impact,
                actual,
                forecast,
                previous
            FROM economic_events
            WHERE currency   = ANY(:currencies)
              AND event_time >= :lookback
              AND event_time <= :now
              AND actual     IS NOT NULL
              AND actual     != ''
            ORDER BY event_time DESC
        """)

        try:
            lookback = now_utc - timedelta(days=LOOKBACK_DAYS)
            with db.connection() as conn:
                result = conn.execute(query, {
                    "currencies": ALL_CURRENCIES,
                    "lookback":   lookback,
                    "now":        now_utc,
                })
                rows = result.fetchall()

            if not rows:
                return pd.DataFrame(
                    columns=["event_time", "currency", "event_name",
                             "impact", "actual", "forecast", "previous"]
                )

            df = pd.DataFrame(
                rows,
                columns=["event_time", "currency", "event_name",
                         "impact", "actual", "forecast", "previous"],
            )
            df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
            return df

        except Exception as exc:
            logger.error(f"EventEngine | Recent events load failed: {exc}")
            return pd.DataFrame()

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute_features(
        self,
        pair:        str,
        currencies:  Tuple[str, str],
        upcoming_df: pd.DataFrame,
        recent_df:   pd.DataFrame,
        now_utc:     datetime,
    ) -> Dict[str, float]:
        """
        Compute all 8 event features for a single pair.

        Args:
            pair:        Currency pair e.g. "EURUSD"
            currencies:  Tuple of (base, quote) e.g. ("EUR", "USD")
            upcoming_df: All upcoming events DataFrame
            recent_df:   All recent events with actuals DataFrame
            now_utc:     Current UTC time

        Returns:
            Complete dict of 8 features.
        """
        feats = dict(NEUTRAL)
        base, quote = currencies

        # Filter upcoming events relevant to this pair
        if not upcoming_df.empty:
            pair_upcoming = upcoming_df[
                upcoming_df["currency"].isin([base, quote])
            ].copy()
        else:
            pair_upcoming = pd.DataFrame()

        # Filter recent events relevant to this pair
        if not recent_df.empty:
            pair_recent = recent_df[
                recent_df["currency"].isin([base, quote])
            ].copy()
        else:
            pair_recent = pd.DataFrame()

        # ── Hours until next HIGH impact event ────────────────────────────────
        if not pair_upcoming.empty:
            high_upcoming = pair_upcoming[pair_upcoming["impact"] == "HIGH"]
            if not high_upcoming.empty:
                next_high_time = high_upcoming["event_time"].iloc[0]
                hours_to_high  = (next_high_time - now_utc).total_seconds() / 3600.0
                feats["hours_until_high_impact"] = round(
                    max(0.0, float(hours_to_high)), 2
                )

            # Hours until next MEDIUM or HIGH
            med_high = pair_upcoming[pair_upcoming["impact"].isin(["HIGH", "MEDIUM"])]
            if not med_high.empty:
                next_any_time = med_high["event_time"].iloc[0]
                hours_to_any  = (next_any_time - now_utc).total_seconds() / 3600.0
                feats["hours_until_any_impact"] = round(
                    max(0.0, float(hours_to_any)), 2
                )

        # ── HIGH impact count in next 24 hours ────────────────────────────────
        if not pair_upcoming.empty:
            horizon_24h = now_utc + timedelta(hours=24)
            high_24h = pair_upcoming[
                (pair_upcoming["impact"] == "HIGH") &
                (pair_upcoming["event_time"] <= horizon_24h)
            ]
            feats["high_impact_count_24h"] = float(len(high_24h))

        # ── Pre-event blackout flag (HARD RISK RULE) ──────────────────────────
        # Flat 2 hours before any HIGH impact event for either currency
        blackout_horizon = now_utc + timedelta(hours=BLACKOUT_HOURS)
        if not pair_upcoming.empty:
            imminent_high = pair_upcoming[
                (pair_upcoming["impact"] == "HIGH") &
                (pair_upcoming["event_time"] <= blackout_horizon)
            ]
            if not imminent_high.empty:
                feats["blackout_flag"] = 1.0
                next_event = imminent_high["event_time"].iloc[0]
                next_name  = imminent_high["event_name"].iloc[0]
                next_curr  = imminent_high["currency"].iloc[0]
                logger.warning(
                    f"EventEngine | {pair} BLACKOUT — {next_curr} {next_name} "
                    f"in {feats['hours_until_high_impact']:.1f}h"
                )

        # ── Post-event flag ───────────────────────────────────────────────────
        # Market reaction window after HIGH impact release
        post_cutoff = now_utc - timedelta(minutes=POST_EVENT_MINUTES)
        if not recent_df.empty:
            pair_recent_high = recent_df[
                (recent_df["currency"].isin([base, quote])) &
                (recent_df["impact"] == "HIGH") &
                (recent_df["event_time"] >= post_cutoff)
            ]
            if not pair_recent_high.empty:
                feats["post_event_flag"] = 1.0

        # ── Event density in next 24h (normalised) ────────────────────────────
        if not pair_upcoming.empty:
            horizon_24h = now_utc + timedelta(hours=24)
            all_24h = pair_upcoming[
                pair_upcoming["event_time"] <= horizon_24h
            ]
            density = len(all_24h) / MAX_EVENTS_24H
            feats["event_density_24h"] = round(min(1.0, float(density)), 4)

        # ── Surprise scoring (most recent event and avg of last 5) ────────────
        if not pair_recent.empty:
            # Most recent event surprise
            last_row = pair_recent.iloc[0]
            feats["surprise_last"] = round(
                _surprise_score(last_row.get("actual"), last_row.get("forecast")), 4
            )

            # Average surprise of last 5 events with parseable values
            scores: List[float] = []
            for _, row in pair_recent.head(5).iterrows():
                s = _surprise_score(row.get("actual"), row.get("forecast"))
                if s != 0.0:   # 0.0 = unparseable, skip
                    scores.append(s)

            if scores:
                feats["surprise_avg_5"] = round(float(np.mean(scores)), 4)

        return feats

    # ── Background scheduler ──────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """Start background thread that recomputes event features every hour."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("EventEngine | Scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="event-feature-scheduler",
        )
        self._scheduler_thread.start()
        logger.info(f"EventEngine | Scheduler started (interval={SCHEDULER_INTERVAL}s)")

    def stop_scheduler(self) -> None:
        """Signal the scheduler thread to stop."""
        self._running = False
        logger.info("EventEngine | Scheduler stop requested")

    def _scheduler_loop(self) -> None:
        """Background loop: compute_all() every SCHEDULER_INTERVAL seconds."""
        logger.info("EventEngine | Scheduler loop running")
        while self._running:
            try:
                self.compute_all()
            except Exception as exc:
                logger.error(f"EventEngine | Scheduler loop error: {exc}")
            time.sleep(SCHEDULER_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Module singleton
# ──────────────────────────────────────────────────────────────────────────────

event_engine = EventFeatureEngine()