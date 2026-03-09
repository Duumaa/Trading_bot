"""
features/orderflow_features.py
===============================
Real-time order flow and microstructure features from tick and L2 data.

Order flow is what separates institutional systems from indicator bots.
Price is a lagging result of order flow — by the time a candle closes,
institutions have already moved. This module reads raw ticks and L2
order book snapshots to measure WHAT IS HAPPENING RIGHT NOW.

Trading rationale:
  - Bid-ask imbalance: more size on bid = buyers dominant = price going up
  - Buy pressure: ratio of aggressive buys tells you who is in control
  - Spread dynamics: widening spread = liquidity withdrawal before big move
  - Tick velocity: fast ticks = momentum; slow ticks = PNS stagnation
  - Flow delta: acceleration of imbalance = early signal of regime change

Graceful degradation:
  When tick/L2 data is sparse (live stream not running, cold start),
  all features return neutral values: imbalance=0.0, pressure=0.5, etc.
  The signal pipeline NEVER crashes because of missing microstructure data.
  This is intentional — orderflow is a confirmation layer, not a gate.

Data sources:
  ticks table               — bid/ask per tick, spread (generated column)
  order_book_snapshots      — pre-computed imbalance + JSONB bids/asks

JPY scaling note:
  ticks table has mixed scaling — some rows stored as actual price (157.92),
  some as cTrader points (15787.3). We detect and normalise at load time.
  Rule: if JPY bid > 1000, divide by 100.

Features produced per pair (12 total):
  imbalance_1m         Bid-ask volume imbalance last 1 min [-1, +1]
  imbalance_5m         Bid-ask volume imbalance last 5 min [-1, +1]
  buy_pressure_100     Aggressive buy ratio last 100 ticks [0, 1]
  buy_pressure_500     Aggressive buy ratio last 500 ticks [0, 1]
  spread_current       Current spread in pips
  spread_zscore        Z-score of current spread vs 100-tick history
  spread_ratio         Current spread / 20-tick avg spread
  tick_velocity_1m     Ticks arriving per minute (last 60 seconds)
  tick_velocity_5m     Ticks per minute avg over last 5 minutes
  velocity_ratio       tick_velocity_1m / tick_velocity_5m
  flow_imbalance_delta imbalance_1m - imbalance_5m (flow acceleration)
  order_book_depth     Total bid+ask volume across top 5 L2 levels

Redis keys:  features:flow:{PAIR}  (e.g. features:flow:EURUSD)
TTL:         300 seconds (5 minutes)
Scheduler:   background thread, runs compute_all() every 5 minutes

Usage:
    from features.orderflow_features import flow_engine
    feats = flow_engine.get_features("EURUSD")
    # {"imbalance_1m": 0.23, "buy_pressure_100": 0.61, ...}
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

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

JPY_PAIRS: set = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

# Pip sizes for spread normalisation
PIP: Dict[str, float] = {p: (0.01 if p in JPY_PAIRS else 0.0001) for p in PAIRS}

REDIS_TTL: int = 300           # 5 minutes
REDIS_PREFIX: str = "features:flow"
SCHEDULER_INTERVAL: int = 300  # recompute every 5 minutes

# Minimum ticks required for meaningful features
MIN_TICKS_PRESSURE: int = 20   # below this → return neutral 0.5
MIN_TICKS_SPREAD:   int = 10   # below this → return spread_zscore=0

# Neutral / fallback values returned when data is insufficient
NEUTRAL: Dict[str, float] = {
    "imbalance_1m":         0.0,
    "imbalance_5m":         0.0,
    "buy_pressure_100":     0.5,
    "buy_pressure_500":     0.5,
    "spread_current":       0.0,
    "spread_zscore":        0.0,
    "spread_ratio":         1.0,
    "tick_velocity_1m":     0.0,
    "tick_velocity_5m":     0.0,
    "velocity_ratio":       1.0,
    "flow_imbalance_delta": 0.0,
    "order_book_depth":     0.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────────────

class OrderFlowFeatureEngine:
    """
    Computes real-time order flow and microstructure features.

    Reads from the ticks and order_book_snapshots tables. When either
    table has insufficient data (live stream not running, cold start),
    returns neutral values so the rest of the pipeline continues normally.

    Feature computation path:
      1. Load last 500 ticks per pair from ticks table
      2. Load last 10 order book snapshots from order_book_snapshots
      3. Compute all 12 features, falling back to NEUTRAL where needed
      4. Store in Redis with 5-minute TTL
    """

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────

    @property
    def redis(self) -> redis.Redis:
        """Lazy Redis connection with password auth."""
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
            logger.warning(f"FlowEngine | Redis GET failed [{key}]: {exc}")
        return None

    def _redis_set(self, key: str, data: Dict[str, float], ttl: int = REDIS_TTL) -> None:
        try:
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning(f"FlowEngine | Redis SET failed [{key}]: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_features(self, pair: str) -> Dict[str, float]:
        """
        Return order flow features for a single pair.

        Always returns a complete dict — falls back to NEUTRAL values
        if tick/L2 data is insufficient.

        Args:
            pair: Currency pair e.g. "EURUSD"

        Returns:
            Dict of 12 feature_name → float values.
        """
        key = f"{REDIS_PREFIX}:{pair}"
        cached = self._redis_get(key)
        if cached:
            logger.debug(f"FlowEngine | Cache HIT: {pair}")
            return cached

        logger.info(f"FlowEngine | Cache MISS for {pair} — computing all pairs")
        return self.compute_all().get(pair, dict(NEUTRAL))

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """
        Compute order flow features for all 12 pairs and cache in Redis.

        Returns:
            Dict of pair → feature dict (always complete, never partial)
        """
        now_utc = datetime.now(timezone.utc)
        logger.info(f"FlowEngine | Computing order flow features | UTC {now_utc.strftime('%H:%M')}")

        results: Dict[str, Dict[str, float]] = {}

        for pair in PAIRS:
            try:
                ticks_df = self._load_ticks(pair, limit=500)
                ob_df    = self._load_order_book(pair, limit=10)
                feats    = self._compute_features(pair, ticks_df, ob_df, now_utc)

                key = f"{REDIS_PREFIX}:{pair}"
                self._redis_set(key, feats)
                results[pair] = feats
                logger.debug(f"FlowEngine | {pair}: cached (ticks={len(ticks_df) if ticks_df is not None else 0})")

            except Exception as exc:
                logger.error(f"FlowEngine | {pair} failed: {exc}")
                results[pair] = dict(NEUTRAL)
                self._redis_set(f"{REDIS_PREFIX}:{pair}", dict(NEUTRAL))

        logger.success(f"FlowEngine | compute_all done: {len(results)}/{len(PAIRS)} pairs")
        return results

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_ticks(self, pair: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Load the most recent N ticks for a pair from the ticks table.

        Handles the mixed JPY scaling issue: rows with bid > 1000 for JPY
        pairs are divided by 100 to normalise to actual price.

        Args:
            pair:  Currency pair symbol
            limit: Max number of ticks to load

        Returns:
            DataFrame with [time, bid, ask, spread] or None.
        """
        query = text("""
            SELECT time, bid, ask, spread
            FROM ticks
            WHERE symbol = :symbol
              AND time   > '2020-01-01'
            ORDER BY time DESC
            LIMIT :limit
        """)
        # Note: time > '2020-01-01' excludes the 1970-01-01 epoch-zero bug rows

        try:
            with db.connection() as conn:
                result = conn.execute(query, {"symbol": pair, "limit": limit})
                rows = result.fetchall()

            if not rows:
                return None

            df = pd.DataFrame(rows, columns=["time", "bid", "ask", "spread"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            df = df.astype({"bid": float, "ask": float, "spread": float})

            # Fix mixed JPY scaling: if bid > 1000 for a JPY pair → divide by 100
            if pair in JPY_PAIRS:
                mask = df["bid"] > 1000
                if mask.any():
                    df.loc[mask, "bid"]  = df.loc[mask, "bid"]  / 100.0
                    df.loc[mask, "ask"]  = df.loc[mask, "ask"]  / 100.0
                    df.loc[mask, "spread"] = df.loc[mask, "spread"] / 100.0
                    logger.debug(f"FlowEngine | {pair}: corrected {mask.sum()} scaled tick rows")

            return df

        except Exception as exc:
            logger.error(f"FlowEngine | ticks load failed for {pair}: {exc}")
            return None

    def _load_order_book(self, pair: str, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Load the most recent N order book snapshots for a pair.

        The order_book_snapshots table stores pre-computed imbalance values
        and total bid/ask volumes. The JSONB bids/asks columns are also
        available but we use the pre-computed columns for efficiency.

        Args:
            pair:  Currency pair symbol
            limit: Max number of snapshots to load

        Returns:
            DataFrame with [time, bid_ask_imbalance, total_bid_volume,
            total_ask_volume] or None if table is empty.
        """
        query = text("""
            SELECT
                time,
                bid_ask_imbalance,
                total_bid_volume,
                total_ask_volume
            FROM order_book_snapshots
            WHERE symbol = :symbol
            ORDER BY time DESC
            LIMIT :limit
        """)

        try:
            with db.connection() as conn:
                result = conn.execute(query, {"symbol": pair, "limit": limit})
                rows = result.fetchall()

            if not rows:
                return None

            df = pd.DataFrame(
                rows,
                columns=["time", "bid_ask_imbalance", "total_bid_volume", "total_ask_volume"],
            )
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            return df

        except Exception as exc:
            logger.error(f"FlowEngine | order_book load failed for {pair}: {exc}")
            return None

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute_features(
        self,
        pair:     str,
        ticks_df: Optional[pd.DataFrame],
        ob_df:    Optional[pd.DataFrame],
        now_utc:  datetime,
    ) -> Dict[str, float]:
        """
        Compute all 12 order flow features. Returns NEUTRAL for any
        feature where data is insufficient.

        Args:
            pair:     Currency pair
            ticks_df: Recent ticks DataFrame (may be None or small)
            ob_df:    Recent order book snapshots (may be None)
            now_utc:  Current UTC time for velocity windows

        Returns:
            Dict of 12 feature_name → float.
        """
        feats = dict(NEUTRAL)  # start with all neutrals, override as data allows
        pip   = PIP.get(pair, 0.0001)

        # ── Tick-derived features ─────────────────────────────────────────────
        if ticks_df is not None and len(ticks_df) >= MIN_TICKS_SPREAD:

            # Spread features
            spread_pips = ticks_df["spread"].abs() / pip
            feats["spread_current"] = round(float(spread_pips.iloc[-1]), 4)

            spread_20   = spread_pips.iloc[-20:] if len(spread_pips) >= 20 else spread_pips
            mean_spread = float(spread_20.mean())
            feats["spread_ratio"] = round(
                feats["spread_current"] / mean_spread if mean_spread > 0 else 1.0, 4
            )

            if len(spread_pips) >= MIN_TICKS_SPREAD:
                std_spread = float(spread_pips.std())
                feats["spread_zscore"] = round(
                    (feats["spread_current"] - float(spread_pips.mean())) / std_spread
                    if std_spread > 0 else 0.0, 4
                )

            # Tick velocity
            feats["tick_velocity_1m"], feats["tick_velocity_5m"] = \
                self._tick_velocity(ticks_df, now_utc)
            feats["velocity_ratio"] = round(
                feats["tick_velocity_1m"] / feats["tick_velocity_5m"]
                if feats["tick_velocity_5m"] > 0 else 1.0, 4
            )

            # Buy pressure (aggressor direction inference)
            # When ask price changes upward → aggressive buyer crossing spread
            # When bid price changes downward → aggressive seller
            if len(ticks_df) >= MIN_TICKS_PRESSURE:
                feats["buy_pressure_100"] = round(
                    self._buy_pressure(ticks_df, n=100), 4
                )
                feats["buy_pressure_500"] = round(
                    self._buy_pressure(ticks_df, n=500), 4
                )

        # ── Order book features ───────────────────────────────────────────────
        if ob_df is not None and len(ob_df) >= 2:

            # Use pre-computed imbalance from the snapshots table
            # imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) in [-1, +1]
            recent_imbalance = ob_df["bid_ask_imbalance"].dropna()

            if len(recent_imbalance) >= 1:
                feats["imbalance_5m"] = round(float(recent_imbalance.mean()), 4)
                feats["imbalance_1m"] = round(float(recent_imbalance.iloc[-1]), 4)
                feats["flow_imbalance_delta"] = round(
                    feats["imbalance_1m"] - feats["imbalance_5m"], 4
                )

            # Total market depth (bid + ask volume across all stored levels)
            if "total_bid_volume" in ob_df.columns and "total_ask_volume" in ob_df.columns:
                last_row = ob_df.iloc[-1]
                bid_vol  = float(last_row["total_bid_volume"] or 0)
                ask_vol  = float(last_row["total_ask_volume"] or 0)
                feats["order_book_depth"] = round(bid_vol + ask_vol, 2)

        # ── Fallback: derive imbalance from ticks when ob_df is empty ─────────
        # If order book is unavailable but ticks exist, approximate imbalance
        # from the buy pressure signal (shifted to [-1, +1] range)
        elif ticks_df is not None and len(ticks_df) >= MIN_TICKS_PRESSURE:
            bp = feats["buy_pressure_100"]
            approx_imbalance = round((bp - 0.5) * 2.0, 4)  # map [0,1] → [-1,+1]
            feats["imbalance_1m"] = approx_imbalance
            feats["imbalance_5m"] = round((feats["buy_pressure_500"] - 0.5) * 2.0, 4)
            feats["flow_imbalance_delta"] = round(
                feats["imbalance_1m"] - feats["imbalance_5m"], 4
            )

        return feats

    # ── Microstructure helpers ────────────────────────────────────────────────

    @staticmethod
    def _buy_pressure(ticks_df: pd.DataFrame, n: int) -> float:
        """
        Estimate aggressive buyer ratio from last N ticks.

        Method: compare each tick's ask price to the previous ask.
        Rising ask = buyer lifted the offer = aggressive buy.
        Falling bid = seller hit the bid = aggressive sell.
        Unchanged = passive / maker-side fill (not counted as directional).

        Args:
            ticks_df: Tick DataFrame sorted ascending by time
            n:        Number of recent ticks to analyse

        Returns:
            Float in [0, 1]. 0.5 = balanced, 1.0 = all aggressive buys.
        """
        df = ticks_df.iloc[-n:] if len(ticks_df) >= n else ticks_df

        ask_diff = df["ask"].diff()
        bid_diff = df["bid"].diff()

        aggressive_buys  = int((ask_diff > 0).sum())
        aggressive_sells = int((bid_diff < 0).sum())
        total_directional = aggressive_buys + aggressive_sells

        if total_directional == 0:
            return 0.5

        return aggressive_buys / total_directional

    @staticmethod
    def _tick_velocity(
        ticks_df: pd.DataFrame,
        now_utc: datetime,
    ) -> Tuple[float, float]:
        """
        Compute tick arrival rate for 1-minute and 5-minute windows.

        Args:
            ticks_df: Tick DataFrame with UTC timestamps as index column 'time'
            now_utc:  Current UTC time

        Returns:
            Tuple of (velocity_1m, velocity_5m) in ticks-per-minute.
        """
        times = pd.to_datetime(ticks_df["time"].values, utc=True)
        cutoff_1m = now_utc - timedelta(minutes=1)
        cutoff_5m = now_utc - timedelta(minutes=5)

        # Convert to numpy for fast comparison
        cutoff_1m_np = np.datetime64(cutoff_1m.replace(tzinfo=None), 'ns')
        cutoff_5m_np = np.datetime64(cutoff_5m.replace(tzinfo=None), 'ns')
        times_np = times.values

        count_1m = int(np.sum(times_np >= cutoff_1m_np))
        count_5m = int(np.sum(times_np >= cutoff_5m_np))

        velocity_1m = round(float(count_1m), 2)          # ticks in last 1 min
        velocity_5m = round(float(count_5m / 5.0), 2)    # avg ticks/min over 5 min

        return velocity_1m, velocity_5m

    # ── Background scheduler ──────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """
        Start background daemon thread that recomputes order flow every 5 min.

        Call once at application startup.
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("FlowEngine | Scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="flow-feature-scheduler",
        )
        self._scheduler_thread.start()
        logger.info(f"FlowEngine | Scheduler started (interval={SCHEDULER_INTERVAL}s)")

    def stop_scheduler(self) -> None:
        """Signal the scheduler thread to stop."""
        self._running = False
        logger.info("FlowEngine | Scheduler stop requested")

    def _scheduler_loop(self) -> None:
        """Background loop: compute_all() every SCHEDULER_INTERVAL seconds."""
        logger.info("FlowEngine | Scheduler loop running")
        while self._running:
            try:
                self.compute_all()
            except Exception as exc:
                logger.error(f"FlowEngine | Scheduler loop error: {exc}")
            time.sleep(SCHEDULER_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Module singleton
# ──────────────────────────────────────────────────────────────────────────────

flow_engine = OrderFlowFeatureEngine()