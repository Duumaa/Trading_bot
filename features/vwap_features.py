"""
features/vwap_features.py
=========================
Session-anchored VWAP features for all 12 G10 pairs.

VWAP (Volume Weighted Average Price) is the primary institutional price
reference. Every bank desk, algo fund, and market maker uses it. This
module computes VWAP anchored to each major session open (Tokyo, London,
New York) and derives normalised distance, band width, and cross-session
momentum features.

Trading rationale:
  - Price above session VWAP = buyers in control for that session
  - Price below session VWAP = sellers in control
  - VWAP bands (±2σ) define the expected session range
  - Price at +2σ band = extended, MR candidate; holding above mid = THV
  - Cross-session VWAP trend reveals overnight sentiment and directional bias
  - London-NY overlap (13:00–16:00 UTC) = peak liquidity, highest signal quality

Session definitions (UTC):
  Tokyo    00:00 – 09:00   JPY pairs, AUD/NZD flow
  London   07:00 – 16:00   Highest volume, EUR/GBP primary
  New York 13:00 – 22:00   USD pairs, overlaps London 13–16
  Overlap  13:00 – 16:00   Peak institutional activity

Features produced per pair (12 total):
  tokyo_vwap              VWAP since today's Tokyo open (00:00 UTC)
  london_vwap             VWAP since today's London open (07:00 UTC)
  ny_vwap                 VWAP since today's NY open (13:00 UTC)
  price_vs_tokyo_vwap     (close - tokyo_vwap) / tokyo_vwap
  price_vs_london_vwap    (close - london_vwap) / london_vwap
  price_vs_ny_vwap        (close - ny_vwap) / ny_vwap
  london_vwap_upper       London VWAP + 2σ
  london_vwap_lower       London VWAP - 2σ
  london_vwap_band_width  (upper - lower) / london_vwap  — normalised session vol
  ny_vwap_band_width      NY band width / ny_vwap (0 if NY not open yet)
  in_london_ny_overlap    1.0 if current UTC hour in [13, 14, 15] else 0.0
  session_vwap_trend      (london_vwap - tokyo_vwap) / tokyo_vwap

Redis keys:  features:vwap:{PAIR}  (e.g. features:vwap:EURUSD)
TTL:         900 seconds (15 minutes — VWAP changes every candle)
Scheduler:   background thread, runs compute_all() every 15 minutes

Usage:
    from features.vwap_features import vwap_engine
    feats = vwap_engine.get_features("EURUSD")
    # {"london_vwap": 1.0821, "price_vs_london_vwap": 0.0003, ...}
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

# Session open hours in UTC
TOKYO_OPEN_HOUR:  int = 0
LONDON_OPEN_HOUR: int = 7
NY_OPEN_HOUR:     int = 13

# London-NY overlap window (UTC hours, inclusive start exclusive end)
OVERLAP_START: int = 13
OVERLAP_END:   int = 16

# VWAP band multiplier (2σ = ~95% of price action inside bands)
BAND_SIGMA: float = 2.0

# How many days of M15 data to load (2 days covers any session start today)
LOOKBACK_DAYS: int = 3

REDIS_TTL: int = 900          # 15 minutes
REDIS_PREFIX: str = "features:vwap"
SCHEDULER_INTERVAL: int = 900  # recompute every 15 minutes

# Minimum bars needed for a valid VWAP (at least 4 × 15min = 1 hour of data)
MIN_SESSION_BARS: int = 4


# ──────────────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────────────

class VWAPFeatureEngine:
    """
    Computes session-anchored VWAP features from M15 OHLCV data.

    For each pair, loads the last 3 days of M15 candles, then computes
    VWAP anchored to each session open for the current UTC day. Results
    are normalised so they are comparable across pairs and price levels.

    A background scheduler thread calls compute_all() every 15 minutes
    so the Redis cache stays fresh across the trading day.
    """

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────

    @property
    def redis(self) -> redis.Redis:
        if self._redis is None:
            from data_pipeline.storage.redis_pool import redis_client
            self._redis = redis_client
        return self._redis

    def _redis_get(self, key: str) -> Optional[Dict[str, float]]:
        """Retrieve cached features. Returns None on miss or error."""
        try:
            raw = self.redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning(f"VWAPEngine | Redis GET failed [{key}]: {exc}")
        return None

    def _redis_set(self, key: str, data: Dict[str, float], ttl: int = REDIS_TTL) -> None:
        """Store features in Redis with TTL."""
        try:
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning(f"VWAPEngine | Redis SET failed [{key}]: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_features(self, pair: str) -> Dict[str, float]:
        """
        Return VWAP features for a single pair.

        Checks Redis cache first. Falls back to compute_all() on miss.

        Args:
            pair: Currency pair e.g. "EURUSD"

        Returns:
            Dict of feature_name → float, or empty dict on failure.
        """
        key = f"{REDIS_PREFIX}:{pair}"
        cached = self._redis_get(key)
        if cached:
            logger.debug(f"VWAPEngine | Cache HIT: {pair}")
            return cached

        logger.info(f"VWAPEngine | Cache MISS for {pair} — computing all pairs")
        return self.compute_all().get(pair, {})

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """
        Compute VWAP features for all 12 pairs and store in Redis.

        Returns:
            Dict of pair → feature dict
        """
        now_utc = datetime.now(timezone.utc)
        logger.info(f"VWAPEngine | Computing VWAP features | UTC {now_utc.strftime('%H:%M')}")

        # Determine session anchors for today
        today = now_utc.date()
        tokyo_anchor  = datetime(today.year, today.month, today.day,
                                 TOKYO_OPEN_HOUR,  0, 0, tzinfo=timezone.utc)
        london_anchor = datetime(today.year, today.month, today.day,
                                 LONDON_OPEN_HOUR, 0, 0, tzinfo=timezone.utc)
        ny_anchor     = datetime(today.year, today.month, today.day,
                                 NY_OPEN_HOUR,     0, 0, tzinfo=timezone.utc)

        results: Dict[str, Dict[str, float]] = {}

        for pair in PAIRS:
            try:
                df = self._load_m15(pair, lookback_days=LOOKBACK_DAYS)
                if df is None or df.empty:
                    logger.warning(f"VWAPEngine | {pair}: no M15 data")
                    continue

                feats = self._compute_features(
                    df, pair, now_utc,
                    tokyo_anchor, london_anchor, ny_anchor,
                )
                key = f"{REDIS_PREFIX}:{pair}"
                self._redis_set(key, feats, ttl=REDIS_TTL)
                results[pair] = feats
                logger.debug(f"VWAPEngine | {pair}: {len(feats)} features cached")

            except Exception as exc:
                logger.error(f"VWAPEngine | {pair} failed: {exc}")

        logger.success(f"VWAPEngine | compute_all done: {len(results)}/{len(PAIRS)} pairs")
        return results

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_m15(self, pair: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """
        Load M15 candles from TimescaleDB for the last N days.

        Uses the confirmed real table name: ohlcv (not ohlcv_data)
        and column name: time (not timestamp).

        Args:
            pair:         Currency pair symbol
            lookback_days: How many days back to fetch

        Returns:
            DataFrame with [time, open, high, low, close, volume] or None.
        """
        query = text("""
            SELECT
                time,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv
            WHERE symbol    = :symbol
              AND timeframe = 'M15'
              AND time      >= NOW() - INTERVAL :interval
            ORDER BY time ASC
        """)

        try:
            interval = f"{lookback_days} days"
            with db.connection() as conn:
                result = conn.execute(query, {"symbol": pair, "interval": interval})
                rows = result.fetchall()

            if not rows:
                return None

            df = pd.DataFrame(
                rows,
                columns=["time", "open", "high", "low", "close", "volume"],
            )
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            df = df.astype({
                "open": float, "high": float,
                "low": float, "close": float, "volume": float,
            })

            # cTrader stores JPY pair prices in points (×100 of actual price)
            # e.g. USDJPY stored as 14900 in DB = 149.00 actual — divide by 100
            if pair in {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] / 100.0

            # Typical price (used in VWAP calculation)
            df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

            # Use volume if available, else treat each bar equally (volume=1)
            # cTrader volume can be 0 for some instruments — fall back to 1
            df["vol_adj"] = df["volume"].where(df["volume"] > 0, 1.0)

            return df

        except Exception as exc:
            logger.error(f"VWAPEngine | DB load failed for {pair}: {exc}")
            return None

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute_features(
        self,
        df: pd.DataFrame,
        pair: str,
        now_utc: datetime,
        tokyo_anchor:  datetime,
        london_anchor: datetime,
        ny_anchor:     datetime,
    ) -> Dict[str, float]:
        """
        Compute all 12 VWAP features from M15 data.

        Args:
            df:             M15 OHLCV DataFrame indexed by UTC timestamp
            pair:           Pair name (for logging)
            now_utc:        Current UTC datetime
            tokyo_anchor:   Today's Tokyo session open (UTC)
            london_anchor:  Today's London session open (UTC)
            ny_anchor:      Today's NY session open (UTC)

        Returns:
            Dict of feature_name → float (rounded to 6 d.p.)
        """
        feats: Dict[str, float] = {}

        last_close = float(df["close"].iloc[-1])

        # ── Per-session VWAP ──────────────────────────────────────────────────
        tokyo_vwap,  _  = self._session_vwap(df, tokyo_anchor,  now_utc)
        london_vwap, london_std = self._session_vwap(df, london_anchor, now_utc)
        ny_vwap,     ny_std     = self._session_vwap(df, ny_anchor,     now_utc)

        feats["tokyo_vwap"]  = round(tokyo_vwap,  6)
        feats["london_vwap"] = round(london_vwap, 6)
        feats["ny_vwap"]     = round(ny_vwap,     6)

        # ── Normalised price distance from each VWAP ──────────────────────────
        # Positive = price above VWAP (buyers in control)
        # Negative = price below VWAP (sellers in control)
        feats["price_vs_tokyo_vwap"]  = round(
            self._safe_div(last_close - tokyo_vwap,  tokyo_vwap),  6
        )
        feats["price_vs_london_vwap"] = round(
            self._safe_div(last_close - london_vwap, london_vwap), 6
        )
        feats["price_vs_ny_vwap"] = round(
            self._safe_div(last_close - ny_vwap, ny_vwap), 6
        )

        # ── London VWAP bands (±2σ) ───────────────────────────────────────────
        # Band width is a real-time session volatility estimate
        london_upper = london_vwap + BAND_SIGMA * london_std
        london_lower = london_vwap - BAND_SIGMA * london_std
        feats["london_vwap_upper"] = round(london_upper, 6)
        feats["london_vwap_lower"] = round(london_lower, 6)
        feats["london_vwap_band_width"] = round(
            self._safe_div(london_upper - london_lower, london_vwap), 6
        )

        # ── NY VWAP band width ────────────────────────────────────────────────
        ny_upper = ny_vwap + BAND_SIGMA * ny_std
        ny_lower = ny_vwap - BAND_SIGMA * ny_std
        feats["ny_vwap_band_width"] = round(
            self._safe_div(ny_upper - ny_lower, ny_vwap), 6
        )

        # ── London-NY overlap flag ────────────────────────────────────────────
        # Peak liquidity window — highest signal quality for entries
        current_hour = now_utc.hour
        feats["in_london_ny_overlap"] = (
            1.0 if OVERLAP_START <= current_hour < OVERLAP_END else 0.0
        )

        # ── Cross-session VWAP trend ──────────────────────────────────────────
        # London VWAP above Tokyo VWAP = sustained buying through AM session
        # London VWAP below Tokyo VWAP = sustained selling into European hours
        feats["session_vwap_trend"] = round(
            self._safe_div(london_vwap - tokyo_vwap, tokyo_vwap), 6
        )

        return feats

    # ── VWAP calculation ──────────────────────────────────────────────────────

    def _session_vwap(
        self,
        df: pd.DataFrame,
        session_open: datetime,
        now_utc: datetime,
    ) -> Tuple[float, float]:
        """
        Compute VWAP and price std since session open.

        Formula: VWAP = Σ(typical_price × volume) / Σ(volume)
        Std:     standard deviation of typical_price weighted by volume,
                 used for band calculation.

        Args:
            df:           Full M15 DataFrame (may span multiple days)
            session_open: UTC datetime of session anchor
            now_utc:      Current UTC time (upper bound)

        Returns:
            Tuple of (vwap, std). Returns (last_close, 0.0) if insufficient
            bars to compute a meaningful VWAP.
        """
        # Slice to session window
        mask = (df.index >= session_open) & (df.index <= now_utc)
        session_df = df[mask]

        last_close = float(df["close"].iloc[-1])

        if len(session_df) < MIN_SESSION_BARS:
            # Session hasn't started yet or too few bars — return last close
            # so downstream features compute as 0 distance (neutral)
            return last_close, 0.0

        tp  = session_df["typical_price"].values
        vol = session_df["vol_adj"].values

        total_vol = vol.sum()
        if total_vol == 0:
            return last_close, 0.0

        vwap = float(np.dot(tp, vol) / total_vol)

        # Volume-weighted standard deviation of typical price
        variance = float(np.dot(vol, (tp - vwap) ** 2) / total_vol)
        std = float(np.sqrt(variance))

        return vwap, std

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        """Division that returns 0.0 instead of inf/nan on zero denominator."""
        if denominator == 0 or not np.isfinite(denominator):
            return 0.0
        result = numerator / denominator
        return float(result) if np.isfinite(result) else 0.0

    # ── Background scheduler ──────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """
        Start background daemon thread that recomputes VWAP every 15 minutes.

        Call once at application startup. Aligned to M15 candle closes.
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("VWAPEngine | Scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="vwap-feature-scheduler",
        )
        self._scheduler_thread.start()
        logger.info(f"VWAPEngine | Scheduler started (interval={SCHEDULER_INTERVAL}s)")

    def stop_scheduler(self) -> None:
        """Signal the scheduler thread to stop."""
        self._running = False
        logger.info("VWAPEngine | Scheduler stop requested")

    def _scheduler_loop(self) -> None:
        """Background loop: compute_all() every SCHEDULER_INTERVAL seconds."""
        logger.info("VWAPEngine | Scheduler loop running")
        while self._running:
            try:
                self.compute_all()
            except Exception as exc:
                logger.error(f"VWAPEngine | Scheduler loop error: {exc}")
            time.sleep(SCHEDULER_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Module singleton
# ──────────────────────────────────────────────────────────────────────────────

vwap_engine = VWAPFeatureEngine()
