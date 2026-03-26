"""
features/statistical_features.py
=================================
Price-derived statistical and volatility features for all 12 G10 pairs.

These features are the PRIMARY inputs for Analyst 1 (LightGBM regime classifier).
The classifier must distinguish 4 regimes: THV / TLV / MR / PNS.
Before any regime label can be assigned, we need objective volatility measurement
across multiple horizons and normalisation methods.

Design rationale:
  - ATR ratio (14/50) is the primary vol regime detector: >1.5 = high vol
  - Rolling std ratio (5d/20d) is a secondary, independent vol regime signal
  - Z-scores of close tell the model whether price is extended (MR) or flat (PNS)
  - Momentum + ROC distinguish THV (accelerating) from TLV (steady trend)
  - ADR in pips is the course-specified non-normal vol measure for forex
  - BB %B combines mean + vol in one [0,1] normalised position

Features produced per pair (17 total):
  atr_14              14-period ATR (pips, using D1 bars)
  atr_pct             ATR as % of current price
  atr_ratio           atr_14 / atr_50  — short vs long vol comparison
  vol_5d              5-day rolling std of log returns
  vol_20d             20-day rolling std of log returns
  vol_ratio           vol_5d / vol_20d  — regime detector
  high_vol_regime     1.0 if vol_ratio > 1.5 else 0.0
  zscore_close_5d     Z-score of close vs 5-day rolling mean/std
  zscore_close_20d    Z-score of close vs 20-day rolling mean/std
  momentum_1d         1-day price return
  momentum_5d         5-day price return
  roc_1d              Rate of change 1 day (%)
  roc_5d              Rate of change 5 days (%)
  adr_5d              Average daily range last 5 days (pips)
  price_vs_week_high  Close position in 5-day range [0, 1]
  price_vs_month_high Close position in 20-day range [0, 1]
  bb_pct_20           Bollinger Band %B, 20-day window [0, 1]

Redis keys:  features:stat:{PAIR}  (e.g. features:stat:EURUSD)
TTL:         3600 seconds (1 hour — price features need freshness)
Scheduler:   background thread, runs compute_all() every 60 minutes

Usage:
    from features.statistical_features import stat_engine
    feats = stat_engine.get_features("EURUSD")
    # {"atr_14": 0.00847, "atr_ratio": 1.23, ...}
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

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

# Pairs whose quote currency is JPY — pip = 0.01 not 0.0001
JPY_PAIRS: set = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

REDIS_TTL: int = 3600          # 1 hour
REDIS_PREFIX: str = "features:stat"
SCHEDULER_INTERVAL: int = 3600  # recompute every hour

# Volatility regime threshold — from course materials
HIGH_VOL_THRESHOLD: float = 1.5  # atr_ratio or vol_ratio > this = high vol

# Lookback in D1 bars needed for longest feature (50-period ATR)
MIN_BARS_REQUIRED: int = 60


# ──────────────────────────────────────────────────────────────────────────────
# Helper: pip size
# ──────────────────────────────────────────────────────────────────────────────

def _pip_size(pair: str) -> float:
    """Return the pip size for a currency pair."""
    return 0.01 if pair in JPY_PAIRS else 0.0001


# ──────────────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────────────

class StatisticalFeatureEngine:
    """
    Computes price-derived statistical and volatility features from OHLCV data.

    Reads M15 candles from TimescaleDB, resamples to D1, then computes
    ATR, rolling vol, Z-scores, momentum, ADR, and Bollinger Band features
    for all 12 G10 pairs. Results are cached in Redis with a 1-hour TTL.

    A background scheduler thread calls compute_all() every hour so that
    the Redis cache is always warm when the signal aggregator reads it.
    """

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ── Redis connection ──────────────────────────────────────────────────────

    @property
    def redis(self) -> redis.Redis:
        if self._redis is None:
            from data_pipeline.storage.redis_pool import redis_client
            self._redis = redis_client
        return self._redis

    def _redis_get(self, key: str) -> Optional[Dict[str, float]]:
        """Retrieve cached features from Redis. Returns None on miss/error."""
        try:
            raw = self.redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning(f"StatEngine | Redis GET failed [{key}]: {exc}")
        return None

    def _redis_set(self, key: str, data: Dict[str, float], ttl: int = REDIS_TTL) -> None:
        """Store features in Redis with TTL."""
        try:
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning(f"StatEngine | Redis SET failed [{key}]: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_features(self, pair: str) -> Dict[str, float]:
        """
        Return statistical features for a single pair.

        Checks Redis cache first. Falls back to compute_all() on cache miss.

        Args:
            pair: Currency pair e.g. "EURUSD"

        Returns:
            Dict of feature_name → float value, or empty dict on failure.
        """
        key = f"{REDIS_PREFIX}:{pair}"
        cached = self._redis_get(key)
        if cached:
            logger.debug(f"StatEngine | Cache HIT: {pair}")
            return cached

        logger.info(f"StatEngine | Cache MISS for {pair} — computing all pairs")
        all_features = self.compute_all()
        return all_features.get(pair, {})

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical features for all 12 pairs, store in Redis.

        Returns:
            Dict of pair → feature dict
        """
        logger.info("StatEngine | Starting full compute for all pairs...")
        results: Dict[str, Dict[str, float]] = {}

        for pair in PAIRS:
            try:
                df = self._load_daily_ohlcv(pair)
                if df is None or len(df) < MIN_BARS_REQUIRED:
                    logger.warning(
                        f"StatEngine | {pair}: insufficient data "
                        f"({len(df) if df is not None else 0} bars, need {MIN_BARS_REQUIRED})"
                    )
                    continue

                feats = self._compute_features(df, pair)
                key = f"{REDIS_PREFIX}:{pair}"
                self._redis_set(key, feats, ttl=REDIS_TTL)
                results[pair] = feats
                logger.debug(f"StatEngine | {pair}: {len(feats)} features cached")

            except Exception as exc:
                logger.error(f"StatEngine | {pair} failed: {exc}")

        logger.success(f"StatEngine | compute_all done: {len(results)}/{len(PAIRS)} pairs")
        return results

    # ── Data loading ──────────────────────────────────────────────────────────

    def _compute_m15_atr(self, pair: str, period: int = 14) -> float:
        """
        Compute ATR_14 directly on M15 bars (not resampled to D1).
        This is the correct stop-loss sizing ATR for an M15 trading system.
        Returns ATR in price units (already JPY-corrected).
        """
        try:
            query = text("""
                SELECT time, high, low, close
                FROM ohlcv
                WHERE symbol    = :symbol
                  AND timeframe = 'M15'
                  AND time      >= NOW() - INTERVAL '7 days'
                ORDER BY time ASC
            """)
            with db.connection() as conn:
                rows = conn.execute(query, {"symbol": pair}).fetchall()

            if len(rows) < period + 1:
                return 0.0

            df = pd.DataFrame(rows, columns=["time", "high", "low", "close"])
            df = df.astype({"high": float, "low": float, "close": float})

            # JPY correction
            if pair in JPY_PAIRS:
                for col in ["high", "low", "close"]:
                    df[col] = df[col] / 100.0

            tr = self._true_range(df)
            atr = self._wilder_smooth(tr, period=period).iloc[-1]
            return float(atr)
        except Exception as exc:
            logger.debug(f"StatEngine | M15 ATR failed {pair}: {exc}")
            return 0.0

    def _load_daily_ohlcv(self, pair: str) -> Optional[pd.DataFrame]:
        """
        Load M15 candles from TimescaleDB and resample to D1.

        We use 120 calendar days of M15 data to ensure ~85+ trading days
        after resampling, which is enough for the 50-period ATR.

        Args:
            pair: Currency pair symbol

        Returns:
            DataFrame with columns [open, high, low, close, volume] indexed
            by date, or None on failure.
        """
        query = text("""
            SELECT
                DATE(time AT TIME ZONE 'UTC')  AS trade_date,
                FIRST(open,  time)             AS open,
                MAX(high)                      AS high,
                MIN(low)                       AS low,
                LAST(close,  time)             AS close,
                SUM(volume)                    AS volume
            FROM ohlcv
            WHERE symbol    = :symbol
              AND timeframe = 'M15'
              AND time      >= NOW() - INTERVAL '120 days'
            GROUP BY trade_date
            ORDER BY trade_date ASC
        """)

        try:
            with db.connection() as conn:
                result = conn.execute(query, {"symbol": pair})
                rows = result.fetchall()

            if not rows:
                logger.warning(f"StatEngine | No OHLCV rows for {pair}")
                return None

            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df.astype({
                "open": float, "high": float,
                "low": float, "close": float, "volume": float,
            })
            # cTrader stores JPY pair prices in points (×100 of actual price)
            # e.g. USDJPY stored as 14900 in DB = 149.00 actual — divide by 100
            if pair in JPY_PAIRS:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] / 100.0
            return df

        except Exception as exc:
            logger.error(f"StatEngine | DB load failed for {pair}: {exc}")
            return None

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute_features(self, df: pd.DataFrame, pair: str) -> Dict[str, float]:
        """
        Compute all 17 statistical features from a D1 OHLCV DataFrame.

        Args:
            df:   D1 OHLCV DataFrame, sorted ascending by date
            pair: Pair name (needed for pip size)

        Returns:
            Dict of feature_name → float (all rounded to 6 d.p.)
        """
        pip = _pip_size(pair)
        feats: Dict[str, float] = {}

        # ── ATR features ──────────────────────────────────────────────────────
        # ATR = Wilder's Average True Range
        # True Range = max(H-L, |H-prev_C|, |L-prev_C|)
        tr = self._true_range(df)

        atr_14 = self._wilder_smooth(tr, period=14).iloc[-1]
        atr_50 = self._wilder_smooth(tr, period=50).iloc[-1]

        feats["atr_14"] = round(float(atr_14), 6)
        feats["atr_pct"] = round(float(atr_14 / df["close"].iloc[-1]), 6)
        feats["atr_ratio"] = round(float(atr_14 / atr_50) if atr_50 > 0 else 1.0, 6)
        # M15 ATR — computed separately, used by risk manager for stop sizing
        # This is the correct ATR for an M15 trading system (not D1)
        feats["atr_14_m15"] = round(float(self._compute_m15_atr(pair)), 6)

        # ── Rolling volatility features ───────────────────────────────────────
        # Log returns are additive and better behaved than simple returns
        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()

        vol_5d  = float(log_returns.rolling(5).std().iloc[-1])
        vol_20d = float(log_returns.rolling(20).std().iloc[-1])

        feats["vol_5d"]  = round(vol_5d,  6)
        feats["vol_20d"] = round(vol_20d, 6)
        feats["vol_ratio"] = round(vol_5d / vol_20d if vol_20d > 0 else 1.0, 6)
        feats["high_vol_regime"] = 1.0 if feats["vol_ratio"] > HIGH_VOL_THRESHOLD else 0.0

        # ── Z-scores of close ─────────────────────────────────────────────────
        # Tells regime classifier whether price is extended (MR) or quiet (PNS)
        close = df["close"]
        feats["zscore_close_5d"]  = round(self._zscore(close, 5),  4)
        feats["zscore_close_20d"] = round(self._zscore(close, 20), 4)

        # ── Momentum & ROC ────────────────────────────────────────────────────
        # THV: strong momentum accelerating | TLV: moderate steady
        # MR: momentum fading | PNS: near-zero momentum
        c_now  = float(close.iloc[-1])
        c_1d   = float(close.iloc[-2])  if len(close) >= 2  else c_now
        c_5d   = float(close.iloc[-6])  if len(close) >= 6  else c_now

        feats["momentum_1d"] = round(c_now - c_1d, 6)
        feats["momentum_5d"] = round(c_now - c_5d, 6)
        feats["roc_1d"] = round((c_now - c_1d) / c_1d * 100 if c_1d > 0 else 0.0, 4)
        feats["roc_5d"] = round((c_now - c_5d) / c_5d * 100 if c_5d > 0 else 0.0, 4)

        # ── ADR — Average Daily Range in pips ─────────────────────────────────
        # Course-specified vol measure: high-to-low range, non-normal distribution
        # Better than std dev for forex because it captures intraday extremes
        daily_range_pips = (df["high"] - df["low"]) / pip
        feats["adr_5d"] = round(float(daily_range_pips.rolling(5).mean().iloc[-1]), 2)

        # ── Price position in range ───────────────────────────────────────────
        # [0 = at period low, 1 = at period high]
        # THV breakout: price near 1.0 with strong momentum
        # MR opportunity: price near 0.0 or 1.0 when momentum fades
        feats["price_vs_week_high"]  = round(self._range_position(close, high=df["high"], low=df["low"], period=5),  4)
        feats["price_vs_month_high"] = round(self._range_position(close, high=df["high"], low=df["low"], period=20), 4)

        # ── Bollinger Band %B ─────────────────────────────────────────────────
        # %B = (price - lower_band) / (upper_band - lower_band)
        # >1 = above upper band (overbought), <0 = below lower (oversold)
        # [0,1] is normal range — values outside indicate breakout or extremity
        feats["bb_pct_20"] = round(self._bollinger_pct(close, period=20, std_mult=2.0), 4)

        return feats

    # ── Mathematical helpers ──────────────────────────────────────────────────

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        """Compute True Range series from OHLCV DataFrame."""
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"]  - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
        """
        Wilder's smoothing (exponential smoothing with alpha=1/period).

        Used for ATR to match the standard Wilder ATR definition.
        Standard EWM uses alpha=2/(n+1); Wilder uses alpha=1/n — different!
        """
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    @staticmethod
    def _zscore(series: pd.Series, period: int) -> float:
        """Rolling Z-score of the last value vs the rolling window."""
        roll_mean = series.rolling(period).mean()
        roll_std  = series.rolling(period).std()
        last_mean = float(roll_mean.iloc[-1])
        last_std  = float(roll_std.iloc[-1])
        last_val  = float(series.iloc[-1])
        if last_std == 0:
            return 0.0
        return (last_val - last_mean) / last_std

    @staticmethod
    def _range_position(
        close: pd.Series,
        high:  pd.Series,
        low:   pd.Series,
        period: int,
    ) -> float:
        """
        Position of last close within the high-low range over last N bars.

        Returns value in [0, 1]: 0 = at period low, 1 = at period high.
        """
        period_high = float(high.rolling(period).max().iloc[-1])
        period_low  = float(low.rolling(period).min().iloc[-1])
        last_close  = float(close.iloc[-1])
        rng = period_high - period_low
        if rng == 0:
            return 0.5
        return max(0.0, min(1.0, (last_close - period_low) / rng))

    @staticmethod
    def _bollinger_pct(series: pd.Series, period: int, std_mult: float) -> float:
        """
        Bollinger Band %B: position of price within the Bollinger Bands.

        %B = (close - lower) / (upper - lower)
        Values outside [0, 1] indicate the price is beyond the bands.
        We do NOT clamp — the extreme values are informative for regime detection.
        """
        roll_mean = series.rolling(period).mean()
        roll_std  = series.rolling(period).std()
        upper = roll_mean + std_mult * roll_std
        lower = roll_mean - std_mult * roll_std
        last_close = float(series.iloc[-1])
        u = float(upper.iloc[-1])
        lo = float(lower.iloc[-1])
        band_width = u - lo
        if band_width == 0:
            return 0.5
        return (last_close - lo) / band_width

    # ── Background scheduler ──────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """
        Start a background daemon thread that recomputes features every hour.

        Call once at application startup. The thread will not block shutdown.
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("StatEngine | Scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="stat-feature-scheduler",
        )
        self._scheduler_thread.start()
        logger.info(f"StatEngine | Scheduler started (interval={SCHEDULER_INTERVAL}s)")

    def stop_scheduler(self) -> None:
        """Signal the scheduler thread to stop on next iteration."""
        self._running = False
        logger.info("StatEngine | Scheduler stop requested")

    def _scheduler_loop(self) -> None:
        """Background loop: compute_all() every SCHEDULER_INTERVAL seconds."""
        logger.info("StatEngine | Scheduler loop running")
        while self._running:
            try:
                self.compute_all()
            except Exception as exc:
                logger.error(f"StatEngine | Scheduler loop error: {exc}")
            time.sleep(SCHEDULER_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Module singleton
# ──────────────────────────────────────────────────────────────────────────────

stat_engine = StatisticalFeatureEngine()
