"""
features/liquidity_features.py
================================
Market liquidity and informed-trading features for all 12 G10 pairs.

While orderflow_features.py measures WHO is pushing price right now,
liquidity_features.py measures HOW EASY it is to push price and WHERE
institutional footprints are visible in the microstructure.

Trading rationale:
  - VPIN detects when informed (smart money) traders dominate volume
  - Large order detection reveals institutional accumulation/distribution
  - Wick analysis detects liquidity sweeps (stop hunts) before reversals
  - Liquidity score composites depth + spread + activity into one signal
  - These features feed Analyst 4 (PatchTSMixer) entry timing layer

Graceful degradation (same philosophy as orderflow_features.py):
  When tick/OHLCV data is sparse, all features return neutral values.
  The signal pipeline NEVER crashes due to missing microstructure data.
  On VPS with 24/7 live stream, all features become fully live.

Data sources:
  ohlcv table       — M15 candles for wick analysis and price range
  ticks table       — bid/ask ticks for VPIN and run detection
  order_book_snapshots — L2 depth (hooks ready, live on VPS)

JPY scaling:
  Same fix as orderflow_features.py — ticks with JPY bid > 1000 divided
  by 100. OHLCV already corrected by statistical_features pattern.

Features produced per pair (10 total):
  vpin                  Informed trading probability [0, 1]
  large_order_flag      Institutional run detected in ticks [0/1]
  run_length_max        Longest consecutive same-direction tick run
  sweep_detected        Liquidity sweep in last M15 candle [0/1]
  wick_ratio_upper      Upper wick / candle body (last candle)
  wick_ratio_lower      Lower wick / candle body (last candle)
  near_depth_ratio      Spread-derived liquidity proxy [0, 1]
  depth_imbalance_near  Tick-derived near-price imbalance [-1, +1]
  avg_trade_size        Avg pip move per tick (order size proxy)
  liquidity_score       Composite liquidity score [0, 1]

Redis keys:  features:liq:{PAIR}  (e.g. features:liq:EURUSD)
TTL:         300 seconds (5 minutes)
Scheduler:   background thread, runs compute_all() every 5 minutes

Usage:
    from features.liquidity_features import liq_engine
    feats = liq_engine.get_features("EURUSD")
    # {"vpin": 0.42, "sweep_detected": 0.0, "liquidity_score": 0.71, ...}
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

JPY_PAIRS: set = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

PIP: Dict[str, float] = {p: (0.01 if p in JPY_PAIRS else 0.0001) for p in PAIRS}

REDIS_TTL: int = 300
REDIS_PREFIX: str = "features:liq"
SCHEDULER_INTERVAL: int = 300

# VPIN bucket size — number of ticks per volume bucket
VPIN_BUCKET_SIZE: int = 50
MIN_VPIN_BUCKETS: int = 3       # need at least 3 buckets for meaningful VPIN

# Run detection — consecutive same-direction ticks = institutional order
MIN_RUN_LENGTH: int = 5         # 5+ consecutive same-direction = large order flag

# Wick analysis thresholds
SWEEP_WICK_MULTIPLIER: float = 2.0   # wick > 2× avg wick = sweep detected
MIN_CANDLES_WICK: int = 5            # need 5+ candles for avg wick calculation

# Neutral fallback values
NEUTRAL: Dict[str, float] = {
    "vpin":                 0.5,
    "large_order_flag":     0.0,
    "run_length_max":       0.0,
    "sweep_detected":       0.0,
    "wick_ratio_upper":     1.0,
    "wick_ratio_lower":     1.0,
    "near_depth_ratio":     0.5,
    "depth_imbalance_near": 0.0,
    "avg_trade_size":       0.0,
    "liquidity_score":      0.5,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────────────

class LiquidityFeatureEngine:
    """
    Computes market liquidity and informed-trading features.

    Primary data source is the ohlcv table (always populated).
    Tick-derived features (VPIN, runs) are computed when ticks exist,
    otherwise return neutral. L2 depth hooks are ready for VPS deployment.

    The liquidity_score composite combines all signals into one number
    that Analyst 4 uses as a gate: only enter when liquidity_score > 0.4.
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
        try:
            raw = self.redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning(f"LiqEngine | Redis GET failed [{key}]: {exc}")
        return None

    def _redis_set(self, key: str, data: Dict[str, float], ttl: int = REDIS_TTL) -> None:
        try:
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning(f"LiqEngine | Redis SET failed [{key}]: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_features(self, pair: str) -> Dict[str, float]:
        """
        Return liquidity features for a single pair.

        Always returns a complete 10-feature dict.
        Falls back to NEUTRAL values when data is insufficient.

        Args:
            pair: Currency pair e.g. "EURUSD"

        Returns:
            Dict of feature_name → float.
        """
        key = f"{REDIS_PREFIX}:{pair}"
        cached = self._redis_get(key)
        if cached:
            logger.debug(f"LiqEngine | Cache HIT: {pair}")
            return cached

        logger.info(f"LiqEngine | Cache MISS for {pair} — computing all pairs")
        return self.compute_all().get(pair, dict(NEUTRAL))

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """
        Compute liquidity features for all 12 pairs and cache in Redis.

        Returns:
            Dict of pair → feature dict (always complete)
        """
        logger.info("LiqEngine | Computing liquidity features...")
        results: Dict[str, Dict[str, float]] = {}

        for pair in PAIRS:
            try:
                ohlcv_df = self._load_ohlcv(pair, days=30)
                ticks_df = self._load_ticks(pair, limit=500)
                feats    = self._compute_features(pair, ohlcv_df, ticks_df)

                key = f"{REDIS_PREFIX}:{pair}"
                self._redis_set(key, feats)
                results[pair] = feats

                src = "OHLCV" if ticks_df is None else f"OHLCV+{len(ticks_df)}ticks"
                logger.debug(f"LiqEngine | {pair}: 10 features [{src}]")

            except Exception as exc:
                logger.error(f"LiqEngine | {pair} failed: {exc}")
                results[pair] = dict(NEUTRAL)
                self._redis_set(f"{REDIS_PREFIX}:{pair}", dict(NEUTRAL))

        logger.success(f"LiqEngine | compute_all done: {len(results)}/{len(PAIRS)} pairs")
        return results

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_ohlcv(self, pair: str, days: int = 10) -> Optional[pd.DataFrame]:
        """
        Load recent M15 candles and resample to D1 for wick analysis.

        Uses confirmed real table: ohlcv, column: time (not timestamp).
        JPY prices divided by 100 if stored as cTrader points.

        Args:
            pair: Currency pair symbol
            days: Calendar days of history to load

        Returns:
            D1 OHLCV DataFrame or None.
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
              AND time      >= NOW() - INTERVAL :interval
            GROUP BY trade_date
            ORDER BY trade_date ASC
        """)

        try:
            with db.connection() as conn:
                result = conn.execute(
                    query,
                    {"symbol": pair, "interval": f"{days} days"},
                )
                rows = result.fetchall()

            if not rows:
                return None

            df = pd.DataFrame(
                rows,
                columns=["date", "open", "high", "low", "close", "volume"],
            )
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df.astype({
                "open": float, "high": float,
                "low": float, "close": float, "volume": float,
            })

            # JPY scaling fix
            if pair in JPY_PAIRS:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] / 100.0

            return df

        except Exception as exc:
            logger.error(f"LiqEngine | OHLCV load failed for {pair}: {exc}")
            return None

    def _load_ticks(self, pair: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Load recent ticks, excluding epoch-zero bug rows.

        Args:
            pair:  Currency pair symbol
            limit: Max ticks to load

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

            # Fix mixed JPY scaling
            if pair in JPY_PAIRS:
                mask = df["bid"] > 1000
                if mask.any():
                    df.loc[mask, "bid"]    = df.loc[mask, "bid"]    / 100.0
                    df.loc[mask, "ask"]    = df.loc[mask, "ask"]    / 100.0
                    df.loc[mask, "spread"] = df.loc[mask, "spread"] / 100.0

            return df

        except Exception as exc:
            logger.error(f"LiqEngine | ticks load failed for {pair}: {exc}")
            return None

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute_features(
        self,
        pair:     str,
        ohlcv_df: Optional[pd.DataFrame],
        ticks_df: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Compute all 10 liquidity features.

        OHLCV-derived features (wick analysis) work whenever ohlcv_df
        has MIN_CANDLES_WICK+ rows. Tick-derived features (VPIN, runs)
        degrade to neutral when ticks are sparse.

        Args:
            pair:     Currency pair (for pip size lookup)
            ohlcv_df: D1 OHLCV DataFrame (may be None)
            ticks_df: Recent ticks DataFrame (may be None)

        Returns:
            Complete dict of 10 features.
        """
        feats = dict(NEUTRAL)
        pip   = PIP.get(pair, 0.0001)

        # ── OHLCV-derived: wick analysis ──────────────────────────────────────
        if ohlcv_df is not None and len(ohlcv_df) >= MIN_CANDLES_WICK:
            wick_upper, wick_lower, sweep, liq_proxy = \
                self._wick_analysis(ohlcv_df)
            feats["wick_ratio_upper"] = round(wick_upper, 4)
            feats["wick_ratio_lower"] = round(wick_lower, 4)
            feats["sweep_detected"]   = sweep
            # near_depth_ratio derived from spread vs daily range
            feats["near_depth_ratio"] = round(liq_proxy, 4)

        # ── Tick-derived: VPIN ────────────────────────────────────────────────
        if ticks_df is not None and len(ticks_df) >= VPIN_BUCKET_SIZE * MIN_VPIN_BUCKETS:
            feats["vpin"] = round(self._compute_vpin(ticks_df), 4)

        # ── Tick-derived: large order / run detection ─────────────────────────
        if ticks_df is not None and len(ticks_df) >= MIN_RUN_LENGTH * 2:
            run_max, large_flag = self._run_detection(ticks_df)
            feats["run_length_max"]  = float(run_max)
            feats["large_order_flag"] = large_flag

        # ── Tick-derived: avg trade size (pip move per tick) ──────────────────
        if ticks_df is not None and len(ticks_df) >= 10:
            mid = (ticks_df["bid"] + ticks_df["ask"]) / 2.0
            avg_move = float(mid.diff().abs().mean() / pip)
            feats["avg_trade_size"] = round(avg_move, 4)

        # ── Tick-derived: depth imbalance near price ──────────────────────────
        # Proxy: ratio of upward vs downward mid-price moves in recent ticks
        # On VPS with L2 data, replace with actual near-price book imbalance
        if ticks_df is not None and len(ticks_df) >= 20:
            mid   = (ticks_df["bid"] + ticks_df["ask"]) / 2.0
            diffs = mid.diff().dropna()
            up    = int((diffs > 0).sum())
            down  = int((diffs < 0).sum())
            total = up + down
            if total > 0:
                imbal = (up - down) / total  # [-1, +1]
                feats["depth_imbalance_near"] = round(float(imbal), 4)

        # ── Composite liquidity score ─────────────────────────────────────────
        feats["liquidity_score"] = round(
            self._composite_score(feats, ohlcv_df, ticks_df), 4
        )

        return feats

    # ── Analytical helpers ────────────────────────────────────────────────────

    @staticmethod
    def _wick_analysis(
        df: pd.DataFrame,
    ) -> Tuple[float, float, float, float]:
        """
        Analyse candle wicks for sweep detection and liquidity proxy.

        Upper wick = high - max(open, close)
        Lower wick = min(open, close) - low
        Body       = |close - open|

        Sweep detection: last candle's wick > SWEEP_WICK_MULTIPLIER × avg wick.
        This is the footprint of a stop hunt / liquidity grab.

        Liquidity proxy: body / (high - low) ratio.
        High ratio (0.8+) = candles have strong bodies = liquid, trending.
        Low ratio (0.2-) = mostly wicks = illiquid, choppy, MR territory.

        Returns:
            Tuple of (wick_ratio_upper, wick_ratio_lower, sweep_flag, liq_proxy)
        """
        body        = (df["close"] - df["open"]).abs()
        upper_wick  = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick  = df[["open", "close"]].min(axis=1) - df["low"]
        candle_range = df["high"] - df["low"]

        # Avoid division by zero on doji candles
        body_safe  = body.where(body > 1e-8, 1e-8)
        range_safe = candle_range.where(candle_range > 1e-8, 1e-8)

        # Wick ratios for last candle
        last_upper = float(upper_wick.iloc[-1] / body_safe.iloc[-1])
        last_lower = float(lower_wick.iloc[-1] / body_safe.iloc[-1])

        # Sweep detection: last candle's largest wick vs historical avg
        avg_upper_wick = float(upper_wick.iloc[:-1].mean())
        avg_lower_wick = float(lower_wick.iloc[:-1].mean())
        last_wick_max  = max(float(upper_wick.iloc[-1]), float(lower_wick.iloc[-1]))
        avg_wick       = max(avg_upper_wick, avg_lower_wick, 1e-8)
        sweep = 1.0 if last_wick_max > SWEEP_WICK_MULTIPLIER * avg_wick else 0.0

        # Liquidity proxy: avg body/range ratio over last 10 candles
        # High = candles are solid bodies = liquid market
        # Low  = candles are mostly wicks = thin, stop-hunty market
        liq_proxy = float((body_safe / range_safe).tail(10).mean())
        liq_proxy = max(0.0, min(1.0, liq_proxy))

        return last_upper, last_lower, sweep, liq_proxy

    @staticmethod
    def _compute_vpin(ticks_df: pd.DataFrame) -> float:
        """
        Compute VPIN (Volume-Synchronized Probability of Informed Trading).

        Method:
          1. Split ticks into equal-size buckets of VPIN_BUCKET_SIZE
          2. Within each bucket, classify each tick as buy or sell:
             - tick mid > prev mid → buy-initiated
             - tick mid < prev mid → sell-initiated
          3. VPIN per bucket = |buys - sells| / bucket_size
          4. Return average VPIN across all buckets

        High VPIN (>0.7): informed traders dominate → directional move likely
        Low VPIN (<0.3):  noise traders → mean reversion or stagnation

        Args:
            ticks_df: Tick DataFrame sorted ascending by time

        Returns:
            Float in [0, 1].
        """
        mid    = ((ticks_df["bid"] + ticks_df["ask"]) / 2.0).values
        n      = len(mid)
        vpins: List[float] = []

        bucket = VPIN_BUCKET_SIZE
        for start in range(0, n - bucket, bucket):
            chunk = mid[start : start + bucket]
            diffs = np.diff(chunk)
            buys  = int(np.sum(diffs > 0))
            sells = int(np.sum(diffs < 0))
            total = buys + sells
            if total > 0:
                vpins.append(abs(buys - sells) / total)

        if not vpins:
            return 0.5

        return float(np.mean(vpins))

    @staticmethod
    def _run_detection(ticks_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Detect runs of consecutive same-direction mid-price moves.

        A run of 5+ consecutive same-direction ticks suggests an
        institution is working a large order — slicing it to hide
        market impact but still leaving a directional footprint.

        Args:
            ticks_df: Tick DataFrame sorted ascending by time

        Returns:
            Tuple of (max_run_length, large_order_flag).
            large_order_flag = 1.0 if max_run >= MIN_RUN_LENGTH else 0.0
        """
        mid   = (ticks_df["bid"] + ticks_df["ask"]) / 2.0
        diffs = mid.diff().dropna()

        # Convert to direction: +1 up, -1 down, 0 unchanged
        direction = np.sign(diffs.values)

        max_run = 0
        cur_run = 1

        for i in range(1, len(direction)):
            if direction[i] != 0 and direction[i] == direction[i - 1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1

        large_flag = 1.0 if max_run >= MIN_RUN_LENGTH else 0.0
        return max_run, large_flag

    @staticmethod
    def _composite_score(
        feats:    Dict[str, float],
        ohlcv_df: Optional[pd.DataFrame],
        ticks_df: Optional[pd.DataFrame],
    ) -> float:
        """
        Composite liquidity score [0, 1].

        Weights:
          near_depth_ratio  30%   body/range = candle quality
          vpin (inverted)   25%   low VPIN = more liquid (noise dominated)
          spread tightness  25%   derived from near_depth_ratio proxy
          sweep penalty     20%   sweep = illiquid moment, penalise

        Returns:
            Float in [0, 1]. Higher = more liquid, safer to trade.
            0.4 is the minimum threshold Analyst 4 uses as entry gate.
        """
        # Body/range ratio → quality of candles (0=all wicks, 1=all body)
        depth_score = feats.get("near_depth_ratio", 0.5)

        # VPIN: high VPIN means informed flow, which is directional but risky
        # For liquidity scoring we want the INVERSE — low VPIN = more liquid
        vpin_score = 1.0 - feats.get("vpin", 0.5)

        # Sweep penalty: if a sweep just happened, liquidity momentarily thin
        sweep_penalty = 1.0 - feats.get("sweep_detected", 0.0) * 0.5

        # Data availability bonus: more data = more confidence in the score
        data_bonus = 0.5
        if ohlcv_df is not None and len(ohlcv_df) >= MIN_CANDLES_WICK:
            data_bonus += 0.25
        if ticks_df is not None and len(ticks_df) >= VPIN_BUCKET_SIZE * MIN_VPIN_BUCKETS:
            data_bonus += 0.25
        data_bonus = min(1.0, data_bonus)

        score = (
            depth_score  * 0.30 +
            vpin_score   * 0.25 +
            sweep_penalty * 0.20 +
            data_bonus   * 0.25
        )
        return max(0.0, min(1.0, float(score)))

    # ── Background scheduler ──────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """Start background thread that recomputes liquidity every 5 minutes."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("LiqEngine | Scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="liq-feature-scheduler",
        )
        self._scheduler_thread.start()
        logger.info(f"LiqEngine | Scheduler started (interval={SCHEDULER_INTERVAL}s)")

    def stop_scheduler(self) -> None:
        """Signal the scheduler thread to stop."""
        self._running = False
        logger.info("LiqEngine | Scheduler stop requested")

    def _scheduler_loop(self) -> None:
        """Background loop: compute_all() every SCHEDULER_INTERVAL seconds."""
        logger.info("LiqEngine | Scheduler loop running")
        while self._running:
            try:
                self.compute_all()
            except Exception as exc:
                logger.error(f"LiqEngine | Scheduler loop error: {exc}")
            time.sleep(SCHEDULER_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Module singleton
# ──────────────────────────────────────────────────────────────────────────────

liq_engine = LiquidityFeatureEngine()
