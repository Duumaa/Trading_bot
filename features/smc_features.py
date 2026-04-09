"""
features/smc_features.py
========================
Smart Money Concept (SMC) feature engine for all 12 G10 pairs.

Computes mechanically-defined SMC features from M15 OHLCV data and caches
them in Redis for use by the Gate 4 entry engine (and A7 SMC model scoring).

Features produced per pair (10 total):
  fvg_bull_dist       Pips to nearest unfilled bullish FVG above price (0 if none)
  fvg_bear_dist       Pips to nearest unfilled bearish FVG below price (0 if none)
  fvg_inside_bull     1 if price is currently inside a bullish FVG (buying area)
  fvg_inside_bear     1 if price is currently inside a bearish FVG (selling area)
  liq_sweep_bull_bars Bars since last bullish liquidity sweep (wick > swing high, close below)
  liq_sweep_bear_bars Bars since last bearish liquidity sweep (wick < swing low, close above)
  ob_bull_dist        Pips to nearest bullish order block above price (0 if none)
  ob_bear_dist        Pips to nearest bearish order block below price (0 if none)
  msb_bull_bars       Bars since last bullish Market Structure Break (swing high broken)
  msb_bear_bars       Bars since last bearish Market Structure Break (swing low broken)

Definitions (all look-back only — no future data used):
  FVG (Fair Value Gap):
    Bullish FVG: bar[i-2].high < bar[i].low  → gap between bar i-2 high and bar i low
    Bearish FVG: bar[i-2].low  > bar[i].high → gap between bar i-2 low and bar i high
    "Unfilled": price has not returned to fill the gap since formation.

  Liquidity Sweep (Stop Hunt):
    Bullish sweep: bar[j].high > most recent swing high AND bar[j].close < swing high
    Bearish sweep: bar[j].low  < most recent swing low  AND bar[j].close > swing low

  Order Block (OB):
    Bullish OB: last bearish (close < open) candle before a bullish impulse (3+ green candles)
    Bearish OB: last bullish (close > open) candle before a bearish impulse (3+ red candles)
    OB is valid until price closes through it.

  Market Structure Break (MSB):
    Bullish MSB: close > most recent significant swing high (last 20 bars)
    Bearish MSB: close < most recent significant swing low (last 20 bars)

Redis key:  features:smc:{PAIR}
TTL:        900s (15 min — M15 granularity)
Scheduler:  background thread, runs compute_all() every 15 minutes
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

# ── Constants ─────────────────────────────────────────────────────────────────

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

REDIS_TTL       = 900     # 15 min
SCHEDULE_SEC    = 900     # recompute every 15 min
LOOKBACK_BARS   = 200     # M15 bars to load (~50h)
SWING_WINDOW    = 5       # bars each side for swing detection
FVG_MAX_AGE     = 96      # max bars age for FVG to stay valid (~24h)
OB_MAX_AGE      = 96      # max bars age for OB to stay valid
MSB_MAX_AGE     = 200     # bars for MSB lookback


def _pip(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else 0.0001


def _redis():
    import redis as _redis_lib
    return _redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


# ── Core computation ──────────────────────────────────────────────────────────

def _find_swing_highs_lows(
    highs: np.ndarray,
    lows: np.ndarray,
    window: int = SWING_WINDOW,
) -> tuple[List[int], List[int]]:
    """Return indices of swing highs and swing lows."""
    sh_idx, sl_idx = [], []
    n = len(highs)
    for i in range(window, n - window):
        if highs[i] > highs[i - window:i].max() and highs[i] > highs[i + 1:i + window + 1].max():
            sh_idx.append(i)
        if lows[i] < lows[i - window:i].min() and lows[i] < lows[i + 1:i + window + 1].min():
            sl_idx.append(i)
    return sh_idx, sl_idx


def compute_smc_features(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Compute SMC features from a DataFrame of M15 OHLCV bars (most recent last).
    Returns a dict of feature_name → float value.
    """
    pip = _pip(pair)
    n   = len(df)

    if n < LOOKBACK_BARS // 2:
        logger.warning(f"SMC | {pair}: only {n} bars — skipping")
        return {}

    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    price  = closes[-1]

    sh_idx, sl_idx = _find_swing_highs_lows(highs, lows, SWING_WINDOW)

    # ── FVG detection ─────────────────────────────────────────────────────────
    fvg_bull_dist   = 0.0
    fvg_bear_dist   = 0.0
    fvg_inside_bull = 0.0
    fvg_inside_bear = 0.0

    bull_fvgs: List[tuple[float, float, int]] = []  # (low, high, formed_idx)
    bear_fvgs: List[tuple[float, float, int]] = []

    last_valid = n - 1  # index of current bar

    for i in range(2, n):
        age = last_valid - i
        if age > FVG_MAX_AGE:
            continue
        # Bullish FVG: gap between bar[i-2].high and bar[i].low
        if highs[i - 2] < lows[i]:
            gap_lo = highs[i - 2]
            gap_hi = lows[i]
            # Check if gap has been filled (any close entered the gap after formation)
            filled = any(lows[j] <= gap_hi and highs[j] >= gap_lo for j in range(i + 1, n))
            if not filled:
                bull_fvgs.append((gap_lo, gap_hi, i))
        # Bearish FVG: gap between bar[i-2].low and bar[i].high
        if lows[i - 2] > highs[i]:
            gap_lo = highs[i]
            gap_hi = lows[i - 2]
            filled = any(lows[j] <= gap_hi and highs[j] >= gap_lo for j in range(i + 1, n))
            if not filled:
                bear_fvgs.append((gap_lo, gap_hi, i))

    # Nearest bullish FVG above price (price is below the gap → approaching it)
    bull_above = [(lo, hi, idx) for lo, hi, idx in bull_fvgs if lo > price]
    if bull_above:
        nearest = min(bull_above, key=lambda x: x[0])
        fvg_bull_dist = (nearest[0] - price) / pip

    # Nearest bearish FVG below price
    bear_below = [(lo, hi, idx) for lo, hi, idx in bear_fvgs if hi < price]
    if bear_below:
        nearest = max(bear_below, key=lambda x: x[1])
        fvg_bear_dist = (price - nearest[1]) / pip

    # Inside a FVG?
    for lo, hi, _ in bull_fvgs:
        if lo <= price <= hi:
            fvg_inside_bull = 1.0
            break
    for lo, hi, _ in bear_fvgs:
        if lo <= price <= hi:
            fvg_inside_bear = 1.0
            break

    # ── Liquidity sweep detection ─────────────────────────────────────────────
    liq_sweep_bull_bars = float(MSB_MAX_AGE)  # bars since last sweep (large = old/none)
    liq_sweep_bear_bars = float(MSB_MAX_AGE)

    # Look at each bar for a sweep of the most recent swing high/low at that point
    for i in range(SWING_WINDOW + 1, n):
        age = last_valid - i
        # Find the most recent swing high before bar i
        sh_before = [idx for idx in sh_idx if idx < i - SWING_WINDOW]
        sl_before = [idx for idx in sl_idx if idx < i - SWING_WINDOW]

        if sh_before:
            recent_sh = highs[sh_before[-1]]
            # Bearish sweep: wick above swing high, close below it
            if highs[i] > recent_sh and closes[i] < recent_sh:
                liq_sweep_bear_bars = min(liq_sweep_bear_bars, float(age))

        if sl_before:
            recent_sl = lows[sl_before[-1]]
            # Bullish sweep: wick below swing low, close above it
            if lows[i] < recent_sl and closes[i] > recent_sl:
                liq_sweep_bull_bars = min(liq_sweep_bull_bars, float(age))

    # ── Order Block detection ─────────────────────────────────────────────────
    ob_bull_dist = 0.0
    ob_bear_dist = 0.0
    IMPULSE_CANDLES = 3  # min consecutive candles to confirm impulse

    bull_obs: List[tuple[float, float, int]] = []  # (ob_low, ob_high, idx)
    bear_obs: List[tuple[float, float, int]] = []

    for i in range(n - IMPULSE_CANDLES - 1):
        age = last_valid - i
        if age > OB_MAX_AGE:
            continue

        # Bullish OB: bearish candle at i, followed by IMPULSE_CANDLES bullish candles
        if closes[i] < opens[i]:  # bearish candle
            is_impulse = all(closes[i + k] > opens[i + k] for k in range(1, IMPULSE_CANDLES + 1))
            if is_impulse:
                ob_lo = lows[i]
                ob_hi = highs[i]
                # Valid only if price hasn't closed below the OB since formation
                broken = any(closes[j] < ob_lo for j in range(i + IMPULSE_CANDLES + 1, n))
                if not broken:
                    bull_obs.append((ob_lo, ob_hi, i))

        # Bearish OB: bullish candle at i, followed by IMPULSE_CANDLES bearish candles
        if closes[i] > opens[i]:  # bullish candle
            is_impulse = all(closes[i + k] < opens[i + k] for k in range(1, IMPULSE_CANDLES + 1))
            if is_impulse:
                ob_lo = lows[i]
                ob_hi = highs[i]
                broken = any(closes[j] > ob_hi for j in range(i + IMPULSE_CANDLES + 1, n))
                if not broken:
                    bear_obs.append((ob_lo, ob_hi, i))

    # Nearest bullish OB below price (price approaching from above → potential support)
    bull_obs_below = [(lo, hi, idx) for lo, hi, idx in bull_obs if hi < price]
    if bull_obs_below:
        nearest = max(bull_obs_below, key=lambda x: x[1])
        ob_bull_dist = (price - nearest[1]) / pip

    # Nearest bearish OB above price
    bear_obs_above = [(lo, hi, idx) for lo, hi, idx in bear_obs if lo > price]
    if bear_obs_above:
        nearest = min(bear_obs_above, key=lambda x: x[0])
        ob_bear_dist = (nearest[0] - price) / pip

    # ── Market Structure Break detection ─────────────────────────────────────
    msb_bull_bars = float(MSB_MAX_AGE)
    msb_bear_bars = float(MSB_MAX_AGE)

    for i in range(SWING_WINDOW + 1, n):
        age = last_valid - i
        sh_before = [idx for idx in sh_idx if idx < i - SWING_WINDOW]
        sl_before = [idx for idx in sl_idx if idx < i - SWING_WINDOW]

        if sh_before:
            recent_sh_val = highs[sh_before[-1]]
            if closes[i] > recent_sh_val:  # bullish MSB
                msb_bull_bars = min(msb_bull_bars, float(age))

        if sl_before:
            recent_sl_val = lows[sl_before[-1]]
            if closes[i] < recent_sl_val:  # bearish MSB
                msb_bear_bars = min(msb_bear_bars, float(age))

    return {
        "fvg_bull_dist":        round(fvg_bull_dist, 2),
        "fvg_bear_dist":        round(fvg_bear_dist, 2),
        "fvg_inside_bull":      fvg_inside_bull,
        "fvg_inside_bear":      fvg_inside_bear,
        "liq_sweep_bull_bars":  liq_sweep_bull_bars,
        "liq_sweep_bear_bars":  liq_sweep_bear_bars,
        "ob_bull_dist":         round(ob_bull_dist, 2),
        "ob_bear_dist":         round(ob_bear_dist, 2),
        "msb_bull_bars":        msb_bull_bars,
        "msb_bear_bars":        msb_bear_bars,
    }


# ── Engine class ──────────────────────────────────────────────────────────────

class SMCFeatureEngine:
    """Computes and caches SMC features for all G10 pairs."""

    def get_features(self, pair: str) -> Dict[str, float]:
        """Return cached SMC features for pair, or compute on cache miss."""
        r = _redis()
        key = f"features:smc:{pair}"
        cached = r.get(key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass
        # Cache miss — compute for this pair only
        data = self._load_bars(pair)
        if data is None or data.empty:
            return {}
        feats = compute_smc_features(data, pair)
        if feats:
            r.setex(key, REDIS_TTL, json.dumps(feats))
        return feats

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """Compute SMC features for all pairs and store in Redis."""
        r = _redis()
        results: Dict[str, Dict[str, float]] = {}

        for pair in PAIRS:
            try:
                data = self._load_bars(pair)
                if data is None or data.empty:
                    continue
                feats = compute_smc_features(data, pair)
                if feats:
                    r.setex(f"features:smc:{pair}", REDIS_TTL, json.dumps(feats))
                    results[pair] = feats
            except Exception as exc:
                logger.error(f"SMCFeatureEngine | {pair} failed: {exc}")

        logger.success(f"SMCFeatureEngine | Computed {len(results)}/{len(PAIRS)} pairs")
        return results

    def _load_bars(self, pair: str) -> Optional[pd.DataFrame]:
        """Load last LOOKBACK_BARS M15 bars from DB."""
        try:
            with db.connection() as conn:
                rows = conn.execute(text("""
                    SELECT time, open, high, low, close
                    FROM ohlcv
                    WHERE symbol = :sym AND timeframe = 'M15'
                    ORDER BY time DESC
                    LIMIT :lim
                """), {"sym": pair, "lim": LOOKBACK_BARS}).fetchall()

            if not rows:
                return None

            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
            df = df.sort_values("time").reset_index(drop=True)

            # JPY price fix
            if pair in JPY_PAIRS and df["close"].iloc[-1] > 1000:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] / 100.0

            return df

        except Exception as exc:
            logger.error(f"SMCFeatureEngine | DB load failed for {pair}: {exc}")
            return None

    def start_scheduler(self) -> None:
        """Start background daemon thread to refresh SMC features every 15 min."""
        def _run():
            logger.info("SMCFeatureEngine | Scheduler started (15 min cycle)")
            while True:
                try:
                    self.compute_all()
                except Exception as exc:
                    logger.error(f"SMCFeatureEngine | Scheduler error: {exc}")
                time.sleep(SCHEDULE_SEC)

        t = threading.Thread(target=_run, daemon=True, name="smc-features")
        t.start()


smc_engine = SMCFeatureEngine()
