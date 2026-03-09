"""
models/analyst1_regime/regime_classifier.py
============================================
Analyst 1: LightGBM Regime Classifier

Classifies each pair's current market state into one of 4 regimes:
  THV  Trending High Volatility   → Full size, trend-follow
  TLV  Trending Low Volatility    → Full size, trend-follow
  MR   Mean Reverting             → Reduced size, fade extremes
  PNS  Pre-News Stagnation        → FLAT — no trades regardless of other signals

Architecture position:
  This is the GATE for all trading. No trade is taken without regime approval.
  signal_aggregator.py checks regime FIRST before consulting Analyst 2–6.

  PNS → forced flat (even if Analysts 2–6 all agree)
  MR  → only mean-reversion setups permitted, reduced sizing
  THV/TLV → trend-following entries permitted, normal sizing

Training:
  Labels generated from 15Y M15 Kaggle data (4.4M rows):
    THV: ATR_ratio > 1.3 AND trending (|slope_20| > threshold)
    TLV: ATR_ratio ≤ 1.3 AND trending
    MR:  ADX < 20 AND mean-reverting price action
    PNS: Within 2h of HIGH-impact event AND low volatility

  Features: 89 per pair across 6 Redis modules
  Model:    LightGBM multiclass (4 classes)
  Per-pair: One model trained per pair = 12 models total

  Train/val/test split: 70/15/15 (chronological — NO shuffling)
  Minimum rows per pair: 50,000 M15 bars

Redis cache:
  Key: features:regime:{PAIR}   TTL: 15 minutes (900s)
  Value: {regime, confidence, probabilities, updated_at}

Usage:
  from models.analyst1_regime.regime_classifier import regime_classifier
  result = regime_classifier.get_regime("AUDUSD")
  # → {"regime": "THV", "confidence": 0.84, "probabilities": {...}}

  regime_classifier.train_all()    # train all 12 models
  regime_classifier.compute_all()  # run inference, cache to Redis
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

from data_pipeline.storage.timeseries_db import db  # noqa: E402

# ── optional LightGBM import ──────────────────────────────────
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("Regime | LightGBM not installed. Run: pip install lightgbm")

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

# Regime labels — integer encoding for LightGBM
REGIME_LABELS = {0: "THV", 1: "TLV", 2: "MR", 3: "PNS"}
REGIME_INT    = {"THV": 0, "TLV": 1, "MR": 2, "PNS": 3}

# Labelling thresholds (tuned for M15 FX data)
ATR_HIGH_VOL_THRESHOLD  = 1.25   # ATR ratio above this = high vol
ADX_TRENDING_THRESHOLD  = 22.0   # ADX above this = trending
BB_SQUEEZE_THRESHOLD    = 0.003  # BB width below this = squeeze / PNS
MOMENTUM_THRESHOLD      = 0.0003 # |momentum_5d| above this = directional
PRE_EVENT_HOURS         = 2.0    # Flag PNS within 2h of HIGH impact event

# Model storage
MODEL_DIR = ROOT / "models" / "analyst1_regime" / "saved_models"

# Cache
CACHE_TTL           = 900    # 15 minutes for regime (fast-changing)
SCHEDULER_INTERVAL  = 900    # Recompute every 15 minutes

# Training
MIN_ROWS_PER_PAIR   = 10_000  # Minimum bars needed to train
TRAIN_RATIO         = 0.70
VAL_RATIO           = 0.15
# Test = remaining 0.15

# LightGBM hyperparameters
LGB_PARAMS = {
    "objective":        "multiclass",
    "num_class":        4,
    "metric":           "multi_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 20,
    "n_estimators":     500,
    "early_stopping_rounds": 50,
    "verbose":          -1,
    "n_jobs":           -1,
    "random_state":     42,
}

# ── feature names (exact keys from each Redis module) ─────────
FEATURE_NAMES: List[str] = [
    # statistical (17)
    "atr_14", "atr_pct", "atr_ratio",
    "vol_5d", "vol_20d", "vol_ratio",
    "high_vol_regime",
    "zscore_close_5d", "zscore_close_20d",
    "momentum_1d", "momentum_5d",
    "roc_1d", "roc_5d",
    "adr_5d",
    "price_vs_week_high", "price_vs_month_high",
    "bb_pct_20",
    # vwap (12)
    "tokyo_vwap", "london_vwap", "ny_vwap",
    "price_vs_tokyo_vwap", "price_vs_london_vwap", "price_vs_ny_vwap",
    "london_vwap_upper", "london_vwap_lower",
    "london_vwap_band_width", "ny_vwap_band_width",
    "in_london_ny_overlap", "session_vwap_trend",
    # event (8)
    "hours_until_high_impact", "hours_until_any_impact",
    "high_impact_count_24h", "blackout_flag",
    "post_event_flag", "surprise_last",
    "surprise_avg_5", "event_density_24h",
    # orderflow (12)
    "imbalance_1m", "imbalance_5m",
    "buy_pressure_100", "buy_pressure_500",
    "spread_current", "spread_zscore", "spread_ratio",
    "tick_velocity_1m", "tick_velocity_5m",
    "velocity_ratio", "flow_imbalance_delta", "order_book_depth",
    # liquidity (10)
    "vpin", "large_order_flag", "run_length_max",
    "sweep_detected",
    "wick_ratio_upper", "wick_ratio_lower",
    "near_depth_ratio", "depth_imbalance_near",
    "avg_trade_size", "liquidity_score",
    # yield (selected — most regime-relevant from 30)
    "yield_diff_10Y", "yield_diff_2Y",
    "carry_net", "rollover_daily_usd",
    "atr_ratio",          # duplicate key — handled in builder (skipped if dup)
    "vol_ratio",          # duplicate key — handled in builder
    "yield_zscore_10Y", "yield_zscore_2Y",
    "curve_slope_diff",
]

# Deduplicated final feature list (preserve order, remove duplicates)
_seen: set = set()
FEATURE_NAMES_DEDUP: List[str] = []
for _f in FEATURE_NAMES:
    if _f not in _seen:
        FEATURE_NAMES_DEDUP.append(_f)
        _seen.add(_f)

# ══════════════════════════════════════════════════════════════
# REDIS
# ══════════════════════════════════════════════════════════════

def _get_redis():
    import redis as redis_lib
    from dotenv import load_dotenv
    load_dotenv(str(ROOT / ".env"))
    return redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


# ══════════════════════════════════════════════════════════════
# LABEL GENERATOR
# ══════════════════════════════════════════════════════════════

class RegimeLabelGenerator:
    """
    Generates 4-class regime labels from raw M15 OHLCV data.

    Label logic (applied to each M15 bar):

    PNS (Pre-News Stagnation) — checked FIRST, highest priority:
      • BB width < 0.003 (squeeze)
      • AND ATR ratio < 1.0 (below-normal volatility)
      • Note: we use a simplified version without live calendar at train time.
        At inference, event_features blackout_flag overrides this.

    THV (Trending High Volatility):
      • ATR ratio ≥ ATR_HIGH_VOL_THRESHOLD (1.25)
      • AND |momentum_5d| ≥ MOMENTUM_THRESHOLD (directional)

    TLV (Trending Low Volatility):
      • ATR ratio < ATR_HIGH_VOL_THRESHOLD
      • AND |momentum_5d| ≥ MOMENTUM_THRESHOLD
      • AND ADX proxy (via ROC consistency) ≥ ADX_TRENDING_THRESHOLD proxy

    MR (Mean Reverting) — default if none of the above:
      • Low momentum, oscillating price action
    """

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate regime labels for a DataFrame of M15 OHLCV bars.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index: DatetimeIndex (UTC)

        Returns:
            pd.Series of int labels {0=THV, 1=TLV, 2=MR, 3=PNS}
        """
        n = len(df)
        labels = np.full(n, REGIME_INT["MR"], dtype=np.int8)  # default = MR

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values

        # ── ATR (14-period) ───────────────────────────────────
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr14 = np.full(n, np.nan)
        if len(tr) >= 14:
            atr14[14] = tr[:14].mean()
            for i in range(15, n):
                atr14[i] = (atr14[i-1] * 13 + tr[i-1]) / 14

        # ATR ratio vs 50-period rolling average
        atr_ma50 = pd.Series(atr14).rolling(50, min_periods=20).mean().values
        atr_ratio = np.where(atr_ma50 > 0, atr14 / atr_ma50, 1.0)

        # ── Momentum (5d = 96 M15 bars) ───────────────────────
        shift96 = 96
        momentum5d = np.full(n, 0.0)
        momentum5d[shift96:] = (close[shift96:] - close[:-shift96]) / (
            close[:-shift96] + 1e-10
        )

        # ── Bollinger Band width ───────────────────────────────
        bb_period = 20
        close_series = pd.Series(close)
        bb_mid = close_series.rolling(bb_period, min_periods=bb_period).mean()
        bb_std = close_series.rolling(bb_period, min_periods=bb_period).std()
        bb_width = (2 * bb_std / (bb_mid + 1e-10)).values

        # ── ADX proxy: ROC consistency (15-period) ────────────
        roc1 = np.full(n, 0.0)
        roc1[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)

        # Count consistent direction over last 15 bars
        roc_series = pd.Series(roc1)
        pos_count = roc_series.rolling(15).apply(
            lambda x: (x > 0).sum(), raw=True
        ).values
        adx_proxy = np.abs(pos_count - 7.5) * (100 / 7.5)  # 0–100 scale

        # ── Apply label logic ─────────────────────────────────
        valid = ~np.isnan(atr14)

        # PNS: squeeze + below-normal vol (priority 1)
        pns_mask = valid & (bb_width < BB_SQUEEZE_THRESHOLD) & (atr_ratio < 1.0)
        labels[pns_mask] = REGIME_INT["PNS"]

        # THV: high vol + trending (priority 2)
        thv_mask = valid & ~pns_mask & (
            (atr_ratio >= ATR_HIGH_VOL_THRESHOLD) &
            (np.abs(momentum5d) >= MOMENTUM_THRESHOLD)
        )
        labels[thv_mask] = REGIME_INT["THV"]

        # TLV: low-mod vol + trending (priority 3)
        tlv_mask = valid & ~pns_mask & ~thv_mask & (
            (np.abs(momentum5d) >= MOMENTUM_THRESHOLD) &
            (adx_proxy >= ADX_TRENDING_THRESHOLD)
        )
        labels[tlv_mask] = REGIME_INT["TLV"]

        # MR: everything else (default, priority 4)
        # already set as default above

        return pd.Series(labels, index=df.index, name="regime_label")

    def label_distribution(self, labels: pd.Series) -> Dict[str, float]:
        """Return % distribution of each regime label."""
        total = len(labels)
        dist = {}
        for i, name in REGIME_LABELS.items():
            count = (labels == i).sum()
            dist[name] = round(count / total * 100, 1)
        return dist


# ══════════════════════════════════════════════════════════════
# FEATURE BUILDER (from OHLCV — for training)
# ══════════════════════════════════════════════════════════════

class RegimeFeatureBuilder:
    """
    Builds training features from raw OHLCV data.

    Used ONLY during training (Kaggle historical data).
    At inference, features come from Redis (live feature engines).

    Produces the same feature names as FEATURE_NAMES_DEDUP
    for features computable from OHLCV. Event/orderflow/liquidity
    features use neutral defaults at training time.
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from OHLCV DataFrame.

        Args:
            df: M15 OHLCV DataFrame, columns: open high low close volume

        Returns:
            DataFrame with columns matching FEATURE_NAMES_DEDUP
            (OHLCV-computable features filled, others = 0.5 neutral)
        """
        n      = len(df)
        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(n)

        feat = {}

        # ── ATR features ──────────────────────────────────────
        tr = np.zeros(n)
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        atr_ma50 = pd.Series(atr14).rolling(50, min_periods=10).mean().values
        feat["atr_14"]   = atr14
        feat["atr_pct"]  = np.where(close > 0, atr14 / close, 0)
        feat["atr_ratio"] = np.where(atr_ma50 > 0, atr14 / atr_ma50, 1.0)

        # ── Volatility features ───────────────────────────────
        ret = pd.Series(np.log(close / np.maximum(np.roll(close, 1), 1e-10)))
        ret.iloc[0] = 0
        feat["vol_5d"]  = ret.rolling(32,  min_periods=5).std().values   # 32 M15 = 8h ≈ 1d
        feat["vol_20d"] = ret.rolling(128, min_periods=20).std().values   # 128 M15 ≈ 5d
        feat["vol_ratio"] = np.where(
            feat["vol_20d"] > 0,
            feat["vol_5d"] / feat["vol_20d"],
            1.0
        )
        feat["high_vol_regime"] = (feat["atr_ratio"] > ATR_HIGH_VOL_THRESHOLD).astype(float)

        # ── Z-score features ──────────────────────────────────
        close_s = pd.Series(close)
        roll5  = close_s.rolling(32,  min_periods=5)
        roll20 = close_s.rolling(128, min_periods=20)
        feat["zscore_close_5d"]  = ((close_s - roll5.mean())  / (roll5.std()  + 1e-10)).values
        feat["zscore_close_20d"] = ((close_s - roll20.mean()) / (roll20.std() + 1e-10)).values

        # ── Momentum / ROC ────────────────────────────────────
        feat["momentum_1d"] = close_s.pct_change(8).values    # 8 M15 bars ≈ 2h
        feat["momentum_5d"] = close_s.pct_change(96).values   # 96 M15 ≈ 1 trading day
        feat["roc_1d"]      = feat["momentum_1d"]
        feat["roc_5d"]      = feat["momentum_5d"]

        # ── ADR (Average Daily Range) ─────────────────────────
        daily_range = high - low
        feat["adr_5d"] = pd.Series(daily_range).rolling(32, min_periods=5).mean().values

        # ── Price position ────────────────────────────────────
        week_high  = pd.Series(high).rolling(672, min_periods=96).max().values   # ~1 week
        month_high = pd.Series(high).rolling(2880, min_periods=672).max().values # ~1 month
        feat["price_vs_week_high"]  = np.where(week_high  > 0, close / week_high  - 1, 0)
        feat["price_vs_month_high"] = np.where(month_high > 0, close / month_high - 1, 0)

        # ── Bollinger Band pct ────────────────────────────────
        bb_mid = close_s.rolling(20, min_periods=10).mean()
        bb_std = close_s.rolling(20, min_periods=10).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = bb_upper - bb_lower
        feat["bb_pct_20"] = ((close_s - bb_lower) / (bb_range + 1e-10)).clip(0, 1).values

        # ── VWAP proxies (OHLCV available) ───────────────────
        typical = (high + low + close) / 3
        if volume.sum() > 0:
            vwap_96 = (pd.Series(typical * volume).rolling(96, min_periods=10).sum() /
                       pd.Series(volume).rolling(96, min_periods=10).sum()).values
        else:
            vwap_96 = typical

        feat["tokyo_vwap"]           = vwap_96
        feat["london_vwap"]          = vwap_96
        feat["ny_vwap"]              = vwap_96
        feat["price_vs_tokyo_vwap"]  = close - vwap_96
        feat["price_vs_london_vwap"] = close - vwap_96
        feat["price_vs_ny_vwap"]     = close - vwap_96
        feat["london_vwap_upper"]    = vwap_96
        feat["london_vwap_lower"]    = vwap_96
        feat["london_vwap_band_width"] = np.zeros(n)
        feat["ny_vwap_band_width"]     = np.zeros(n)
        feat["in_london_ny_overlap"]   = np.zeros(n)
        feat["session_vwap_trend"]     = np.sign(close - vwap_96)

        # ── Event features — neutral defaults at train time ───
        feat["hours_until_high_impact"] = np.full(n, 24.0)
        feat["hours_until_any_impact"]  = np.full(n, 24.0)
        feat["high_impact_count_24h"]   = np.zeros(n)
        feat["blackout_flag"]           = np.zeros(n)
        feat["post_event_flag"]         = np.zeros(n)
        feat["surprise_last"]           = np.zeros(n)
        feat["surprise_avg_5"]          = np.zeros(n)
        feat["event_density_24h"]       = np.full(n, 0.5)

        # ── Order flow — neutral defaults ─────────────────────
        feat["imbalance_1m"]         = np.full(n, 0.0)
        feat["imbalance_5m"]         = np.full(n, 0.0)
        feat["buy_pressure_100"]     = np.full(n, 0.5)
        feat["buy_pressure_500"]     = np.full(n, 0.5)
        feat["spread_current"]       = np.zeros(n)
        feat["spread_zscore"]        = np.zeros(n)
        feat["spread_ratio"]         = np.ones(n)
        feat["tick_velocity_1m"]     = np.zeros(n)
        feat["tick_velocity_5m"]     = np.zeros(n)
        feat["velocity_ratio"]       = np.ones(n)
        feat["flow_imbalance_delta"] = np.zeros(n)
        feat["order_book_depth"]     = np.full(n, 0.5)

        # ── Liquidity — wick-based proxies from OHLCV ─────────
        body   = np.abs(close - df["open"].values)
        candle = high - low + 1e-10
        feat["vpin"]               = np.full(n, 0.5)
        feat["large_order_flag"]   = np.zeros(n)
        feat["run_length_max"]     = np.zeros(n)
        feat["sweep_detected"]     = np.zeros(n)
        feat["wick_ratio_upper"]   = (high - np.maximum(close, df["open"].values)) / candle
        feat["wick_ratio_lower"]   = (np.minimum(close, df["open"].values) - low) / candle
        feat["near_depth_ratio"]   = np.full(n, 0.35)
        feat["depth_imbalance_near"] = np.zeros(n)
        feat["avg_trade_size"]     = np.full(n, 0.5)
        feat["liquidity_score"]    = np.full(n, 0.5)

        # ── Yield features — neutral defaults at train time ───
        # (Not available in OHLCV — will be filled at inference from Redis)
        feat["yield_diff_10Y"]   = np.zeros(n)
        feat["yield_diff_2Y"]    = np.zeros(n)
        feat["carry_net"]        = np.zeros(n)
        feat["rollover_daily_usd"] = np.zeros(n)
        feat["yield_zscore_10Y"] = np.zeros(n)
        feat["yield_zscore_2Y"]  = np.zeros(n)
        feat["curve_slope_diff"] = np.zeros(n)

        result = pd.DataFrame(feat, index=df.index)

        # Return only columns in FEATURE_NAMES_DEDUP, in order
        final_cols = [c for c in FEATURE_NAMES_DEDUP if c in result.columns]
        return result[final_cols].replace([np.inf, -np.inf], 0).fillna(0)


# ══════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════

def _load_ohlcv(pair: str, timeframe: str = "M15",
                min_date: str = "2011-01-01") -> pd.DataFrame:
    """
    Load M15 OHLCV from TimescaleDB for training.

    Args:
        pair:      e.g. "AUDUSD"
        timeframe: "M15" or "D1"
        min_date:  Start date string

    Returns:
        DataFrame indexed by UTC timestamp with columns:
        open, high, low, close, volume
    """
    sql = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol    = :symbol
          AND timeframe = :tf
          AND time      >= CAST(:min_date AS timestamptz)
        ORDER BY time ASC
    """
    with db.connection() as conn:
        rows = conn.execute(
            text(sql),
            {"symbol": pair, "tf": timeframe, "min_date": min_date}
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    # JPY pair scaling correction (cTrader stores ×100)
    if "JPY" in pair:
        for col in ["open", "high", "low", "close"]:
            if df[col].mean() > 500:
                df[col] = df[col] / 100.0

    return df


# ══════════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════

class RegimeClassifier:
    """
    Analyst 1: LightGBM Regime Classifier.

    Trains one LightGBM model per pair on 15Y of M15 data.
    At inference, reads live features from Redis and returns
    regime + confidence for each pair.

    Usage pattern:
      1. regime_classifier.train_all()         # once, before Phase 4 live
      2. regime_classifier.compute_all()        # every 15 minutes via scheduler
      3. result = regime_classifier.get_regime("AUDUSD")  # in signal_aggregator
    """

    def __init__(self) -> None:
        self._models:    Dict[str, "lgb.Booster"] = {}
        self._label_gen  = RegimeLabelGenerator()
        self._feat_build = RegimeFeatureBuilder()
        self._redis      = None
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._load_saved_models()

    # ── Redis ─────────────────────────────────────────────────

    @property
    def r(self):
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    def _redis_get(self, key: str) -> Optional[Dict]:
        try:
            val = self.r.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def _redis_set(self, key: str, value: Dict, ttl: int = CACHE_TTL) -> None:
        try:
            self.r.setex(key, ttl, json.dumps(value))
        except Exception as exc:
            logger.warning(f"Regime | Redis write failed: {exc}")

    # ── Public API ────────────────────────────────────────────

    def get_regime(self, pair: str) -> Dict:
        """
        Get current regime for a pair.
        Returns cached value if available, else computes fresh.

        Returns dict:
          regime      : "THV" | "TLV" | "MR" | "PNS"
          confidence  : 0.0–1.0
          probabilities: {THV: float, TLV: float, MR: float, PNS: float}
          model_available: bool
          updated_at  : ISO timestamp
        """
        cached = self._redis_get(f"features:regime:{pair}")
        if cached:
            return cached
        return self._infer_pair(pair)

    def compute_all(self) -> Dict[str, Dict]:
        """
        Run inference for all 12 pairs and cache results.
        Called every 15 minutes by scheduler.
        """
        logger.info("Regime | Computing regimes for all 12 pairs...")
        results = {}
        for pair in PAIRS:
            try:
                result = self._infer_pair(pair)
                self._redis_set(f"features:regime:{pair}", result)
                results[pair] = result
                logger.info(
                    f"Regime | {pair}: {result['regime']} "
                    f"conf={result['confidence']:.2f} "
                    f"({'model' if result['model_available'] else 'rule'})"
                )
            except Exception as exc:
                logger.error(f"Regime | {pair} inference failed: {exc}")
                results[pair] = self._rule_based_regime(pair)

        regimes = {p: r["regime"] for p, r in results.items()}
        logger.success(f"Regime | Done: {regimes}")
        return results

    def train_all(self, timeframe: str = "M15") -> None:
        """
        Train one LightGBM model per pair on historical OHLCV data.
        Saves models to MODEL_DIR.

        Run this ONCE before live trading. Takes ~5–15 minutes.
        Requires 4.4M rows of M15 data in ohlcv table (Kaggle import done).
        """
        if not LGB_AVAILABLE:
            logger.error("Regime | LightGBM not installed. pip install lightgbm")
            return

        logger.info("=" * 60)
        logger.info("Regime | TRAINING ALL REGIME MODELS")
        logger.info("=" * 60)

        for pair in PAIRS:
            try:
                self._train_pair(pair, timeframe)
            except Exception as exc:
                logger.error(f"Regime | {pair} training failed: {exc}")

        logger.success("Regime | Training complete for all pairs")

    def start_scheduler(self) -> None:
        """Start background thread to refresh regimes every 15 minutes."""
        def _loop():
            import time
            logger.info("Regime | Scheduler started (15min interval)")
            while True:
                try:
                    self.compute_all()
                except Exception as exc:
                    logger.error(f"Regime | Scheduler error: {exc}")
                time.sleep(SCHEDULER_INTERVAL)

        t = threading.Thread(target=_loop, daemon=True, name="regime-scheduler")
        t.start()
        logger.info("Regime | Scheduler thread launched")

    # ── Training ──────────────────────────────────────────────

    def _train_pair(self, pair: str, timeframe: str) -> None:
        """Train and save LightGBM model for one pair."""
        logger.info(f"Regime | {pair}: Loading OHLCV data...")
        df = _load_ohlcv(pair, timeframe)

        if len(df) < MIN_ROWS_PER_PAIR:
            logger.warning(
                f"Regime | {pair}: Only {len(df)} rows — "
                f"need {MIN_ROWS_PER_PAIR}. Skipping."
            )
            return

        logger.info(f"Regime | {pair}: {len(df):,} rows — generating labels...")

        # Generate labels
        labels = self._label_gen.generate_labels(df)
        dist   = self._label_gen.label_distribution(labels)
        logger.info(f"Regime | {pair}: Label dist = {dist}")

        # Build features
        logger.info(f"Regime | {pair}: Building features ({len(FEATURE_NAMES_DEDUP)} cols)...")
        X = self._feat_build.build(df)
        y = labels.values

        # Drop NaN rows (first N rows before rolling windows fill)
        valid_mask = ~np.isnan(X.values).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        n = len(X)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

        X_train, y_train = X.iloc[:train_end], y[:train_end]
        X_val,   y_val   = X.iloc[train_end:val_end], y[train_end:val_end]
        X_test,  y_test  = X.iloc[val_end:], y[val_end:]

        logger.info(
            f"Regime | {pair}: train={len(X_train):,} "
            f"val={len(X_val):,} test={len(X_test):,}"
        )

        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train,
                                  feature_name=list(X.columns))
        val_data   = lgb.Dataset(X_val,   label=y_val,
                                  feature_name=list(X.columns),
                                  reference=train_data)

        params = {k: v for k, v in LGB_PARAMS.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}

        callbacks = [
            lgb.early_stopping(LGB_PARAMS["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=50),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=LGB_PARAMS["n_estimators"],
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Evaluate on test set
        preds     = model.predict(X_test)
        pred_cls  = preds.argmax(axis=1)
        accuracy  = (pred_cls == y_test).mean()
        logger.success(
            f"Regime | {pair}: Test accuracy={accuracy:.3f} "
            f"best_iter={model.best_iteration}"
        )

        # Save model
        model_path = MODEL_DIR / f"regime_{pair}.lgb"
        model.save_model(str(model_path))
        self._models[pair] = model
        logger.success(f"Regime | {pair}: Saved → {model_path}")

    # ── Inference ─────────────────────────────────────────────

    def _infer_pair(self, pair: str) -> Dict:
        """
        Run regime inference for one pair using live Redis features.
        Falls back to rule-based regime if model not available.
        """
        if pair in self._models and LGB_AVAILABLE:
            return self._model_inference(pair)
        return self._rule_based_regime(pair)

    def _model_inference(self, pair: str) -> Dict:
        """Run LightGBM model inference using live Redis features."""
        feats = self._get_live_features(pair)
        if feats is None:
            return self._rule_based_regime(pair)

        # Build feature vector in correct order
        x = np.array([[feats.get(f, 0.0) for f in FEATURE_NAMES_DEDUP]])
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        probs    = self._models[pair].predict(x)[0]
        pred_cls = int(probs.argmax())
        regime   = REGIME_LABELS[pred_cls]
        conf     = float(probs[pred_cls])

        return {
            "pair":            pair,
            "regime":          regime,
            "confidence":      round(conf, 4),
            "probabilities":   {REGIME_LABELS[i]: round(float(p), 4)
                                for i, p in enumerate(probs)},
            "model_available": True,
            "updated_at":      datetime.now(timezone.utc).isoformat(),
        }

    def _rule_based_regime(self, pair: str) -> Dict:
        """
        Rule-based regime fallback when LightGBM model not trained yet.

        Uses only statistical + event features from Redis.
        Less accurate than model but always available.

        Rules (same logic as label generator, applied to live features):
          PNS: blackout_flag=1 OR hours_until_high_impact < 2
          THV: atr_ratio > 1.25 AND |momentum_5d| > threshold
          TLV: |momentum_5d| > threshold AND NOT high vol
          MR:  default
        """
        feats = self._get_live_features(pair) or {}

        # PNS check — event blackout always wins
        blackout        = feats.get("blackout_flag", 0.0)
        hours_to_event  = feats.get("hours_until_high_impact", 999.0)
        if blackout > 0.5 or hours_to_event < PRE_EVENT_HOURS:
            return self._make_result(pair, "PNS", 0.90, False)

        atr_ratio   = feats.get("atr_ratio",   1.0)
        momentum_5d = feats.get("momentum_5d", 0.0)
        vol_ratio   = feats.get("vol_ratio",   1.0)
        bb_pct      = feats.get("bb_pct_20",   0.5)
        zscore_20d  = feats.get("zscore_close_20d", 0.0)

        abs_mom = abs(momentum_5d)

        # THV: high vol trending
        if atr_ratio >= ATR_HIGH_VOL_THRESHOLD and abs_mom >= MOMENTUM_THRESHOLD:
            conf = min(0.5 + (atr_ratio - 1.25) * 0.3, 0.85)
            return self._make_result(pair, "THV", conf, False)

        # TLV: low-mod vol trending
        if abs_mom >= MOMENTUM_THRESHOLD and vol_ratio < ATR_HIGH_VOL_THRESHOLD:
            conf = min(0.5 + abs_mom * 100, 0.80)
            return self._make_result(pair, "TLV", conf, False)

        # MR: default
        # High confidence MR if zscore is extreme (mean reversion setup)
        mr_conf = 0.55 + min(abs(zscore_20d) * 0.05, 0.20)
        return self._make_result(pair, "MR", mr_conf, False)

    def _make_result(self, pair: str, regime: str,
                     confidence: float, model_available: bool) -> Dict:
        """Build standardised result dict."""
        probs = {r: 0.05 for r in ["THV", "TLV", "MR", "PNS"]}
        probs[regime] = confidence
        return {
            "pair":            pair,
            "regime":          regime,
            "confidence":      round(confidence, 4),
            "probabilities":   {k: round(v, 4) for k, v in probs.items()},
            "model_available": model_available,
            "updated_at":      datetime.now(timezone.utc).isoformat(),
        }

    # ── Feature aggregation ───────────────────────────────────

    def _get_live_features(self, pair: str) -> Optional[Dict[str, float]]:
        """
        Aggregate features from all Redis modules for one pair.

        Reads from:
          features:stat:{pair}      → statistical
          features:vwap:{pair}      → vwap
          features:event:{pair}     → event
          features:flow:{pair}      → orderflow
          features:liq:{pair}       → liquidity
          features:yield:{pair}     → yield

        Returns flat dict of all features, or None if no data.
        """
        module_keys = {
            "features:stat:{pair}":   "stat",
            "features:vwap:{pair}":   "vwap",
            "features:event:{pair}":  "event",
            "features:flow:{pair}":   "flow",
            "features:liq:{pair}":    "liq",
            "features:yield:{pair}":  "yield",
        }

        all_feats: Dict[str, float] = {}
        found_any = False

        for key_template, label in module_keys.items():
            key = key_template.replace("{pair}", pair)
            try:
                val = self.r.get(key)
                if val:
                    feats = json.loads(val)
                    if isinstance(feats, dict):
                        all_feats.update(feats)
                        found_any = True
            except Exception as exc:
                logger.debug(f"Regime | {pair} {label} Redis read: {exc}")

        # Also check analyst2 for composite score
        try:
            a2 = self.r.get(f"features:analyst2:{pair}")
            if a2:
                a2_data = json.loads(a2)
                drivers = a2_data.get("drivers", {})
                all_feats["analyst2_score"]      = a2_data.get("score", 0.0)
                all_feats["analyst2_yield"]       = drivers.get("yield_score", 0.0)
                all_feats["analyst2_cot"]         = drivers.get("cot_score", 0.0)
                all_feats["analyst2_carry"]       = drivers.get("carry_score", 0.0)
                all_feats["analyst2_commodity"]   = drivers.get("commodity_score", 0.0)
        except Exception:
            pass

        return all_feats if found_any else None

    # ── Model persistence ─────────────────────────────────────

    def _load_saved_models(self) -> None:
        """Load any previously saved LightGBM models from disk."""
        if not LGB_AVAILABLE:
            return
        loaded = 0
        for pair in PAIRS:
            model_path = MODEL_DIR / f"regime_{pair}.lgb"
            if model_path.exists():
                try:
                    self._models[pair] = lgb.Booster(model_file=str(model_path))
                    loaded += 1
                except Exception as exc:
                    logger.warning(f"Regime | Could not load {pair} model: {exc}")
        if loaded:
            logger.info(f"Regime | Loaded {loaded} saved models from {MODEL_DIR}")

    # ── Display ───────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print formatted summary of all pair regimes."""
        print("\n" + "=" * 65)
        print("ANALYST 1 — REGIME SUMMARY")
        print("=" * 65)
        print(f"{'PAIR':<10} {'REGIME':<6} {'CONF':>6}  "
              f"{'THV':>6} {'TLV':>6} {'MR':>6} {'PNS':>6}  {'SRC':<6}")
        print("-" * 65)

        for pair in PAIRS:
            r = self.get_regime(pair)
            p = r.get("probabilities", {})
            src = "model" if r.get("model_available") else "rule"
            print(
                f"{pair:<10} {r['regime']:<6} {r['confidence']:>6.2f}  "
                f"{p.get('THV',0):>6.2f} {p.get('TLV',0):>6.2f} "
                f"{p.get('MR',0):>6.2f} {p.get('PNS',0):>6.2f}  {src:<6}"
            )

        print("=" * 65)
        print("Gate: PNS=FLAT | MR=fade+half-size | THV/TLV=trend+full-size")
        print("=" * 65 + "\n")


# ── Module singleton ──────────────────────────────────────────
regime_classifier = RegimeClassifier()


# ── Direct run ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")

    parser = argparse.ArgumentParser(description="Regime Classifier")
    parser.add_argument("--train",   action="store_true", help="Train all models")
    parser.add_argument("--infer",   action="store_true", help="Run inference only")
    parser.add_argument("--summary", action="store_true", help="Print summary table")
    args = parser.parse_args()

    if args.train:
        regime_classifier.train_all()

    if args.train or args.infer:
        regime_classifier.compute_all()

    regime_classifier.print_summary()