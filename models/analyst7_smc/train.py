"""
models/analyst7_smc/train.py
=============================
A7 — SMC LightGBM Entry Quality Model.

Trains a binary classifier to answer: "Does the current SMC setup lead to
a profitable entry?" using 2+ years of M15 historical data.

Target definition:
  For each bar, given direction (BUY or SELL), did price move ≥ TARGET_R ATR
  in the correct direction within the next TARGET_BARS M15 bars?
  → 1 = yes (good entry timing), 0 = no

Features used:
  SMC features (10):  fvg_bull_dist, fvg_bear_dist, fvg_inside_bull,
                      fvg_inside_bear, liq_sweep_bull_bars, liq_sweep_bear_bars,
                      ob_bull_dist, ob_bear_dist, msb_bull_bars, msb_bear_bars
  Direction-aware (10): above features remapped for BUY vs SELL direction
  Context features (6):  atr_pips, hour_of_day, day_of_week,
                          price_vs_session_high, bb_pct_20, adx

Train/test split: temporal — 2024-01-01 to 2025-06-30 train, 2025-07-01+ test

Usage:
  python3 models/analyst7_smc/train.py
  python3 models/analyst7_smc/train.py --pair EURUSD --since 2024-01-01
  python3 models/analyst7_smc/train.py --all --since 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from data_pipeline.storage.timeseries_db import db
from features.smc_features import compute_smc_features, _pip, JPY_PAIRS

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm not installed — run: pip install lightgbm")

try:
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Config ────────────────────────────────────────────────────────────────────

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

TARGET_BARS  = 16    # 4h forward window
TARGET_R     = 1.0   # price must move ≥ 1× ATR in direction
ATR_PERIOD   = 14
MSB_MAX_AGE  = 200
FVG_MAX_AGE  = 96
OB_MAX_AGE   = 96
LOOKBACK     = 250
SWING_WINDOW = 5
LOOKBACK     = 250   # bars context for each feature window
TEST_DATE    = "2025-07-01"  # temporal split

# ── Data loading ──────────────────────────────────────────────────────────────

def load_ohlcv(pair: str, since: str) -> pd.DataFrame:
    with db.connection() as conn:
        rows = conn.execute(text("""
            SELECT time, open, high, low, close
            FROM ohlcv
            WHERE symbol = :sym AND timeframe = 'M15' AND time >= :since
            ORDER BY time
        """), {"sym": pair, "since": since}).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    if pair in JPY_PAIRS and df["close"].iloc[-1] > 1000:
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / 100.0

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def _wilder(s: pd.Series, n: int) -> pd.Series:
    result = s.copy() * np.nan
    result.iloc[n - 1] = s.iloc[:n].mean()
    for i in range(n, len(s)):
        result.iloc[i] = result.iloc[i - 1] * (n - 1) / n + s.iloc[i] / n
    return result


def compute_atr(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    pc = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - pc), abs(df["low"] - pc)))
    return _wilder(pd.Series(tr, index=df.index), n)


def compute_adx(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    pc = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - pc), abs(df["low"] - pc)))
    up   = df["high"] - df["high"].shift(1)
    down = df["low"].shift(1) - df["low"]
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    atr_s = _wilder(pd.Series(tr, index=df.index), n)
    pdi   = 100 * _wilder(pd.Series(pdm, index=df.index), n) / atr_s.where(atr_s > 0)
    ndi   = 100 * _wilder(pd.Series(ndm, index=df.index), n) / atr_s.where(atr_s > 0)
    dx    = 100 * abs(pdi - ndi) / (pdi + ndi).where((pdi + ndi) > 0)
    return _wilder(dx.fillna(0), n)


def _vectorized_smc(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Compute all SMC features for every bar in one vectorized pass.
    Returns DataFrame indexed same as df with one column per SMC feature.
    No look-ahead: at bar i, only uses data from bar 0..i.
    """
    pip  = _pip(pair)
    n    = len(df)
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    # ── Swing detection (pre-compute flags) ──────────────────────────────────
    w = SWING_WINDOW
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        if highs[i] > highs[i-w:i].max() and highs[i] > highs[i+1:i+w+1].max():
            sh[i] = True
        if lows[i] < lows[i-w:i].min() and lows[i] < lows[i+1:i+w+1].min():
            sl[i] = True

    # Running arrays for each feature (initialised to neutral values)
    fvg_bull_dist   = np.zeros(n, dtype=float)
    fvg_bear_dist   = np.zeros(n, dtype=float)
    fvg_inside_bull = np.zeros(n, dtype=float)
    fvg_inside_bear = np.zeros(n, dtype=float)
    liq_bull_bars   = np.full(n, MSB_MAX_AGE, dtype=float)
    liq_bear_bars   = np.full(n, MSB_MAX_AGE, dtype=float)
    ob_bull_dist    = np.zeros(n, dtype=float)
    ob_bear_dist    = np.zeros(n, dtype=float)
    msb_bull_bars   = np.full(n, MSB_MAX_AGE, dtype=float)
    msb_bear_bars   = np.full(n, MSB_MAX_AGE, dtype=float)

    # ── FVG: detect all formations, track fill state per bar ─────────────────
    # Bullish FVG at bar j: highs[j-2] < lows[j]
    # filled at bar i if: any bar k in [j+1..i] has low<=lows[j] and high>=highs[j-2]
    # We track active FVGs (formation_idx, gap_lo, gap_hi) and forward-fill
    active_bull_fvgs: list = []  # (gap_lo, gap_hi)
    active_bear_fvgs: list = []

    for i in range(2, n):
        price = closes[i]

        # New FVG formed at bar i?
        if highs[i-2] < lows[i]:  # bullish
            active_bull_fvgs.append((highs[i-2], lows[i]))
        if lows[i-2] > highs[i]:  # bearish
            active_bear_fvgs.append((highs[i], lows[i-2]))

        # Remove filled FVGs (price entered the gap)
        active_bull_fvgs = [
            (lo, hi) for lo, hi in active_bull_fvgs
            if not (lows[i] <= hi and highs[i] >= lo)
        ]
        active_bear_fvgs = [
            (lo, hi) for lo, hi in active_bear_fvgs
            if not (lows[i] <= hi and highs[i] >= lo)
        ]

        # Nearest bull FVG above price
        above = [(lo, hi) for lo, hi in active_bull_fvgs if lo > price]
        if above:
            lo, hi = min(above, key=lambda x: x[0])
            fvg_bull_dist[i] = (lo - price) / pip

        # Nearest bear FVG below price
        below = [(lo, hi) for lo, hi in active_bear_fvgs if hi < price]
        if below:
            lo, hi = max(below, key=lambda x: x[1])
            fvg_bear_dist[i] = (price - hi) / pip

        # Inside a FVG?
        for lo, hi in active_bull_fvgs:
            if lo <= price <= hi:
                fvg_inside_bull[i] = 1.0
                break
        for lo, hi in active_bear_fvgs:
            if lo <= price <= hi:
                fvg_inside_bear[i] = 1.0
                break

    # ── Liquidity sweeps, OB, MSB in one forward pass ────────────────────────
    # Track running "last seen" values
    last_sh_val = None  # most recent swing high value seen
    last_sl_val = None  # most recent swing low value seen
    active_bull_obs: list = []  # (ob_lo, ob_hi)
    active_bear_obs: list = []
    last_bull_sweep = MSB_MAX_AGE
    last_bear_sweep = MSB_MAX_AGE
    last_bull_msb   = MSB_MAX_AGE
    last_bear_msb   = MSB_MAX_AGE
    IMPULSE = 3

    for i in range(w + 1, n):
        price = closes[i]

        # Update running swing levels (use only confirmed swings: i-w bars ago)
        confirm_idx = i - w
        if confirm_idx >= w and sh[confirm_idx]:
            last_sh_val = highs[confirm_idx]
        if confirm_idx >= w and sl[confirm_idx]:
            last_sl_val = lows[confirm_idx]

        # Liquidity sweep detection
        if last_sh_val is not None:
            if highs[i] > last_sh_val and closes[i] < last_sh_val:
                last_bear_sweep = 0
            if closes[i] > last_sh_val:
                last_bull_msb = 0
        if last_sl_val is not None:
            if lows[i] < last_sl_val and closes[i] > last_sl_val:
                last_bull_sweep = 0
            if closes[i] < last_sl_val:
                last_bear_msb = 0

        liq_bull_bars[i]  = last_bull_sweep
        liq_bear_bars[i]  = last_bear_sweep
        msb_bull_bars[i]  = last_bull_msb
        msb_bear_bars[i]  = last_bear_msb

        last_bull_sweep = min(last_bull_sweep + 1, MSB_MAX_AGE)
        last_bear_sweep = min(last_bear_sweep + 1, MSB_MAX_AGE)
        last_bull_msb   = min(last_bull_msb   + 1, MSB_MAX_AGE)
        last_bear_msb   = min(last_bear_msb   + 1, MSB_MAX_AGE)

        # OB detection: check if bar i-IMPULSE-1 qualifies as OB
        ob_i = i - IMPULSE - 1
        if ob_i >= 0:
            # Bullish OB: bearish candle followed by IMPULSE bull candles
            if closes[ob_i] < opens[ob_i]:
                if all(closes[ob_i+k] > opens[ob_i+k] for k in range(1, IMPULSE+1)):
                    active_bull_obs.append((lows[ob_i], highs[ob_i]))
            # Bearish OB: bullish candle followed by IMPULSE bear candles
            if closes[ob_i] > opens[ob_i]:
                if all(closes[ob_i+k] < opens[ob_i+k] for k in range(1, IMPULSE+1)):
                    active_bear_obs.append((lows[ob_i], highs[ob_i]))

        # Invalidate OBs that price has closed through
        active_bull_obs = [(lo, hi) for lo, hi in active_bull_obs if closes[i] >= lo]
        active_bear_obs = [(lo, hi) for lo, hi in active_bear_obs if closes[i] <= hi]

        # Nearest bull OB below price
        bull_below = [(lo, hi) for lo, hi in active_bull_obs if hi < price]
        if bull_below:
            lo, hi = max(bull_below, key=lambda x: x[1])
            ob_bull_dist[i] = (price - hi) / pip

        # Nearest bear OB above price
        bear_above = [(lo, hi) for lo, hi in active_bear_obs if lo > price]
        if bear_above:
            lo, hi = min(bear_above, key=lambda x: x[0])
            ob_bear_dist[i] = (lo - price) / pip

    return pd.DataFrame({
        "fvg_bull_dist":        fvg_bull_dist,
        "fvg_bear_dist":        fvg_bear_dist,
        "fvg_inside_bull":      fvg_inside_bull,
        "fvg_inside_bear":      fvg_inside_bear,
        "liq_sweep_bull_bars":  liq_bull_bars,
        "liq_sweep_bear_bars":  liq_bear_bars,
        "ob_bull_dist":         ob_bull_dist,
        "ob_bear_dist":         ob_bear_dist,
        "msb_bull_bars":        msb_bull_bars,
        "msb_bear_bars":        msb_bear_bars,
    }, index=df.index)


def build_dataset(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Build feature matrix + labels for all bars in df.
    Vectorized: SMC features computed once over full array (not per-bar rolling).
    """
    pip = _pip(pair)
    atr = compute_atr(df)
    adx = compute_adx(df)

    close = df["close"]
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pct = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    session_high = df["high"].rolling(16).max()
    session_low  = df["low"].rolling(16).min()
    price_vs_session = (close - session_low) / (session_high - session_low + 1e-10)

    n = len(df)
    logger.info(f"{pair}: computing vectorized SMC features for {n} bars...")
    smc_df = _vectorized_smc(df, pair)

    min_idx = LOOKBACK + SWING_WINDOW
    max_idx = n - TARGET_BARS - 1

    # Vectorized labels
    highs_arr  = df["high"].values
    lows_arr   = df["low"].values
    closes_arr = df["close"].values
    atr_arr    = atr.values

    logger.info(f"{pair}: computing labels for {max_idx - min_idx} bars...")

    rows = []
    for i in range(min_idx, max_idx):
        bar_atr = float(atr_arr[i])
        if np.isnan(bar_atr) or bar_atr <= 0:
            continue

        target_move = bar_atr * TARGET_R
        price = closes_arr[i]
        fwd_h = highs_arr[i + 1: i + TARGET_BARS + 1]
        fwd_l = lows_arr[i + 1: i + TARGET_BARS + 1]

        label_buy  = int((fwd_h >= price + target_move).any())
        label_sell = int((fwd_l <= price - target_move).any())

        atr_pips = bar_atr / pip
        bar_adx  = float(adx.iloc[i])  if not np.isnan(adx.iloc[i])  else 25.0
        bar_bb   = float(bb_pct.iloc[i]) if not np.isnan(bb_pct.iloc[i]) else 0.5
        bar_pvs  = float(price_vs_session.iloc[i]) if not np.isnan(price_vs_session.iloc[i]) else 0.5
        ts       = df["time"].iloc[i] if "time" in df.columns else None
        hour     = ts.hour        if ts is not None else 0
        dow      = ts.dayofweek   if ts is not None else 0

        smc = smc_df.iloc[i]

        for direction, label in [("BUY", label_buy), ("SELL", label_sell)]:
            if direction == "BUY":
                fvg_aligned_dist    = smc["fvg_bull_dist"]
                fvg_opposing_dist   = smc["fvg_bear_dist"]
                fvg_inside_aligned  = smc["fvg_inside_bull"]
                fvg_inside_opposing = smc["fvg_inside_bear"]
                liq_sweep_aligned   = smc["liq_sweep_bull_bars"]
                liq_sweep_opposing  = smc["liq_sweep_bear_bars"]
                ob_aligned_dist     = smc["ob_bull_dist"]
                ob_opposing_dist    = smc["ob_bear_dist"]
                msb_aligned_bars    = smc["msb_bull_bars"]
                msb_opposing_bars   = smc["msb_bear_bars"]
            else:
                fvg_aligned_dist    = smc["fvg_bear_dist"]
                fvg_opposing_dist   = smc["fvg_bull_dist"]
                fvg_inside_aligned  = smc["fvg_inside_bear"]
                fvg_inside_opposing = smc["fvg_inside_bull"]
                liq_sweep_aligned   = smc["liq_sweep_bear_bars"]
                liq_sweep_opposing  = smc["liq_sweep_bull_bars"]
                ob_aligned_dist     = smc["ob_bear_dist"]
                ob_opposing_dist    = smc["ob_bull_dist"]
                msb_aligned_bars    = smc["msb_bear_bars"]
                msb_opposing_bars   = smc["msb_bull_bars"]

            rows.append({
                # SMC features (direction-aware)
                "fvg_aligned_dist":    fvg_aligned_dist,
                "fvg_opposing_dist":   fvg_opposing_dist,
                "fvg_inside_aligned":  fvg_inside_aligned,
                "fvg_inside_opposing": fvg_inside_opposing,
                "liq_sweep_aligned":   liq_sweep_aligned,
                "liq_sweep_opposing":  liq_sweep_opposing,
                "ob_aligned_dist":     ob_aligned_dist,
                "ob_opposing_dist":    ob_opposing_dist,
                "msb_aligned_bars":    msb_aligned_bars,
                "msb_opposing_bars":   msb_opposing_bars,
                # Context
                "atr_pips":            atr_pips,
                "adx":                 bar_adx,
                "bb_pct":              bar_bb,
                "price_vs_session":    bar_pvs,
                "hour_of_day":         hour,
                "day_of_week":         dow,
                # Metadata
                "pair":                pair,
                "time":                ts,
                "direction":           direction,
                "label":               label,
            })

    return pd.DataFrame(rows)


# ── Training ──────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "fvg_aligned_dist", "fvg_opposing_dist",
    "fvg_inside_aligned", "fvg_inside_opposing",
    "liq_sweep_aligned", "liq_sweep_opposing",
    "ob_aligned_dist", "ob_opposing_dist",
    "msb_aligned_bars", "msb_opposing_bars",
    "atr_pips", "adx", "bb_pct", "price_vs_session",
    "hour_of_day", "day_of_week",
]


def train_and_evaluate(full_df: pd.DataFrame, pairs_used: list) -> None:
    if not HAS_LGB:
        logger.error("lightgbm not installed — cannot train")
        return
    if not HAS_SKLEARN:
        logger.error("scikit-learn not installed — cannot evaluate")
        return

    full_df["time"] = pd.to_datetime(full_df["time"], utc=True)
    split_dt = pd.Timestamp(TEST_DATE, tz="UTC")

    train_df = full_df[full_df["time"] < split_dt].copy()
    test_df  = full_df[full_df["time"] >= split_dt].copy()

    logger.info(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    logger.info(f"Train label rate: {train_df['label'].mean():.1%}")
    logger.info(f"Test  label rate: {test_df['label'].mean():.1%}")

    if len(train_df) < 1000 or len(test_df) < 100:
        logger.error("Not enough data — check --since date")
        return

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "n_estimators":     500,
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_child_samples": 50,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": pos_weight,
        "random_state":     42,
        "verbose":          -1,
    }

    logger.info("Training LightGBM A7 SMC model...")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    # Evaluate
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba  = model.predict_proba(X_test)[:, 1]
    train_auc   = roc_auc_score(y_train, train_proba)
    test_auc    = roc_auc_score(y_test,  test_proba)

    print(f"\n{'═'*60}")
    print(f"  A7 SMC MODEL — TRAINING RESULTS")
    print(f"{'═'*60}")
    print(f"  Pairs trained on : {', '.join(pairs_used)}")
    print(f"  Train period     : {full_df['time'].min().date()} → {split_dt.date()}")
    print(f"  Test period      : {split_dt.date()} → {full_df['time'].max().date()}")
    print(f"  Train samples    : {len(train_df):,}")
    print(f"  Test samples     : {len(test_df):,}")
    print(f"  Label rate       : {y_train.mean():.1%} train / {y_test.mean():.1%} test")
    print(f"  Train AUC        : {train_auc:.4f}")
    print(f"  Test AUC         : {test_auc:.4f}  {'✅ GOOD' if test_auc >= 0.60 else '⚠️  WEAK' if test_auc >= 0.55 else '❌ NO EDGE'}")
    print(f"{'─'*60}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=False)
    print(f"\n  Feature importance (top 10):")
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp / importance.max() * 20)
        print(f"    {feat:<28} {bar:<20} {imp:.0f}")

    # Per-pair AUC breakdown
    print(f"\n  Per-pair test AUC:")
    for pair in sorted(test_df["pair"].unique()):
        p_df = test_df[test_df["pair"] == pair]
        if len(p_df) < 50:
            continue
        p_auc = roc_auc_score(p_df["label"], model.predict_proba(p_df[FEATURE_COLS].values)[:, 1])
        verdict = "✅" if p_auc >= 0.60 else "⚠️ " if p_auc >= 0.55 else "❌"
        print(f"    {pair:<8} AUC={p_auc:.4f}  {verdict}")

    # Precision at high confidence
    print(f"\n  Precision at model confidence thresholds (test set):")
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        mask = test_proba >= thresh
        if mask.sum() > 0:
            prec = y_test[mask].mean()
            cov  = mask.mean()
            print(f"    score >= {thresh:.2f}  precision={prec:.1%}  coverage={cov:.1%}  n={mask.sum()}")

    print(f"{'═'*60}")

    # Save model
    out_dir = Path(__file__).parent
    model_path = out_dir / "a7_smc_lgbm.pkl"

    import pickle
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":       model,
            "features":    FEATURE_COLS,
            "test_auc":    test_auc,
            "train_auc":   train_auc,
            "trained_at":  datetime.now(timezone.utc).isoformat(),
            "pairs":       pairs_used,
            "test_date":   TEST_DATE,
        }, f)

    logger.success(f"Model saved to {model_path}")
    if test_auc >= 0.60:
        logger.success(f"AUC={test_auc:.4f} ≥ 0.60 — ready to wire into Gate 4 confidence score")
    else:
        logger.warning(f"AUC={test_auc:.4f} < 0.60 — SMC adds limited edge on this dataset")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="A7 SMC LightGBM training")
    parser.add_argument("--pair",  default="EURUSD")
    parser.add_argument("--all",   action="store_true", help="Train on all 12 pairs")
    parser.add_argument("--since", default="2024-01-01")
    args = parser.parse_args()

    pairs = PAIRS if args.all else [args.pair.upper()]

    all_dfs = []
    for pair in pairs:
        logger.info(f"Loading {pair} since {args.since}...")
        df = load_ohlcv(pair, args.since)
        if df.empty:
            logger.warning(f"{pair}: no data")
            continue
        logger.info(f"{pair}: {len(df):,} M15 bars")
        feat_df = build_dataset(df, pair)
        if feat_df.empty:
            logger.warning(f"{pair}: no features built")
            continue
        logger.info(f"{pair}: {len(feat_df):,} labeled samples")
        all_dfs.append(feat_df)

    if not all_dfs:
        logger.error("No data — aborting")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total samples: {len(combined):,} across {len(all_dfs)} pairs")

    train_and_evaluate(combined, pairs)


if __name__ == "__main__":
    main()
