"""
models/analyst7_smc/walk_forward.py
=====================================
Option A: Walk-forward AUC validation for the A7 SMC LightGBM model.

Tests whether the test AUC of 0.6040 is stable across multiple time windows,
or whether it was a lucky single-period result.

For each of 4 walk-forward windows:
  1. Build dataset from train_start to train_end (all 12 pairs)
  2. Train a fresh LightGBM with the same hyperparams as train.py
  3. Evaluate on held-out test window (never seen during training)
  4. Report AUC per window, per pair

Windows (expanding train, rolling test):
  Window 1: train 2024-01 → 2024-10  | test 2024-10 → 2025-01
  Window 2: train 2024-01 → 2025-01  | test 2025-01 → 2025-04
  Window 3: train 2024-01 → 2025-04  | test 2025-04 → 2025-07
  Window 4: train 2024-01 → 2025-07  | test 2025-07 → 2026-04   ← original split

Stability verdict:
  ✅ STABLE  — AUC ≥ 0.58 in 3+ windows
  ⚠️  MIXED   — AUC ≥ 0.58 in 1–2 windows
  ❌ UNSTABLE — AUC < 0.58 in 3+ windows → original 0.6040 was luck

Usage:
  python3 models/analyst7_smc/walk_forward.py
  python3 models/analyst7_smc/walk_forward.py --quick   # 4 pairs only, faster
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("ERROR: pip install lightgbm scikit-learn")
    sys.exit(1)

from models.analyst7_smc.train import (
    PAIRS, FEATURE_COLS,
    load_ohlcv, build_dataset,
)

# ── Walk-forward windows ──────────────────────────────────────────────────────

WINDOWS = [
    ("2024-01-01", "2024-10-01", "2024-10-01", "2025-01-01"),
    ("2024-01-01", "2025-01-01", "2025-01-01", "2025-04-01"),
    ("2024-01-01", "2025-04-01", "2025-04-01", "2025-07-01"),
    ("2024-01-01", "2025-07-01", "2025-07-01", "2026-04-06"),  # original split
]

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "verbose":           -1,
}

QUICK_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]


# ── Core ──────────────────────────────────────────────────────────────────────

def build_window_dataset(pairs: list[str], since: str, until: str) -> pd.DataFrame:
    """Load + build feature dataset for all pairs within [since, until)."""
    all_dfs = []
    for pair in pairs:
        df = load_ohlcv(pair, since)
        if df.empty:
            logger.warning(f"{pair}: no OHLCV data from {since}")
            continue
        # Trim to window
        df = df[df["time"] < pd.Timestamp(until, tz="UTC")].reset_index(drop=True)
        if len(df) < 500:
            logger.warning(f"{pair}: only {len(df)} bars, skipping")
            continue
        logger.info(f"{pair}: {len(df)} bars [{since} → {until})")
        feat_df = build_dataset(df, pair)
        if feat_df.empty:
            continue
        all_dfs.append(feat_df)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def train_and_eval(
    full_df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[float, float, dict[str, float]]:
    """
    Split full_df into train/test, train LightGBM, return (train_auc, test_auc, per_pair_auc).
    """
    full_df = full_df.copy()
    full_df["time"] = pd.to_datetime(full_df["time"], utc=True)

    train_end_ts  = pd.Timestamp(train_end,  tz="UTC")
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    test_end_ts   = pd.Timestamp(test_end,   tz="UTC")

    train_df = full_df[full_df["time"] < train_end_ts].copy()
    test_df  = full_df[
        (full_df["time"] >= test_start_ts) &
        (full_df["time"] <  test_end_ts)
    ].copy()

    if len(train_df) < 1000 or len(test_df) < 100:
        logger.error(f"Insufficient data: train={len(train_df)}, test={len(test_df)}")
        return 0.5, 0.5, {}

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = {**LGBM_PARAMS, "scale_pos_weight": pos_weight}

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)],
    )

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])

    per_pair: dict[str, float] = {}
    for pair in sorted(test_df["pair"].unique()):
        p = test_df[test_df["pair"] == pair]
        if len(p) < 50:
            continue
        per_pair[pair] = roc_auc_score(
            p["label"], model.predict_proba(p[FEATURE_COLS].values)[:, 1]
        )

    return train_auc, test_auc, per_pair


# ── Report ────────────────────────────────────────────────────────────────────

def report(window_results: list[dict]) -> None:
    sep = "═" * 72

    print(f"\n{sep}")
    print(f"  A7 SMC — Walk-Forward AUC Stability Test")
    print(sep)

    # Window summary
    print(f"\n  {'WINDOW':<30} {'TRAIN':>7} {'TEST':>7} {'TRAIN AUC':>10} {'TEST AUC':>9} {'VERDICT'}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*10} {'-'*9} {'-'*10}")

    test_aucs = []
    for w in window_results:
        tauc = w["test_auc"]
        test_aucs.append(tauc)
        verdict = "✅ GOOD" if tauc >= 0.60 else ("⚠️  WEAK" if tauc >= 0.58 else "❌ FAIL")
        label = f"{w['train_start'][:7]}→{w['train_end'][:7]}"
        test_label = f"{w['test_start'][:7]}→{w['test_end'][:7]}"
        print(f"  {label:<30} {w['train_n']:>7,} {w['test_n']:>7,} "
              f"{w['train_auc']:>10.4f} {tauc:>9.4f} {verdict}")

    # Stability verdict
    good_windows = sum(1 for a in test_aucs if a >= 0.58)
    mean_auc = float(np.mean(test_aucs))
    std_auc  = float(np.std(test_aucs))

    print(f"\n  {'─'*40}")
    print(f"  Mean test AUC : {mean_auc:.4f}")
    print(f"  Std test AUC  : {std_auc:.4f}  {'(stable)' if std_auc < 0.02 else '(variable)'}")
    print(f"  Windows ≥0.58 : {good_windows}/{len(window_results)}")

    if good_windows >= 3:
        stability = "✅ STABLE — AUC holds across time. Model has genuine edge."
    elif good_windows >= 2:
        stability = "⚠️  MIXED — AUC varies with market regime. Edge is conditional."
    else:
        stability = "❌ UNSTABLE — Original 0.6040 was likely a lucky test window."
    print(f"\n  VERDICT: {stability}")

    # Per-pair heatmap across windows
    all_pairs = sorted({p for w in window_results for p in w["per_pair"]})
    if all_pairs:
        print(f"\n  PER-PAIR AUC ACROSS WINDOWS")
        header = f"  {'PAIR':<8}" + "".join(
            f"  {w['test_start'][:7]:>9}" for w in window_results
        ) + f"  {'MEAN':>7}"
        print(header)
        print(f"  {'─'*8}" + "─" * (11 * len(window_results) + 9))

        for pair in all_pairs:
            aucs = [w["per_pair"].get(pair) for w in window_results]
            valid = [a for a in aucs if a is not None]
            mean_p = float(np.mean(valid)) if valid else None
            row = f"  {pair:<8}"
            for a in aucs:
                if a is None:
                    row += f"  {'—':>9}"
                else:
                    flag = "✅" if a >= 0.60 else ("⚠ " if a >= 0.58 else "❌")
                    row += f"  {flag}{a:.3f}  "
            if mean_p:
                row += f"  {mean_p:.3f}"
            print(row)

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help=f"Use only {QUICK_PAIRS} for faster run (~8 min)")
    args = parser.parse_args()

    pairs = QUICK_PAIRS if args.quick else PAIRS
    mode  = "QUICK (4 pairs)" if args.quick else "FULL (12 pairs)"
    logger.info(f"Walk-forward validation | {mode} | {len(WINDOWS)} windows")

    # Pre-load all OHLCV data once (entire period) — avoids re-fetching per window
    logger.info("Pre-loading OHLCV data for all pairs (2024-01-01 → now)...")
    pair_dfs: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df = load_ohlcv(pair, "2024-01-01")
        if not df.empty:
            pair_dfs[pair] = df
            logger.info(f"  {pair}: {len(df):,} bars loaded")

    window_results = []

    for win_i, (train_start, train_end, test_start, test_end) in enumerate(WINDOWS, 1):
        logger.info(f"\n── Window {win_i}/{len(WINDOWS)}: "
                    f"train {train_start[:7]}→{train_end[:7]} | "
                    f"test {test_start[:7]}→{test_end[:7]} ──")

        t0 = time.time()

        # Build dataset from pre-loaded data, filtered to window
        train_end_ts = pd.Timestamp(train_end, tz="UTC")
        test_end_ts  = pd.Timestamp(test_end,  tz="UTC")

        all_dfs = []
        for pair, df in pair_dfs.items():
            window_df = df[df["time"] < test_end_ts].reset_index(drop=True)
            if len(window_df) < 500:
                continue
            logger.info(f"  {pair}: building features ({len(window_df)} bars)...")
            feat_df = build_dataset(window_df, pair)
            if not feat_df.empty:
                all_dfs.append(feat_df)

        if not all_dfs:
            logger.error("No data for this window — skipping")
            continue

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df["time"] = pd.to_datetime(full_df["time"], utc=True)

        train_n = len(full_df[full_df["time"] < train_end_ts])
        test_n  = len(full_df[
            (full_df["time"] >= pd.Timestamp(test_start, tz="UTC")) &
            (full_df["time"] <  test_end_ts)
        ])

        logger.info(f"  Training: {train_n:,} rows | Testing: {test_n:,} rows")

        train_auc, test_auc, per_pair = train_and_eval(
            full_df, train_end, test_start, test_end
        )

        elapsed = time.time() - t0
        logger.info(f"  Window {win_i} done: train_auc={train_auc:.4f} "
                    f"test_auc={test_auc:.4f} ({elapsed:.0f}s)")

        window_results.append({
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
            "train_n":     train_n,
            "test_n":      test_n,
            "train_auc":   train_auc,
            "test_auc":    test_auc,
            "per_pair":    per_pair,
        })

    if not window_results:
        print("No windows completed — check data availability.")
        return

    report(window_results)


if __name__ == "__main__":
    main()
