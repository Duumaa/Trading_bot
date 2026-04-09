"""
models/analyst7_smc/backtest_a7.py
====================================
Option B: Retroactively score every historical trade with the A7 SMC model
and measure whether high-SMC trades actually outperformed low-SMC trades.

For each closed trade in the DB:
  1. Fetch M15 bars up to entry_time (with LOOKBACK context)
  2. Run _vectorized_smc() on those bars (same logic as training)
  3. Compute context indicators (ATR, ADX, BB%, price_vs_session)
  4. Call a7_scorer.predict() → smc_prob
  5. Classify into HIGH (≥0.55), NEUTRAL, LOW (≤0.35)

Output:
  - Per-tier: count, win rate, avg pnl_pips, avg pnl_usd
  - Per-pair × tier breakdown
  - Distribution of smc_prob across all scored trades
  - Calibration check: precision at each 0.05 threshold bucket

AUC reliability verdict: if HIGH tier meaningfully outperforms LOW tier
on real closed trades, the 0.6040 test AUC is translating to live edge.

Usage:
  python3 models/analyst7_smc/backtest_a7.py
  python3 models/analyst7_smc/backtest_a7.py --since 2026-03-27
  python3 models/analyst7_smc/backtest_a7.py --pair EURUSD
  python3 models/analyst7_smc/backtest_a7.py --verbose   # print each trade
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
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
from models.analyst7_smc.train import (
    load_ohlcv,
    _vectorized_smc,
    compute_atr,
    compute_adx,
)
from models.analyst7_smc.score import a7_scorer
from features.smc_features import JPY_PAIRS, _pip

# ── Config ────────────────────────────────────────────────────────────────────

LOOKBACK_BARS   = 400     # M15 bars before entry to load (100h context)
ATR_PERIOD      = 14
SWING_WINDOW    = 5

TIER_HIGH    = 0.55
TIER_LOW     = 0.35

CLOSED_STATUSES = ("CLOSED_SL", "CLOSED_TP", "CLOSED_MANUAL", "CLOSED_TECH_FLIP")

# ── Data ──────────────────────────────────────────────────────────────────────

def load_trades(since: str | None, pair: str | None) -> pd.DataFrame:
    filters = ["status = ANY(:statuses)"]
    params: dict = {"statuses": list(CLOSED_STATUSES)}
    if since:
        filters.append("entry_time >= :since")
        params["since"] = since
    if pair:
        filters.append("symbol = :pair")
        params["pair"] = pair.upper()

    where = " AND ".join(filters)
    q = f"""
        SELECT id, symbol, direction, status,
               entry_price, exit_price, pnl_pips, pnl_usd,
               entry_time, exit_time, regime, signal_path
        FROM trades
        WHERE {where}
        ORDER BY entry_time
    """
    with db.connection() as conn:
        rows = conn.execute(text(q), params).fetchall()

    cols = ["id", "symbol", "direction", "status",
            "entry_price", "exit_price", "pnl_pips", "pnl_usd",
            "entry_time", "exit_time", "regime", "signal_path"]
    df = pd.DataFrame(rows, columns=cols)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["pnl_pips"]   = pd.to_numeric(df["pnl_pips"], errors="coerce")
    df["pnl_usd"]    = pd.to_numeric(df["pnl_usd"],  errors="coerce")
    return df


def load_bars_before(pair: str, entry_time: datetime, n_bars: int) -> pd.DataFrame:
    """Fetch the last n_bars M15 bars strictly before entry_time."""
    with db.connection() as conn:
        rows = conn.execute(text("""
            SELECT time, open, high, low, close
            FROM ohlcv
            WHERE symbol = :sym AND timeframe = 'M15'
              AND time < :before
            ORDER BY time DESC
            LIMIT :lim
        """), {"sym": pair, "before": entry_time, "lim": n_bars}).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    if pair in JPY_PAIRS and df["close"].iloc[-1] > 1000:
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / 100.0

    return df


# ── Scoring ───────────────────────────────────────────────────────────────────

def _wilder(s: pd.Series, n: int) -> pd.Series:
    result = s.copy() * np.nan
    result.iloc[n - 1] = s.iloc[:n].mean()
    for i in range(n, len(s)):
        result.iloc[i] = result.iloc[i - 1] * (n - 1) / n + s.iloc[i] / n
    return result


def score_trade(pair: str, direction: str, entry_time: datetime) -> float | None:
    """
    Return A7 SMC probability for a given trade entry.
    Returns None if insufficient bars to score.
    """
    bars = load_bars_before(pair, entry_time, LOOKBACK_BARS)
    if len(bars) < SWING_WINDOW * 2 + ATR_PERIOD + 10:
        return None

    pip = _pip(pair)

    # ATR
    pc = bars["close"].shift(1)
    tr = np.maximum(bars["high"] - bars["low"],
         np.maximum(abs(bars["high"] - pc), abs(bars["low"] - pc)))
    atr_s = _wilder(pd.Series(tr.values, index=bars.index), ATR_PERIOD)
    atr_pips = float(atr_s.iloc[-1]) / pip
    if np.isnan(atr_pips) or atr_pips <= 0:
        return None

    # ADX
    up   = bars["high"] - bars["high"].shift(1)
    down = bars["low"].shift(1) - bars["low"]
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    atr_w = _wilder(pd.Series(tr.values, index=bars.index), ATR_PERIOD)
    pdi   = 100 * _wilder(pd.Series(pdm, index=bars.index), ATR_PERIOD) / atr_w.where(atr_w > 0)
    ndi   = 100 * _wilder(pd.Series(ndm, index=bars.index), ATR_PERIOD) / atr_w.where(atr_w > 0)
    dx    = 100 * abs(pdi - ndi) / (pdi + ndi).where((pdi + ndi) > 0)
    adx_s = _wilder(dx.fillna(0), ATR_PERIOD)
    adx_val = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 25.0

    # BB%
    close = bars["close"]
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pct = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)
    bb_val = float(bb_pct.iloc[-1]) if not np.isnan(bb_pct.iloc[-1]) else 0.5

    # Price vs session high/low
    sh = bars["high"].rolling(16).max()
    sl = bars["low"].rolling(16).min()
    pvs = (close - sl) / (sh - sl + 1e-10)
    pvs_val = float(pvs.iloc[-1]) if not np.isnan(pvs.iloc[-1]) else 0.5

    # SMC features (vectorized, same as training)
    smc_df = _vectorized_smc(bars, pair)
    smc = smc_df.iloc[-1].to_dict()

    # hour/dow from entry_time
    hour = entry_time.hour
    dow  = entry_time.weekday()

    prob = a7_scorer.predict(
        direction=direction,
        smc=smc,
        atr_pips=atr_pips,
        adx=adx_val,
        bb_pct=bb_val,
        price_vs_session=pvs_val,
        hour=hour,
        dow=dow,
    )
    return prob


# ── Reporting ─────────────────────────────────────────────────────────────────

def _tier(prob: float) -> str:
    if prob >= TIER_HIGH:
        return "HIGH"
    if prob <= TIER_LOW:
        return "LOW"
    return "NEUTRAL"


def _win(row: pd.Series) -> bool:
    if row["status"] == "CLOSED_TP":
        return True
    if row["status"] in ("CLOSED_MANUAL", "CLOSED_TECH_FLIP"):
        return (row["pnl_pips"] or 0) > 0
    return False


def _pct(n: int, total: int) -> str:
    return f"{n/total*100:.0f}%" if total else "—"


def report(results: pd.DataFrame, verbose: bool) -> None:
    sep = "═" * 68

    if verbose:
        print(f"\n{'Trade-by-trade':}")
        print(f"{'ID':<6} {'PAIR':<8} {'DIR':<5} {'PROB':>5} {'TIER':<8} {'PIPS':>7} {'STATUS'}")
        print("─" * 60)
        for _, r in results.iterrows():
            print(f"{r['id']:<6} {r['symbol']:<8} {r['direction']:<5} "
                  f"{r['smc_prob']:>5.3f} {r['tier']:<8} "
                  f"{r['pnl_pips']:>+7.1f} {r['status']}")

    print(f"\n{sep}")
    print(f"  A7 SMC Backtest — Option B: Retroactive Trade Scoring")
    print(f"  Trades scored: {len(results)}  |  Model loaded: {a7_scorer._loaded}")
    print(sep)

    # ── Tier summary ──────────────────────────────────────────────────────────
    print(f"\n  {'TIER':<10} {'COUNT':>6} {'WIN%':>6} {'AVG PIPS':>10} {'AVG USD':>9} {'TOTAL USD':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*10} {'-'*9} {'-'*10}")

    tier_order = ["HIGH", "NEUTRAL", "LOW"]
    tier_data  = {}
    for tier in tier_order:
        t = results[results["tier"] == tier]
        if len(t) == 0:
            continue
        n    = len(t)
        wins = t["is_win"].sum()
        avg_pips = t["pnl_pips"].mean()
        avg_usd  = t["pnl_usd"].mean()
        tot_usd  = t["pnl_usd"].sum()
        tier_data[tier] = {"n": n, "wins": wins, "avg_pips": avg_pips, "avg_usd": avg_usd, "tot_usd": tot_usd}

        wr  = _pct(wins, n)
        tag = "  ← model predicts GOOD" if tier == "HIGH" else ("  ← model predicts BAD" if tier == "LOW" else "")
        print(f"  {tier:<10} {n:>6} {wr:>6} {avg_pips:>+10.1f} {avg_usd:>+9.2f} {tot_usd:>+10.2f}{tag}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  {'VERDICT':}")
    high = tier_data.get("HIGH")
    low  = tier_data.get("LOW")
    if high and low:
        pips_delta = high["avg_pips"] - low["avg_pips"]
        wr_high = high["wins"] / high["n"] * 100 if high["n"] else 0
        wr_low  = low["wins"]  / low["n"]  * 100 if low["n"] else 0
        wr_delta = wr_high - wr_low

        if pips_delta > 3 and wr_delta > 5:
            verdict = "✅ STRONG — HIGH tier outperforms LOW on both WR and pips. AUC is translating."
        elif pips_delta > 0 and wr_delta > 0:
            verdict = "⚠️  WEAK — HIGH tier slightly better but edge is small. More data needed."
        elif pips_delta < 0 or wr_delta < 0:
            verdict = "❌ REVERSED — LOW tier outperforms HIGH. Model may be inverted or overfit."
        else:
            verdict = "➖ FLAT — No meaningful difference between tiers."

        print(f"  WR delta  (HIGH−LOW): {wr_delta:+.1f}pp")
        print(f"  Pip delta (HIGH−LOW): {pips_delta:+.1f} pips")
        print(f"  {verdict}")
    else:
        print("  Insufficient data in one or both extreme tiers.")

    # ── Per-pair breakdown ────────────────────────────────────────────────────
    print(f"\n  PER-PAIR BREAKDOWN")
    print(f"  {'PAIR':<8} {'N':>4} {'HIGH_WR':>8} {'NEUT_WR':>8} {'LOW_WR':>8} {'HIGH_PIPS':>10} {'LOW_PIPS':>9}")
    print(f"  {'-'*8} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*9}")

    for pair in sorted(results["symbol"].unique()):
        pt = results[results["symbol"] == pair]
        row_parts = [f"  {pair:<8} {len(pt):>4}"]
        for tier in ["HIGH", "NEUTRAL", "LOW"]:
            t = pt[pt["tier"] == tier]
            if len(t) == 0:
                row_parts.append(f"{'—':>8}")
            else:
                row_parts.append(f"{_pct(t['is_win'].sum(), len(t)):>8}")
        for tier in ["HIGH", "LOW"]:
            t = pt[pt["tier"] == tier]
            if len(t) == 0:
                row_parts.append(f"{'—':>9}" if tier == "LOW" else f"{'—':>10}")
            else:
                w = int(tier == "LOW")
                row_parts.append(f"{t['pnl_pips'].mean():>+{9+w}.1f}")
        print("".join(row_parts))

    # ── Calibration: probability distribution ────────────────────────────────
    print(f"\n  PROBABILITY DISTRIBUTION (bucket = 0.05 wide)")
    print(f"  {'BUCKET':<14} {'N':>5} {'WIN%':>6} {'AVG PIPS':>10}")
    print(f"  {'-'*14} {'-'*5} {'-'*6} {'-'*10}")

    buckets = np.arange(0.30, 0.76, 0.05)
    for lo in buckets:
        hi   = lo + 0.05
        mask = (results["smc_prob"] >= lo) & (results["smc_prob"] < hi)
        t    = results[mask]
        if len(t) == 0:
            continue
        wr = _pct(t["is_win"].sum(), len(t))
        print(f"  [{lo:.2f}–{hi:.2f})  {len(t):>5} {wr:>6} {t['pnl_pips'].mean():>+10.1f}")

    # ── Skipped trades ────────────────────────────────────────────────────────
    skip_n = results["smc_prob"].isna().sum()
    if skip_n:
        print(f"\n  Skipped (insufficient bars): {skip_n} trades")

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=None, help="Filter trades after date (YYYY-MM-DD)")
    parser.add_argument("--pair",  default=None, help="Filter to single pair")
    parser.add_argument("--verbose", action="store_true", help="Print each trade row")
    args = parser.parse_args()

    if not a7_scorer._loaded:
        logger.error("A7 model not loaded — run train.py first")
        sys.exit(1)

    logger.info("Loading trades from DB...")
    trades = load_trades(args.since, args.pair)
    logger.info(f"Loaded {len(trades)} trades to score")

    if trades.empty:
        print("No trades found matching filters.")
        return

    # Score each trade
    scored_rows = []
    for i, row in trades.iterrows():
        pair      = row["symbol"]
        direction = row["direction"]  # BUY or SELL
        entry_t   = row["entry_time"]

        try:
            prob = score_trade(pair, direction, entry_t)
        except Exception as exc:
            logger.debug(f"Trade {row['id']} | {pair} {direction} | score failed: {exc}")
            prob = None

        scored_rows.append({
            **row.to_dict(),
            "smc_prob": prob,
            "tier":     _tier(prob) if prob is not None else "SKIP",
            "is_win":   _win(row),
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(trades):
            logger.info(f"Scored {i + 1}/{len(trades)}...")

    results = pd.DataFrame(scored_rows)
    results = results[results["tier"] != "SKIP"].copy()

    if results.empty:
        print("No trades could be scored (all lacked sufficient historical bars).")
        return

    report(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
