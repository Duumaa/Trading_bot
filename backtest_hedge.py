"""
backtest_hedge.py
=================
Hedge strategy backtest — accurate structural S/R version.

Simulates HedgeManager bar-by-bar on historical M15 OHLCV data using the same
S/R computation as sr_features.py (swing highs/lows, round numbers, daily pivots,
EMAs, weekly/daily high-low). Replaces the earlier rolling-max/min proxy.

Sources of S/R (matching sr_features.py)
-----------------------------------------
  H1 swing highs/lows   — last 120 H1 bars (5-bar window)
  Round numbers          — every 25 pips (JPY 0.25 / others 0.0025)
  Daily pivot + R1/R2/S1/S2 — from yesterday's M15 data
  5-day high/low         — rolling
  Weekly high/low        — Mon 00:00 UTC → current bar
  H1 EMA20 + EMA50       — for cycling walls only (not structural gate)
  M15 EMA20              — for cycling walls only

Two wall types (matching hedge_manager.py usage):
  structural: swing + round + pivots + daily/weekly HL (used for open gate + SL sizing)
  nearest:    structural + EMAs  (used for cycle trigger)

MR regime proxy: ADX(14) < 25 on M15

What is NOT simulated:
  - A1 LightGBM regime (ADX proxy used)
  - Event blackout
  - Cascade pyramid tiers
  - Re-entry after WINNER_RUNNING loser close

Usage
-----
  python3 backtest_hedge.py                          # EURUSD, last 6 months
  python3 backtest_hedge.py --pair AUDJPY --since 2025-01-01
  python3 backtest_hedge.py --all --since 2025-10-01
  python3 backtest_hedge.py --pair USDJPY --since 2024-01-01 --balance 2000
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from data_pipeline.storage.timeseries_db import db

# ── Constants (mirror hedge_manager.py) ──────────────────────────────────────

JPY_PAIRS    = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}
ALL_PAIRS    = [
    "EURUSD", "USDJPY",  # low-spread pairs only for hedge strategy
]

CYCLE_PIPS            = 5
SR_RANGE_MIN_PIPS     = 25      # min structural range (round-number based — keep original)
SR_OPEN_MIN_PIPS      = 10      # don't open if sitting on a structural wall
SWING_MIN_PIPS        = 30      # min distance to nearest SWING-ONLY wall (no round nums) each side
                                # → total 60-pip swing channel requirement
MIN_CYCLE_PROFIT_PIPS = 5       # min pip gain on leg before cycling
ATR_STOP_MULT         = 2.0
ATR_TP_SAFETY_MULT    = 12.0
RISK_PER_LEG_PCT      = 0.01
PROFIT_TARGET_MULT    = 2.0
HEDGE_MAX_HOURS       = 48
SL_WALK_FRACTION      = 0.33
MAX_CYCLE_COUNT       = 6
MAX_WASH_LOSS_USD     = 25.0
WINNER_TRAIL_PIPS     = 15
LOSER_INVALIDATE_MULT = 1.5
WINNER_BREAKOUT_PIPS  = 3      # pips past structural wall → WINNER_RUNNING

# MR detection (ADX)
ADX_MR_THRESHOLD = 18  # stricter: genuine range, not slow drift
ADX_PERIODS      = 14

# S/R computation (matching sr_features.py)
SWING_WINDOW          = 5      # bars each side for H1 swing detection
H1_BARS_LOOKBACK      = 120    # last N H1 bars for swing levels
DAILY_BARS_LOOKBACK   = 5      # days for rolling daily high/low

# Round number interval (price units)
ROUND_NUM_INTERVAL = {True: 0.25, False: 0.0025}  # JPY / non-JPY

# Spread (one-side, pips)
SPREAD_PIPS: Dict[str, float] = {
    "EURUSD": 0.8, "GBPUSD": 1.0, "AUDUSD": 0.9, "NZDUSD": 1.0,
    "USDCAD": 1.2, "USDCHF": 1.1, "EURGBP": 1.0,
    "USDJPY": 0.8, "EURJPY": 1.1, "GBPJPY": 1.5,
    "AUDJPY": 1.2, "NZDJPY": 1.5,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else 0.0001


def pip_value_per_unit(pair: str, price: float) -> float:
    p = pip_size(pair)
    usd_quote = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURGBP"}
    if pair in usd_quote:
        return p
    return p / price if price > 0 else p


def calc_volume(pair: str, price: float, atr: float, balance: float) -> int:
    pip       = pip_size(pair)
    stop_pips = (atr * ATR_STOP_MULT) / pip
    pip_val   = pip_value_per_unit(pair, price)
    if stop_pips <= 0 or pip_val <= 0:
        return 1000
    raw = (balance * RISK_PER_LEG_PCT) / (stop_pips * pip_val)
    return max(1000, int(raw / 1000) * 1000)


def calc_pnl_usd(pair: str, entry: float, exit_p: float, direction: str, vol: int) -> float:
    pip   = pip_size(pair)
    pip_v = pip_value_per_unit(pair, exit_p)
    pips  = (exit_p - entry) / pip if direction == "BUY" else (entry - exit_p) / pip
    return round(pips * pip_v * vol, 4)


def spread_cost_usd(pair: str, vol: int, price: float) -> float:
    sp    = SPREAD_PIPS.get(pair, 1.0)
    pip_v = pip_value_per_unit(pair, price)
    return round(sp * pip_v * vol, 4)


# ── S/R helper functions (ported from sr_features.py) ───────────────────────

def _round_number_levels(price: float, pair: str, n: int = 8) -> Tuple[List[float], List[float]]:
    interval = ROUND_NUM_INTERVAL[pair in JPY_PAIRS]
    base     = round(price / interval) * interval
    above    = [round(base + interval * i, 6) for i in range(1, n + 1)]
    below    = [round(base - interval * i, 6) for i in range(1, n + 1)]
    return above, below


def _find_swing_levels(highs: np.ndarray, lows: np.ndarray, window: int = SWING_WINDOW):
    swing_highs, swing_lows = [], []
    n = len(highs)
    for i in range(window, n - window):
        lh = highs[i - window:i]
        rh = highs[i + 1:i + window + 1]
        if highs[i] > lh.max() and highs[i] > rh.max():
            swing_highs.append(float(highs[i]))
        ll = lows[i - window:i]
        rl = lows[i + 1:i + window + 1]
        if lows[i] < ll.min() and lows[i] < rl.min():
            swing_lows.append(float(lows[i]))
    return swing_highs, swing_lows


# ── ADX computation (for MR detection) ───────────────────────────────────────

def _wilder(series: pd.Series, n: int) -> pd.Series:
    result = series.copy() * np.nan
    seed   = series.iloc[:n].mean()
    result.iloc[n - 1] = seed
    for i in range(n, len(series)):
        result.iloc[i] = result.iloc[i - 1] * (n - 1) / n + series.iloc[i] / n
    return result


def compute_adx(df: pd.DataFrame, n: int = ADX_PERIODS) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - prev_close), abs(df["low"] - prev_close)))
    up   = df["high"] - df["high"].shift(1)
    down = df["low"].shift(1) - df["low"]
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    atr_s = _wilder(pd.Series(tr, index=df.index), n)
    pdi   = 100 * _wilder(pd.Series(pdm, index=df.index), n) / atr_s.where(atr_s > 0)
    ndi   = 100 * _wilder(pd.Series(ndm, index=df.index), n) / atr_s.where(atr_s > 0)
    dx    = 100 * abs(pdi - ndi) / (pdi + ndi).where((pdi + ndi) > 0)
    adx   = _wilder(dx.fillna(0), n)
    return adx, atr_s


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ohlcv(pair: str, since: str) -> pd.DataFrame:
    query = """
        SELECT time, open, high, low, close
        FROM ohlcv
        WHERE symbol = :sym AND timeframe = 'M15' AND time >= :since
        ORDER BY time
    """
    with db.connection() as conn:
        rows = conn.execute(text(query), {"sym": pair, "since": since}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if pair in JPY_PAIRS:
            df.loc[df[col] > 1000, col] /= 100
    return df.dropna(subset=["close"])


# ── Structural S/R pre-computation ───────────────────────────────────────────

def precompute_sr_table(m15_df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Pre-compute S/R levels at every H1 boundary over the backtest window.
    Returns DataFrame indexed by H1 timestamp with columns:
      struct_res, struct_sup       — structural walls (open gate + SL sizing)
      struct_dist_res, struct_dist_sup  — distance in pips
      nearest_res, nearest_sup    — cycling walls (structural + EMAs)
      dist_res, dist_sup           — distance in pips
      atr                          — M15 ATR-14 at this H1 boundary
      is_mr                        — ADX < 25
    """
    pip = pip_size(pair)
    logger.info(f"{pair}: pre-computing structural S/R table…")

    # ── ADX + ATR on M15 ─────────────────────────────────────────────────────
    adx_series, atr_series = compute_adx(m15_df, ADX_PERIODS)
    m15_df = m15_df.copy()
    m15_df["adx"]   = adx_series
    m15_df["atr"]   = atr_series
    m15_df["is_mr"] = adx_series < ADX_MR_THRESHOLD

    # ── M15 EMA20 ─────────────────────────────────────────────────────────────
    m15_df["m15_ema20"] = m15_df["close"].ewm(span=20, adjust=False).mean()

    # ── Resample M15 → H1 ────────────────────────────────────────────────────
    h1_df = (
        m15_df.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "adx": "last", "atr": "last",
              "is_mr": "last", "m15_ema20": "last"})
        .dropna(subset=["close"])
    )

    # ── H1 EMAs ──────────────────────────────────────────────────────────────
    h1_df["h1_ema20"] = h1_df["close"].ewm(span=20, adjust=False).mean()
    h1_df["h1_ema50"] = h1_df["close"].ewm(span=50, adjust=False).mean()

    # ── Pre-flag H1 swing highs/lows ─────────────────────────────────────────
    h1_highs = h1_df["high"].values
    h1_lows  = h1_df["low"].values
    n        = len(h1_df)
    swing_high_flag = np.zeros(n, dtype=bool)
    swing_low_flag  = np.zeros(n, dtype=bool)
    w = SWING_WINDOW
    for i in range(w, n - w):
        if h1_highs[i] > h1_highs[i-w:i].max() and h1_highs[i] > h1_highs[i+1:i+w+1].max():
            swing_high_flag[i] = True
        if h1_lows[i]  < h1_lows[i-w:i].min()  and h1_lows[i]  < h1_lows[i+1:i+w+1].min():
            swing_low_flag[i]  = True
    h1_df["swing_high"] = swing_high_flag
    h1_df["swing_low"]  = swing_low_flag

    # ── Compute daily pivots once per day ────────────────────────────────────
    # Group M15 bars by date → pivot from previous day's OHLC
    m15_df_reset = m15_df.reset_index()
    m15_df_reset["date"] = m15_df_reset["time"].dt.date
    daily_ohlc = m15_df_reset.groupby("date").agg(
        H=("high", "max"), L=("low", "min"), C=("close", "last")
    )
    daily_pivots: Dict = {}  # date → {pivot, r1, r2, s1, s2}
    dates = sorted(daily_ohlc.index)
    for i in range(1, len(dates)):
        today = dates[i]
        yday  = dates[i - 1]
        H, L, C = daily_ohlc.loc[yday, ["H", "L", "C"]]
        P = (H + L + C) / 3.0
        daily_pivots[today] = {
            "pivot": P,
            "r1": 2 * P - L,  "r2": P + (H - L),
            "s1": 2 * P - H,  "s2": P - (H - L),
        }

    # ── Build SR record for every H1 bar ─────────────────────────────────────
    records = []
    h1_timestamps = h1_df.index.tolist()

    for bar_idx, ts in enumerate(h1_timestamps):
        row         = h1_df.iloc[bar_idx]
        price       = float(row["close"])
        current_date = ts.date()

        # ADX / ATR from M15 at this point
        bar_adx  = float(row["adx"]) if not np.isnan(row["adx"]) else 50.0
        bar_atr  = float(row["atr"]) if not np.isnan(row["atr"]) else 0.001
        bar_is_mr = bool(row["is_mr"]) if not np.isnan(row["adx"]) else False

        # ── H1 swing levels (last 120 bars including current) ─────────────────
        slice_start = max(0, bar_idx - H1_BARS_LOOKBACK + 1)
        h1_slice    = h1_df.iloc[slice_start:bar_idx + 1]
        swing_highs = h1_slice.loc[h1_slice["swing_high"], "high"].tolist()
        swing_lows  = h1_slice.loc[h1_slice["swing_low"],  "low"].tolist()

        # ── Round numbers ────────────────────────────────────────────────────
        rn_above, rn_below = _round_number_levels(price, pair)

        # ── 5-day high/low ───────────────────────────────────────────────────
        cutoff_5d = ts - timedelta(days=DAILY_BARS_LOOKBACK)
        m15_5d    = m15_df.loc[cutoff_5d:ts]
        day_high  = float(m15_5d["high"].max()) if not m15_5d.empty else price
        day_low   = float(m15_5d["low"].min())  if not m15_5d.empty else price

        # ── Weekly high/low (Mon 00:00 UTC → current bar) ───────────────────
        days_since_mon = ts.weekday()
        week_start     = (ts - timedelta(days=days_since_mon)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        m15_week  = m15_df.loc[week_start:ts] if week_start <= ts else m15_df.iloc[:0]
        week_high = float(m15_week["high"].max()) if not m15_week.empty else day_high
        week_low  = float(m15_week["low"].min())  if not m15_week.empty else day_low

        # ── Daily pivot (from yesterday) ─────────────────────────────────────
        pivots = daily_pivots.get(current_date, {})

        # ── Structural candidates (swing + round + pivots + day/week HL) ─────
        struct_above: List[float] = []
        struct_below: List[float] = []

        struct_above.extend(h for h in swing_highs if h > price)
        struct_above.extend(rn_above)
        if day_high  > price: struct_above.append(day_high)
        if week_high > price: struct_above.append(week_high)
        for k in ("r1", "r2", "pivot"):
            v = pivots.get(k)
            if v is not None and v > price:
                struct_above.append(v)

        struct_below.extend(l for l in swing_lows if l < price)
        struct_below.extend(rn_below)
        if day_low  < price: struct_below.append(day_low)
        if week_low < price: struct_below.append(week_low)
        for k in ("s1", "s2", "pivot"):
            v = pivots.get(k)
            if v is not None and v < price:
                struct_below.append(v)

        struct_res = min(struct_above) if struct_above else price + 50 * pip
        struct_sup = max(struct_below) if struct_below else price - 50 * pip

        # ── Swing-only walls (no round numbers) for wide-range gate ──────────
        swing_above = [h for h in swing_highs if h > price]
        swing_below = [l for l in swing_lows  if l < price]
        if day_high  > price: swing_above.append(day_high)
        if week_high > price: swing_above.append(week_high)
        if day_low   < price: swing_below.append(day_low)
        if week_low  < price: swing_below.append(week_low)
        for k in ("r1", "r2", "pivot"):
            v = pivots.get(k)
            if v is not None and v > price:
                swing_above.append(v)
        for k in ("s1", "s2", "pivot"):
            v = pivots.get(k)
            if v is not None and v < price:
                swing_below.append(v)
        swing_res = min(swing_above) if swing_above else price + 100 * pip
        swing_sup = max(swing_below) if swing_below else price - 100 * pip

        # ── Full candidates (add EMAs for cycling) ────────────────────────────
        full_above = list(struct_above)
        full_below = list(struct_below)

        for ema in (row.get("h1_ema20"), row.get("h1_ema50"), row.get("m15_ema20")):
            if ema is not None and not np.isnan(ema):
                if ema > price:
                    full_above.append(float(ema))
                elif ema < price:
                    full_below.append(float(ema))

        nearest_res = min(full_above) if full_above else struct_res
        nearest_sup = max(full_below) if full_below else struct_sup

        struct_dist_res = (struct_res  - price) / pip
        struct_dist_sup = (price - struct_sup)   / pip
        dist_res        = (nearest_res - price)   / pip
        dist_sup        = (price - nearest_sup)   / pip
        swing_dist_res  = (swing_res - price)     / pip
        swing_dist_sup  = (price - swing_sup)     / pip

        records.append({
            "time":            ts,
            "struct_res":      struct_res,
            "struct_sup":      struct_sup,
            "struct_dist_res": struct_dist_res,
            "struct_dist_sup": struct_dist_sup,
            "nearest_res":     nearest_res,
            "nearest_sup":     nearest_sup,
            "dist_res":        dist_res,
            "dist_sup":        dist_sup,
            "swing_dist_res":  swing_dist_res,
            "swing_dist_sup":  swing_dist_sup,
            "atr":             bar_atr,
            "is_mr":           bar_is_mr,
        })

    sr_df = pd.DataFrame(records).set_index("time")
    logger.info(f"{pair}: S/R table ready — {len(sr_df):,} H1 bars, "
                f"MR={sr_df['is_mr'].mean()*100:.0f}% of bars")
    return sr_df


# ── Simulation state + records ────────────────────────────────────────────────

@dataclass
class HedgeState:
    pair:             str
    entry_price:      float      # original fixed entry (for SL walk reference)
    buy_entry:        float      # updated per BUY cycle
    sell_entry:       float      # updated per SELL cycle
    buy_sl:           float
    sell_sl:          float
    atr:              float
    volume:           int
    opened_at:        pd.Timestamp
    initial_risk_usd: float
    struct_res:       float      # structural resistance at open
    struct_sup:       float      # structural support at open
    cycle_count:      int   = 0
    realized_pnl:     float = 0.0
    hedge_mode:       str   = "CYCLING"
    winning_leg:      Optional[str] = None
    winner_sl:        Optional[float] = None
    loser_entry:      Optional[float] = None
    wash_mode:        bool  = False


@dataclass
class TradeRecord:
    pair:          str
    opened_at:     pd.Timestamp
    closed_at:     pd.Timestamp
    close_reason:  str
    cycle_count:   int
    realized_pnl:  float
    total_pnl:     float
    duration_h:    float


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(
    m15_df: pd.DataFrame,
    sr_df:  pd.DataFrame,
    pair:   str,
    balance: float,
) -> Tuple[List[TradeRecord], pd.DataFrame]:
    """
    Bar-by-bar hedge simulation using pre-computed structural S/R.
    sr_df is indexed at H1 boundaries; we forward-fill to M15 granularity.
    """
    pip      = pip_size(pair)
    records: List[TradeRecord] = []
    equity:  List[Dict]        = []
    state:   Optional[HedgeState] = None
    running_balance = balance

    # Forward-fill H1 S/R onto M15 bars (each M15 bar uses the last H1 entry)
    sr_ffill = sr_df.reindex(sr_df.index.union(m15_df.index)).ffill()
    sr_ffill = sr_ffill.reindex(m15_df.index)

    for idx, row in m15_df.iterrows():
        price    = float(row["close"])
        bar_high = float(row["high"])
        bar_low  = float(row["low"])

        # Look up the forward-filled S/R for this M15 bar
        sr = sr_ffill.loc[idx] if idx in sr_ffill.index else None
        if sr is None or sr.isnull().any():
            continue

        is_mr       = bool(sr["is_mr"])
        atr         = float(sr["atr"])
        struct_res  = float(sr["struct_res"])
        struct_sup  = float(sr["struct_sup"])
        nearest_res = float(sr["nearest_res"])
        nearest_sup = float(sr["nearest_sup"])
        struct_dist_res = float(sr["struct_dist_res"])
        struct_dist_sup = float(sr["struct_dist_sup"])
        dist_res    = float(sr["dist_res"])
        dist_sup    = float(sr["dist_sup"])
        swing_dist_res = float(sr["swing_dist_res"])
        swing_dist_sup = float(sr["swing_dist_sup"])

        # ── No active hedge: try to open ──────────────────────────────────────
        if state is None:
            if not is_mr:
                continue

            # Structural range check (matching hedge_manager.consider_opening)
            sr_range_pips = struct_dist_res + struct_dist_sup
            if sr_range_pips < SR_RANGE_MIN_PIPS:
                continue
            if struct_dist_res < SR_OPEN_MIN_PIPS or struct_dist_sup < SR_OPEN_MIN_PIPS:
                continue
            # Swing-only wide-range gate: nearest major wall (no round nums) ≥ 30 pips each side
            # This ensures a genuine 60-pip ranging channel exists (not just round-number proximity)
            if swing_dist_res < SWING_MIN_PIPS or swing_dist_sup < SWING_MIN_PIPS:
                continue

            # SL must cover at least 1.5× the structural range
            sr_range_price = sr_range_pips * pip
            min_atr = (sr_range_price * 1.5) / ATR_STOP_MULT
            atr_eff = max(atr, min_atr)
            sl_dist = atr_eff * ATR_STOP_MULT

            volume           = calc_volume(pair, price, atr_eff, running_balance)
            pip_val          = pip_value_per_unit(pair, price)
            sl_pips          = sl_dist / pip
            initial_risk_usd = sl_pips * pip_val * volume

            # Spread for opening both legs
            running_balance -= spread_cost_usd(pair, volume, price) * 2

            state = HedgeState(
                pair             = pair,
                entry_price      = price,
                buy_entry        = price,
                sell_entry       = price,
                buy_sl           = price - sl_dist,
                sell_sl          = price + sl_dist,
                atr              = atr_eff,
                volume           = volume,
                opened_at        = idx,
                initial_risk_usd = initial_risk_usd,
                struct_res       = struct_res,
                struct_sup       = struct_sup,
            )
            continue

        # ── Active hedge checks ───────────────────────────────────────────────

        # 1. SL hit (bar high/low crosses SL)
        buy_sl_hit  = bar_low  <= state.buy_sl
        sell_sl_hit = bar_high >= state.sell_sl

        if buy_sl_hit or sell_sl_hit:
            pnl = state.realized_pnl
            pnl += (calc_pnl_usd(pair, state.buy_entry,  state.buy_sl,  "BUY",  state.volume)
                    if buy_sl_hit else
                    calc_pnl_usd(pair, state.buy_entry,  price,         "BUY",  state.volume))
            pnl -= spread_cost_usd(pair, state.volume, price)
            pnl += (calc_pnl_usd(pair, state.sell_entry, state.sell_sl, "SELL", state.volume)
                    if sell_sl_hit else
                    calc_pnl_usd(pair, state.sell_entry, price,         "SELL", state.volume))
            pnl -= spread_cost_usd(pair, state.volume, price)
            dur  = (idx - state.opened_at).total_seconds() / 3600
            records.append(TradeRecord(pair, state.opened_at, idx, "SL_HIT",
                                       state.cycle_count, state.realized_pnl, round(pnl,2), round(dur,1)))
            running_balance += pnl
            state = None
            continue

        # 2. Profit target (realized ≥ 2× initial risk)
        if state.initial_risk_usd > 0 and state.realized_pnl >= state.initial_risk_usd * PROFIT_TARGET_MULT:
            b = calc_pnl_usd(pair, state.buy_entry,  price, "BUY",  state.volume)
            s = calc_pnl_usd(pair, state.sell_entry, price, "SELL", state.volume)
            c = spread_cost_usd(pair, state.volume, price) * 2
            pnl = state.realized_pnl + b + s - c
            dur = (idx - state.opened_at).total_seconds() / 3600
            records.append(TradeRecord(pair, state.opened_at, idx, "PROFIT_TARGET",
                                       state.cycle_count, state.realized_pnl, round(pnl,2), round(dur,1)))
            running_balance += pnl
            state = None
            continue

        # 3. Time stop (48h) — only if NET PnL > 0
        age_h = (idx - state.opened_at).total_seconds() / 3600
        if age_h > HEDGE_MAX_HOURS:
            b    = calc_pnl_usd(pair, state.buy_entry,  price, "BUY",  state.volume)
            s    = calc_pnl_usd(pair, state.sell_entry, price, "SELL", state.volume)
            cost = spread_cost_usd(pair, state.volume, price) * 2
            pnl  = state.realized_pnl + b + s - cost
            if pnl > 0:
                records.append(TradeRecord(pair, state.opened_at, idx, "TIME_STOP",
                                           state.cycle_count, state.realized_pnl, round(pnl,2), round(age_h,1)))
                running_balance += pnl
                state = None
                continue

        # 4. WINNER_RUNNING: triggers when price breaks OUTSIDE structural walls
        buy_breakout  = price > state.struct_res + WINNER_BREAKOUT_PIPS * pip
        sell_breakout = price < state.struct_sup - WINNER_BREAKOUT_PIPS * pip

        if state.hedge_mode == "CYCLING" and (buy_breakout or sell_breakout):
            state.hedge_mode  = "WINNER_RUNNING"
            state.winning_leg = "BUY" if buy_breakout else "SELL"
            # Start trail at max(breakeven, price - trail) so we never lock in a loss
            if state.winning_leg == "BUY":
                state.winner_sl = max(price - WINNER_TRAIL_PIPS * pip, state.buy_entry)
            else:
                state.winner_sl = min(price + WINNER_TRAIL_PIPS * pip, state.sell_entry)
            state.loser_entry = state.sell_entry if state.winning_leg == "BUY" else state.buy_entry

        if state.hedge_mode == "WINNER_RUNNING":
            wl = state.winning_leg
            if wl == "BUY":
                new_trail = price - WINNER_TRAIL_PIPS * pip
                if state.winner_sl is None or new_trail > state.winner_sl:
                    state.winner_sl = new_trail
                if bar_low <= state.winner_sl:
                    w_pnl  = calc_pnl_usd(pair, state.buy_entry,  state.winner_sl, "BUY",  state.volume)
                    lo_pnl = calc_pnl_usd(pair, state.sell_entry, price,           "SELL", state.volume)
                    pnl    = state.realized_pnl + w_pnl + lo_pnl - spread_cost_usd(pair, state.volume, price)*2
                    dur    = (idx - state.opened_at).total_seconds() / 3600
                    records.append(TradeRecord(pair, state.opened_at, idx, "WINNER_TRAIL_HIT",
                                               state.cycle_count, state.realized_pnl, round(pnl,2), round(dur,1)))
                    running_balance += pnl
                    state = None
                else:
                    loser_move = price - (state.loser_entry or state.entry_price)
                    if loser_move >= state.atr * LOSER_INVALIDATE_MULT:
                        w_pnl  = calc_pnl_usd(pair, state.buy_entry,  price, "BUY",  state.volume)
                        lo_pnl = calc_pnl_usd(pair, state.sell_entry, price, "SELL", state.volume)
                        pnl    = state.realized_pnl + w_pnl + lo_pnl - spread_cost_usd(pair, state.volume, price)*2
                        dur    = (idx - state.opened_at).total_seconds() / 3600
                        records.append(TradeRecord(pair, state.opened_at, idx, "LOSER_INVALIDATED",
                                                   state.cycle_count, state.realized_pnl, round(pnl,2), round(dur,1)))
                        running_balance += pnl
                        state = None
            else:  # SELL winning
                new_trail = price + WINNER_TRAIL_PIPS * pip
                if state.winner_sl is None or new_trail < state.winner_sl:
                    state.winner_sl = new_trail
                if bar_high >= state.winner_sl:
                    w_pnl  = calc_pnl_usd(pair, state.sell_entry, state.winner_sl, "SELL", state.volume)
                    lo_pnl = calc_pnl_usd(pair, state.buy_entry,  price,           "BUY",  state.volume)
                    pnl    = state.realized_pnl + w_pnl + lo_pnl - spread_cost_usd(pair, state.volume, price)*2
                    dur    = (idx - state.opened_at).total_seconds() / 3600
                    records.append(TradeRecord(pair, state.opened_at, idx, "WINNER_TRAIL_HIT",
                                               state.cycle_count, state.realized_pnl, round(pnl,2), round(dur,1)))
                    running_balance += pnl
                    state = None
                else:
                    loser_move = (state.loser_entry or state.entry_price) - price
                    if loser_move >= state.atr * LOSER_INVALIDATE_MULT:
                        w_pnl  = calc_pnl_usd(pair, state.sell_entry, price, "SELL", state.volume)
                        lo_pnl = calc_pnl_usd(pair, state.buy_entry,  price, "BUY",  state.volume)
                        pnl    = state.realized_pnl + w_pnl + lo_pnl - spread_cost_usd(pair, state.volume, price)*2
                        dur    = (idx - state.opened_at).total_seconds() / 3600
                        records.append(TradeRecord(pair, state.opened_at, idx, "LOSER_INVALIDATED",
                                                   state.cycle_count, state.realized_pnl, round(pnl,2), round(dur,1)))
                        running_balance += pnl
                        state = None
            continue

        # 5. Max cycles + wash mode
        if state.cycle_count >= MAX_CYCLE_COUNT:
            buy_upnl  = calc_pnl_usd(pair, state.buy_entry,  price, "BUY",  state.volume)
            sell_upnl = calc_pnl_usd(pair, state.sell_entry, price, "SELL", state.volume)
            net = state.realized_pnl + buy_upnl + sell_upnl
            cost = spread_cost_usd(pair, state.volume, price) * 2
            dur  = (idx - state.opened_at).total_seconds() / 3600
            if net >= 0:
                records.append(TradeRecord(pair, state.opened_at, idx, "WASH_COMPLETE",
                                           state.cycle_count, state.realized_pnl, round(net-cost,2), round(dur,1)))
                running_balance += net - cost
                state = None
                continue
            if net <= -MAX_WASH_LOSS_USD:
                records.append(TradeRecord(pair, state.opened_at, idx, "WASH_STOPPED",
                                           state.cycle_count, state.realized_pnl, round(net-cost,2), round(dur,1)))
                running_balance += net - cost
                state = None
                continue
            state.cycle_count = 0
            state.wash_mode   = True

        # 6. Cycling: use nearest_res/sup (all levels including EMAs)
        dist_res_now = (nearest_res - price) / pip
        dist_sup_now = (price - nearest_sup) / pip

        if dist_res_now <= CYCLE_PIPS:
            buy_gain = (price - state.buy_entry) / pip
            if buy_gain >= MIN_CYCLE_PROFIT_PIPS:
                leg_pnl = calc_pnl_usd(pair, state.buy_entry, price, "BUY", state.volume)
                cost    = spread_cost_usd(pair, state.volume, price) * 2
                state.realized_pnl += leg_pnl - cost
                state.cycle_count  += 1
                state.buy_entry     = price
                state.buy_sl        = price - state.atr * ATR_STOP_MULT
                # Walk SELL SL toward original entry_price (1/3 of gap)
                new_sell_sl = state.sell_sl + (state.entry_price - state.sell_sl) * SL_WALK_FRACTION
                if new_sell_sl > state.entry_price:
                    state.sell_sl = new_sell_sl

        elif dist_sup_now <= CYCLE_PIPS:
            sell_gain = (state.sell_entry - price) / pip
            if sell_gain >= MIN_CYCLE_PROFIT_PIPS:
                leg_pnl = calc_pnl_usd(pair, state.sell_entry, price, "SELL", state.volume)
                cost    = spread_cost_usd(pair, state.volume, price) * 2
                state.realized_pnl += leg_pnl - cost
                state.cycle_count  += 1
                state.sell_entry    = price
                state.sell_sl       = price + state.atr * ATR_STOP_MULT
                # Walk BUY SL toward original entry_price (1/3 of gap)
                new_buy_sl = state.buy_sl + (state.entry_price - state.buy_sl) * SL_WALK_FRACTION
                if new_buy_sl < state.entry_price:
                    state.buy_sl = new_buy_sl

        # Equity snapshot
        if state:
            u = (calc_pnl_usd(pair, state.buy_entry,  price, "BUY",  state.volume)
               + calc_pnl_usd(pair, state.sell_entry, price, "SELL", state.volume))
            equity.append({"time": idx, "balance": running_balance,
                           "equity": running_balance + u, "in_trade": True})

    # Close any remaining open hedge at end-of-data
    if state is not None:
        lp  = float(m15_df["close"].iloc[-1])
        b   = calc_pnl_usd(pair, state.buy_entry,  lp, "BUY",  state.volume)
        s   = calc_pnl_usd(pair, state.sell_entry, lp, "SELL", state.volume)
        pnl = state.realized_pnl + b + s - spread_cost_usd(pair, state.volume, lp)*2
        dur = (m15_df.index[-1] - state.opened_at).total_seconds() / 3600
        records.append(TradeRecord(pair, state.opened_at, m15_df.index[-1], "END_OF_DATA",
                                   state.cycle_count, state.realized_pnl, round(pnl,2), round(dur,1)))
        running_balance += pnl

    eq_df = pd.DataFrame(equity) if equity else pd.DataFrame()
    return records, eq_df


# ── Reporting ─────────────────────────────────────────────────────────────────

def _bar(val: float, mx: float, w: int = 20) -> str:
    f = int(round(val / mx * w)) if mx > 0 else 0
    return "█" * f + "░" * (w - f)


def _pct(n: int, total: int) -> str:
    return f"{n/total*100:.0f}%" if total else "—"


def print_report(records: List[TradeRecord], pair: str, balance: float,
                 eq_df: pd.DataFrame, mr_pct: float) -> None:
    if not records:
        print(f"\n  {pair}: no hedge trades triggered.")
        return

    df   = pd.DataFrame([vars(r) for r in records])
    n    = len(df)
    wins = (df["total_pnl"] > 0).sum()
    wr   = wins / n * 100
    pnl  = df["total_pnl"].sum()
    avg_win  = df.loc[df["total_pnl"] > 0, "total_pnl"].mean() if wins else 0
    avg_loss = df.loc[df["total_pnl"] <= 0, "total_pnl"].mean() if (n - wins) else 0
    avg_dur  = df["duration_h"].mean()
    avg_cyc  = df["cycle_count"].mean()
    avg_range = df.get("sr_range_pips", pd.Series([0]*n)).mean() if "sr_range_pips" in df else 0

    print(f"\n{'═'*64}")
    print(f"  HEDGE BACKTEST — {pair}")
    print(f"{'═'*64}")
    print(f"  MR regime detected  : {mr_pct:.0f}% of bars")
    print(f"  Trades              : {n}")
    print(f"  Win rate            : {wr:.1f}%  ({wins}W / {n-wins}L)")
    print(f"  Total P&L           : ${pnl:+.2f}")
    print(f"  Starting balance    : ${balance:,.2f}  →  ${balance+pnl:,.2f}")
    print(f"  Avg P&L/trade       : ${pnl/n:+.2f}")
    if wins:     print(f"  Avg win $           : ${avg_win:+.2f}")
    if n-wins:   print(f"  Avg loss $          : ${avg_loss:+.2f}")
    if wins and n-wins: print(f"  Win/Loss ratio      : {abs(avg_win/avg_loss):.2f}×")
    print(f"  Avg duration        : {avg_dur:.1f}h")
    print(f"  Avg cycles/trade    : {avg_cyc:.1f}")

    if not eq_df.empty and "equity" in eq_df.columns:
        eq  = eq_df["equity"]
        rmx = eq.cummax()
        dd  = ((eq - rmx) / rmx * 100).min()
        print(f"  Max drawdown        : {dd:.1f}%")

    print(f"\n  Close reasons:")
    reasons = df["close_reason"].value_counts()
    mx = reasons.max()
    for reason, cnt in reasons.items():
        sub = df.loc[df["close_reason"] == reason, "total_pnl"].sum()
        print(f"    {reason:<24} {_bar(cnt, mx, 16)}  {cnt:>3} ({_pct(cnt,n)})  ${sub:+.2f}")

    print(f"\n  Cycles per trade:")
    cdist = df["cycle_count"].value_counts().sort_index()
    mx = cdist.max()
    for cyc, cnt in cdist.items():
        print(f"    {cyc} cycles  {_bar(cnt, mx, 16)}  {cnt:>3}")

    if n >= 4:
        print(f"\n  Monthly P&L:")
        df["month"] = pd.to_datetime(df["closed_at"]).dt.to_period("M")
        monthly = df.groupby("month")["total_pnl"].agg(["sum","count"]).reset_index()
        for _, row in monthly.iterrows():
            sign = "✅" if row["sum"] > 0 else "🔴"
            print(f"    {row['month']}  {sign}  ${row['sum']:>+8.2f}  ({row['count']} trades)")


def print_summary(all_results: Dict[str, List[TradeRecord]], balance: float) -> None:
    print(f"\n{'═'*70}")
    print("  MULTI-PAIR SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Pair':<10} {'N':>4}  {'WR':>6}  {'PnL $':>9}  {'Avg $':>8}  {'Cycles':>6}")
    print("  " + "─"*58)
    all_dfs = []
    for pair in sorted(all_results):
        recs = all_results[pair]
        if not recs:
            print(f"  {pair:<10}    0     —         —          —        —")
            continue
        df   = pd.DataFrame([vars(r) for r in recs])
        n    = len(df); wins = (df["total_pnl"] > 0).sum()
        wr   = wins/n*100; pnl = df["total_pnl"].sum()
        avg  = pnl/n; cyc = df["cycle_count"].mean()
        all_dfs.append(df)
        print(f"  {pair:<10} {n:>4}  {wr:>5.0f}%  {pnl:>+9.2f}  {avg:>+8.2f}  {cyc:>6.1f}")
    if all_dfs:
        combined = pd.concat(all_dfs)
        n = len(combined); wins = (combined["total_pnl"] > 0).sum()
        wr = wins/n*100; pnl = combined["total_pnl"].sum()
        avg = pnl/n; cyc = combined["cycle_count"].mean()
        print("  " + "─"*58)
        print(f"  {'TOTAL':<10} {n:>4}  {wr:>5.0f}%  {pnl:>+9.2f}  {avg:>+8.2f}  {cyc:>6.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="G10 Hedge Strategy Backtest (accurate S/R)")
    parser.add_argument("--pair",    default="EURUSD")
    parser.add_argument("--all",     action="store_true", help="Run all 11 pairs")
    parser.add_argument("--since",   default="2025-10-01")
    parser.add_argument("--balance", type=float, default=2000.0)
    args = parser.parse_args()

    pairs = ALL_PAIRS if args.all else [args.pair.upper()]

    print(f"\n{'═'*64}")
    print("  G10 BOT — HEDGE STRATEGY BACKTEST (accurate S/R)")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"  Generated : {ts}")
    print(f"  Since     : {args.since}")
    print(f"  Balance   : ${args.balance:,.2f}")
    print(f"  Pairs     : {', '.join(pairs)}")
    print(f"{'═'*64}")

    all_results: Dict[str, List[TradeRecord]] = {}

    for pair in pairs:
        # Load extra history for S/R warm-up (swing lookback needs 120 H1 = 5 days)
        warmup_since = (
            datetime.strptime(args.since, "%Y-%m-%d") - timedelta(days=10)
        ).strftime("%Y-%m-%d")

        df = load_ohlcv(pair, warmup_since)
        if df.empty:
            logger.warning(f"{pair}: no data"); all_results[pair] = []; continue

        logger.info(f"{pair}: {len(df):,} M15 bars (since {warmup_since})")

        # Pre-compute full structural S/R table
        sr_df = precompute_sr_table(df, pair)

        # Trim simulation to requested start date (keep warm-up only for S/R)
        sim_start = pd.Timestamp(args.since, tz="UTC")
        df_sim    = df.loc[df.index >= sim_start]
        mr_pct    = float(sr_df["is_mr"].mean() * 100)

        records, eq_df = run_simulation(df_sim, sr_df, pair, args.balance)
        all_results[pair] = records

        if len(pairs) == 1:
            print_report(records, pair, args.balance, eq_df, mr_pct)
        else:
            n   = len(records)
            pnl = sum(r.total_pnl for r in records)
            wins = sum(1 for r in records if r.total_pnl > 0)
            wr   = wins/n*100 if n else 0
            print(f"  {pair}: {n} trades | WR={wr:.0f}% | PnL=${pnl:+.2f} | MR={mr_pct:.0f}%")

    if len(pairs) > 1:
        print_summary(all_results, args.balance)

    print(f"\n{'═'*64}")
    print("  END OF REPORT")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
