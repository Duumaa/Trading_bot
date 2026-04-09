"""
features/sr_features.py
========================
Support & Resistance Level Engine.

Detects significant price levels above and below current price for each G10
pair. Used by Gate 4 (entry_engine) to block entries too close to a wall, by
the price-target engine to snap take-profit levels, and by the hedge_manager
to determine cycling walls.

Level sources (7 types)
-----------------------
  H1 swing highs/lows   — last 120 H1 bars (5 days), window=5
  Daily high/low         — rolling 5-day range
  Weekly high/low        — Mon 00:00 UTC → now
  Round numbers          — every 25 pips (JPY) / 25 pips (non-JPY)
  H1 EMA20 + EMA50       — dynamic intraday walls (widely respected by algos)
  M15 EMA20              — tighter short-term dynamic level
  Daily pivot + R1/R2/S1/S2 — fixed levels from yesterday's OHLC, very watched
  Session high/low       — Asia/London/NY session boundaries (previous + current)

Output (Redis key  features:sr:{PAIR}, TTL=1h):
  nearest_resistance        float   price of nearest resistance above current
  nearest_support           float   price of nearest support below current
  distance_to_resistance    float   pips to nearest resistance (≥0)
  distance_to_support       float   pips to nearest support (≥0)
  resistance_score          float   0–1  (1 = price touching resistance)
  support_score             float   0–1  (1 = price touching support)
  at_resistance             int     1 if distance_to_resistance < 15 pips
  at_support                int     1 if distance_to_support < 15 pips
  all_resistance_levels     list    all resistance levels above price (sorted asc)
  all_support_levels        list    all support levels below price (sorted desc)
  pivot                     float   daily pivot point
  pivot_r1, pivot_r2        float   daily resistance pivots
  pivot_s1, pivot_s2        float   daily support pivots
  h1_ema20, h1_ema50        float   H1 EMA levels
  m15_ema20                 float   M15 EMA level
  session_high, session_low float   current session range
  week_high, week_low       float
  day_high, day_low         float
  level_count_above         int     total resistance candidates
  level_count_below         int     total support candidates
  computed_at               str     ISO timestamp

Gate 4 rule (applied in entry_engine.py):
  Block LONG  if distance_to_resistance < 15 pips
  Block SHORT if distance_to_support    < 15 pips

Schedule: refresh every 1h (EMAs drift — 4h is too stale for dynamic levels).
Redis TTL: 3600s (1h)
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text

load_dotenv("/workspaces/Trading_bot/.env")

# ── constants ────────────────────────────────────────────────────────────────

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

# Round number interval in price units — 25 pips for both JPY and non-JPY
ROUND_NUM_INTERVAL = {True: 0.25, False: 0.0025}   # JPY=0.25 (25 pip), non-JPY=0.0025

SWING_WINDOW         = 5       # bars each side for swing detection
H1_BARS_LOOKBACK     = 120     # ~5 days of H1 data
DAILY_BARS_LOOKBACK  = 5       # days for rolling daily high/low
PROXIMITY_BLOCK_PIPS = 15      # Gate 4: block entry within this many pips of wall
CACHE_TTL            = 3_600   # 1h — EMAs drift, must stay fresh
SCHEDULER_INTERVAL   = 3_600   # 1h

# Session boundaries (UTC hours)
SESSIONS = {
    "ASIA":   (0,  8),
    "LONDON": (8,  16),
    "NY":     (16, 24),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else 0.0001


def _price_to_pips(diff: float, pair: str) -> float:
    return abs(diff) / _pip_size(pair)


def _redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


def _db_connection():
    from data_pipeline.storage.timeseries_db import db
    return db.connection()


def _current_price(pair: str, r: redis.Redis) -> Optional[float]:
    """Get latest mid price from Redis tick key."""
    raw = r.get(f"tick:{pair}")
    if not raw:
        return None
    try:
        d   = json.loads(raw)
        bid = float(d.get("bid", 0))
        ask = float(d.get("ask", 0))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        if bid > 0:
            return bid
    except Exception:
        pass
    return None


def _current_session(now: datetime) -> str:
    h = now.hour
    if 0 <= h < 8:
        return "ASIA"
    if 8 <= h < 16:
        return "LONDON"
    return "NY"


# ── swing level detection ─────────────────────────────────────────────────────

def _find_swing_levels(
    highs: np.ndarray,
    lows:  np.ndarray,
    window: int = SWING_WINDOW,
) -> Tuple[List[float], List[float]]:
    """
    Classic N-bar swing high/low detection.
    A bar i is a swing HIGH if its high > all highs in [i-window, i+window].
    A bar i is a swing LOW  if its low  < all lows  in [i-window, i+window].
    """
    swing_highs: List[float] = []
    swing_lows:  List[float] = []
    n = len(highs)
    for i in range(window, n - window):
        left_h  = highs[i - window:i]
        right_h = highs[i + 1:i + window + 1]
        if highs[i] > left_h.max() and highs[i] > right_h.max():
            swing_highs.append(float(highs[i]))
        left_l  = lows[i - window:i]
        right_l = lows[i + 1:i + window + 1]
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            swing_lows.append(float(lows[i]))
    return swing_highs, swing_lows


def _round_number_levels(
    price: float,
    pair:  str,
    n_above: int = 6,
    n_below: int = 6,
) -> Tuple[List[float], List[float]]:
    """
    Return round number levels above and below price at 25-pip density.
    JPY: 0.25 interval  →  109.00, 109.25, 109.50, 109.75, 110.00
    Non-JPY: 0.0025     →  1.0800, 1.0825, 1.0850, 1.0875, 1.0900
    """
    interval = ROUND_NUM_INTERVAL[pair in JPY_PAIRS]
    base     = round(price / interval) * interval
    above    = [round(base + interval * i, 6) for i in range(1, n_above + 1)]
    below    = [round(base - interval * i, 6) for i in range(1, n_below + 1)]
    return above, below


def _ema(series: pd.Series, span: int) -> float:
    """Return the latest EMA value for a price series."""
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _daily_pivot_levels(
    df_m15: pd.DataFrame,
    now:    datetime,
) -> Dict[str, float]:
    """
    Compute standard daily pivot points from yesterday's OHLC.
    P  = (H + L + C) / 3
    R1 = 2P - L  |  R2 = P + (H - L)
    S1 = 2P - H  |  S2 = P - (H - L)
    Returns dict with keys: pivot, pivot_r1, pivot_r2, pivot_s1, pivot_s2.
    Empty dict if no data.
    """
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    df_yday = df_m15[
        (df_m15["time"] >= yesterday_start) &
        (df_m15["time"] <  today_start)
    ]
    if df_yday.empty:
        return {}

    H = float(df_yday["high"].max())
    L = float(df_yday["low"].min())
    C = float(df_yday["close"].iloc[-1])
    P = (H + L + C) / 3.0
    return {
        "pivot":    round(P, 6),
        "pivot_r1": round(2 * P - L, 6),
        "pivot_r2": round(P + (H - L), 6),
        "pivot_s1": round(2 * P - H, 6),
        "pivot_s2": round(P - (H - L), 6),
    }


def _session_levels(
    df_m15: pd.DataFrame,
    now:    datetime,
) -> Dict[str, float]:
    """
    Compute high/low for the current and previous trading session.
    Sessions (UTC): ASIA 00–08, LONDON 08–16, NY 16–24.
    Returns current session high/low + previous session high/low.
    """
    current = _current_session(now)
    s_start, s_end = SESSIONS[current]
    session_start = now.replace(hour=s_start, minute=0, second=0, microsecond=0)

    df_cur = df_m15[df_m15["time"] >= session_start]
    result: Dict[str, float] = {}

    if not df_cur.empty:
        result["session_high"] = float(df_cur["high"].max())
        result["session_low"]  = float(df_cur["low"].min())

    # Previous session
    prev_sessions = [s for s in SESSIONS if s != current]
    best_prev_end: Optional[datetime] = None
    for s in prev_sessions:
        ps, pe = SESSIONS[s]
        # Build candidate end time (today or yesterday)
        cand = now.replace(hour=pe % 24, minute=0, second=0, microsecond=0)
        if pe == 24:
            cand = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        if cand > session_start:
            cand -= timedelta(days=1)
        if cand <= session_start and (best_prev_end is None or cand > best_prev_end):
            best_prev_end = cand

    if best_prev_end:
        for s, (ps, pe) in SESSIONS.items():
            pe_dt = best_prev_end
            ps_dt = best_prev_end - timedelta(hours=(pe - ps))
            df_prev = df_m15[
                (df_m15["time"] >= ps_dt) &
                (df_m15["time"] <  pe_dt)
            ]
            if not df_prev.empty:
                result["prev_session_high"] = float(df_prev["high"].max())
                result["prev_session_low"]  = float(df_prev["low"].min())
                break

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SRFeatureEngine:
    """
    Support/Resistance feature engine for all 12 G10 pairs.

    Singleton — import and call:
        from features.sr_features import sr_engine
        feats = sr_engine.get_features("EURUSD")
    """

    def __init__(self) -> None:
        self._scheduler_thread: Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────────────

    def get_features(self, pair: str) -> Dict[str, Any]:
        """Return cached S/R features for pair (recomputes on cache miss)."""
        r = _redis_client()
        raw = r.get(f"features:sr:{pair}")
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return self.compute_pair(pair)

    def compute_all(self) -> Dict[str, Dict[str, Any]]:
        """Compute and cache S/R features for all pairs."""
        logger.info("SREngine | Computing S/R levels for all pairs...")
        results: Dict[str, Dict[str, Any]] = {}
        r = _redis_client()
        for pair in PAIRS:
            try:
                feats = self._compute(pair, r)
                r.set(f"features:sr:{pair}", json.dumps(feats), ex=CACHE_TTL)
                results[pair] = feats
                logger.debug(
                    f"SREngine | {pair}: "
                    f"res={feats.get('nearest_resistance', 0):.5f} "
                    f"({feats.get('distance_to_resistance', 0):.1f}pip) | "
                    f"sup={feats.get('nearest_support', 0):.5f} "
                    f"({feats.get('distance_to_support', 0):.1f}pip) | "
                    f"levels ↑{feats.get('level_count_above', 0)} ↓{feats.get('level_count_below', 0)}"
                )
            except Exception as exc:
                logger.error(f"SREngine | {pair} failed: {exc}")
                results[pair] = {}
        logger.success(f"SREngine | Done: {len(results)} pairs computed")
        return results

    def compute_pair(self, pair: str) -> Dict[str, Any]:
        """Compute and cache S/R features for one pair."""
        r = _redis_client()
        feats = self._compute(pair, r)
        r.set(f"features:sr:{pair}", json.dumps(feats), ex=CACHE_TTL)
        return feats

    # ── core computation ──────────────────────────────────────────────────────

    def _compute(self, pair: str, r: redis.Redis) -> Dict[str, Any]:
        """Derive all S/R levels and compute output features."""
        price = _current_price(pair, r)
        if price is None:
            logger.warning(f"SREngine | {pair}: no live price, skipping")
            return {}

        pip    = _pip_size(pair)
        is_jpy = pair in JPY_PAIRS
        now    = datetime.now(timezone.utc)

        # ── Fetch M15 bars ────────────────────────────────────────────────────
        lookback_start = now - timedelta(days=8)
        rows = self._fetch_m15(pair, lookback_start)
        if len(rows) < 20:
            logger.warning(f"SREngine | {pair}: only {len(rows)} M15 bars")
            return {}

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)

        # Normalise JPY ×100 storage
        if is_jpy and df["close"].iloc[-1] > 1000:
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] / 100.0

        # ── Resample M15 → H1 ────────────────────────────────────────────────
        df_h1 = (
            df.resample("1h", on="time")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )

        # ── Daily high/low (rolling 5 days) ──────────────────────────────────
        cutoff_day = now - timedelta(days=DAILY_BARS_LOOKBACK)
        df_recent  = df[df["time"] >= cutoff_day]
        day_high   = float(df_recent["high"].max()) if len(df_recent) else price
        day_low    = float(df_recent["low"].min())  if len(df_recent) else price

        # ── Weekly high/low ───────────────────────────────────────────────────
        days_since_monday = now.weekday()
        week_start = (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        df_week   = df[df["time"] >= week_start]
        week_high = float(df_week["high"].max()) if len(df_week) else price
        week_low  = float(df_week["low"].min())  if len(df_week) else price

        # ── H1 swing highs/lows ───────────────────────────────────────────────
        h1_slice   = df_h1.tail(H1_BARS_LOOKBACK)
        swing_highs, swing_lows = _find_swing_levels(
            h1_slice["high"].values, h1_slice["low"].values
        )

        # ── Round numbers (25-pip density) ───────────────────────────────────
        rn_above, rn_below = _round_number_levels(price, pair)

        # ── EMAs ─────────────────────────────────────────────────────────────
        h1_ema20 = _ema(df_h1["close"], 20) if len(df_h1) >= 20 else None
        h1_ema50 = _ema(df_h1["close"], 50) if len(df_h1) >= 50 else None
        m15_ema20 = _ema(df["close"],   20) if len(df)    >= 20 else None

        # ── Daily pivot points ────────────────────────────────────────────────
        pivots = _daily_pivot_levels(df, now)

        # ── Session high/low ─────────────────────────────────────────────────
        session_lvls = _session_levels(df, now)

        # ── Structural candidates (swing + round + daily/weekly + pivots) ────────
        # Used for: hedge open decision, SL sizing, price target snapping.
        # Excludes EMAs and session micro-levels — these move too frequently to
        # define the tradeable range.
        structural_above: List[float] = []
        structural_below: List[float] = []

        structural_above.extend(h for h in swing_highs if h > price)
        structural_above.extend(rn_above)
        if day_high  > price: structural_above.append(day_high)
        if week_high > price: structural_above.append(week_high)
        for key in ("pivot_r1", "pivot_r2", "pivot"):
            v = pivots.get(key)
            if v is not None and v > price:
                structural_above.append(v)

        structural_below.extend(l for l in swing_lows if l < price)
        structural_below.extend(rn_below)
        if day_low  < price: structural_below.append(day_low)
        if week_low < price: structural_below.append(week_low)
        for key in ("pivot_s1", "pivot_s2", "pivot"):
            v = pivots.get(key)
            if v is not None and v < price:
                structural_below.append(v)

        structural_resistance = min(structural_above) if structural_above else price + 50 * pip
        structural_support    = max(structural_below) if structural_below else price - 50 * pip

        # ── Full candidates — adds EMAs + session levels for cycling ──────────
        # Used by hedge_manager for cycle trigger (nearest wall of any type).
        candidates_above: List[float] = list(structural_above)
        candidates_below: List[float] = list(structural_below)

        for ema_val in (h1_ema20, h1_ema50, m15_ema20):
            if ema_val is not None and ema_val > price:
                candidates_above.append(ema_val)
            elif ema_val is not None and ema_val < price:
                candidates_below.append(ema_val)

        for key in ("session_high", "prev_session_high"):
            v = session_lvls.get(key)
            if v is not None and v > price:
                candidates_above.append(v)

        for key in ("session_low", "prev_session_low"):
            v = session_lvls.get(key)
            if v is not None and v < price:
                candidates_below.append(v)

        # ── Nearest levels (cycling — all sources) ────────────────────────────
        nearest_resistance = min(candidates_above) if candidates_above else price + 50 * pip
        nearest_support    = max(candidates_below) if candidates_below else price - 50 * pip

        dist_resistance = _price_to_pips(nearest_resistance - price, pair)
        dist_support    = _price_to_pips(price - nearest_support,    pair)

        # Proximity score: 1.0 at 0 pips → 0 at 50+ pips (exponential decay)
        res_score = float(np.exp(-dist_resistance / 20.0))
        sup_score = float(np.exp(-dist_support    / 20.0))

        # Sorted level lists (for hedge_manager to see all walls)
        all_res = sorted(set(round(v, 6) for v in candidates_above))
        all_sup = sorted(set(round(v, 6) for v in candidates_below), reverse=True)

        struct_dist_res = _price_to_pips(structural_resistance - price, pair)
        struct_dist_sup = _price_to_pips(price - structural_support,    pair)

        result: Dict[str, Any] = {
            "pair":                   pair,
            "current_price":          round(price, 6),
            # ── Cycling walls (all levels — for hedge cycle trigger) ───────────
            "nearest_resistance":     round(nearest_resistance, 6),
            "nearest_support":        round(nearest_support, 6),
            "distance_to_resistance": round(dist_resistance, 2),
            "distance_to_support":    round(dist_support, 2),
            "resistance_score":       round(res_score, 4),
            "support_score":          round(sup_score, 4),
            "at_resistance":          1 if dist_resistance < PROXIMITY_BLOCK_PIPS else 0,
            "at_support":             1 if dist_support    < PROXIMITY_BLOCK_PIPS else 0,
            # ── Structural walls (swing/round/pivot only — for SL + open guard) ─
            "structural_resistance":             round(structural_resistance, 6),
            "structural_support":                round(structural_support, 6),
            "structural_distance_to_resistance": round(struct_dist_res, 2),
            "structural_distance_to_support":    round(struct_dist_sup, 2),
            "all_resistance_levels":  all_res[:10],   # nearest 10
            "all_support_levels":     all_sup[:10],   # nearest 10
            "week_high":              round(week_high, 6),
            "week_low":               round(week_low, 6),
            "day_high":               round(day_high, 6),
            "day_low":                round(day_low, 6),
            "level_count_above":      len(candidates_above),
            "level_count_below":      len(candidates_below),
            "computed_at":            now.isoformat(),
        }

        # Optional fields — only include when present
        if h1_ema20  is not None: result["h1_ema20"]  = round(h1_ema20,  6)
        if h1_ema50  is not None: result["h1_ema50"]  = round(h1_ema50,  6)
        if m15_ema20 is not None: result["m15_ema20"] = round(m15_ema20, 6)
        if pivots:
            result.update(pivots)
        if "session_high" in session_lvls:
            result["session_high"] = round(session_lvls["session_high"], 6)
            result["session_low"]  = round(session_lvls["session_low"],  6)
        if "prev_session_high" in session_lvls:
            result["prev_session_high"] = round(session_lvls["prev_session_high"], 6)
            result["prev_session_low"]  = round(session_lvls["prev_session_low"],  6)

        return result

    def _fetch_m15(self, pair: str, since: datetime) -> List[Tuple]:
        """Fetch M15 OHLC bars from TimescaleDB."""
        with _db_connection() as conn:
            rows = conn.execute(text("""
                SELECT time, open, high, low, close
                FROM ohlcv
                WHERE symbol = :sym
                  AND timeframe = 'M15'
                  AND time >= :since
                ORDER BY time ASC
            """), {"sym": pair, "since": since}).fetchall()
        return rows

    # ── scheduler ─────────────────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """Start background thread refreshing S/R levels every 1h."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return

        def _loop() -> None:
            logger.info("SREngine | Scheduler started (interval=1h)")
            while True:
                time.sleep(SCHEDULER_INTERVAL)
                try:
                    self.compute_all()
                except Exception as exc:
                    logger.error(f"SREngine | compute_all failed: {exc}")

        self.compute_all()
        self._scheduler_thread = threading.Thread(
            target=_loop, name="sr-scheduler", daemon=True
        )
        self._scheduler_thread.start()
        logger.success("SREngine | Scheduler started (interval=1h)")


# ── module singleton ──────────────────────────────────────────────────────────

sr_engine = SRFeatureEngine()
