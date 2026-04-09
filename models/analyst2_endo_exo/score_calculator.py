"""
models/analyst2_endo_exo/score_calculator.py
=============================================
Analyst 2: Exogenous Score Calculator

Produces a composite fundamental score (-1.0 to +1.0) for each of the
12 trading pairs, based on four exogenous driver groups drawn directly
from the course methodology:

  Driver Group 1 — Yield Differential & Momentum     (weight: 35%)
    "Direction matters more than absolute level"
    • 2Y and 10Y spread in bps (from yield_spreads table)
    • 4-week and 13-week trend of the spread
    • Z-score vs 5-day and 20-day history
    • Non-linear scoring: |spread| > 400bps → conviction drops
    • CIP deviation as carry distortion signal

  Driver Group 2 — COT Positioning & Flip            (weight: 30%)
    "Hot money follows carry — hedge funds track yield differential"
    • Net positioning % of open interest (nc_net_pct_oi)
    • Flip detection: net crossed zero in last 4 weeks
    • Commitment level: extreme longs/shorts → lower conviction
    • Week-on-week change direction

  Driver Group 3 — Carry Ranking & Rollover          (weight: 20%)
    "Positive carry = credit, negative = debit"
    • Daily rollover cost direction (long vs short)
    • Carry rank differential between base and quote
    • 10Y yield gap as primary carry proxy

  Driver Group 4 — Commodity Correlation             (weight: 15%)
    "Commodity currencies have concentrated exposure"
    Only applied to commodity-linked pairs:
    • WTI Oil  → CAD (positive), NOK
    • Gold     → AUD (positive), NZD (moderate), CHF/JPY (safe haven)
    Non-commodity pairs → this driver = 0 (neutral)

Output per pair:
  {
    "pair":        "AUDUSD",
    "score":       0.65,          # -1.0 to +1.0
    "bias":        "LONG_BASE",   # LONG_BASE | SHORT_BASE | NEUTRAL
    "confidence":  0.72,          # 0.0 to 1.0
    "drivers": {
      "yield_score":     0.71,
      "cot_score":       0.58,
      "carry_score":     0.80,
      "commodity_score": 0.60,
    },
    "signal_valid": True,         # False if data too stale
    "updated_at":  "2026-03-09T06:00:00Z"
  }

Redis cache:
  Key: features:analyst2:{PAIR}   TTL: 4 hours (14400s)

Architecture rule:
  This is Analyst 2 of the FUNDAMENTAL GROUP.
  Output feeds signal_aggregator.py gate logic ONLY.
  NEVER ensemble with Analyst 3/4 (technical) or 5/6 (sentiment).

Usage:
  from models.analyst2_endo_exo.score_calculator import analyst2
  score = analyst2.get_score("AUDUSD")
  all_scores = analyst2.compute_all()
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis
from loguru import logger
from sqlalchemy import text

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

from data_pipeline.storage.timeseries_db import db  # noqa: E402
from models.analyst2_endo_exo.macro_scorer import macro_scorer as _macro_scorer

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

# Base / quote currency for each pair
PAIR_CURRENCIES: Dict[str, Tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"), "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
    "AUDJPY": ("AUD", "JPY"), "NZDJPY": ("NZD", "JPY"),
}

# Carry rank (higher = higher yielding) — updated from live yields 2026-03-08
# AUD(1) > NZD(2) > GBP(3) > USD(4) > CAD(5) > EUR(6) > JPY(7) > CHF(8)
CARRY_RANK: Dict[str, int] = {
    "AUD": 1, "NZD": 2, "GBP": 3, "USD": 4,
    "CAD": 5, "EUR": 6, "JPY": 7, "CHF": 8,
}

# Commodity links: currency → (commodity, direction)
# direction: +1 = currency rises when commodity rises
#            -1 = currency falls when commodity rises
COMMODITY_LINKS: Dict[str, Tuple[str, float]] = {
    "CAD": ("OIL", +1.0),   # Canada major oil exporter
    "AUD": ("XAU", +1.0),   # Australia major gold exporter
    "NZD": ("XAU", +0.5),   # NZD moderate gold correlation
    "CHF": ("XAU", -0.3),   # CHF safe haven (inverse to risk-off gold)
    "JPY": ("XAU", -0.3),   # JPY safe haven (inverse to risk-off gold)
}

# Driver weights — must sum to 1.0
WEIGHTS = {
    "yield":     0.30,
    "cot":       0.25,
    "carry":     0.15,
    "commodity": 0.10,
    "endo":      0.20,
}

# Thresholds
YIELD_EXTREME_BPS      = 400.0   # Above this → conviction drops (already priced in)
COT_EXTREME_PCT        = 35.0    # Above this → commitment extreme, lower conviction
CACHE_TTL              = 14400   # 4 hours in seconds
STALE_DATA_HOURS       = 48      # Data older than this → signal_valid = False
SCHEDULER_INTERVAL_SEC = 14400   # Recompute every 4 hours


# ══════════════════════════════════════════════════════════════
# REDIS CONNECTION
# ══════════════════════════════════════════════════════════════

def _get_redis() -> redis.Redis:
    from dotenv import load_dotenv
    load_dotenv(str(ROOT / ".env"))
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


# ══════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════

def _load_yield_features(pair: str) -> Dict[str, float]:
    """
    Load yield features for a pair from Redis (populated by yield_feature_engine).

    Keys used (from features/yield_features.py — 30 features):
      yield_diff_2Y, yield_diff_10Y       — spread in percentage points
      yield_score_2Y, yield_score_10Y     — pre-scaled scores (-1 to +1)
      yield_trend_4w_2Y, yield_trend_4w_10Y   — 4-week momentum
      yield_trend_13w_2Y, yield_trend_13w_10Y — 13-week momentum
      yield_zscore_2Y, yield_zscore_10Y   — z-scores
      carry_net                            — net carry rank
      rollover_daily_usd                   — daily rollover in USD
      cip_deviation                        — CIP deviation
      curve_slope_diff                     — curve shape differential
    """
    try:
        r = _get_redis()
        cached = r.get(f"features:yield:{pair}")
        if cached:
            return json.loads(cached)
    except Exception as exc:
        logger.warning(f"Analyst2 | Redis yield read failed for {pair}: {exc}")

    # Cache miss — trigger compute
    try:
        from features.yield_features import yield_feature_engine
        feats = yield_feature_engine.get_features(pair)
        return feats if feats else {}
    except Exception as exc:
        logger.warning(f"Analyst2 | yield_feature_engine fallback failed: {exc}")
        return {}


def _load_yield_history(currency: str, tenor: str, days: int = 100) -> pd.DataFrame:
    """Load raw yield history for momentum calculation."""
    sql = """
        SELECT date, yield_pct
        FROM yield_data
        WHERE currency = :currency
          AND tenor = :tenor
          AND date >= NOW() - INTERVAL ':days days'
        ORDER BY date DESC
        LIMIT :days
    """
    with db.connection() as conn:
        rows = conn.execute(
            text("SELECT date, yield_pct FROM yield_data "
                 "WHERE currency = :currency AND tenor = :tenor "
                 "AND date >= NOW() - INTERVAL '100 days' "
                 "ORDER BY date DESC LIMIT :days"),
            {"currency": currency, "tenor": tenor, "days": days}
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "yield_pct"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def _load_cot(currency: str, weeks: int = 26) -> pd.DataFrame:
    """Load COT report history for a currency."""
    sql = """
        SELECT report_date, nc_net, nc_net_pct_oi, nc_net_change,
               nc_long, nc_short, open_interest
        FROM cot_reports
        WHERE currency = :currency
          AND report_date >= NOW() - INTERVAL ':weeks weeks'
        ORDER BY report_date DESC
        LIMIT :weeks
    """
    with db.connection() as conn:
        rows = conn.execute(
            text("SELECT report_date, nc_net, nc_net_pct_oi, nc_net_change, "
                 "nc_long, nc_short, open_interest "
                 "FROM cot_reports "
                 "WHERE currency = :currency "
                 "AND report_date >= NOW() - INTERVAL '200 days' "
                 "ORDER BY report_date DESC LIMIT :weeks"),
            {"currency": currency, "weeks": weeks}
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["report_date", "nc_net", "nc_net_pct_oi",
                                      "nc_net_change", "nc_long", "nc_short",
                                      "open_interest"])
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.sort_values("report_date")


def _load_commodity_prices(commodity: str, days: int = 90) -> pd.DataFrame:
    """Load commodity price history from macro_indicators."""
    indicator = "WTI_CLOSE" if commodity == "OIL" else "GOLD_CLOSE"
    with db.connection() as conn:
        rows = conn.execute(
            text("SELECT time, actual FROM macro_indicators "
                 "WHERE country = :country AND indicator = :indicator "
                 "AND time >= NOW() - INTERVAL '90 days' "
                 "ORDER BY time DESC LIMIT 90"),
            {"country": commodity, "indicator": indicator}
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time", "price"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time")


# ══════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _nonlinear_scale(value: float, extreme_threshold: float) -> float:
    """
    Course principle: "Extreme differentials = lower conviction — already priced in"
    Maps |value| to conviction score:
      0       → 0.0  (no differential)
      moderate→ 1.0  (high conviction)
      extreme → 0.6  (already priced in, lower conviction)
    Returns signed score: positive = bullish base, negative = bearish base.
    """
    sign = np.sign(value)
    abs_val = abs(value)

    if abs_val < 1e-9:
        return 0.0

    # Ramp up to 80% of threshold → max conviction 1.0
    ramp_top = extreme_threshold * 0.80
    if abs_val <= ramp_top:
        raw = abs_val / ramp_top
    else:
        # Step down from 1.0 to 0.6 as we go further extreme
        excess = (abs_val - ramp_top) / (extreme_threshold * 0.5)
        raw = 1.0 - (0.4 * min(excess, 1.0))

    return float(sign * np.clip(raw, -1.0, 1.0))


def _score_yield(pair: str, base: str, quote: str) -> Tuple[float, bool]:
    """
    Driver 1: Yield Differential & Momentum

    Reads from Redis (populated by yield_feature_engine.compute_all()).
    Falls back to yield_feature_engine.get_features() on cache miss.

    Redis key: features:yield:{pair}
    All 30 features pre-computed by features/yield_features.py.

    Combines:
      a) Pre-scaled yield scores for 2Y and 10Y (yield_score_2Y / _10Y)
      b) Raw differential for non-linear re-scoring
      c) 4-week and 13-week trend momentum
      d) Z-scores (yield_zscore_2Y / _10Y)
      e) CIP deviation and curve slope differential

    Returns (score -1 to +1, data_valid bool)
    Positive score = bullish base currency.
    """
    feats = _load_yield_features(pair)

    if not feats:
        logger.warning(f"Analyst2 | {pair}: No yield features in Redis — run yield_feature_engine.compute_all() first")
        return 0.0, False

    # ── 2Y component (40% weight) ─────────────────────────────
    diff_2y       = feats.get("yield_diff_2Y", 0.0)       # pct points
    score_2y_raw  = feats.get("yield_score_2Y", 0.0)      # pre-scaled [-1,1]
    trend_4w_2y   = feats.get("yield_trend_4w_2Y", 0.0)   # pct change
    trend_13w_2y  = feats.get("yield_trend_13w_2Y", 0.0)
    zscore_2y     = feats.get("yield_zscore_2Y", 0.0)

    # Non-linear re-score on raw diff (convert pct to bps)
    nonlin_2y    = _nonlinear_scale(diff_2y * 100.0, YIELD_EXTREME_BPS)
    combined_2y  = score_2y_raw * 0.50 + nonlin_2y * 0.50

    # Momentum: direction of trend, not magnitude

    mom_2y = (
        np.sign(trend_4w_2y)  * 0.60 +
        np.sign(trend_13w_2y) * 0.40
    ) * 0.40

    # Course principle: "Direction matters more than absolute level"
    # When ALL trend signals oppose the absolute spread → compress conviction
    # This prevents a wide-but-compressing spread from generating false confidence
    trend_4w_2y_sig  = trend_4w_2y  if abs(trend_4w_2y)  > 0.02 else 0.0
    trend_13w_2y_sig = trend_13w_2y if abs(trend_13w_2y) > 0.02 else 0.0
    _2y_level = np.sign(combined_2y)
    _2y_trends_sig = [trend_4w_2y_sig, trend_13w_2y_sig]
    _2y_opposing = sum(1 for t in _2y_trends_sig
                       if t != 0 and np.sign(t) != _2y_level)
    _2y_nonzero  = sum(1 for t in _2y_trends_sig if t != 0)
    if abs(combined_2y) > 0.25 and _2y_nonzero > 0 and _2y_opposing >= _2y_nonzero * 0.5:
        combined_2y *= 0.35
    z2y_score  = np.clip(zscore_2y / 3.0, -0.30, 0.30)
    tenor_2y   = float(np.clip(combined_2y * 0.60 + mom_2y * 0.35 + z2y_score * 0.15,
                               -1.0, 1.0))
    # ── 10Y component (60% weight) ────────────────────────────
    diff_10y      = feats.get("yield_diff_10Y", 0.0)
    score_10y_raw = feats.get("yield_score_10Y", 0.0)
    trend_4w_10y  = feats.get("yield_trend_4w_10Y", 0.0)
    trend_13w_10y = feats.get("yield_trend_13w_10Y", 0.0)
    zscore_10y    = feats.get("yield_zscore_10Y", 0.0)

    nonlin_10y   = _nonlinear_scale(diff_10y * 100.0, YIELD_EXTREME_BPS)
    combined_10y = score_10y_raw * 0.50 + nonlin_10y * 0.50

    mom_10y = (
        np.sign(trend_4w_10y)  * 0.60 +
        np.sign(trend_13w_10y) * 0.40
    ) * 0.40

    # Compression override: 10Y is primary carry tenor — even more important
    # that compressing spreads reduce conviction
    # Dead zone: treat |trend| < 0.02 as noise, not signal
    trend_4w_10y_sig  = trend_4w_10y  if abs(trend_4w_10y)  > 0.02 else 0.0
    trend_13w_10y_sig = trend_13w_10y if abs(trend_13w_10y) > 0.02 else 0.0
    _10y_level = np.sign(combined_10y)

    # Count how many non-noise trends oppose the absolute level
    _10y_trends_sig = [trend_4w_10y_sig, trend_13w_10y_sig]
    _opposing = sum(1 for t in _10y_trends_sig
                    if t != 0 and np.sign(t) != _10y_level)
    _nonzero  = sum(1 for t in _10y_trends_sig if t != 0)

    _momentum_10y = feats.get("yield_momentum_10Y", 0.0)
    _momentum_agrees_compression = (
        abs(_momentum_10y) > 0.02 and
        np.sign(_momentum_10y) != _10y_level
    )

    # Compress when MAJORITY of non-noise trends oppose the level
    # NZDJPY example: 4w=+0.005(noise→0), 13w=-0.130(opposes) → 1/1 = 100% opposing
    if abs(combined_10y) > 0.25 and _nonzero > 0 and _opposing >= _nonzero * 0.5:
        compression = 0.30 if _momentum_agrees_compression else 0.45
        combined_10y *= compression
        logger.debug(
            f"  yield compress 10Y: {_opposing}/{_nonzero} trends oppose "
            f"level={_10y_level:+.0f} → x{compression}"
        )
    z10y_score  = np.clip(zscore_10y / 3.0, -0.30, 0.30)
    tenor_10y   = float(np.clip(combined_10y * 0.60 + mom_10y * 0.35 + z10y_score * 0.15,
                                -1.0, 1.0))

    # ── Auxiliary signals ─────────────────────────────────────
    # CIP deviation: negative CIP for base = carry is cheaper than it looks
    cip_dev     = feats.get("cip_deviation", 0.0)
    cip_score   = np.clip(cip_dev * 2.0, -0.30, 0.30)

    # Curve slope differential: steeper base curve = growth expectations higher
    slope_diff  = feats.get("curve_slope_diff", 0.0)
    slope_score = np.clip(slope_diff * 2.0, -0.20, 0.20)

    # ── Real rate differential (40% blend when available) ─────────────────────
    # Real rate = nominal yield − inflation. Capital responds to real returns,
    # not nominal. 5% yield with 4% inflation = 1% real return.
    real_diff_10y  = feats.get("real_rate_diff_10Y")
    real_score_10y = feats.get("real_rate_score_10Y")
    real_diff_2y   = feats.get("real_rate_diff_2Y")
    real_score_2y  = feats.get("real_rate_score_2Y")
    real_premium   = feats.get("real_rate_premium", 0.0)
    real_momentum  = feats.get("real_rate_momentum_10Y", 0.0)

    if real_diff_10y is not None and real_score_10y is not None:
        nonlin_real10 = _nonlinear_scale(real_diff_10y * 100.0, YIELD_EXTREME_BPS)
        real_10y_base = float(np.clip(
            real_score_10y * 0.50 + nonlin_real10 * 0.50 +
            np.sign(real_momentum) * 0.40 * 0.40,
            -1.0, 1.0
        ))
        if real_diff_2y is not None and real_score_2y is not None:
            nonlin_real2  = _nonlinear_scale(real_diff_2y * 100.0, YIELD_EXTREME_BPS)
            real_2y_base  = float(np.clip(
                real_score_2y * 0.50 + nonlin_real2 * 0.50, -1.0, 1.0
            ))
            real_rate_score = real_2y_base * 0.40 + real_10y_base * 0.60
        else:
            real_rate_score = real_10y_base

        nominal_yield_score = tenor_2y * 0.40 + tenor_10y * 0.60
        yield_score = nominal_yield_score * 0.60 + real_rate_score * 0.40

        # Conviction cut when real and nominal signals diverge materially
        if abs(real_premium) > 0.5 and np.sign(real_rate_score) != np.sign(nominal_yield_score):
            yield_score *= 0.65
            logger.debug(
                f"Analyst2 | {pair}: real/nominal diverge "
                f"(premium={real_premium:+.2f}) → conviction −35%"
            )
    else:
        yield_score = tenor_2y * 0.40 + tenor_10y * 0.60

    yield_score += cip_score * 0.05 + slope_score * 0.05
    yield_score  = float(np.clip(yield_score, -1.0, 1.0))

    data_valid = True   # Redis data freshness guaranteed by 4h TTL
    return yield_score, data_valid


def _score_cot(base: str, quote: str) -> Tuple[float, bool]:
    """
    Driver 2: COT Positioning & Flip Detection

    Course principle: COT tells us:
      1) Direction the wind is blowing (net long/short)
      2) Flip: when hedge funds switch from net long→short or vice versa
      3) Commitment level: extremes signal caution

    Net positive base COT + negative quote COT → bullish base.
    Flip in last 4 weeks for base → strong directional signal.
    Extreme commitment (|pct_oi| > 35%) → reduce score by 30%.

    Returns (score -1 to +1, data_valid bool)
    """
    base_cot  = _load_cot(base)
    quote_cot = _load_cot(quote)

    if base_cot.empty and quote_cot.empty:
        return 0.0, False

    def _currency_cot_score(cot_df: pd.DataFrame) -> float:
        if cot_df.empty:
            return 0.0

        latest     = cot_df.iloc[-1]
        pct_oi     = latest["nc_net_pct_oi"] or 0.0
        net_change = latest["nc_net_change"] or 0

        # Base direction score from net % OI
        direction = _nonlinear_scale(pct_oi, COT_EXTREME_PCT)

        # Flip detection: check if net crossed zero in last 4 rows
        flip_bonus = 0.0
        if len(cot_df) >= 4:
            recent_nets = cot_df["nc_net"].tail(4).values
            signs = np.sign(recent_nets)
            non_zero_signs = signs[signs != 0]
            if len(non_zero_signs) >= 2:
                if non_zero_signs[-1] != non_zero_signs[0]:
                    # Flip occurred — adds 20% to direction conviction
                    flip_bonus = 0.2 * np.sign(pct_oi)

        # Momentum: week-on-week direction of change
        change_score = np.clip(net_change / 10000.0, -0.2, 0.2)

        # Extreme commitment penalty
        abs_pct = abs(pct_oi)
        if abs_pct > COT_EXTREME_PCT:
            extreme_penalty = 1.0 - ((abs_pct - COT_EXTREME_PCT) / COT_EXTREME_PCT * 0.3)
            direction *= max(extreme_penalty, 0.5)

        score = direction + flip_bonus + change_score
        return float(np.clip(score, -1.0, 1.0))

    base_score  = _currency_cot_score(base_cot)
    quote_score = _currency_cot_score(quote_cot)

    # Pair score: base bullish - quote bullish
    pair_score = np.clip((base_score - quote_score) / 2.0, -1.0, 1.0)

    # Check data freshness (COT is weekly — allow up to 14 days)
    data_valid = True
    for cot_df in [base_cot, quote_cot]:
        if not cot_df.empty:
            latest_date = pd.to_datetime(cot_df["report_date"].max())
            age_days = (datetime.now(timezone.utc).date() - latest_date.date()).days
            if age_days > 14:
                data_valid = False

    return float(pair_score), data_valid


def _score_carry(base: str, quote: str) -> Tuple[float, bool]:
    """
    Driver 3: Carry Ranking & Rollover Direction

    Course principle: "Hot money follows carry"
    • Higher yielding currency attracts institutional flows
    • Carry rank differential drives position bias
    • 10Y yield gap is the primary carry proxy at macro horizon

    Returns (score -1 to +1, data_valid bool)
    Positive = long base is positive carry (receive rollover).
    """
    # Static carry rank component
    base_rank  = CARRY_RANK.get(base, 4)
    quote_rank = CARRY_RANK.get(quote, 4)
    # Lower rank = higher yield. Score: positive if base higher yielding.
    rank_diff  = quote_rank - base_rank   # positive = base higher yielding
    rank_score = np.clip(rank_diff / 7.0, -1.0, 1.0)
    # Dynamic 10Y yield differential from DB
    base_yield_df  = _load_yield_history(base,  "10Y", days=30)
    quote_yield_df = _load_yield_history(quote, "10Y", days=30)

    data_valid = not base_yield_df.empty and not quote_yield_df.empty

    if data_valid:
        base_y  = base_yield_df["yield_pct"].iloc[-1]
        quote_y = quote_yield_df["yield_pct"].iloc[-1]
        yield_gap = base_y - quote_y   # positive = base higher yield

        # Non-linear scoring — extreme gaps already priced in
        dynamic_score = _nonlinear_scale(yield_gap * 100, YIELD_EXTREME_BPS)
    else:
        dynamic_score = 0.0

    # Blend static rank + dynamic yield gap
    # Course principle: compressing carry differential = deteriorating trade
    # Check if the yield gap itself is trending against current level
    if data_valid:
        try:
            base_hist  = _load_yield_history(base,  "10Y", days=100)
            quote_hist = _load_yield_history(quote, "10Y", days=100)

            if len(base_hist) >= 2 and len(quote_hist) >= 2:
                # Date-based lookback — works for both daily AND monthly data
                # Monthly OECD series only has ~3 rows per 100 days
                # so iloc[-20] would go back 20 months, not 4 weeks
                def _get_nearest(df: pd.DataFrame, target_date) -> float:
                    idx = (df["date"] - target_date).abs().idxmin()
                    return df.loc[idx, "yield_pct"]

                now    = base_hist["date"].max()
                t_4w   = now - pd.Timedelta(weeks=4)
                t_13w  = now - pd.Timedelta(weeks=13)

                b_now  = base_hist["yield_pct"].iloc[-1]
                q_now  = quote_hist["yield_pct"].iloc[-1]
                b_4w   = _get_nearest(base_hist,  t_4w)
                q_4w   = _get_nearest(quote_hist, t_4w)
                b_13w  = _get_nearest(base_hist,  t_13w)
                q_13w  = _get_nearest(quote_hist, t_13w)
                diff_now = b_now  - q_now
                diff_4w  = b_4w   - q_4w
                diff_13w = b_13w  - q_13w

                # Positive = spread widening (carry improving)
                # Negative = spread compressing (carry deteriorating)
                trend_4w  = diff_now - diff_4w
                trend_13w = diff_now - diff_13w

                t4_sig  = trend_4w  if abs(trend_4w)  > 0.02 else 0.0
                t13_sig = trend_13w if abs(trend_13w) > 0.02 else 0.0

                _level    = np.sign(dynamic_score)
                _trends   = [t4_sig, t13_sig]
                _opposing = sum(1 for t in _trends if t != 0 and np.sign(t) != _level)
                _nonzero  = sum(1 for t in _trends if t != 0)

                if abs(dynamic_score) > 0.25 and _nonzero > 0 and _opposing >= _nonzero * 0.5:
                    dynamic_score *= 0.40
                    logger.debug(
                        f"  carry compress {base}/{quote}: "
                        f"4w={trend_4w:+.3f} 13w={trend_13w:+.3f} → x0.40"
                    )
        except Exception as _e:
            logger.debug(f"  carry trend check failed: {_e}")

    # Blend: static rank (structural) + dynamic yield gap (current reality)
    carry_score = rank_score * 0.40 + dynamic_score * 0.60
    return float(np.clip(carry_score, -1.0, 1.0)), data_valid

def _score_commodity(base: str, quote: str) -> Tuple[float, bool]:
    """
    Driver 4: Commodity Correlation

    Only applies to commodity-linked currencies (CAD, AUD, NZD, CHF, JPY).
    Non-commodity currencies → returns 0.0 (neutral, no penalty).

    Measures:
      • 4-week momentum of commodity price
      • 13-week trend of commodity price
      • Recent acceleration (1-week vs 4-week rate of change)

    Returns (score -1 to +1, data_valid bool)
    """
    base_link  = COMMODITY_LINKS.get(base)
    quote_link = COMMODITY_LINKS.get(quote)

    # If neither currency has commodity link → neutral
    if base_link is None and quote_link is None:
        return 0.0, True   # valid but neutral

    def _comm_score(link: Optional[Tuple[str, float]]) -> float:
        if link is None:
            return 0.0
        commodity, direction = link

        comm_df = _load_commodity_prices(commodity)
        if comm_df.empty or len(comm_df) < 5:
            return 0.0

        prices = comm_df["price"].values

        # 4-week momentum (20 trading days)
        if len(prices) >= 20:
            mom_4w = (prices[-1] - prices[-20]) / prices[-20]
        elif len(prices) >= 5:
            mom_4w = (prices[-1] - prices[0]) / prices[0]
        else:
            mom_4w = 0.0

        # 13-week momentum (65 trading days)
        if len(prices) >= 65:
            mom_13w = (prices[-1] - prices[-65]) / prices[-65]
        else:
            mom_13w = mom_4w

        # 1-week acceleration
        if len(prices) >= 5:
            mom_1w = (prices[-1] - prices[-5]) / prices[-5]
        else:
            mom_1w = 0.0

        # Combine momentum timeframes
        raw_score = (
            np.sign(mom_4w)  * 0.50 +
            np.sign(mom_13w) * 0.30 +
            np.sign(mom_1w)  * 0.20
        )

        # Scale by magnitude (stronger momentum = higher conviction)
        magnitude = abs(mom_4w) * 10.0   # 10% move → full conviction
        scaled    = raw_score * min(magnitude, 1.0)

        return float(np.clip(scaled * direction, -1.0, 1.0))

    base_comm_score  = _comm_score(base_link)
    quote_comm_score = _comm_score(quote_link)

    # Pair score: base commodity effect - quote commodity effect
    pair_score = np.clip((base_comm_score - quote_comm_score) / 2.0, -1.0, 1.0)

    # Data validity: at least one commodity has data
    data_valid = True
    for link in [base_link, quote_link]:
        if link is not None:
            comm_df = _load_commodity_prices(link[0])
            if comm_df.empty:
                data_valid = False

    return float(pair_score), data_valid

def _score_endo(base: str, quote: str) -> Tuple[float, bool]:
    """
    Driver 5: Endogenous policy direction differential.

    Course principle: A CB hiking into strength is fundamentally different
    from a CB cutting into weakness — even at the same yield level.
    RBNZ cutting + BoJ hiking = bearish NZD/JPY regardless of absolute spread.

    Returns (score -1 to +1, data_valid bool)
    Positive = base currency more hawkish than quote.
    """
    try:
        base_data  = _macro_scorer.get_score(base)
        quote_data = _macro_scorer.get_score(quote)

        base_endo  = base_data.get("endo_score", 0.0)
        quote_endo = quote_data.get("endo_score", 0.0)
        base_conv  = base_data.get("conviction", 0.0)
        quote_conv = quote_data.get("conviction", 0.0)

        # Pair score: base hawkish vs quote hawkish
        pair_score = float(np.clip((base_endo - quote_endo) / 2.0, -1.0, 1.0))

        # Data valid if at least one currency has real conviction
        data_valid = (base_conv > 0.05 or quote_conv > 0.05)

        logger.debug(
            f"  endo {base}/{quote}: {base_endo:+.3f} vs {quote_endo:+.3f} "
            f"→ pair={pair_score:+.3f} valid={data_valid}"
        )
        return pair_score, data_valid

    except Exception as exc:
        logger.warning(f"Analyst2 | endo score failed {base}/{quote}: {exc}")
        return 0.0, False

# ══════════════════════════════════════════════════════════════
# MAIN CALCULATOR CLASS
# ══════════════════════════════════════════════════════════════

class EndoExoScoreCalculator:
    """
    Analyst 2: Exogenous Score Calculator.

    Computes a composite fundamental score per pair using 4 driver groups.
    Results cached in Redis for 4 hours with background refresh scheduler.

    The score gates entry in signal_aggregator.py:
      score > +0.20  → LONG_BASE signal valid for Analysts 3+4 to time
      score < -0.20  → SHORT_BASE signal valid for Analysts 3+4 to time
      |score| < 0.20 → NEUTRAL — no fundamental predisposition, skip trade
    """

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._lock  = threading.Lock()

    @property
    def r(self) -> redis.Redis:
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    # ── public API ────────────────────────────────────────────

    def get_score(self, pair: str) -> Dict:
        """
        Get exogenous score for a pair. Returns cached value if available,
        else computes fresh.

        Args:
            pair: e.g. "AUDUSD"

        Returns:
            Score dict with keys: pair, score, bias, confidence,
            drivers, signal_valid, updated_at
        """
        cache_key = f"features:analyst2:{pair}"
        try:
            cached = self.r.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as exc:
            logger.warning(f"Analyst2 | Redis read failed: {exc}")

        # Cache miss — compute fresh
        result = self._compute_pair(pair)
        self._cache_result(cache_key, result)
        return result

    def compute_all(self) -> Dict[str, Dict]:
        """
        Compute scores for all 12 pairs. Stores results in Redis.
        Called by background scheduler every 4 hours.

        Returns:
            {pair: score_dict} for all pairs
        """
        logger.info("Analyst2 | Computing scores for all 12 pairs...")
        results: Dict[str, Dict] = {}

        for pair in PAIRS:
            try:
                result = self._compute_pair(pair)
                cache_key = f"features:analyst2:{pair}"
                self._cache_result(cache_key, result)
                results[pair] = result
                logger.info(
                    f"Analyst2 | {pair}: score={result['score']:+.3f} "
                    f"bias={result['bias']} conf={result['confidence']:.2f}"
                )
            except Exception as exc:
                logger.error(f"Analyst2 | {pair} failed: {exc}")
                results[pair] = self._neutral_result(pair)

        # Summary
        long_pairs  = [p for p, r in results.items() if r["bias"] == "LONG_BASE"]
        short_pairs = [p for p, r in results.items() if r["bias"] == "SHORT_BASE"]
        logger.success(
            f"Analyst2 | Done — "
            f"LONG: {long_pairs} | SHORT: {short_pairs}"
        )
        return results

    def start_scheduler(self) -> None:
        """
        Start background daemon thread that recomputes scores every 4 hours.
        Call once at bot startup.
        """
        def _loop() -> None:
            import time
            logger.info("Analyst2 | Background scheduler started (4h interval)")
            while True:
                try:
                    self.compute_all()
                except Exception as exc:
                    logger.error(f"Analyst2 | Scheduler error: {exc}")
                time.sleep(SCHEDULER_INTERVAL_SEC)

        thread = threading.Thread(target=_loop, daemon=True, name="analyst2-scheduler")
        thread.start()
        logger.info("Analyst2 | Scheduler thread launched")

    # ── internal computation ──────────────────────────────────

    def _compute_pair(self, pair: str) -> Dict:
        """
        Compute full exogenous score for one pair.
        Combines 4 driver groups with weighted blend.
        """
        if pair not in PAIR_CURRENCIES:
            logger.error(f"Analyst2 | Unknown pair: {pair}")
            return self._neutral_result(pair)

        base, quote = PAIR_CURRENCIES[pair]
        logger.debug(f"Analyst2 | Computing {pair} ({base}/{quote})...")

        # ── Driver 1: Yield ───────────────────────────────────
        yield_score, yield_valid = _score_yield(pair, base, quote)

        # ── Driver 2: COT ─────────────────────────────────────
        cot_score, cot_valid = _score_cot(base, quote)

        # ── Driver 3: Carry ───────────────────────────────────
        carry_score, carry_valid = _score_carry(base, quote)

        # ── Driver 4: Commodity ───────────────────────────────
        commodity_score, comm_valid = _score_commodity(base, quote)

        # ── Driver 5: Endogenous policy direction ─────────────  ← ADD THIS
        endo_score, endo_valid = _score_endo(base, quote)

        # ── Weighted composite ────────────────────────────────
        # Adjust weights if data missing
        active_weights = {
            "yield":     WEIGHTS["yield"]     if yield_valid    else 0.0,
            "cot":       WEIGHTS["cot"]       if cot_valid      else 0.0,
            "carry":     WEIGHTS["carry"]     if carry_valid    else 0.0,
            "commodity": WEIGHTS["commodity"] if comm_valid     else 0.0,
            "endo":      WEIGHTS["endo"]      if endo_valid     else 0.0,
        }
        total_weight = sum(active_weights.values())

        if total_weight < 0.01:
            return self._neutral_result(pair)

        # Renormalise weights
        norm_weights = {k: v / total_weight for k, v in active_weights.items()}

        composite = (
            yield_score     * norm_weights["yield"]     +
            cot_score       * norm_weights["cot"]       +
            carry_score     * norm_weights["carry"]     +
            commodity_score * norm_weights["commodity"] +
            endo_score      * norm_weights.get("endo", 0.0)
        )
        composite = float(np.clip(composite, -1.0, 1.0))

        # ── Confidence ────────────────────────────────────────
        # How many drivers agree on direction?
        driver_scores = [yield_score, cot_score, carry_score, commodity_score, endo_score]
        valid_scores  = [s for s, v in zip(driver_scores,
                         [yield_valid, cot_valid, carry_valid, comm_valid, endo_valid]) if v]

        if valid_scores:
            agreement = sum(1 for s in valid_scores if np.sign(s) == np.sign(composite))
            confidence = agreement / len(valid_scores)
            # Scale by |composite| — strong score + high agreement = high confidence
            confidence = confidence * min(abs(composite) * 2.0, 1.0)
        else:
            confidence = 0.0

        # ── Bias classification ───────────────────────────────
        if composite > 0.20:
            bias = "LONG_BASE"
        elif composite < -0.20:
            bias = "SHORT_BASE"
        else:
            bias = "NEUTRAL"

        signal_valid = yield_valid or cot_valid or carry_valid

        return {
            "pair":        pair,
            "score":       round(composite, 4),
            "bias":        bias,
            "confidence":  round(confidence, 4),
            "drivers": {
                "yield_score":     round(yield_score, 4),
                "cot_score":       round(cot_score, 4),
                "carry_score":     round(carry_score, 4),
                "commodity_score": round(commodity_score, 4),
                "endo_score":      round(endo_score, 4),
            },
            "weights_used": {k: round(v, 3) for k, v in norm_weights.items()},
            "signal_valid": signal_valid,
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        }

    def _cache_result(self, key: str, result: Dict) -> None:
        """Store result in Redis with 4h TTL."""
        try:
            self.r.setex(key, CACHE_TTL, json.dumps(result))
        except Exception as exc:
            logger.warning(f"Analyst2 | Redis write failed: {exc}")

    @staticmethod
    def _neutral_result(pair: str) -> Dict:
        """Return a safe neutral result when data is unavailable."""
        return {
            "pair":        pair,
            "score":       0.0,
            "bias":        "NEUTRAL",
            "confidence":  0.0,
            "drivers": {
                "yield_score":     0.0,
                "cot_score":       0.0,
                "carry_score":     0.0,
                "commodity_score": 0.0,
            },
            "weights_used":  {},
            "signal_valid":  False,
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }

    def print_summary(self) -> None:
        """
        Print a formatted summary table of all pair scores.
        Useful for manual inspection at session start.
        """
        print("\n" + "=" * 70)
        print("ANALYST 2 — EXOGENOUS SCORE SUMMARY")
        print("=" * 70)
        print(f"{'PAIR':<10} {'SCORE':>7} {'BIAS':<12} {'CONF':>6} "
              f"{'YIELD':>7} {'COT':>7} {'CARRY':>7} {'COMM':>7}")
        print("-" * 70)

        for pair in PAIRS:
            r = self.get_score(pair)
            d = r.get("drivers", {})
            print(
                f"{pair:<10} "
                f"{r['score']:>+7.3f} "
                f"{r['bias']:<12} "
                f"{r['confidence']:>6.2f} "
                f"{d.get('yield_score', 0):>+7.3f} "
                f"{d.get('cot_score', 0):>+7.3f} "
                f"{d.get('carry_score', 0):>+7.3f} "
                f"{d.get('commodity_score', 0):>+7.3f}"
            )
        print("=" * 70)
        print("Bias gate: |score| > 0.20 → tradeable signal")
        print("=" * 70 + "\n")


# ── Module singleton ──────────────────────────────────────────
analyst2 = EndoExoScoreCalculator()


# ── Direct run ────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")

    logger.info("Analyst 2 | Computing all pair scores...")
    analyst2.compute_all()
    analyst2.print_summary()
