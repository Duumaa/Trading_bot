"""
features/yield_features.py
===========================
Phase 3 Step 1 — Yield & Carry Feature Engineering

Transforms raw yield data (from yield_scraper) into ML-ready features.
Pre-computed on schedule, cached in Redis (TTL 4h), read instantly at signal time.

Course principles implemented:
    1. "It's not just the absolute level — it's the DIRECTION moving in"
       → 4w and 13w differential change features
    2. "When differential widens to extreme levels, conviction becomes lower"
       → Non-linear scoring that reduces at extremes
    3. "Hot money follows carry — speculative money from hedge funds"
       → Carry attractiveness ranking feature
    4. "Positive carry = credit, negative carry = debit"
       → Daily rollover cost feature (affects trade viability)

Features produced (per pair):
    yield_diff_2Y_{pair}        2Y yield differential (rate expectations)
    yield_diff_10Y_{pair}       10Y yield differential (growth expectations)
    yield_diff_score_{pair}     Normalized score -1 to +1
    yield_trend_4w_{pair}       4-week change in differential
    yield_trend_13w_{pair}      13-week change in differential
    yield_momentum_{pair}       Trend direction: +1 widening, -1 narrowing
    curve_slope_{ccy}           10Y - 2Y per currency (recession indicator)
    curve_inverted_{ccy}        1 if inverted (risk-off signal)
    carry_rank_{ccy}            Rank 1-10 by yield (1=highest carry)
    carry_score_{ccy}           Normalized carry score -1 to +1
    rollover_daily_{pair}       Estimated daily rollover pct (annualized/365)
    cip_deviation_{pair}        Covered Interest Parity deviation (arbitrage signal)

Redis keys:
    features:yield:{pair}       Dict of yield features for that pair
    features:yield:summary      All pairs combined
    features:carry:ranking      Sorted carry ranking list
TTL: 4 hours (re-computed by scheduler)

Usage:
    from features.yield_features import yield_feature_engine

    # Get all yield features for a pair
    feats = yield_feature_engine.get_features("EURUSD")

    # Get carry ranking
    ranking = yield_feature_engine.get_carry_ranking()

    # Start background scheduler (every 4h)
    yield_feature_engine.start_scheduler()
"""

from __future__ import annotations

import json
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

UTC = timezone.utc

# ── Constants ──────────────────────────────────────────────────────────────────

# Contract size for rollover calculation (1 standard lot)
LOT_SIZE = 100_000

# Redis TTL for cached features
CACHE_TTL_S = 14_400   # 4 hours

# G10 pairs with base/quote
G10_PAIRS: Dict[str, tuple] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"),
    "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"),
    "GBPJPY": ("GBP", "JPY"),
    "AUDJPY": ("AUD", "JPY"),
    "NZDJPY": ("NZD", "JPY"),
}

ALL_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]

# Extreme differential threshold — course says conviction drops above this
EXTREME_DIFF_THRESHOLD = 4.0   # %


# ── Main feature engine ────────────────────────────────────────────────────────

class YieldFeatureEngine:
    """
    Computes yield/carry features from TimescaleDB yield_data.
    Caches results to Redis for instant retrieval by signal aggregator.

    Schedule: compute every 4 hours (yields change slowly, daily is enough
    for regime features but 4h keeps features fresh after intraday CB moves).
    """

    def __init__(self) -> None:
        self._last_compute: Optional[datetime] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def get_features(self, pair: str) -> Dict[str, float]:
        """
        Get all yield features for a pair.
        Returns from Redis cache if available, else computes fresh.

        Args:
            pair: e.g. "EURUSD"

        Returns:
            Flat feature dict ready for ML model input.
        """
        pair = pair.upper()
        cached = self._redis_get(f"features:yield:{pair}")
        if cached:
            return cached

        # Cache miss — compute all and return this pair
        all_features = self.compute_all()
        return all_features.get(pair, {})

    def get_carry_ranking(self) -> List[Dict[str, Any]]:
        """Get carry ranking from cache."""
        cached = self._redis_get("features:carry:ranking")
        if cached:
            return cached
        self.compute_all()
        return self._redis_get("features:carry:ranking") or []

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all yield features for all G10 pairs.
        Stores results to Redis. Returns full feature dict.
        """
        logger.info("YieldFeatures | Computing all features...")

        try:
            yields      = self._get_latest_yields()
            yield_hist  = self._get_yield_history(lookback_days=200)
        except Exception as exc:
            logger.error(f"YieldFeatures | DB fetch failed: {exc}")
            return {}

        if not yields:
            logger.warning("YieldFeatures | No yield data in DB")
            return {}

        # Carry ranking (per currency)
        carry_ranking = self._compute_carry_ranking(yields)
        carry_scores  = self._compute_carry_scores(carry_ranking)

        # Curve slopes per currency
        curve_features = self._compute_curve_features(yields)

        # Per-pair features
        all_features: Dict[str, Dict[str, float]] = {}

        for pair, (base_ccy, quote_ccy) in G10_PAIRS.items():
            try:
                pair_feats = self._compute_pair_features(
                    pair        = pair,
                    base_ccy    = base_ccy,
                    quote_ccy   = quote_ccy,
                    yields      = yields,
                    yield_hist  = yield_hist,
                    carry_scores= carry_scores,
                    curve_feats = curve_features,
                )
                all_features[pair] = pair_feats

                # Cache per-pair
                self._redis_set(f"features:yield:{pair}", pair_feats, CACHE_TTL_S)

            except Exception as exc:
                logger.error(f"YieldFeatures | {pair} failed: {exc}")
                all_features[pair] = {}

        # Cache full summary and carry ranking
        self._redis_set("features:yield:summary",  all_features,  CACHE_TTL_S)
        self._redis_set("features:carry:ranking",  carry_ranking, CACHE_TTL_S)

        self._last_compute = datetime.now(UTC)

        logger.success(
            f"YieldFeatures | {len(all_features)} pairs computed | "
            f"USD10Y={yields.get('USD_10Y', 'N/A')}% "
            f"EUR10Y={yields.get('EUR_10Y', 'N/A')}%"
        )
        return all_features

    # =========================================================================
    # Feature computation
    # =========================================================================

    def _compute_pair_features(
        self,
        pair:         str,
        base_ccy:     str,
        quote_ccy:    str,
        yields:       Dict[str, float],
        yield_hist:   Dict[str, pd.Series],
        carry_scores: Dict[str, float],
        curve_feats:  Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Compute all yield features for one currency pair.

        Feature naming convention: feature_name (no pair suffix in dict —
        the dict is already pair-specific, suffix added when merging all pairs).
        """
        feats: Dict[str, float] = {}

        # ── 1. Yield differentials (2Y and 10Y) ──────────────────────────────
        # Course: "The direction of the interest rate differential is as
        #          important as the absolute level"
        for tenor in ["2Y", "10Y", "3M"]:
            base_y  = yields.get(f"{base_ccy}_{tenor}")
            quote_y = yields.get(f"{quote_ccy}_{tenor}")

            if base_y is not None and quote_y is not None:
                diff = round(base_y - quote_y, 4)
                feats[f"yield_diff_{tenor}"] = diff
                feats[f"yield_diff_{tenor}_abs"] = abs(diff)

                # Non-linear score: -1 to +1, reduced at extremes
                # Course: "When differential widens to extreme levels,
                #          conviction scoring becomes lower"
                feats[f"yield_score_{tenor}"] = self._diff_to_score(diff)
            else:
                feats[f"yield_diff_{tenor}"]     = 0.0
                feats[f"yield_diff_{tenor}_abs"] = 0.0
                feats[f"yield_score_{tenor}"]    = 0.0

        # ── 2. Yield differential trend (4w and 13w) ─────────────────────────
        # Most important feature — direction of change
        for tenor in ["2Y", "10Y"]:
            chg_4w, chg_13w, momentum = self._compute_diff_trend(
                base_ccy, quote_ccy, tenor, yield_hist
            )
            feats[f"yield_trend_4w_{tenor}"]  = chg_4w
            feats[f"yield_trend_13w_{tenor}"] = chg_13w
            feats[f"yield_momentum_{tenor}"]  = momentum

        # ── 3. Carry features ─────────────────────────────────────────────────
        # Course: "Hot money follows carry"
        # Net carry = base_carry_score - quote_carry_score
        base_carry  = carry_scores.get(base_ccy, 0.0)
        quote_carry = carry_scores.get(quote_ccy, 0.0)
        feats["carry_base"]    = base_carry
        feats["carry_quote"]   = quote_carry
        feats["carry_net"]     = round(base_carry - quote_carry, 4)

        # Daily rollover estimate (annualized diff / 365)
        diff_10y = feats.get("yield_diff_10Y", 0.0)
        feats["rollover_daily_pct"] = round(diff_10y / 365, 6)
        # In dollar terms per lot (approximate)
        feats["rollover_daily_usd"] = round(
            (diff_10y / 100 / 365) * LOT_SIZE, 4
        )

        # ── 4. Curve slope features ───────────────────────────────────────────
        # Inverted curve = recession fear = risk-off = JPY/CHF/USD bid
        for ccy in [base_ccy, quote_ccy]:
            cf = curve_feats.get(ccy, {})
            prefix = "base" if ccy == base_ccy else "quote"
            feats[f"curve_slope_{prefix}"]    = cf.get("slope", 0.0)
            feats[f"curve_inverted_{prefix}"] = float(cf.get("inverted", False))
            feats[f"curve_signal_{prefix}"]   = cf.get("signal_num", 0.0)

        # Relative curve: base steeper than quote = base currency growth optimism
        feats["curve_slope_diff"] = round(
            feats.get("curve_slope_base", 0.0) -
            feats.get("curve_slope_quote", 0.0), 4
        )

        # ── 5. CIP deviation (approximate) ───────────────────────────────────
        # Covered Interest Parity: in theory diff_2Y ≈ 0 after hedging
        # Deviation from CIP = arbitrage opportunity / stress signal
        # Approximate: large 2Y diff that persists = market stress
        diff_2y = feats.get("yield_diff_2Y", 0.0)
        diff_10y_val = feats.get("yield_diff_10Y", 0.0)
        # CIP deviation ≈ short-term diff - long-term diff (term premium)
        feats["cip_deviation"] = round(diff_2y - diff_10y_val, 4)

        # ── 6. Z-score of current differential vs 1Y history ──────────────────
        feats["yield_zscore_10Y"] = self._compute_diff_zscore(
            base_ccy, quote_ccy, "10Y", yield_hist, window=252
        )
        feats["yield_zscore_2Y"] = self._compute_diff_zscore(
            base_ccy, quote_ccy, "2Y", yield_hist, window=252
        )

        return {k: float(v) if not np.isnan(float(v)) else 0.0
                for k, v in feats.items()}

    def _compute_carry_ranking(
        self, yields: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Rank currencies by 10Y yield — high yield attracts hot money flows.
        Course: "Speculative money from hedge funds follows carry."
        """
        ranking = []
        for ccy in ALL_CURRENCIES:
            y10 = yields.get(f"{ccy}_10Y")
            y2  = yields.get(f"{ccy}_2Y")
            if y10 is not None:
                ranking.append({
                    "currency":    ccy,
                    "yield_10Y":   y10,
                    "yield_2Y":    y2,
                    "slope":       round(y10 - y2, 4) if y2 else None,
                })

        ranking.sort(key=lambda x: x["yield_10Y"] or 0, reverse=True)
        for i, item in enumerate(ranking):
            item["rank"] = i + 1

        return ranking

    def _compute_carry_scores(
        self, ranking: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Normalize carry ranking to [-1, +1] score.
        Rank 1 (highest yield) → +1.0
        Rank N (lowest yield)  → -1.0
        """
        n = len(ranking)
        if n == 0:
            return {}

        scores = {}
        for item in ranking:
            rank = item["rank"]
            # Linear mapping: rank 1 → +1.0, rank n → -1.0
            score = 1.0 - 2.0 * (rank - 1) / max(n - 1, 1)
            scores[item["currency"]] = round(score, 4)

        return scores

    def _compute_curve_features(
        self, yields: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compute curve slope and inversion status per currency."""
        result = {}
        for ccy in ALL_CURRENCIES:
            y10 = yields.get(f"{ccy}_10Y")
            y2  = yields.get(f"{ccy}_2Y")

            if y10 is None:
                result[ccy] = {"slope": 0.0, "inverted": False, "signal_num": 0.0}
                continue

            if y2 is not None:
                slope    = round(y10 - y2, 4)
                inverted = slope < 0
            else:
                slope    = 0.0
                inverted = False

            # Numeric signal: -1 risk_off, 0 neutral, +1 risk_on
            if inverted:
                signal_num = -1.0
            elif slope < 0.5:
                signal_num = 0.0
            else:
                signal_num = 1.0

            result[ccy] = {
                "slope":      slope,
                "inverted":   inverted,
                "signal_num": signal_num,
            }

        return result

    def _compute_diff_trend(
        self,
        base_ccy:   str,
        quote_ccy:  str,
        tenor:      str,
        yield_hist: Dict[str, pd.Series],
    ) -> tuple:
        """
        Compute 4-week and 13-week trend in yield differential.

        Course: "Direction of the differential is as important as the level."
        Returns: (chg_4w, chg_13w, momentum)
        momentum: +1.0 widening (base favored), -1.0 narrowing, 0.0 stable
        """
        base_series  = yield_hist.get(f"{base_ccy}_{tenor}")
        quote_series = yield_hist.get(f"{quote_ccy}_{tenor}")

        if base_series is None or quote_series is None:
            return (0.0, 0.0, 0.0)

        try:
            # Align on common dates
            common = base_series.index.intersection(quote_series.index)
            if len(common) < 30:
                return (0.0, 0.0, 0.0)

            diff = base_series.loc[common] - quote_series.loc[common]
            diff = diff.sort_index()

            current = float(diff.iloc[-1])
            val_4w  = float(diff.iloc[-28]) if len(diff) >= 28 else current
            val_13w = float(diff.iloc[-91]) if len(diff) >= 91 else current

            chg_4w  = round(current - val_4w,  4)
            chg_13w = round(current - val_13w, 4)

            # Momentum: direction of 4w change normalized
            if abs(chg_4w) < 0.05:
                momentum = 0.0
            elif chg_4w > 0:
                momentum = min(1.0, chg_4w / 0.5)   # cap at +1
            else:
                momentum = max(-1.0, chg_4w / 0.5)  # cap at -1

            return (chg_4w, chg_13w, round(momentum, 4))

        except Exception as exc:
            logger.debug(f"Trend compute failed [{base_ccy}/{quote_ccy} {tenor}]: {exc}")
            return (0.0, 0.0, 0.0)

    def _compute_diff_zscore(
        self,
        base_ccy:   str,
        quote_ccy:  str,
        tenor:      str,
        yield_hist: Dict[str, pd.Series],
        window:     int = 252,
    ) -> float:
        """
        Z-score of current yield differential vs rolling 1Y history.
        High z-score → differential unusually wide (mean reversion risk).
        """
        base_s  = yield_hist.get(f"{base_ccy}_{tenor}")
        quote_s = yield_hist.get(f"{quote_ccy}_{tenor}")

        if base_s is None or quote_s is None:
            return 0.0

        try:
            common = base_s.index.intersection(quote_s.index)
            diff   = (base_s.loc[common] - quote_s.loc[common]).sort_index()

            if len(diff) < window // 2:
                return 0.0

            recent = diff.iloc[-window:]
            mean   = float(recent.mean())
            std    = float(recent.std())

            if std < 1e-6:
                return 0.0

            z = (float(diff.iloc[-1]) - mean) / std
            return round(float(np.clip(z, -4, 4)), 4)  # clip to [-4, +4]

        except Exception:
            return 0.0

    def _diff_to_score(self, diff: float) -> float:
        """
        Convert yield differential to ML score [-1, +1].

        Non-linear scaling matching course conviction levels:
            |diff| < 0.05%  → 0.0  (NEUTRAL)
            |diff| 0.3-1%   → 0.3  (WEAK)
            |diff| 1-2%     → 0.6  (MODERATE)
            |diff| > 2%     → 0.9  (STRONG)
            |diff| > 4%     → 0.6  (EXTREME — lower conviction, priced in)
        """
        sign     = 1.0 if diff > 0 else -1.0
        abs_diff = abs(diff)

        if abs_diff < 0.05:
            score = 0.0
        elif abs_diff >= EXTREME_DIFF_THRESHOLD:
            score = sign * 0.6
        elif abs_diff >= 2.0:
            score = sign * 0.9
        elif abs_diff >= 1.0:
            score = sign * 0.6
        elif abs_diff >= 0.3:
            score = sign * 0.3
        else:
            score = sign * 0.1

        return round(score, 4)

    # =========================================================================
    # DB access
    # =========================================================================

    def _get_latest_yields(self) -> Dict[str, float]:
        """Get most recent yield per (currency, tenor) from DB."""
        from data_pipeline.storage.timeseries_db import db
        from sqlalchemy import text

        with db.connection() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT ON (currency, tenor)
                    currency, tenor, yield_pct
                FROM yield_data
                ORDER BY currency, tenor, date DESC
            """))
            return {
                f"{r[0]}_{r[1]}": round(float(r[2]), 4)
                for r in result.fetchall()
            }

    def _get_yield_history(self, lookback_days: int = 200) -> Dict[str, pd.Series]:
        """Get yield history as dict of pd.Series indexed by date."""
        from data_pipeline.storage.timeseries_db import db
        from sqlalchemy import text

        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        with db.connection() as conn:
            result = conn.execute(text("""
                SELECT date, currency, tenor, yield_pct
                FROM yield_data
                WHERE date >= :cutoff
                  AND tenor IN ('2Y', '10Y', '3M', 'SLOPE')
                ORDER BY currency, tenor, date
            """), {"cutoff": cutoff})
            rows = result.fetchall()

        if not rows:
            return {}

        df = pd.DataFrame(rows, columns=["date", "currency", "tenor", "yield_pct"])
        df["date"] = pd.to_datetime(df["date"])
        df["key"]  = df["currency"] + "_" + df["tenor"]

        result_dict: Dict[str, pd.Series] = {}
        for key, group in df.groupby("key"):
            s = group.set_index("date")["yield_pct"].astype(float)
            s = s.sort_index()
            result_dict[key] = s

        return result_dict

    # =========================================================================
    # Redis helpers
    # =========================================================================

    def _get_redis(self):
        """Lazy Redis connection."""
        try:
            # NEW — shares the pool
            from data_pipeline.storage.redis_pool import redis_client
            return redis_client
        except Exception:
            return None

    def _redis_get(self, key: str) -> Any:
        """Get JSON-decoded value from Redis."""
        try:
            r = self._get_redis()
            if r is None:
                return None
            val = r.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def _redis_set(self, key: str, value: Any, ttl: int) -> None:
        """Set JSON-encoded value in Redis with TTL."""
        try:
            r = self._get_redis()
            if r:
                r.set(key, json.dumps(value), ex=ttl)
        except Exception as exc:
            logger.debug(f"Redis set failed [{key}]: {exc}")

    # =========================================================================
    # Scheduler
    # =========================================================================

    def start_scheduler(self) -> None:
        """
        Background thread: re-compute yield features every 4 hours.
        Runs immediately on start, then every 4h.
        """
        def _run():
            logger.info("YieldFeatures scheduler started — every 4h")
            while True:
                try:
                    self.compute_all()
                except Exception as exc:
                    logger.error(f"YieldFeatures scheduler error: {exc}")
                time.sleep(CACHE_TTL_S)

        t = threading.Thread(target=_run, name="yield-features", daemon=True)
        t.start()
        logger.info("YieldFeatures scheduler started ✅")


# ── Module singleton ───────────────────────────────────────────────────────────
yield_feature_engine = YieldFeatureEngine()
