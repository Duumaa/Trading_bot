"""
models/analyst2_endo_exo/macro_scorer.py
=========================================
Endogenous Policy Direction Scorer — Analyst 2 Driver 5.

Answers the question A2 previously could not:
  "Is this central bank hiking, cutting, or on hold RIGHT NOW?"

A 4.5% NZD yield from a CB that just cut 50bps is fundamentally
different from a 4.5% yield from a CB that just hiked 25bps.
Yield levels alone cannot distinguish these. Policy direction can.

Three data sources, all already in the DB:
  1. 2Y yield trend (yield_data table)
     The market's real-time estimate of where CB rates will be in 2Y.
     Rising 2Y = hikes priced in = hawkish = positive endo score.
     Falling 2Y = cuts priced in = dovish = negative endo score.
     This is the PRIMARY signal — most current, most sensitive.

  2. CB rate decisions (economic_events table)
     Actual rate changes in last 6 months = confirmed cycle direction.
     3× cuts in 6 months = clearly cutting. Overrides 2Y if diverging.

  3. CPI + employment surprises (economic_events table)
     Persistent upside inflation surprises = hawkish pressure building.
     Weak jobs = dovish pressure. Secondary signal, modifies conviction.

Output per currency:
  {
    "currency":       "NZD",
    "endo_score":     -0.55,      # -1.0 (very dovish) to +1.0 (very hawkish)
    "policy_stance":  "CUTTING",  # HIKING | CUTTING | HOLD | UNKNOWN
    "conviction":     0.72,
    "signals": {
      "2y_trend_8w":    -0.18,    # 8-week 2Y yield change
      "2y_trend_26w":   -0.41,    # 26-week 2Y yield change
      "cb_decisions":   -0.60,    # from actual rate decisions
      "cpi_surprise":   -0.10,    # inflation surprise direction
      "employ_surprise": 0.00,
    },
    "updated_at": "2026-03-17T..."
  }

Redis cache:
  Key: features:macro:{CURRENCY}   TTL: 14400s (4 hours)

Wire into score_calculator.py:
  Called by _score_endo() — returns (base_score, quote_score)
  Pair endo score = base_endo - quote_endo
  Weight in composite: 20% (yields 30%, COT 25%, carry 15%, comm 10%)

Usage:
  from models.analyst2_endo_exo.macro_scorer import macro_scorer
  score = macro_scorer.get_score("NZD")   # single currency
  all   = macro_scorer.compute_all()       # all 10 currencies
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
from loguru import logger
from sqlalchemy import text

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

from data_pipeline.storage.timeseries_db import db  # noqa: E402

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

CURRENCIES: List[str] = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# CB decision event names as stored in economic_events.event_name
# These are the exact strings ForexFactory uses
CB_EVENT_KEYWORDS: Dict[str, List[str]] = {
    "USD": ["Federal Funds Rate", "Fed Funds Rate", "FOMC"],
    "EUR": ["ECB Deposit Facility Rate", "ECB Rate", "Minimum Bid Rate",
            "Main Refinancing Rate"],
    "GBP": ["Official Bank Rate", "Bank Rate"],
    "JPY": ["BOJ Policy Rate", "BoJ Rate", "Overnight Call Rate",
            "Uncollateralized Overnight"],
    "CHF": ["SNB Policy Rate", "Libor Rate", "SNB Rate"],
    "CAD": ["BOC Rate", "BoC Rate", "Overnight Rate Target"],
    "AUD": ["RBA Rate", "Cash Rate"],
    "NZD": ["RBNZ Rate", "Official Cash Rate"],
}

# CPI event keywords per currency
CPI_KEYWORDS: Dict[str, List[str]] = {
    "USD": ["CPI", "Core CPI", "Consumer Price Index"],
    "EUR": ["CPI Flash", "German CPI", "Euro CPI", "Inflation"],
    "GBP": ["CPI", "Consumer Price Index"],
    "JPY": ["Tokyo CPI", "National CPI"],
    "CHF": ["CPI", "Consumer Price Index"],
    "CAD": ["CPI", "Consumer Price Index"],
    "AUD": ["CPI", "Trimmed Mean CPI"],
    "NZD": ["CPI", "Consumer Price Index"],
}

# Employment event keywords
EMPLOY_KEYWORDS: Dict[str, List[str]] = {
    "USD": ["Non-Farm Payrolls", "NFP", "Unemployment Rate"],
    "EUR": ["German Employment", "Unemployment Rate"],
    "GBP": ["Claimant Count", "Employment Change", "Unemployment Rate"],
    "JPY": ["Unemployment Rate", "Jobs"],
    "CHF": ["Unemployment Rate"],
    "CAD": ["Employment Change", "Unemployment Rate"],
    "AUD": ["Employment Change", "Unemployment Rate"],
    "NZD": ["Employment Change", "Unemployment Rate"],
}

# Signal weights within endo score
SIGNAL_WEIGHTS = {
    "2y_trend":   0.55,   # primary — most current CB expectation signal
    "cb_decision": 0.30,  # confirmed cycle direction from actual decisions
    "cpi_surprise": 0.10, # inflation pressure modifier
    "employ":      0.05,  # employment modifier
}

CACHE_TTL = 14400   # 4 hours


# ══════════════════════════════════════════════════════════════
# REDIS
# ══════════════════════════════════════════════════════════════

def _get_redis():
    import redis as _redis
    from dotenv import load_dotenv
    load_dotenv(str(ROOT / ".env"))
    return _redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


# ══════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════

def _load_2y_yield_history(currency: str, days: int = 200) -> pd.DataFrame:
    """
    Load 2Y yield history for a currency.
    2Y yield = best market proxy for CB rate expectations 2 years out.
    Already in yield_data for all G10 currencies.
    """
    try:
        with db.connection() as conn:
            rows = conn.execute(text("""
                SELECT date, yield_pct
                FROM yield_data
                WHERE currency = :currency
                  AND tenor = '2Y'
                  AND date >= NOW() - INTERVAL '250 days'
                ORDER BY date ASC
            """), {"currency": currency}).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "yield_pct"])
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.warning(f"MacroScorer | {currency} 2Y yield load failed: {exc}")
        return pd.DataFrame()

def _load_cb_decisions(currency: str, months: int = 6) -> pd.DataFrame:
    """Load CB rate decisions — deduplicated, handles < prefix in JPY rates."""
    keywords = CB_EVENT_KEYWORDS.get(currency, [])
    if not keywords:
        return pd.DataFrame()

    conditions = " OR ".join([f"event_name ILIKE :kw{i}"
                               for i in range(len(keywords))])
    params = {"currency": currency}
    for i, kw in enumerate(keywords):
        params[f"kw{i}"] = f"%{kw}%"

    try:
        with db.connection() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT ON (event_time::date, actual)
                    event_time, event_name, actual, forecast, previous
                FROM economic_events
                WHERE country = :currency
                  AND impact = 'HIGH'
                  AND event_time >= NOW() - INTERVAL '{months} months'
                  AND actual IS NOT NULL
                  AND previous IS NOT NULL
                  AND ({conditions})
                ORDER BY event_time::date, actual, event_time DESC
            """), params).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["event_time", "event_name",
                                          "actual", "forecast", "previous"])
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

        # Parse rate values — handles "5.50%", "5.50", "<0.75%", ">0.10%"
        def _parse_rate(val) -> float:
            if val is None:
                return np.nan
            s = str(val).replace("%", "").replace("<", "").replace(">", "").strip()
            try:
                return float(s)
            except (ValueError, TypeError):
                return np.nan

        for col in ["actual", "previous"]:
            df[col] = df[col].apply(_parse_rate)

        df = df.dropna(subset=["actual", "previous"])
        df["rate_change"] = df["actual"] - df["previous"]

        # Keep only one row per date (highest |rate_change| if still dupes)
        df = df.sort_values("event_time")
        df["date_only"] = df["event_time"].dt.date
        df = df.loc[df.groupby("date_only")["rate_change"].apply(
            lambda x: x.abs().idxmax()
        )]
        return df.reset_index(drop=True)

    except Exception as exc:
        logger.warning(f"MacroScorer | {currency} CB decisions load failed: {exc}")
        return pd.DataFrame()


def _load_event_surprises(currency: str, keywords: List[str],
                           months: int = 3) -> pd.DataFrame:
    """
    Load economic event surprises (actual vs forecast) for CPI or employment.
    Positive surprise = actual > forecast = inflationary / strong.
    """
    if not keywords:
        return pd.DataFrame()

    conditions = " OR ".join([f"event_name ILIKE :kw{i}" for i in range(len(keywords))])
    params = {"currency": currency}
    for i, kw in enumerate(keywords):
        params[f"kw{i}"] = f"%{kw}%"

    try:
        with db.connection() as conn:
            rows = conn.execute(text(f"""                
                SELECT event_time, event_name, actual, forecast
                FROM economic_events
                WHERE country = :currency
                  AND impact IN ('HIGH', 'MEDIUM')
                  AND event_time >= NOW() - INTERVAL '{months} months'
                  AND actual IS NOT NULL
                  AND ({conditions})
                ORDER BY event_time ASC
            """), params).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["event_time", "event_name",
                                          "actual", "forecast"])
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
        return df.reset_index(drop=True)

    except Exception as exc:
        logger.warning(f"MacroScorer | {currency} surprise load failed: {exc}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# SIGNAL SCORERS
# ══════════════════════════════════════════════════════════════

def _score_2y_trend(currency: str) -> Tuple[float, float, float]:
    """
    Score CB policy direction from 2Y yield trend.

    Returns (score, trend_8w, trend_26w)
    score: -1.0 (strong cutting) to +1.0 (strong hiking)

    Logic:
      8-week trend  = recent repricing (last 2 months)
      26-week trend = cycle direction  (last 6 months)
      8w weighted 60%, 26w weighted 40%
      Magnitude matters: bigger yield move = stronger signal
    """
    df = _load_2y_yield_history(currency, days=200)

    if df.empty or len(df) < 5:
        return 0.0, 0.0, 0.0

    latest = df["yield_pct"].iloc[-1]
    now_dt = df["date"].max()

    # Date-based lookback — works for both daily and monthly data
    def _get_at_date(target_dt) -> Optional[float]:
        mask = df["date"] <= target_dt
        if not mask.any():
            return None
        return df.loc[mask, "yield_pct"].iloc[-1]

    y_8w  = _get_at_date(now_dt - pd.Timedelta(weeks=8))
    y_26w = _get_at_date(now_dt - pd.Timedelta(weeks=26))

    trend_8w  = (latest - y_8w)  if y_8w  is not None else 0.0
    trend_26w = (latest - y_26w) if y_26w is not None else 0.0

    # Scale: 0.50% move in 2Y over 8w = full signal (±50bps is a full rate cycle move)
    # Non-linear: diminishing returns above 50bps (already priced in)
    def _scale_trend(t: float, scale: float = 0.50) -> float:
        sign = np.sign(t)
        magnitude = min(abs(t) / scale, 1.0)
        # Slight sigmoid to reduce overconfidence at extremes
        return float(sign * (1 - np.exp(-3 * magnitude)) / (1 - np.exp(-3)))

    s_8w  = _scale_trend(trend_8w,  scale=0.50)
    s_26w = _scale_trend(trend_26w, scale=1.00)  # bigger scale for 26w

    score = s_8w * 0.60 + s_26w * 0.40
    return float(np.clip(score, -1.0, 1.0)), round(trend_8w, 4), round(trend_26w, 4)


def _score_cb_decisions(currency: str) -> Tuple[float, str]:
    """
    Score CB policy cycle from actual rate decisions in last 6 months.

    Returns (score, stance_label)
    score: -1.0 (all cuts) to +1.0 (all hikes)
    stance: "HIKING" | "CUTTING" | "HOLD" | "UNKNOWN"

    Logic:
      Count hikes (positive rate changes) vs cuts (negative) in last 6 months.
      Weight recent decisions more than older ones (exponential decay).
      Magnitude matters: 50bps cut >> 25bps cut.
    """
    df = _load_cb_decisions(currency, months=6)

    if df.empty:
        return 0.0, "UNKNOWN"

    # Filter to actual rate changes (not just holds)
    decisions = df[df["rate_change"].abs() > 0.001].copy()

    if decisions.empty:
        # All holds — check if that's consistent or just recent
        return 0.0, "HOLD"

    # Exponential recency weighting — decisions in last 30 days = weight 1.0
    now = datetime.now(timezone.utc)
    decisions["age_days"] = decisions["event_time"].apply(
        lambda t: (now - t).days
    )
    decisions["weight"] = np.exp(-decisions["age_days"] / 60.0)  # 60-day half-life

    # Weighted sum of rate changes
    weighted_sum = (decisions["rate_change"] * decisions["weight"]).sum()
    total_weight = decisions["weight"].sum()

    if total_weight < 0.01:
        return 0.0, "UNKNOWN"

    avg_change = weighted_sum / total_weight

    # Scale: 0.25% (25bps) average change = moderate signal
    score = float(np.clip(avg_change / 0.25, -1.0, 1.0))

    # Determine stance label
    n_hikes = (decisions["rate_change"] > 0.001).sum()
    n_cuts  = (decisions["rate_change"] < -0.001).sum()

    if n_hikes > n_cuts:
        stance = "HIKING"
    elif n_cuts > n_hikes:
        stance = "CUTTING"
    else:
        stance = "HOLD"

    return score, stance


def _score_cpi_surprise(currency: str) -> float:
    """
    Score inflation surprise direction over last 3 months.

    Persistent above-forecast CPI = hawkish pressure building.
    Persistent below-forecast CPI = dovish pressure.

    Uses surprise_score if available, else computes actual vs forecast.
    Returns -1.0 to +1.0
    """
    keywords = CPI_KEYWORDS.get(currency, [])
    df = _load_event_surprises(currency, keywords, months=3)

    if df.empty:
        return 0.0

    surprises = []

    for _, row in df.iterrows():
        if pd.notna(row.get("actual")) and pd.notna(row.get("forecast")):
            try:
                actual   = float(str(row["actual"]).replace("%","").replace("<","").replace(">","").strip())
                forecast = float(str(row["forecast"]).replace("%","").replace("<","").replace(">","").strip())
                surprise = (actual - forecast) / 0.20
                surprises.append(float(np.clip(surprise, -2.0, 2.0)))
            except (ValueError, TypeError):
                pass
        # Use pre-computed surprise_score if available
        if pd.notna(row.get("surprise_score")) and row["surprise_score"] != 0:
            surprises.append(float(row["surprise_score"]))
        elif pd.notna(row.get("actual")) and pd.notna(row.get("forecast")):
            try:
                actual   = float(str(row["actual"]).replace("%", "").strip())
                forecast = float(str(row["forecast"]).replace("%", "").strip())
                # Normalize surprise by typical CPI volatility (~0.2%)
                surprise = (actual - forecast) / 0.20
                surprises.append(float(np.clip(surprise, -2.0, 2.0)))
            except (ValueError, TypeError):
                pass

    if not surprises:
        return 0.0

    # Average surprise direction — persistent misses matter more than one-offs
    avg = np.mean(surprises)
    return float(np.clip(avg / 2.0, -1.0, 1.0))


def _score_employment_surprise(currency: str) -> float:
    """
    Score employment surprise direction over last 3 months.
    Strong jobs = hawkish pressure. Weak jobs = dovish.
    """
    keywords = EMPLOY_KEYWORDS.get(currency, [])
    df = _load_event_surprises(currency, keywords, months=3)

    if df.empty:
        return 0.0

    surprises = []
    for _, row in df.iterrows():
        if pd.notna(row.get("actual")) and pd.notna(row.get("forecast")):
            try:
                actual   = float(str(row["actual"]).replace("%", "").strip())
                forecast = float(str(row["forecast"]).replace("%", "").strip())
                surprise = (actual - forecast) / 0.20
                surprises.append(float(np.clip(surprise, -2.0, 2.0)))
            except (ValueError, TypeError):
                pass

    if not surprises:
        return 0.0

    avg = np.mean(surprises)
    return float(np.clip(avg / 2.0, -1.0, 1.0))


# ══════════════════════════════════════════════════════════════
# MAIN SCORER CLASS
# ══════════════════════════════════════════════════════════════

class MacroScorer:
    """
    Endogenous policy direction scorer for all G10 currencies.

    Computes a per-currency score (-1 to +1) representing:
      +1.0 = strongly hawkish CB, hiking cycle, strong fundamentals
      -1.0 = strongly dovish CB, cutting cycle, weak fundamentals
       0.0 = on hold / neutral

    This is the endogenous layer missing from A2's original design.
    Wire into score_calculator.py as Driver 5 (weight: 20%).
    """

    def __init__(self) -> None:
        self._redis = None

    @property
    def r(self):
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    def get_score(self, currency: str) -> Dict:
        """
        Get endogenous policy score for a currency.
        Returns cached value if fresh, else recomputes.
        """
        cache_key = f"features:macro:{currency}"
        try:
            cached = self.r.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as exc:
            logger.warning(f"MacroScorer | Redis read failed: {exc}")

        result = self._compute(currency)
        self._cache(cache_key, result)
        return result

    def compute_all(self) -> Dict[str, Dict]:
        """
        Compute endogenous scores for all G10 currencies.
        Called by A2 scheduler every 4 hours.
        """
        logger.info("MacroScorer | Computing endogenous scores for all currencies...")
        results = {}

        for currency in CURRENCIES:
            try:
                result = self._compute(currency)
                self._cache(f"features:macro:{currency}", result)
                results[currency] = result
                logger.info(
                    f"MacroScorer | {currency}: score={result['endo_score']:+.3f} "
                    f"stance={result['policy_stance']} conf={result['conviction']:.2f}"
                )
            except Exception as exc:
                logger.error(f"MacroScorer | {currency} failed: {exc}")
                results[currency] = self._neutral(currency)

        self._print_summary(results)
        return results

    def _compute(self, currency: str) -> Dict:
        """
        Compute full endogenous score for one currency.
        """
        logger.debug(f"MacroScorer | Computing {currency}...")

        # ── Signal 1: 2Y yield trend (PRIMARY) ───────────────
        s_2y, trend_8w, trend_26w = _score_2y_trend(currency)

        # ── Signal 2: CB rate decisions ───────────────────────
        s_cb, stance = _score_cb_decisions(currency)

        # ── Signal 3: CPI surprises ───────────────────────────
        s_cpi = _score_cpi_surprise(currency)

        # ── Signal 4: Employment surprises ────────────────────
        s_emp = _score_employment_surprise(currency)

        # ── Conflict check: 2Y trend vs CB decisions ──────────
        # If 2Y says cutting but CB has been hiking (or vice versa),
        # the 2Y is forward-looking — trust it more, but flag the conflict
        conflict = False
        if abs(s_2y) > 0.2 and abs(s_cb) > 0.2:
            if np.sign(s_2y) != np.sign(s_cb):
                conflict = True
                logger.debug(
                    f"MacroScorer | {currency}: 2Y trend ({s_2y:+.2f}) conflicts "
                    f"with CB decisions ({s_cb:+.2f}) — 2Y wins (forward-looking)"
                )
                # 2Y is forward-looking, CB decisions are backward-looking
                # Trust 2Y more but reduce conviction
                s_cb = s_cb * 0.30

        # ── Composite ─────────────────────────────────────────
        endo_score = (
            s_2y * SIGNAL_WEIGHTS["2y_trend"]    +
            s_cb * SIGNAL_WEIGHTS["cb_decision"] +
            s_cpi * SIGNAL_WEIGHTS["cpi_surprise"] +
            s_emp * SIGNAL_WEIGHTS["employ"]
        )
        endo_score = float(np.clip(endo_score, -1.0, 1.0))

        # ── Conviction ────────────────────────────────────────
        # High when 2Y and CB decisions agree AND magnitude is clear
        signals = [s for s in [s_2y, s_cb] if abs(s) > 0.05]
        if len(signals) >= 2:
            agree = all(np.sign(s) == np.sign(endo_score) for s in signals)
            conviction = min(abs(endo_score) * 1.5, 1.0) if agree else abs(endo_score) * 0.6
        elif len(signals) == 1:
            conviction = abs(endo_score) * 0.7
        else:
            conviction = 0.0

        if conflict:
            conviction *= 0.7  # reduce conviction when signals conflict

        # ── Stance label from best available source ───────────
        if stance == "UNKNOWN":
            # Fall back to 2Y trend interpretation
            if endo_score > 0.15:
                stance = "HIKING"
            elif endo_score < -0.15:
                stance = "CUTTING"
            else:
                stance = "HOLD"

        return {
            "currency":      currency,
            "endo_score":    round(endo_score, 4),
            "policy_stance": stance,
            "conviction":    round(float(conviction), 4),
            "conflict":      conflict,
            "signals": {
                "2y_trend_score":  round(s_2y, 4),
                "2y_trend_8w":     trend_8w,
                "2y_trend_26w":    trend_26w,
                "cb_decision_score": round(s_cb, 4),
                "cb_stance":       stance,
                "cpi_surprise":    round(s_cpi, 4),
                "employ_surprise": round(s_emp, 4),
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _cache(self, key: str, result: Dict) -> None:
        try:
            self.r.setex(key, CACHE_TTL, json.dumps(result))
        except Exception as exc:
            logger.warning(f"MacroScorer | Redis write failed: {exc}")

    @staticmethod
    def _neutral(currency: str) -> Dict:
        return {
            "currency":      currency,
            "endo_score":    0.0,
            "policy_stance": "UNKNOWN",
            "conviction":    0.0,
            "conflict":      False,
            "signals": {
                "2y_trend_score": 0.0, "2y_trend_8w": 0.0, "2y_trend_26w": 0.0,
                "cb_decision_score": 0.0, "cb_stance": "UNKNOWN",
                "cpi_surprise": 0.0, "employ_surprise": 0.0,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _print_summary(results: Dict[str, Dict]) -> None:
        print("\n" + "=" * 65)
        print("MACRO SCORER — ENDOGENOUS POLICY DIRECTION SUMMARY")
        print("=" * 65)
        print(f"{'CCY':<6} {'SCORE':>7} {'STANCE':<10} {'CONF':>6} "
              f"{'2Y_8W':>7} {'2Y_26W':>8} {'CB':>7} {'CPI':>6}")
        print("-" * 65)
        for ccy, r in results.items():
            s = r["signals"]
            print(
                f"{ccy:<6} "
                f"{r['endo_score']:>+7.3f} "
                f"{r['policy_stance']:<10} "
                f"{r['conviction']:>6.2f} "
                f"{s['2y_trend_8w']:>+7.3f} "
                f"{s['2y_trend_26w']:>+8.3f} "
                f"{s['cb_decision_score']:>+7.3f} "
                f"{s['cpi_surprise']:>+6.3f}"
            )
        print("=" * 65)
        print("Score: +1.0=strongly hawkish | -1.0=strongly dovish")
        print("=" * 65 + "\n")


# ── Module singleton ──────────────────────────────────────────
macro_scorer = MacroScorer()


# ── Direct run ────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")
    logger.info("MacroScorer | Running full computation...")
    macro_scorer.compute_all()
