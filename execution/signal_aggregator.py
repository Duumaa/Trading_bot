"""
execution/signal_aggregator.py
===============================
Signal Aggregator — Gate Logic Engine

Wires all 6 analysts into a final LONG/SHORT/FLAT decision per pair.
This is NOT an ensemble. Each analyst group gates independently.

Gate logic (must pass ALL 3 gates):
  Gate 1 — FUNDAMENTAL  (Analysts 1 + 2)
    • Regime ≠ PNS
    • Analyst 2 bias is LONG_BASE or SHORT_BASE (not NEUTRAL)
    • Analyst 2 confidence ≥ threshold (default 0.20)
    • Analyst 1 regime is consistent with direction
      THV/TLV → trend-follow (agree with A2 direction)
      MR      → fade (opposing direction allowed)
      PNS     → FLAT always

  Gate 2 — SENTIMENT (Analysts 5 + 6)
    • Sentiment does NOT actively oppose fundamental direction
    • Opus conviction ≥ 0.25 AND not opposing (>0.3 conviction)
    • Both can be NEUTRAL — that passes (abstain = allow)
    • Only BLOCKS if actively contradicts with high conviction

  Gate 3 — TECHNICAL (Analysts 3 + 4)  [OPTIONAL — skipped if unavailable]
    • When models trained: both must confirm direction
    • When unavailable: gate auto-passes (paper trading mode)

Final decision per pair:
  {
    "pair":        "AUDJPY",
    "action":      "SHORT",       # LONG | SHORT | FLAT
    "direction":   "SHORT_BASE",  # SHORT_BASE | LONG_BASE | FLAT
    "confidence":  0.72,          # 0.0 to 1.0
    "size_mult":   1.0,           # 0.5 = half size (MR regime)
    "gate1":       "PASS",        # PASS | FAIL | reason
    "gate2":       "PASS",
    "gate3":       "SKIP",        # SKIP when models unavailable
    "regime":      "THV",
    "a2_score":    0.535,
    "a2_bias":     "LONG_BASE",
    "opus_bias":   "SHORT_BASE",
    "opus_conv":   0.70,
    "sent_bias":   "NEUTRAL",
    "conflict":    "A2_VS_OPUS",  # conflict flag if present
    "reason":      "A2 LONG vs Opus SHORT — conflict, FLAT",
    "updated_at":  "2026-03-09T..."
  }

Conflict resolution (CRITICAL — no ensembling):
  A2 LONG_BASE + Opus SHORT_BASE + conviction > 0.5 → FLAT (conflict)
  A2 LONG_BASE + Opus SHORT_BASE + conviction ≤ 0.5 → defer to A2
  A2 LONG_BASE + Opus NEUTRAL                       → follow A2
  A2 NEUTRAL   + Opus SHORT_BASE                    → FLAT (no A2 confirm)

Redis key: signal:{PAIR}   TTL=15min (re-evaluates every bar)

Usage:
  from execution.signal_aggregator import signal_aggregator
  decisions = signal_aggregator.evaluate_all()
  decision  = signal_aggregator.evaluate_pair("AUDJPY")
  signal_aggregator.print_summary()
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

PAIR_CURRENCIES: Dict[str, Tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"), "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
    "AUDJPY": ("AUD", "JPY"), "NZDJPY": ("NZD", "JPY"),
}

# ── Gate thresholds ───────────────────────────────────────────
A2_MIN_CONFIDENCE     = 0.20   # Analyst 2 min confidence to trade
A2_MIN_SCORE          = 0.20   # |score| must exceed this to be directional
OPUS_BLOCK_CONVICTION = 0.50   # Opus conviction above this actively BLOCKS
OPUS_MIN_CONVICTION   = 0.25   # Opus conviction below this = abstain (neutral)
SENT_BLOCK_THRESHOLD  = 0.30   # |sentiment| above this can block if opposing

# ── Regime → sizing multiplier ────────────────────────────────
REGIME_SIZE: Dict[str, float] = {
    "THV": 1.0,   # Trending High Vol  → full size
    "TLV": 1.0,   # Trending Low Vol   → full size
    "MR":  0.5,   # Mean Reverting     → half size
    "PNS": 0.0,   # Pre-News Stagnation→ flat
}

# ── Regime → allowed directions ──────────────────────────────
# THV/TLV: trend-follow (A2 direction) — MR: fade (opposite direction)
REGIME_DIRECTION_RULE: Dict[str, str] = {
    "THV": "TREND",   # follow A2
    "TLV": "TREND",   # follow A2
    "MR":  "FADE",    # opposite of A2 (mean revert)
    "PNS": "FLAT",
}

# Technical analysts available flag
# Set to True when TFT + PatchTSMixer models are trained and deployed
TECHNICAL_ANALYSTS_AVAILABLE = False

CACHE_TTL_SIGNAL = 900   # 15 min

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
# DATA READERS — one per analyst
# ══════════════════════════════════════════════════════════════

def _read_regime(r, pair: str) -> Dict:
    """Read Analyst 1 regime from Redis. Re-triggers compute if missing."""
    val = r.get(f"features:regime:{pair}")
    if val:
        return json.loads(val)

    # Cache miss — re-trigger regime computation
    logger.debug(f"Aggregator | Regime cache miss for {pair} — recomputing...")
    try:
        from models.analyst1_regime.regime_classifier import regime_classifier
        results = regime_classifier.compute_all()
        val = r.get(f"features:regime:{pair}")
        if val:
            return json.loads(val)
    except Exception as exc:
        logger.warning(f"Aggregator | Regime recompute failed: {exc}")

    # Fallback: unknown regime → treat as TLV (conservative trend-follow)
    return {
        "pair":       pair,
        "regime":     "TLV",
        "confidence": 0.0,
        "source":     "fallback",
    }


def _read_analyst2(r, pair: str) -> Dict:
    """Read Analyst 2 exogenous score from Redis."""
    val = r.get(f"features:analyst2:{pair}")
    if val:
        return json.loads(val)

    logger.debug(f"Aggregator | A2 cache miss for {pair} — recomputing...")
    try:
        from models.analyst2_endo_exo.score_calculator import analyst2
        analyst2.compute_all()
        val = r.get(f"features:analyst2:{pair}")
        if val:
            return json.loads(val)
    except Exception as exc:
        logger.warning(f"Aggregator | A2 recompute failed: {exc}")

    return {
        "pair": pair, "score": 0.0, "bias": "NEUTRAL",
        "confidence": 0.0, "signal_valid": False,
    }


def _read_sentiment(r, pair: str) -> Dict:
    """Read Analyst 5 Groq sentiment from Redis."""
    val = r.get(f"features:sentiment:{pair}")
    if val:
        return json.loads(val)
    return {
        "pair": pair, "sentiment": 0.0,
        "bias": "NEUTRAL", "confidence": 0.0,
    }


def _read_opus_pair(r, pair: str) -> Dict:
    """Read Analyst 6 Opus weekly conviction from Redis."""
    val = r.get(f"features:opus:pair:{pair}")
    if val:
        return json.loads(val)
    return {"bias": "NEUTRAL", "conviction": 0.0, "rationale": "No analysis"}


def _read_technical(r, pair: str) -> Dict:
    """
    Read Analysts 3+4 technical signals from Redis.
    Returns neutral pass when models not yet trained.
    """
    if not TECHNICAL_ANALYSTS_AVAILABLE:
        return {"available": False, "bias": "NEUTRAL", "confidence": 0.0}

    # When models exist: keys are features:tft:{pair} and features:tcn:{pair}
    tft = r.get(f"features:tft:{pair}")
    tcn = r.get(f"features:tcn:{pair}")
    if tft and tcn:
        tft_d = json.loads(tft)
        tcn_d = json.loads(tcn)
        return {
            "available":  True,
            "tft_bias":   tft_d.get("bias", "NEUTRAL"),
            "tft_conf":   tft_d.get("confidence", 0.0),
            "tcn_bias":   tcn_d.get("bias", "NEUTRAL"),
            "tcn_conf":   tcn_d.get("confidence", 0.0),
        }
    return {"available": False, "bias": "NEUTRAL", "confidence": 0.0}


# ══════════════════════════════════════════════════════════════
# GATE LOGIC
# ══════════════════════════════════════════════════════════════

def _gate1_fundamental(
    regime_data: Dict,
    a2_data: Dict,
    pair: str,
) -> Tuple[str, str, str, float]:
    """
    Gate 1: Fundamental confirmation.

    Returns: (gate_result, direction, regime, size_mult)
      gate_result: "PASS" | reason string for FAIL
      direction:   "LONG_BASE" | "SHORT_BASE" | "FLAT"
      regime:      "THV" | "TLV" | "MR" | "PNS"
      size_mult:   0.0 to 1.0
    """
    regime     = regime_data.get("regime", "TLV")
    regime_conf = regime_data.get("confidence", 0.0)
    a2_bias    = a2_data.get("bias", "NEUTRAL")
    a2_score   = a2_data.get("score", 0.0)
    a2_conf    = a2_data.get("confidence", 0.0)
    a2_valid   = a2_data.get("signal_valid", False)

    # PNS → always flat
    if regime == "PNS":
        return "FAIL:PNS_REGIME", "FLAT", regime, 0.0

    # A2 signal must be valid and directional
    if not a2_valid:
        return "FAIL:A2_INVALID", "FLAT", regime, 0.0

    if a2_bias == "NEUTRAL" or abs(a2_score) < A2_MIN_SCORE:
        return "FAIL:A2_NEUTRAL", "FLAT", regime, 0.0

    if a2_conf < A2_MIN_CONFIDENCE:
        return f"FAIL:A2_LOW_CONF({a2_conf:.2f})", "FLAT", regime, 0.0

    # Determine raw direction from A2
    raw_direction = a2_bias  # LONG_BASE or SHORT_BASE

    # Apply regime rule
    rule = REGIME_DIRECTION_RULE.get(regime, "TREND")
    size = REGIME_SIZE.get(regime, 1.0)

    if rule == "FLAT":
        return "FAIL:PNS_REGIME", "FLAT", regime, 0.0

    elif rule == "TREND":
        # Trend-follow: trade in A2 direction
        direction = raw_direction

    elif rule == "FADE":
        # MR regime: fade A2 direction (mean revert)
        direction = (
            "SHORT_BASE" if raw_direction == "LONG_BASE" else "LONG_BASE"
        )

    return "PASS", direction, regime, size


def _gate2_sentiment(
    sentiment_data: Dict,
    opus_data: Dict,
    direction: str,
) -> Tuple[str, str]:
    """
    Gate 2: Sentiment validation.

    Sentiment BLOCKS only if actively opposing with sufficient conviction.
    NEUTRAL from either analyst = abstain = allow trade.

    Returns: (gate_result, conflict_flag)
    """
    sent_bias   = sentiment_data.get("bias", "NEUTRAL")
    sent_conf   = sentiment_data.get("confidence", 0.0)
    sent_score  = sentiment_data.get("sentiment", 0.0)
    opus_bias   = opus_data.get("bias", "NEUTRAL")
    opus_conv   = opus_data.get("conviction", 0.0)

    # Determine if each analyst opposes direction
    def _opposes(bias: str, direction: str) -> bool:
        if bias == "NEUTRAL":
            return False
        if direction == "LONG_BASE" and bias == "SHORT_BASE":
            return True
        if direction == "SHORT_BASE" and bias == "LONG_BASE":
            return True
        return False

    opus_opposes = _opposes(opus_bias, direction)
    sent_opposes = _opposes(
        "SHORT_BASE" if sent_score < -SENT_BLOCK_THRESHOLD
        else "LONG_BASE" if sent_score > SENT_BLOCK_THRESHOLD
        else "NEUTRAL",
        direction
    )

    # Opus blocks if high conviction and opposing
    if opus_opposes and opus_conv >= OPUS_BLOCK_CONVICTION:
        return (
            f"FAIL:OPUS_BLOCKS({opus_bias},conv={opus_conv:.2f})",
            "A2_VS_OPUS"
        )

    # Both opposing with moderate conviction → conflict, be cautious
    if opus_opposes and sent_opposes and opus_conv >= OPUS_MIN_CONVICTION:
        return (
            f"FAIL:BOTH_SENTIMENT_OPPOSE",
            "A2_VS_SENTIMENT"
        )

    # Low-conviction Opus opposition — log but allow
    conflict_flag = ""
    if opus_opposes and opus_conv < OPUS_BLOCK_CONVICTION:
        conflict_flag = "A2_VS_OPUS_WEAK"
        logger.debug(
            f"Aggregator | Weak Opus opposition (conv={opus_conv:.2f}) — "
            f"allowing A2 direction"
        )

    return "PASS", conflict_flag


def _gate3_technical(technical_data: Dict, direction: str) -> str:
    """
    Gate 3: Technical confirmation (Analysts 3+4).

    When TECHNICAL_ANALYSTS_AVAILABLE=False: always SKIP (paper trading).
    When available: both TFT and TCN must agree with direction.
    """
    if not technical_data.get("available", False):
        return "SKIP"

    tft_bias = technical_data.get("tft_bias", "NEUTRAL")
    tcn_bias = technical_data.get("tcn_bias", "NEUTRAL")
    tft_conf = technical_data.get("tft_conf", 0.0)
    tcn_conf = technical_data.get("tcn_conf", 0.0)

    def _agrees(bias: str, direction: str) -> bool:
        return bias == direction or bias == "NEUTRAL"

    if not _agrees(tft_bias, direction):
        return f"FAIL:TFT_DISAGREES({tft_bias})"
    if not _agrees(tcn_bias, direction):
        return f"FAIL:TCN_DISAGREES({tcn_bias})"

    return "PASS"


def _compute_confidence(
    a2_data: Dict,
    opus_data: Dict,
    sentiment_data: Dict,
    regime_data: Dict,
    conflict_flag: str,
) -> float:
    """
    Compute final trade confidence score.

    Sources:
      - Analyst 2 confidence (base)
      - Opus conviction (boosts confidence when aligned)
      - Sentiment confidence (minor boost)
      - Regime confidence (minor boost)
      - Conflict penalty (reduces if weak conflict present)
    """
    a2_conf      = a2_data.get("confidence", 0.0)
    opus_conv    = opus_data.get("conviction", 0.0)
    sent_conf    = sentiment_data.get("confidence", 0.0)
    regime_conf  = regime_data.get("confidence", 0.0)

    opus_bias    = opus_data.get("bias", "NEUTRAL")
    a2_bias      = a2_data.get("bias", "NEUTRAL")

    # Base: A2 confidence
    conf = a2_conf * 0.50

    # Opus boost: aligned and high conviction
    opus_aligned = (
        (a2_bias == "LONG_BASE"  and opus_bias == "LONG_BASE") or
        (a2_bias == "SHORT_BASE" and opus_bias == "SHORT_BASE")
    )
    if opus_aligned:
        conf += opus_conv * 0.30
    elif opus_bias == "NEUTRAL":
        conf += opus_conv * 0.10   # abstain = small boost

    # Sentiment boost (minor)
    conf += sent_conf * 0.10

    # Regime confidence boost (minor)
    conf += regime_conf * 0.10

    # Conflict penalty
    if conflict_flag == "A2_VS_OPUS_WEAK":
        conf *= 0.85   # -15% for weak conflict

    return float(min(round(conf, 4), 1.0))


# ══════════════════════════════════════════════════════════════
# MAIN AGGREGATOR
# ══════════════════════════════════════════════════════════════

class SignalAggregator:
    """
    Signal Aggregator — Final trade decision engine.

    Reads all analyst outputs from Redis, applies 3-gate logic,
    produces a final LONG/SHORT/FLAT decision per pair.

    Architecture rule:
      This is NOT an ensemble. Gates are sequential and independent.
      A trade fires ONLY when all available gates PASS.
      Each gate can independently veto a trade.

    The aggregator does NOT execute trades.
    It produces decisions which are consumed by order_executor.py.
    """

    def __init__(self) -> None:
        self._redis = None

    @property
    def r(self):
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    # ── Public API ────────────────────────────────────────────

    def evaluate_pair(self, pair: str) -> Dict:
        """
        Evaluate a single pair through all gates.
        Returns the full decision dict.
        """
        # Load all analyst data
        regime_data    = _read_regime(self.r, pair)
        a2_data        = _read_analyst2(self.r, pair)
        sentiment_data = _read_sentiment(self.r, pair)
        opus_data      = _read_opus_pair(self.r, pair)
        tech_data      = _read_technical(self.r, pair)

        now = datetime.now(timezone.utc).isoformat()

        # ── Gate 1: Fundamental ───────────────────────────────
        g1_result, direction, regime, size_mult = _gate1_fundamental(
            regime_data, a2_data, pair
        )

        if not g1_result.startswith("PASS"):
            decision = self._flat_decision(
                pair, g1_result, "PASS_SKIPPED", "SKIP",
                regime_data, a2_data, opus_data, sentiment_data,
                g1_result, now
            )
            self._cache(pair, decision)
            return decision

        # ── Gate 2: Sentiment ─────────────────────────────────
        g2_result, conflict_flag = _gate2_sentiment(
            sentiment_data, opus_data, direction
        )

        if not g2_result.startswith("PASS"):
            decision = self._flat_decision(
                pair, "PASS", g2_result, "SKIP",
                regime_data, a2_data, opus_data, sentiment_data,
                conflict_flag, now
            )
            self._cache(pair, decision)
            return decision

        # ── Gate 3: Technical ─────────────────────────────────
        g3_result = _gate3_technical(tech_data, direction)

        if g3_result.startswith("FAIL"):
            decision = self._flat_decision(
                pair, "PASS", "PASS", g3_result,
                regime_data, a2_data, opus_data, sentiment_data,
                "TECH_BLOCK", now
            )
            self._cache(pair, decision)
            return decision

        # ── All gates passed — compute final decision ─────────
        confidence = _compute_confidence(
            a2_data, opus_data, sentiment_data,
            regime_data, conflict_flag
        )

        # Convert direction to action
        action = "LONG" if direction == "LONG_BASE" else "SHORT"

        decision = {
            "pair":       pair,
            "action":     action,
            "direction":  direction,
            "confidence": confidence,
            "size_mult":  size_mult,
            "gate1":      g1_result,
            "gate2":      g2_result,
            "gate3":      g3_result,
            "regime":     regime,
            "a2_score":   round(a2_data.get("score", 0.0), 4),
            "a2_bias":    a2_data.get("bias", "NEUTRAL"),
            "a2_conf":    round(a2_data.get("confidence", 0.0), 4),
            "opus_bias":  opus_data.get("bias", "NEUTRAL"),
            "opus_conv":  round(opus_data.get("conviction", 0.0), 4),
            "sent_bias":  sentiment_data.get("bias", "NEUTRAL"),
            "sent_score": round(sentiment_data.get("sentiment", 0.0), 4),
            "conflict":   conflict_flag,
            "reason":     f"All gates passed | {direction} | regime={regime}",
            "updated_at": now,
        }

        self._cache(pair, decision)
        return decision

    def evaluate_all(self) -> Dict[str, Dict]:
        """Evaluate all 12 pairs and return full decision dict."""
        logger.info("Aggregator | Evaluating all 12 pairs...")
        results = {}
        for pair in PAIRS:
            try:
                results[pair] = self.evaluate_pair(pair)
            except Exception as exc:
                logger.error(f"Aggregator | {pair} failed: {exc}")
                results[pair] = self._emergency_flat(pair)
        return results

    def get_tradeable(self) -> List[Dict]:
        """
        Return only pairs with actionable signals (LONG or SHORT).
        Sorted by confidence descending.
        """
        results = self.evaluate_all()
        tradeable = [
            d for d in results.values()
            if d.get("action") in ("LONG", "SHORT")
        ]
        return sorted(tradeable, key=lambda x: x["confidence"], reverse=True)

    def get_cached_decision(self, pair: str) -> Optional[Dict]:
        """Get cached signal from Redis without recomputing."""
        try:
            val = self.r.get(f"signal:{pair}")
            if val:
                return json.loads(val)
        except Exception:
            pass
        return None

    def start_scheduler(self) -> None:
        """Evaluate all pairs every 15 minutes in background."""
        def _loop() -> None:
            logger.info("Aggregator | Scheduler started (15min interval)")
            while True:
                try:
                    results = self.evaluate_all()
                    tradeable = [
                        p for p, d in results.items()
                        if d["action"] in ("LONG", "SHORT")
                    ]
                    logger.info(
                        f"Aggregator | Evaluation complete | "
                        f"Tradeable: {tradeable}"
                    )
                except Exception as exc:
                    logger.error(f"Aggregator | Scheduler error: {exc}")
                time.sleep(CACHE_TTL_SIGNAL)

        t = threading.Thread(
            target=_loop, daemon=True, name="aggregator-scheduler"
        )
        t.start()
        logger.info("Aggregator | Scheduler thread launched")

    def print_summary(self) -> None:
        """Print formatted signal summary table for all pairs."""
        results = self.evaluate_all()

        print("\n" + "=" * 85)
        print("SIGNAL AGGREGATOR — FINAL DECISIONS")
        print("=" * 85)
        print(
            f"{'PAIR':<10} {'ACTION':<7} {'CONF':>5} {'SIZE':>5} "
            f"{'REG':<5} {'A2':>6} {'OPUS':<12} {'SENT':<10} "
            f"{'G1':<6} {'G2':<6} {'G3':<6} CONFLICT"
        )
        print("-" * 85)

        for pair in PAIRS:
            d = results.get(pair, {})
            action    = d.get("action", "FLAT")
            conf      = d.get("confidence", 0.0)
            size      = d.get("size_mult", 0.0)
            regime    = d.get("regime", "?")
            a2_score  = d.get("a2_score", 0.0)
            opus_bias = d.get("opus_bias", "?")[:10]
            sent_bias = d.get("sent_bias", "?")[:8]
            g1        = "PASS" if d.get("gate1","").startswith("PASS") else "FAIL"
            g2        = "PASS" if d.get("gate2","").startswith("PASS") else (
                        "SKIP" if d.get("gate2","") == "PASS_SKIPPED" else "FAIL"
            )
            g3        = d.get("gate3", "SKIP")[:6]
            conflict  = d.get("conflict", "")[:15]

            # Colour coding via text markers
            action_str = (
                f"▲{action}" if action == "LONG"
                else f"▼{action}" if action == "SHORT"
                else f" {action}"
            )

            print(
                f"{pair:<10} {action_str:<7} {conf:>5.2f} {size:>5.1f} "
                f"{regime:<5} {a2_score:>+6.3f} {opus_bias:<12} {sent_bias:<10} "
                f"{g1:<6} {g2:<6} {g3:<6} {conflict}"
            )

        print("=" * 85)
        tradeable = [
            p for p, d in results.items()
            if d.get("action") in ("LONG", "SHORT")
        ]
        print(f"Tradeable pairs: {tradeable}")
        print(
            f"Technical gate: "
            f"{'ACTIVE' if TECHNICAL_ANALYSTS_AVAILABLE else 'SKIP (models pending)'}"
        )
        print("=" * 85 + "\n")

    # ── Internal helpers ──────────────────────────────────────

    def _cache(self, pair: str, decision: Dict) -> None:
        """Cache signal in Redis with 15min TTL."""
        try:
            self.r.setex(
                f"signal:{pair}",
                CACHE_TTL_SIGNAL,
                json.dumps(decision)
            )
        except Exception as exc:
            logger.warning(f"Aggregator | Cache write failed {pair}: {exc}")

    @staticmethod
    def _flat_decision(
        pair: str,
        g1: str, g2: str, g3: str,
        regime_data: Dict,
        a2_data: Dict,
        opus_data: Dict,
        sentiment_data: Dict,
        conflict: str,
        now: str,
    ) -> Dict:
        """Build a FLAT decision dict."""
        return {
            "pair":       pair,
            "action":     "FLAT",
            "direction":  "FLAT",
            "confidence": 0.0,
            "size_mult":  0.0,
            "gate1":      g1,
            "gate2":      g2,
            "gate3":      g3,
            "regime":     regime_data.get("regime", "?"),
            "a2_score":   round(a2_data.get("score", 0.0), 4),
            "a2_bias":    a2_data.get("bias", "NEUTRAL"),
            "a2_conf":    round(a2_data.get("confidence", 0.0), 4),
            "opus_bias":  opus_data.get("bias", "NEUTRAL"),
            "opus_conv":  round(opus_data.get("conviction", 0.0), 4),
            "sent_bias":  sentiment_data.get("bias", "NEUTRAL"),
            "sent_score": round(sentiment_data.get("sentiment", 0.0), 4),
            "conflict":   conflict,
            "reason":     f"FLAT | g1={g1} g2={g2} g3={g3}",
            "updated_at": now,
        }

    @staticmethod
    def _emergency_flat(pair: str) -> Dict:
        """Emergency flat on unexpected error."""
        return {
            "pair": pair, "action": "FLAT", "direction": "FLAT",
            "confidence": 0.0, "size_mult": 0.0,
            "gate1": "FAIL:ERROR", "gate2": "FAIL:ERROR", "gate3": "SKIP",
            "regime": "?", "a2_score": 0.0, "a2_bias": "NEUTRAL",
            "a2_conf": 0.0, "opus_bias": "NEUTRAL", "opus_conv": 0.0,
            "sent_bias": "NEUTRAL", "sent_score": 0.0,
            "conflict": "ERROR", "reason": "Emergency flat — exception",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


# ── Module singleton ──────────────────────────────────────────
signal_aggregator = SignalAggregator()


# ── Direct run ────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")

    logger.info("Signal Aggregator | Running evaluation...")
    signal_aggregator.print_summary()

    tradeable = signal_aggregator.get_tradeable()
    if tradeable:
        logger.success(
            f"Signal Aggregator | {len(tradeable)} tradeable signals:"
        )
        for d in tradeable:
            logger.success(
                f"  {d['pair']} {d['action']} | "
                f"conf={d['confidence']:.2f} | "
                f"size={d['size_mult']:.1f}x | "
                f"regime={d['regime']} | "
                f"{d['reason']}"
            )
    else:
        logger.info("Signal Aggregator | No tradeable signals right now")