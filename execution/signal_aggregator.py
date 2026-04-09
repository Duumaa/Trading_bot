"""
execution/signal_aggregator.py
===============================
Signal Aggregator — Gate Logic Engine

Wires all 6 analysts into a final LONG/SHORT/FLAT decision per pair.
This is NOT an ensemble. Each analyst group gates independently.

Gate logic (2-gate minimum, 2026-04-07):
  Gate 1 — FUNDAMENTAL (A1 LightGBM OR HMM)
    • Entry fires when EITHER A1 or HMM detects THV/TLV trending regime
    • PNS → always flat regardless
    • HMM requires conf ≥ 0.40 to lead
    • MACRO_OVERRIDE: A5+A6 strong conviction on MR → upgrade to TLV (existing)

  Gate 2 — TECHNICAL (A4: TCN or PatchTSMixer)  [DIRECTION SOURCE + HARD GATE]
    • A4 now determines trade direction (not A2)
    • PASS = both models agree | PASS_SINGLE = one model agrees
    • FAIL = no direction → FLAT

  Soft modifiers — reduce size_mult, NEVER block:
    • A2 opposing directional view   → ×0.80
    • A5 Groq opposing (conf ≥ 0.30) → ×0.85
    • A6 Opus opposing (conv ≥ 0.25) → ×0.90

  Gate 4 — ENTRY TIMING (rule-based, still hard gate)
    • Entry engine score ≥ 0.40 required

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

from intelligence.apex_framework import apex_framework   # noqa: E402  (import after sys.path)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY",
    # "USDCHF" — SUSPENDED 2026-03-27: 5× MODEL_FAILURE (-$45)
    # SNB intervention risk makes this pair unmodelable. Re-enable after retrain.
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
A2_MIN_CONFIDENCE     = 0.20   # Analyst 2 min confidence to trade (default)

# Per-pair minimum confidence overrides — raised for pairs with high MODEL_FAILURE rate
# USD pairs (USDJPY, USDCAD) raised 2026-03-27 after postmortem: 12 MODEL_FAILUREs, -$115
PAIR_MIN_CONFIDENCE: Dict[str, float] = {
    "USDJPY": 0.72,
    "USDCAD": 0.72,
    "EURUSD": 0.20,   # default — GBPJPY/EURUSD MODEL_FAILURE were profitable
    "GBPUSD": 0.20,
    "AUDUSD": 0.20,
    "NZDUSD": 0.20,
    "EURGBP": 0.20,
    "EURJPY": 0.20,
    "GBPJPY": 0.20,
    "AUDJPY": 0.20,
    "NZDJPY": 0.20,
}
# LightGBM per-pair accuracy dampener (from Kaggle training results 2026-03-16)
# Tier 1 (88–90% accuracy) → 1.00 | Tier 2 (81–83%) → 0.94–0.95
# Tier 3 (79–81%) → 0.90–0.92

PAIR_MODEL_QUALITY: Dict[str, float] = {
    "EURGBP": 1.00, "GBPUSD": 1.00, "USDCAD": 1.00,
    "EURUSD": 1.00, "USDCHF": 1.00, "AUDUSD": 0.95,
    "EURJPY": 0.95, "NZDUSD": 0.95, "NZDJPY": 0.94,
    "USDJPY": 0.90, "GBPJPY": 0.92, "AUDJPY": 0.92,
}

A2_MIN_SCORE          = 0.20   # |score| must exceed this to be directional

# ── Gate 1 combined (A1 × A2) thresholds ──────────────────────────────────
# combined_conf = A1_regime_conf × 0.60 + |A2_score| × 0.40
# Replaces A2 standalone veto: A2 neutral no longer blocks if A1 is confident.
# Direction deferred to A4 when A2 is neutral (same as existing MR_DEFER path).
GATE1_MIN_CONF: float = 0.40
GATE1_PAIR_MIN_CONF: Dict[str, float] = {
    "USDJPY": 0.52,   # raised — high MODEL_FAILURE rate (postmortem 2026-03-27)
    "USDCAD": 0.52,   # raised — high MODEL_FAILURE rate
}
PASS_QUEUE_THRESHOLD  = 0.55   # below this, entry is queued (not executed) — recovery next bar
OPUS_BLOCK_CONVICTION = 0.50   # Opus conviction above this actively BLOCKS
OPUS_MIN_CONVICTION   = 0.25   # Opus conviction below this = abstain (neutral)
SENT_BLOCK_THRESHOLD  = 0.30   # |sentiment| above this can block if opposing

# ── Macro conviction override thresholds ──────────────────────────────────
# When A1=MR but A5+A6 strongly agree on direction, override MR → TLV.
# Both analysts must agree and independently meet their thresholds.
MACRO_OVERRIDE_SENT_MIN  = 0.20   # |sentiment| minimum for A5 to contribute
MACRO_OVERRIDE_SENT_CONF = 0.30   # A5 confidence minimum
MACRO_OVERRIDE_OPUS_CONV = 0.45   # A6 conviction minimum (weekly macro view)

# ── Regime → sizing multiplier ────────────────────────────────
REGIME_SIZE: Dict[str, float] = {
    "THV": 1.0,   # Trending High Vol  → full size
    "TLV": 0.5,   # Trending Low Vol   → Trending Low Vol   → half size (62.3% precision, reduced risk)
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
TECHNICAL_ANALYSTS_AVAILABLE = True

CACHE_TTL_SIGNAL = 1200  # 20 min — intentional overlap with M15 cycle to avoid stale-key gap

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

def _read_regime_hmm(r, pair: str) -> Optional[Dict]:
    """
    Read HMM regime from Redis (features:regime_hmm:{pair}).

    Returns None if no HMM model trained for this pair (graceful degradation).
    HMM is a parallel validator — missing = no penalty, not an error.
    """
    try:
        val = r.get(f"features:regime_hmm:{pair}")
        if val:
            return json.loads(val)
    except Exception:
        pass
    return None


def _update_regime_history(r, pair: str, regime_data: Dict) -> None:
    """
    Push current regime probability snapshot to Redis history list.
    Keeps last 6 readings (= 90 minutes of M15 candles).
    Key: regime:stability:{pair}
    """
    try:
        probs = regime_data.get("probabilities", {})
        if not probs:
            return

        top_regime  = regime_data.get("regime", "TLV")
        top_prob    = probs.get(top_regime, 0.0)
        sorted_prob = sorted(probs.values(), reverse=True)
        second_prob = sorted_prob[1] if len(sorted_prob) > 1 else 0.0
        gap         = round(top_prob - second_prob, 4)

        entry = {
            "top_regime":  top_regime,
            "top_prob":    round(top_prob, 4),
            "second_prob": round(second_prob, 4),
            "gap":         gap,
            "ts":          datetime.now(timezone.utc).isoformat(),
        }

        key = f"regime:stability:{pair}"
        # Read existing history
        raw = r.get(key)
        history = json.loads(raw) if raw else []
        history.append(entry)
        history = history[-6:]   # keep last 6 candles only

        r.setex(key, 7200, json.dumps(history))  # TTL=2h

    except Exception as exc:
        logger.debug(f"Aggregator | Regime history update failed {pair}: {exc}")


def _check_regime_stability(r, pair: str, regime_data: Dict) -> tuple:
    """
    Detect if the current regime is about to shift by tracking
    the gap between top-2 regime probabilities over the last 6 candles.

    Instability signals:
      1. Current gap < 0.25  → two regimes nearly tied RIGHT NOW
      2. Gap shrinking > 0.08/reading over last 4 readings → converging

    Returns:
        (is_unstable: bool, reason: str, current_gap: float)

    Trading response: size_mult × 0.25 when unstable (skip marginal trades)
    """
    try:
        probs = regime_data.get("probabilities", {})
        if not probs:
            return False, "", 1.0

        top_regime  = regime_data.get("regime", "TLV")
        top_prob    = probs.get(top_regime, 0.0)
        sorted_prob = sorted(probs.values(), reverse=True)
        second_prob = sorted_prob[1] if len(sorted_prob) > 1 else 0.0
        current_gap = round(top_prob - second_prob, 4)

        # Rule 1 — gap too small right now
        if current_gap < 0.25:
            reason = (
                f"REGIME_UNSTABLE: gap={current_gap:.2f} "
                f"({top_regime}={top_prob:.2f} vs 2nd={second_prob:.2f})"
            )
            return True, reason, current_gap

        # Rule 2 — gap shrinking fast over last 4 readings
        key = f"regime:stability:{pair}"
        raw = r.get(key)
        if raw:
            history = json.loads(raw)
            if len(history) >= 4:
                gaps = [h["gap"] for h in history[-4:]]
                # Check if each reading is lower than the previous
                shrinking = all(
                    gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)
                )
                total_shrink = gaps[0] - gaps[-1]
                if shrinking and total_shrink > 0.20:
                    reason = (
                        f"REGIME_SHIFTING: gap shrinking "
                        f"{gaps[0]:.2f}→{gaps[-1]:.2f} over 4 candles"
                    )
                    return True, reason, current_gap

        return False, "", current_gap

    except Exception as exc:
        logger.debug(f"Aggregator | Regime stability check failed {pair}: {exc}")
        return False, "", 1.0



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
    Read Analyst 4 TCN + PatchTSMixer signal from Redis.
    Key: features:analyst4:{pair}  TTL=900s
    Returns neutral unavailable when flag is False or key missing.
    """
    if not TECHNICAL_ANALYSTS_AVAILABLE:
        return {"available": False, "direction": "FLAT", "confidence": 0.0}

    val = r.get(f"features:analyst4:{pair}")
    if val:
        d = json.loads(val)
        return {
            "available"  : True,
            "direction"  : d.get("direction",  "FLAT"),
            "confidence" : d.get("confidence", 0.0),
            "tcn_prob"   : d.get("tcn_prob",   0.5),
            "patch_prob" : d.get("patch_prob",  0.5),
            "agreed"     : d.get("agreed",      False),
            "gate"       : d.get("gate",        "FAIL"),
            "regime"     : d.get("regime",      "MR"),
        }
    return {"available": False, "direction": "FLAT", "confidence": 0.0}

def _read_entry(r, pair: str) -> Dict:
    """Read Analyst 4 entry engine signal from Redis."""
    val = r.get(f"features:entry:{pair}")
    if val:
        return json.loads(val)
    # Cache miss — compute now
    try:
        from features.entry_engine import entry_engine
        results = entry_engine.compute_all()
        val = r.get(f"features:entry:{pair}")
        if val:
            return json.loads(val)
    except Exception as exc:
        logger.warning(f"Aggregator | Entry engine recompute failed: {exc}")
    return {
        "entry_score": 0.5, "entry_type": "WAIT",
        "wait": True, "reason": "No entry data",
        "direction": "FLAT", "available": False,
    }

# ══════════════════════════════════════════════════════════════
# GATE LOGIC
# ══════════════════════════════════════════════════════════════

def _check_macro_conviction(
    sentiment_data: Dict,
    opus_data: Dict,
) -> Tuple[str, str]:
    """
    Evaluate whether A5 (Groq sentiment) + A6 (Opus weekly) have enough
    combined conviction to override A1's MR regime classification.

    Both analysts must independently meet their thresholds AND agree on
    direction. If they conflict, return FLAT — do not override.

    Returns: (direction "LONG_BASE"|"SHORT_BASE"|"FLAT", reason_str)
    """
    sent_score = sentiment_data.get("sentiment", 0.0)
    sent_conf  = sentiment_data.get("confidence", 0.0)
    opus_bias  = opus_data.get("bias", "NEUTRAL")
    opus_conv  = opus_data.get("conviction", 0.0)

    # A5 must have enough signal and confidence
    if abs(sent_score) < MACRO_OVERRIDE_SENT_MIN or sent_conf < MACRO_OVERRIDE_SENT_CONF:
        return "FLAT", ""

    # A6 must have a directional view with sufficient conviction
    if opus_bias == "NEUTRAL" or opus_conv < MACRO_OVERRIDE_OPUS_CONV:
        return "FLAT", ""

    # Determine A5 direction
    a5_dir = "LONG_BASE" if sent_score > 0 else "SHORT_BASE"

    # Both must agree — conflict = no override
    if a5_dir != opus_bias:
        return "FLAT", ""

    reason = (
        f"A5={sent_score:+.2f}(conf={sent_conf:.2f}),"
        f"A6={opus_bias}(conv={opus_conv:.2f})"
    )
    return opus_bias, reason


def _gate1_fundamental(
    r,
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

    # Always update history — even on FAIL paths
    # Stability tracking needs data regardless of trade outcome
    _update_regime_history(r, pair, regime_data)

    # PNS → always flat
    if regime == "PNS":
        return "FAIL:PNS_REGIME", "FLAT", regime, 0.0

    # A2 signal must be valid and directional
    if not a2_valid:
        return "FAIL:A2_INVALID", "FLAT", regime, 0.0

    size = REGIME_SIZE.get(regime, 1.0)

    # ── MR regime special handling ────────────────────────────────────────
    # In MR, A2 neutral is EXPECTED. Direction deferred to A4 entry engine.
    if regime == "MR":
        if a2_bias != "NEUTRAL" and abs(a2_score) >= A2_MIN_SCORE:
            direction = "SHORT_BASE" if a2_bias == "LONG_BASE" else "LONG_BASE"
            return "PASS:MR_FADE", direction, regime, size
        else:
            return "PASS:MR_DEFER", "FLAT", regime, size

    # ── THV / TLV — combined A1 × A2 single gate ─────────────────────────
    # A2 NEUTRAL no longer hard-blocks. Gate score = A1_conf × 0.60 + |A2_score| × 0.40.
    # When A2 has no macro view, A1 regime confidence carries the gate alone.
    # Direction: A2-derived when directional; deferred to A4 when neutral.
    size = REGIME_SIZE.get(regime, 1.0)

    a2_magnitude  = abs(a2_score) if (a2_bias != "NEUTRAL" and abs(a2_score) >= A2_MIN_SCORE) else 0.0
    combined_conf = regime_conf * 0.60 + a2_magnitude * 0.40
    gate1_min     = GATE1_PAIR_MIN_CONF.get(pair, GATE1_MIN_CONF)

    if combined_conf < gate1_min:
        return (
            f"FAIL:GATE1({combined_conf:.2f}<{gate1_min:.2f})",
            "FLAT", regime, 0.0,
        )

    # Determine direction and size multiplier
    if a2_bias != "NEUTRAL" and abs(a2_score) >= A2_MIN_SCORE:
        direction = a2_bias          # THV/TLV: trend-follow A2
        size_mult = size
    else:
        direction = "FLAT"           # resolved from A4 in _fundamental_path
        size_mult = round(size * 0.75, 2)   # no macro backing → 0.75×

    # ── Regime stability check ────────────────────────────────────────────
    _update_regime_history(r, pair, regime_data)
    is_unstable, unstable_reason, gap = _check_regime_stability(r, pair, regime_data)
    if is_unstable:
        reduced_size = round(size_mult * 0.25, 2)
        logger.warning(
            f"Aggregator | {pair} {unstable_reason} "
            f"— size reduced {size_mult}x → {reduced_size}x"
        )
        tag = "PASS:A2_NEUTRAL_DEFER" if direction == "FLAT" else "PASS"
        return f"{tag}+UNSTABLE(gap={gap:.2f})", direction, regime, reduced_size

    if direction == "FLAT":
        return "PASS:A2_NEUTRAL_DEFER", "FLAT", regime, size_mult

    return "PASS", direction, regime, size_mult


def _hmm_regime_tag(hmm_regime: str, a1_regime: str, upgraded: bool) -> str:
    """
    Descriptive tag for the gate1b signal field — NOT a gate, purely for postmortem.

    After postmortem accumulates trades we can see:
      HMM_UPGRADE  — HMM detected trend A1 missed. Did these trades win?
      A1_LEADS     — A1 detected trend HMM missed. Did these trades win?
      AGREE        — both models agree on category.
      SKIP         — no HMM model available.

    This attribution lets us decide later whether to trust A1, HMM, or require both.
    """
    if not hmm_regime:
        return "SKIP"
    if upgraded:
        return f"HMM_UPGRADE(a1=MR,hmm={hmm_regime})"
    TRENDING = {"THV", "TLV"}
    a1_trend  = a1_regime  in TRENDING
    hmm_trend = hmm_regime in TRENDING
    if a1_trend and not hmm_trend:
        return f"A1_LEADS(a1={a1_regime},hmm={hmm_regime})"
    if hmm_trend and not a1_trend:
        return f"HMM_LEADS(hmm={hmm_regime},a1={a1_regime})"   # shouldn't happen post-upgrade
    return "AGREE"


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
    Gate 3: Technical ML confirmation (Analyst 4 — TCN + PatchTSMixer).
    When TECHNICAL_ANALYSTS_AVAILABLE=False: always SKIP.
    When available: either model agreeing is sufficient (OR logic).
      - Both agree  → PASS       (full confidence, unchanged)
      - One agrees  → PASS_SINGLE (85% confidence, set in analyst4_inference)
      - Conflict    → FAIL
      - Neither     → FAIL
    """
    if not technical_data.get("available", False):
        return "SKIP"

    a4_direction = technical_data.get("direction", "FLAT")
    a4_gate      = technical_data.get("gate", "FAIL")
    a4_conf      = technical_data.get("confidence", 0.0)
    tcn_prob     = technical_data.get("tcn_prob", 0.5)
    patch_prob   = technical_data.get("patch_prob", 0.5)

    if a4_gate not in ("PASS", "PASS_SINGLE"):
        return f"FAIL:A4_NOAGREE(tcn={tcn_prob:.2f},patch={patch_prob:.2f})"

    if a4_direction == "FLAT":
        return "FAIL:A4_FLAT"

    # Normalise direction strings — A4 uses LONG/SHORT, Gate1 uses LONG_BASE/SHORT_BASE
    a4_norm = "LONG_BASE" if a4_direction == "LONG" else ("SHORT_BASE" if a4_direction == "SHORT" else "FLAT")
    if a4_norm != direction:
        return f"FAIL:A4_DISAGREES({a4_direction}_vs_{direction})"

    if a4_gate == "PASS_SINGLE":
        return f"PASS_SINGLE:A4({a4_direction},conf={a4_conf:.2f})"

    return f"PASS:A4({a4_direction},conf={a4_conf:.2f})"
def _apply_soft_modifiers(
    size: float,
    direction: str,
    a2_data: Optional[Dict],
    sentiment_data: Optional[Dict],
    opus_data: Optional[Dict],
) -> Tuple[float, List[str]]:
    """
    Apply soft size penalties when A2/A5/A6 oppose the trade direction.
    These analysts reduce size_mult but never block a trade.

    Penalties (multiplicative):
      A2 directional macro opposing    → ×0.80
      A5 Groq opposing (conf ≥ 0.30)  → ×0.85
      A6 Opus opposing (conv ≥ 0.25)  → ×0.90

    Returns: (new_size_mult, list_of_modifier_reasons)
    """
    reasons: List[str] = []
    mult = size

    def _opposes(bias: str, direction: str) -> bool:
        if bias == "NEUTRAL":
            return False
        return (direction == "LONG_BASE"  and bias == "SHORT_BASE") or \
               (direction == "SHORT_BASE" and bias == "LONG_BASE")

    if a2_data:
        a2_bias  = a2_data.get("bias", "NEUTRAL")
        a2_score = a2_data.get("score", 0.0)
        if a2_bias != "NEUTRAL" and abs(a2_score) >= A2_MIN_SCORE and _opposes(a2_bias, direction):
            mult = round(mult * 0.80, 4)
            reasons.append(f"A2_OPPOSES({a2_bias},score={a2_score:+.2f},×0.80)")

    if sentiment_data:
        sent_score = sentiment_data.get("sentiment", 0.0)
        sent_conf  = sentiment_data.get("confidence", 0.0)
        sent_bias  = (
            "SHORT_BASE" if sent_score < -SENT_BLOCK_THRESHOLD
            else "LONG_BASE" if sent_score > SENT_BLOCK_THRESHOLD
            else "NEUTRAL"
        )
        if sent_conf >= 0.30 and _opposes(sent_bias, direction):
            mult = round(mult * 0.85, 4)
            reasons.append(f"A5_OPPOSES(score={sent_score:+.2f},conf={sent_conf:.2f},×0.85)")

    if opus_data:
        opus_bias = opus_data.get("bias", "NEUTRAL")
        opus_conv = opus_data.get("conviction", 0.0)
        if opus_conv >= OPUS_MIN_CONVICTION and _opposes(opus_bias, direction):
            mult = round(mult * 0.90, 4)
            reasons.append(f"A6_OPPOSES({opus_bias},conv={opus_conv:.2f},×0.90)")

    return mult, reasons


def _gate3_entry(entry_data: Dict, direction: str) -> str:
    """
    Gate 3: Price action entry timing (Analyst 4 — EntryEngine).

    Validates WHEN to enter based on VWAP, RSI, OBI, and momentum.
    Direction already confirmed by Gates 1+2. This gate ensures we
    don't enter at a bad price (extended, wide spread, OBI opposing).

    Returns:
      PASS            — entry_score >= 0.65, enter this bar
      PASS_WAIT       — entry_score 0.55–0.65, enter now (marginal but acceptable)
      PASS_QUEUE      — entry_score 0.40–0.55, do NOT enter, queue for next bar
      FAIL:reason     — entry_score < 0.40, skip this bar entirely
    """
    if not entry_data.get("available", False):
        # Entry engine not yet computed — allow trade (conservative default)
        return "PASS_WAIT"

    score      = entry_data.get("entry_score", 0.0)
    entry_type = entry_data.get("entry_type", "WAIT")

    # Hard fail — setup is actively bad
    if score < 0.40:
        return f"FAIL:ENTRY_SCORE_LOW({score:.2f})|{entry_type}"

    # Good entry — fire now
    if score >= 0.65:
        return f"PASS|{entry_type}|score={score:.2f}"

    # Acceptable entry — execute with awareness of timing risk
    if score >= PASS_QUEUE_THRESHOLD:
        return f"PASS_WAIT|{entry_type}|score={score:.2f}"

    # Marginal — queue for next bar, do not execute yet
    return f"PASS_QUEUE|{entry_type}|score={score:.2f}"

def _compute_confidence(
    a2_data: Dict,
    opus_data: Dict,
    sentiment_data: Dict,
    regime_data: Dict,
    conflict_flag: str,
    tech_data: Optional[Dict] = None,
) -> float:
    """
    Compute final trade confidence score.

    Sources:
      - Analyst 2 confidence (base)
      - Opus conviction (boosts confidence when aligned)
      - Sentiment confidence (minor boost)
      - Regime confidence (minor boost)
      - A4 TCN+PatchTSMixer confidence (boost when strong)
      - Conflict penalty (reduces if weak conflict present)
    """
    a2_conf      = a2_data.get("confidence", 0.0)
    opus_conv    = opus_data.get("conviction", 0.0)
    sent_conf    = sentiment_data.get("confidence", 0.0)
    regime_conf  = regime_data.get("confidence", 0.0)

    opus_bias    = opus_data.get("bias", "NEUTRAL")
    a2_bias      = a2_data.get("bias", "NEUTRAL")

    # Base: A2 confidence when A2 has a view; A1 regime confidence when A2 neutral
    a2_score_val = a2_data.get("score", 0.0)
    if a2_bias == "NEUTRAL" or abs(a2_score_val) < A2_MIN_SCORE:
        conf = regime_conf * 0.50   # A1 carries the base when A2 has no macro view
    else:
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

    # ── A4 TCN + PatchTSMixer confidence boost ─────────────────────────
    if tech_data and tech_data.get("agreed", False):
        a4_conf = tech_data.get("confidence", 0.0)
        if a4_conf > 0.5:
            conf *= 1.10
        elif a4_conf > 0.3:
            conf *= 1.05

    # Conflict penalty
    if conflict_flag == "A2_VS_OPUS_WEAK":
        conf *= 0.85   # -15% for weak conflict

    return float(min(round(conf, 4), 1.0))


def _compute_confidence_v2(
    a4_conf: float,
    a1_conf: float,
    hmm_conf: float,
    hmm_contributed: bool,
    direction: str,
    a2_data: Dict,
    sentiment_data: Optional[Dict],
    opus_data: Optional[Dict],
) -> float:
    """
    Confidence under 2-gate architecture.

    A4 carries 60% (it is the direction gate).
    A1 regime confidence: 15%. HMM if it led: +10%.
    Aligned A2/A5/A6 each add a minor boost.
    """
    conf = a4_conf * 0.60 + a1_conf * 0.15

    if hmm_contributed:
        conf += hmm_conf * 0.10

    a2_bias  = (a2_data or {}).get("bias",  "NEUTRAL")
    a2_score = (a2_data or {}).get("score", 0.0)
    a2_c     = (a2_data or {}).get("confidence", 0.0)
    if a2_bias != "NEUTRAL" and abs(a2_score) >= A2_MIN_SCORE and (
        (direction == "LONG_BASE"  and a2_bias == "LONG_BASE") or
        (direction == "SHORT_BASE" and a2_bias == "SHORT_BASE")
    ):
        conf += a2_c * 0.15

    sent_score = (sentiment_data or {}).get("sentiment",   0.0)
    sent_conf  = (sentiment_data or {}).get("confidence",  0.0)
    if (direction == "LONG_BASE"  and sent_score >  SENT_BLOCK_THRESHOLD) or \
       (direction == "SHORT_BASE" and sent_score < -SENT_BLOCK_THRESHOLD):
        conf += sent_conf * 0.05

    opus_bias = (opus_data or {}).get("bias",       "NEUTRAL")
    opus_conv = (opus_data or {}).get("conviction", 0.0)
    if (direction == "LONG_BASE"  and opus_bias == "LONG_BASE") or \
       (direction == "SHORT_BASE" and opus_bias == "SHORT_BASE"):
        conf += opus_conv * 0.05

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
        Evaluate a single pair.

        Path A     — 4-gate fundamental: A1+A2 → Sentiment → A4 → Entry.
        Path B-lite— A1 confirms trending regime, A2 neutral, D1+H1+A4 agree.
                     Fires before Path B when Gate 1 fails only on A2_NEUTRAL.
                     Stricter entry threshold (0.55 vs Path B's 0.40). tag=A1_TREND_TECH
        Path B     — Technical confluence: D1+H1+A4 agree regardless of fundamentals.
                     Fires when Path A returns FLAT and Path B-lite didn't fire. tag=TECH_CONFLUENCE
        Recovery   — When PASS_QUEUE queued a signal last bar, re-check and execute if
                     entry score improved to ≥ 0.55. tag=PASS_WAIT_RECOVERY
        """
        # 0a. Skip pairs managed by hedge_manager (perpetual hedge mode)
        if self.r.get(f"hedge:active:{pair}"):
            d = {"pair": pair, "action": "FLAT", "reason": "HEDGE_ACTIVE",
                 "confidence": 0.0, "size_multiplier": 0.0}
            self._cache(pair, d)
            return d

        # 0. Check PASS_WAIT recovery queue (queued from previous bar)
        recovery = self._check_recovery_queue(pair)
        if recovery is not None:
            self._cache(pair, recovery)
            return recovery

        # 1. Run Path A
        decision = self._fundamental_path(pair)

        # 1b. PASS_QUEUE: entry score 0.40–0.55 — queue and return FLAT
        if decision["action"] != "FLAT":
            gate4 = decision.get("gate4", "")
            if gate4.startswith("PASS_QUEUE"):
                self._set_watch_queue(pair, decision)
                flat_d = dict(decision)
                flat_d["action"] = "FLAT"
                flat_d["reason"] = f"PASS_QUEUE: entry marginal ({gate4}), watching next bar"
                self._cache(pair, flat_d)
                return flat_d

        if decision["action"] == "FLAT":
            # 2a. Path B-lite: fundamental model sees trend but A4 failed or entry low.
            # Try when A1 LightGBM OR HMM regime was trending (Gate 1 passed but Gate 2+ failed).
            regime  = decision.get("regime", "?")
            hmm_reg = decision.get("hmm_regime", "")
            if regime in ("THV", "TLV") or hmm_reg in ("THV", "TLV"):
                blit = self._path_b_lite(pair)
                if blit is not None:
                    self._cache(pair, blit)
                    return blit

            # 2b. Path B: full technical confluence (all A1/A2 fail cases)
            tc = self._technical_confluence_path(pair)
            if tc is not None:
                self._cache(pair, tc)
                return tc

        return decision

    # ── PASS_WAIT recovery queue ──────────────────────────────────────────────

    def _set_watch_queue(self, pair: str, decision: Dict) -> None:
        """Store pair in PASS_QUEUE watch — re-check next bar."""
        watch_data = {
            "direction": decision.get("direction"),
            "regime":    decision.get("regime"),
            "a2_score":  decision.get("a2_score"),
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }
        self.r.setex(f"signal:watch:{pair}", 1800, json.dumps(watch_data))
        logger.debug(
            f"Aggregator | {pair} PASS_QUEUE: "
            f"entry {decision.get('gate4', '?')} — watching next bar"
        )

    def _check_recovery_queue(self, pair: str) -> Optional[Dict]:
        """
        If pair is in watch queue and entry score now ≥ 0.55, execute recovery.

        Re-runs the full fundamental path — gates must still pass. Only the
        entry score gating changes (accept PASS_WAIT that was previously PASS_QUEUE).
        """
        if not self.r.get(f"signal:watch:{pair}"):
            return None

        # Re-evaluate fundamental path fresh
        decision = self._fundamental_path(pair)

        if decision["action"] == "FLAT":
            # Fundamentals degraded — discard queue
            self.r.delete(f"signal:watch:{pair}")
            return None

        gate4 = decision.get("gate4", "")

        if gate4.startswith("PASS_QUEUE"):
            # Entry still marginal — stay queued (TTL handles expiry)
            logger.debug(
                f"Aggregator | {pair}: recovery pending, "
                f"entry score still {decision.get('entry_score', 0):.2f}"
            )
            return None

        if gate4.startswith("FAIL"):
            # Entry got worse — abandon queue
            self.r.delete(f"signal:watch:{pair}")
            return None

        # Gate4 is PASS or PASS_WAIT — entry improved, execute
        self.r.delete(f"signal:watch:{pair}")
        decision["path"]   = "PASS_WAIT_RECOVERY"
        decision["reason"] = decision.get("reason", "") + " [RECOVERY]"
        logger.info(
            f"Aggregator | {pair} RECOVERY: "
            f"entry improved → {gate4} — executing"
        )
        return decision

    # ── Path B-lite: A1 trend confirmed, A2 neutral ───────────────────────────

    def _path_b_lite(self, pair: str) -> Optional[Dict]:
        """
        Path B-lite (tag: A1_TREND_TECH).

        Fires when A1 confirms a trending regime (THV/TLV) but A2 has no macro
        signal (neutral score), AND D1+H1+A4 all agree on direction.

        Stricter than full Path B:
          - Regime gate: THV or TLV only (trending confirmed by A1)
          - Entry threshold: 0.55 (vs 0.40 for Path B)
          - Base confidence: 0.65 (vs 0.62 for Path B)
        Size: 0.75× (same as Path B — no A2 macro backing).
        """
        try:
            stat_raw = self.r.get(f"features:stat:{pair}")
            if not stat_raw:
                return None
            stat_data = json.loads(stat_raw)

            d1_trend = int(stat_data.get("d1_trend_dir", 0))
            h1_trend = int(stat_data.get("h1_trend_dir", 0))
            d1_slope = float(stat_data.get("d1_ema20_slope", 0.0))
            h1_slope = float(stat_data.get("h1_ema20_slope", 0.0))

            if d1_trend == 0 or h1_trend == 0 or d1_trend != h1_trend:
                return None

            tc_direction = "LONG_BASE" if d1_trend == 1 else "SHORT_BASE"
            action       = "LONG"      if d1_trend == 1 else "SHORT"

            tech_data = _read_technical(self.r, pair)
            if not tech_data:
                return None

            a4_dir     = tech_data.get("direction", "FLAT")
            a4_gate    = tech_data.get("gate", "FAIL")
            tcn_prob   = float(tech_data.get("tcn_prob",   0.5))
            patch_prob = float(tech_data.get("patch_prob", 0.5))

            if a4_gate != "PASS" or a4_dir != action:
                return None
            if action == "LONG"  and (tcn_prob < 0.52 or patch_prob < 0.52):
                return None
            if action == "SHORT" and (tcn_prob > 0.48 or patch_prob > 0.48):
                return None

            # Stricter entry threshold (0.55 vs Path B's 0.40)
            entry_data  = _read_entry(self.r, pair)
            entry_score = float(entry_data.get("entry_score", 0.0))
            entry_type  = entry_data.get("entry_type", "WAIT")

            if entry_score < 0.55:
                return None
            if entry_type == "BLOCKED_SR":
                return None

            regime_data = _read_regime(self.r, pair)
            regime      = regime_data.get("regime", "?")
            a2_data     = _read_analyst2(self.r, pair)

            # HMM tag — observability only on bypass paths (no gate)
            hmm_raw_bl    = _read_regime_hmm(self.r, pair)
            hmm_regime_bl = (hmm_raw_bl or {}).get("regime", "")
            hmm_conf_bl   = round((hmm_raw_bl or {}).get("confidence", 0.0), 4)
            g1b_bl        = _hmm_regime_tag(hmm_regime_bl, regime, False)

            # Confidence: higher base (0.65) because A1 regime confirmed
            slope_boost = min(0.05, abs(d1_slope) * 0.01 + abs(h1_slope) * 0.02)
            a4_boost    = (abs(tcn_prob - 0.5) + abs(patch_prob - 0.5)) * 0.1
            confidence  = round(min(0.80, 0.65 + slope_boost + a4_boost), 4)

            try:
                apex_ctx = {
                    "a2_fund_score": a2_data.get("score", 0.0),
                    "a1_confidence": regime_data.get("confidence", 0.0),
                    "a1_regime":     regime,
                    "a5_sentiment":  0.0,
                    "direction":     "LONG" if tc_direction == "LONG_BASE" else "SHORT",
                }
                apex_mult, apex_reasons = apex_framework.get_penalty(pair, apex_ctx)
                if apex_mult < 1.0:
                    confidence = round(confidence * apex_mult, 4)
            except Exception:
                apex_mult    = 1.0
                apex_reasons = []

            now = datetime.now(timezone.utc).isoformat()

            logger.info(
                f"Aggregator | {pair} PATH-BL A1_TREND_TECH: {action} "
                f"regime={regime} D1={d1_slope:+.3f}% H1={h1_slope:+.3f}% "
                f"A4={a4_dir} tcn={tcn_prob:.2f} patch={patch_prob:.2f} "
                f"entry={entry_score:.2f} conf={confidence:.2f}"
            )

            return {
                "pair":         pair,
                "action":       action,
                "direction":    tc_direction,
                "confidence":   confidence,
                "size_mult":    0.75,
                "gate1":        f"PASS:A1_ONLY(regime={regime})",
                "gate2":        "BYPASS:A2_NEUTRAL",
                "gate3":        f"PASS:A4({a4_dir},tcn={tcn_prob:.2f},patch={patch_prob:.2f})",
                "gate4":        f"PASS:ENTRY({entry_score:.2f})",
                "entry_score":  round(entry_score, 4),
                "entry_type":   entry_type,
                "entry_reason": entry_data.get("reason", ""),
                "regime":       regime,
                "hmm_regime":   hmm_regime_bl,
                "hmm_conf":     hmm_conf_bl,
                "gate1b":       g1b_bl,
                "a2_score":     round(a2_data.get("score", 0.0), 4),
                "a2_bias":      a2_data.get("bias", "NEUTRAL"),
                "a2_conf":      round(a2_data.get("confidence", 0.0), 4),
                "opus_bias":    "NEUTRAL",
                "opus_conv":    0.0,
                "sent_bias":    "NEUTRAL",
                "sent_score":   0.0,
                "tcn_prob":     round(tcn_prob,   4),
                "patch_prob":   round(patch_prob, 4),
                "a4_conf":      round(tech_data.get("confidence", 0.0), 4),
                "conflict":     "",
                "apex_mult":    apex_mult,
                "apex_reasons": apex_reasons,
                "path":         "A1_TREND_TECH",
                "reason": (
                    f"PATH-BL | regime={regime} D1={d1_slope:+.3f}% "
                    f"H1={h1_slope:+.3f}% A4={a4_dir} entry={entry_score:.2f}"
                ),
                "updated_at":   now,
            }

        except Exception as exc:
            logger.warning(f"Aggregator | {pair} Path B-lite failed: {exc}")
            return None

    def _fundamental_path(self, pair: str) -> Dict:
        """
        Path A: 2-gate fundamental + technical evaluation (2026-04-07).

        Entry fires when ANY paired condition is met:
          • A1 LightGBM trending (THV/TLV) + A4 (TCN or PatchTSMixer) directional
          • HMM trending (THV/TLV, conf≥0.40) + A4 (TCN or PatchTSMixer) directional

        A2/A5/A6 are SOFT size modifiers only — reduce size_mult when opposing, never block.
        Gate 4 (entry engine) is still a hard gate.
        Direction is always determined by A4 (the required technical gate).
        Exception: MACRO_OVERRIDE (A5+A6 on MR) sets direction from macro consensus.
        """
        # ── Load all analyst data ─────────────────────────────────────────────
        regime_data    = _read_regime(self.r, pair)
        a2_data        = _read_analyst2(self.r, pair)
        sentiment_data = _read_sentiment(self.r, pair)
        opus_data      = _read_opus_pair(self.r, pair)

        # ── Opus staleness + conflict guard ──────────────────────────────────
        _opus_generated = opus_data.get("generated_at", "")
        if not _opus_generated:
            try:
                _weekly = self.r.get("features:opus:weekly")
                if _weekly:
                    import json as _j
                    _opus_generated = _j.loads(_weekly).get("generated_at", "")
            except Exception:
                pass
        _opus_age_days = 999
        if _opus_generated:
            try:
                import datetime as _dt
                _opus_dt       = _dt.datetime.fromisoformat(
                    _opus_generated.replace("Z", "+00:00"))
                _opus_age_days = (_dt.datetime.now(_dt.timezone.utc) - _opus_dt).days
            except Exception:
                pass
        if _opus_age_days > 5:
            logger.warning(f"Aggregator | {pair}: Opus {_opus_age_days}d old → NEUTRAL")
            opus_data = {"bias": "NEUTRAL", "conviction": 0.0}
        else:
            try:
                _yf = self.r.get(f"features:yield:{pair}")
                if _yf:
                    import json as _json
                    _yf_data   = _json.loads(_yf)
                    _trend_4w  = _yf_data.get("yield_trend_4w_10Y", 0.0)
                    _trend_13w = _yf_data.get("yield_trend_13w_10Y", 0.0)
                    _obias     = opus_data.get("bias", "NEUTRAL")
                    _oconv     = opus_data.get("conviction", 0.0)
                    _opus_long = _obias == "LONG_BASE"
                    if (_opus_long and _trend_4w < -0.15 and _trend_13w < -0.15) or \
                       (not _opus_long and _trend_4w > 0.15 and _trend_13w > 0.15):
                        logger.warning(
                            f"Aggregator | {pair}: Opus conflicts yield trend "
                            f"(4w={_trend_4w:+.3f} 13w={_trend_13w:+.3f}) "
                            f"→ conviction {_oconv:.2f}→{_oconv*0.5:.2f}"
                        )
                        opus_data = dict(opus_data)
                        opus_data["conviction"] = _oconv * 0.5
            except Exception as _e:
                logger.warning(f"Aggregator | {pair}: Opus conflict check failed: {_e}")

        tech_data  = _read_technical(self.r, pair)
        entry_data = _read_entry(self.r, pair)
        now        = datetime.now(timezone.utc).isoformat()

        # ── HMM regime ────────────────────────────────────────────────────────
        hmm_raw        = _read_regime_hmm(self.r, pair)
        hmm_regime     = (hmm_raw or {}).get("regime", "")
        hmm_conf       = round((hmm_raw or {}).get("confidence", 0.0), 4)
        a1_regime_raw  = regime_data.get("regime", "")
        a1_regime_conf = regime_data.get("confidence", 0.0)

        _update_regime_history(self.r, pair, regime_data)

        # ── PNS: always flat ─────────────────────────────────────────────────
        if a1_regime_raw == "PNS":
            g1b = _hmm_regime_tag(hmm_regime, a1_regime_raw, False)
            decision = self._flat_decision(
                pair, "FAIL:PNS_REGIME", "MODIFIER:SKIP", "SKIP",
                regime_data, a2_data, opus_data, sentiment_data, "PNS", now,
                hmm_regime=hmm_regime, gate1b=g1b,
            )
            self._cache(pair, decision)
            return decision

        # ── Gate 1: At least one fundamental model sees trending regime ───────
        # Either A1 LightGBM OR HMM detecting THV/TLV is enough to pass.
        TRENDING     = {"THV", "TLV"}
        a1_trending  = a1_regime_raw in TRENDING
        hmm_trending = hmm_regime in TRENDING and hmm_conf >= 0.40
        hmm_upgraded = not a1_trending and hmm_trending
        g1b_result   = _hmm_regime_tag(hmm_regime, a1_regime_raw, hmm_upgraded)

        if a1_trending:
            effective_regime = a1_regime_raw
            g1_result        = f"PASS:A1({a1_regime_raw},conf={a1_regime_conf:.2f})"
        elif hmm_trending:
            effective_regime = hmm_regime
            g1_result        = f"PASS:HMM({hmm_regime},conf={hmm_conf:.2f})"
            logger.info(
                f"Aggregator | {pair} HMM_LEADS: A1={a1_regime_raw} → "
                f"HMM={hmm_regime}(conf={hmm_conf:.2f})"
            )
        else:
            # Neither sees trend — try MACRO_OVERRIDE (A5+A6 strong on MR)
            if a1_regime_raw == "MR":
                macro_dir, macro_reason = _check_macro_conviction(sentiment_data, opus_data)
                if macro_dir != "FLAT":
                    # Direction from A5+A6. A4 must confirm. A2 is soft modifier.
                    effective_regime = "TLV"
                    g1_result        = f"PASS:MACRO_OVERRIDE({macro_reason})"
                    direction        = macro_dir
                    base_size        = REGIME_SIZE["TLV"]
                    logger.info(
                        f"Aggregator | {pair} MACRO_OVERRIDE: MR → TLV "
                        f"dir={macro_dir} [{macro_reason}]"
                    )

                    # A4 must confirm MACRO_OVERRIDE direction
                    g3_ml = _gate3_technical(tech_data, direction)
                    if g3_ml.startswith("FAIL"):
                        decision = self._flat_decision(
                            pair, g1_result, "MODIFIER:MACRO_OVERRIDE", g3_ml,
                            regime_data, a2_data, opus_data, sentiment_data,
                            "A4_BLOCK", now, hmm_regime=hmm_regime, gate1b=g1b_result,
                        )
                        self._cache(pair, decision)
                        return decision

                    # Stability check
                    is_unstable, unstable_reason, gap = _check_regime_stability(
                        self.r, pair, regime_data
                    )
                    if is_unstable:
                        base_size = round(base_size * 0.25, 2)
                        g1_result += f"+UNSTABLE(gap={gap:.2f})"

                    # Soft modifier: A2 only (A5/A6 confirmed direction for MACRO_OVERRIDE)
                    size_mult, mod_reasons = _apply_soft_modifiers(
                        base_size, direction, a2_data, None, None
                    )
                    g2_result = f"MODIFIER:{','.join(mod_reasons) if mod_reasons else 'ALIGNED'}"

                    # Gate 4: entry engine
                    g4_result = _gate3_entry(entry_data, direction)
                    if g4_result.startswith("FAIL"):
                        decision = self._flat_decision(
                            pair, g1_result, g2_result, g3_ml,
                            regime_data, a2_data, opus_data, sentiment_data,
                            "TECH_BLOCK", now, hmm_regime=hmm_regime, gate1b=g1b_result,
                            gate4=g4_result,
                            entry_score=entry_data.get("entry_score", 0.0),
                            entry_type=entry_data.get("entry_type", ""),
                        )
                        self._cache(pair, decision)
                        return decision

                    tcn_prob   = tech_data.get("tcn_prob",   0.5)
                    patch_prob = tech_data.get("patch_prob", 0.5)
                    a4_conf    = tech_data.get("confidence", 0.0)
                    confidence = _compute_confidence_v2(
                        a4_conf, a1_regime_conf, hmm_conf, hmm_upgraded,
                        direction, a2_data, sentiment_data, opus_data,
                    )
                    confidence = round(confidence * PAIR_MODEL_QUALITY.get(pair, 1.0), 4)
                    apex_ctx = {
                        "a2_fund_score": a2_data.get("score", 0.0),
                        "a1_confidence": a1_regime_conf,
                        "a1_regime":     a1_regime_raw,
                        "a5_sentiment":  sentiment_data.get("sentiment", 0.0),
                        "direction":     "LONG" if direction == "LONG_BASE" else "SHORT",
                    }
                    try:
                        apex_mult, apex_reasons = apex_framework.get_penalty(pair, apex_ctx)
                        if apex_mult < 1.0:
                            confidence = round(confidence * apex_mult, 4)
                    except Exception:
                        apex_mult, apex_reasons = 1.0, []

                    action   = "LONG" if direction == "LONG_BASE" else "SHORT"
                    mod_str  = " | ".join(mod_reasons) if mod_reasons else "ALL_ALIGNED"
                    decision = {
                        "pair": pair, "action": action, "direction": direction,
                        "confidence": confidence, "size_mult": size_mult,
                        "gate1": g1_result, "gate1b": g1b_result,
                        "gate2": g2_result, "gate3": g3_ml, "gate4": g4_result,
                        "entry_score":    round(entry_data.get("entry_score", 0.0), 4),
                        "entry_type":     entry_data.get("entry_type", "WAIT"),
                        "entry_reason":   entry_data.get("reason", ""),
                        "regime":         effective_regime,
                        "hmm_regime":     hmm_regime, "hmm_conf": hmm_conf,
                        "a2_score":       round(a2_data.get("score", 0.0), 4),
                        "a2_bias":        a2_data.get("bias", "NEUTRAL"),
                        "a2_conf":        round(a2_data.get("confidence", 0.0), 4),
                        "opus_bias":      opus_data.get("bias", "NEUTRAL"),
                        "opus_conv":      round(opus_data.get("conviction", 0.0), 4),
                        "sent_bias":      sentiment_data.get("bias", "NEUTRAL"),
                        "sent_score":     round(sentiment_data.get("sentiment", 0.0), 4),
                        "tcn_prob":       round(tcn_prob, 4),
                        "patch_prob":     round(patch_prob, 4),
                        "a4_conf":        round(a4_conf, 4),
                        "size_modifiers": mod_reasons,
                        "conflict":       "",
                        "apex_mult":      apex_mult, "apex_reasons": apex_reasons,
                        "reason": f"MACRO_OVERRIDE | {action} | {mod_str}",
                        "updated_at":     now,
                    }
                    self._cache(pair, decision)
                    return decision

            decision = self._flat_decision(
                pair,
                f"FAIL:NO_TREND(A1={a1_regime_raw},HMM={hmm_regime or 'N/A'})",
                "MODIFIER:SKIP", "SKIP",
                regime_data, a2_data, opus_data, sentiment_data, "NO_TREND", now,
                hmm_regime=hmm_regime, gate1b=g1b_result,
            )
            self._cache(pair, decision)
            return decision

        # ── Gate 2 (A4 Technical): direction source + hard gate ───────────────
        # A4 (TCN or PatchTSMixer) determines trade direction.
        # Passes on PASS (both agree) or PASS_SINGLE (one agrees).
        a4_gate    = tech_data.get("gate",       "FAIL")
        a4_dir     = tech_data.get("direction",  "FLAT")
        tcn_prob   = tech_data.get("tcn_prob",   0.5)
        patch_prob = tech_data.get("patch_prob", 0.5)
        a4_conf    = tech_data.get("confidence", 0.0)

        if (not tech_data.get("available", False)
                or a4_gate not in ("PASS", "PASS_SINGLE")
                or a4_dir == "FLAT"):
            g3_ml = f"FAIL:A4(gate={a4_gate},dir={a4_dir})"
            decision = self._flat_decision(
                pair, g1_result, "MODIFIER:SKIP", g3_ml,
                regime_data, a2_data, opus_data, sentiment_data, "A4_BLOCK", now,
                hmm_regime=hmm_regime, gate1b=g1b_result,
            )
            self._cache(pair, decision)
            return decision

        direction = "LONG_BASE" if a4_dir == "LONG" else "SHORT_BASE"
        g3_ml     = (
            f"PASS_SINGLE:A4({a4_dir},tcn={tcn_prob:.2f},patch={patch_prob:.2f})"
            if a4_gate == "PASS_SINGLE"
            else f"PASS:A4({a4_dir},tcn={tcn_prob:.2f},patch={patch_prob:.2f})"
        )

        # ── Soft modifiers: A2, A5, A6 (reduce size_mult, never block) ────────
        base_size = REGIME_SIZE.get(effective_regime, 1.0)

        is_unstable, unstable_reason, gap = _check_regime_stability(
            self.r, pair, regime_data
        )
        if is_unstable:
            base_size = round(base_size * 0.25, 2)
            logger.warning(
                f"Aggregator | {pair} {unstable_reason} — base size → {base_size}x"
            )
            g1_result += f"+UNSTABLE(gap={gap:.2f})"

        size_mult, mod_reasons = _apply_soft_modifiers(
            base_size, direction, a2_data, sentiment_data, opus_data
        )
        g2_result = f"MODIFIER:{','.join(mod_reasons) if mod_reasons else 'ALIGNED'}"

        if mod_reasons:
            logger.info(
                f"Aggregator | {pair} soft modifiers: {' | '.join(mod_reasons)} "
                f"→ {base_size:.2f}x → {size_mult:.2f}x"
            )

        # ── Gate 4: Entry engine (hard gate) ──────────────────────────────────
        g4_result = _gate3_entry(entry_data, direction)
        if g4_result.startswith("FAIL"):
            decision = self._flat_decision(
                pair, g1_result, g2_result, g3_ml,
                regime_data, a2_data, opus_data, sentiment_data, "TECH_BLOCK", now,
                hmm_regime=hmm_regime, gate1b=g1b_result,
                gate4=g4_result,
                entry_score=entry_data.get("entry_score", 0.0),
                entry_type=entry_data.get("entry_type", ""),
            )
            self._cache(pair, decision)
            return decision

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = _compute_confidence_v2(
            a4_conf, a1_regime_conf, hmm_conf, hmm_upgraded,
            direction, a2_data, sentiment_data, opus_data,
        )
        confidence = round(confidence * PAIR_MODEL_QUALITY.get(pair, 1.0), 4)

        # ── APEX penalty ──────────────────────────────────────────────────────
        apex_ctx = {
            "a2_fund_score": a2_data.get("score", 0.0),
            "a1_confidence": a1_regime_conf,
            "a1_regime":     a1_regime_raw,
            "a5_sentiment":  sentiment_data.get("sentiment", 0.0),
            "direction":     "LONG" if direction == "LONG_BASE" else "SHORT",
        }
        try:
            apex_mult, apex_reasons = apex_framework.get_penalty(pair, apex_ctx)
            if apex_mult < 1.0:
                confidence = round(confidence * apex_mult, 4)
                logger.debug(
                    f"Aggregator | {pair} APEX penalty {apex_mult:.4f}: {apex_reasons}"
                )
        except Exception:
            apex_mult, apex_reasons = 1.0, []

        action  = "LONG" if direction == "LONG_BASE" else "SHORT"
        mod_str = " | ".join(mod_reasons) if mod_reasons else "ALL_ALIGNED"

        decision = {
            "pair":          pair,
            "action":        action,
            "direction":     direction,
            "confidence":    confidence,
            "size_mult":     size_mult,
            "gate1":         g1_result,
            "gate1b":        g1b_result,
            "gate2":         g2_result,
            "gate3":         g3_ml,
            "gate4":         g4_result,
            "entry_score":   round(entry_data.get("entry_score", 0.0), 4),
            "entry_type":    entry_data.get("entry_type", "WAIT"),
            "entry_reason":  entry_data.get("reason", ""),
            "regime":        effective_regime,
            "hmm_regime":    hmm_regime,
            "hmm_conf":      hmm_conf,
            "a2_score":      round(a2_data.get("score", 0.0), 4),
            "a2_bias":       a2_data.get("bias", "NEUTRAL"),
            "a2_conf":       round(a2_data.get("confidence", 0.0), 4),
            "opus_bias":     opus_data.get("bias", "NEUTRAL"),
            "opus_conv":     round(opus_data.get("conviction", 0.0), 4),
            "sent_bias":     sentiment_data.get("bias", "NEUTRAL"),
            "sent_score":    round(sentiment_data.get("sentiment", 0.0), 4),
            "tcn_prob":      round(tcn_prob, 4),
            "patch_prob":    round(patch_prob, 4),
            "a4_conf":       round(a4_conf, 4),
            "size_modifiers": mod_reasons,
            "conflict":      "",
            "apex_mult":     apex_mult,
            "apex_reasons":  apex_reasons,
            "reason":        f"PASS | {action} | regime={effective_regime} | {mod_str}",
            "updated_at":    now,
        }

        self._cache(pair, decision)
        return decision

    # ── Path B: Technical Confluence ──────────────────────────────────────────

    def _technical_confluence_path(self, pair: str) -> Optional[Dict]:
        """
        Path B — Technical Confluence entry.

        Fires when D1 trend + H1 trend + A4 TCN + A4 PatchTSMixer all agree
        on the same direction, even if fundamentals (A1/A2) did not confirm.

        Requirements:
          - D1 EMA20 trend clearly BULL or BEAR (d1_trend_dir ≠ 0)
          - H1 EMA20 trend same direction (h1_trend_dir matches d1_trend_dir)
          - A4 TCN direction matches trend
          - A4 PatchTSMixer direction matches trend
          - Entry engine score ≥ 0.40 (relaxed from fundamental path 0.65)
          - All risk gates still apply (cooldown, session cap, correlation)

        Returns decision dict with path="TECH_CONFLUENCE", size_mult=0.75,
        or None if conditions not met.
        """
        try:
            # ── Load stat features (D1/H1 trend) ─────────────────────────────
            stat_raw = self.r.get(f"features:stat:{pair}")
            if not stat_raw:
                return None
            stat_data = json.loads(stat_raw)

            d1_trend = int(stat_data.get("d1_trend_dir", 0))
            h1_trend = int(stat_data.get("h1_trend_dir", 0))
            d1_slope = float(stat_data.get("d1_ema20_slope", 0.0))
            h1_slope = float(stat_data.get("h1_ema20_slope", 0.0))

            # Both timeframes must agree and be non-flat
            if d1_trend == 0 or h1_trend == 0:
                return None
            if d1_trend != h1_trend:
                return None

            tc_direction = "LONG_BASE" if d1_trend == 1 else "SHORT_BASE"
            action       = "LONG"     if d1_trend == 1 else "SHORT"

            # ── A4 must confirm same direction ────────────────────────────────
            tech_data = _read_technical(self.r, pair)
            if not tech_data:
                return None

            a4_dir  = tech_data.get("direction", "FLAT")
            a4_gate = tech_data.get("gate", "FAIL")
            tcn_prob   = float(tech_data.get("tcn_prob",   0.5))
            patch_prob = float(tech_data.get("patch_prob", 0.5))

            if a4_gate != "PASS":
                return None
            if a4_dir != action:
                return None

            # Both A4 model probabilities must lean toward direction (>0.52)
            if action == "LONG"  and (tcn_prob < 0.52 or patch_prob < 0.52):
                return None
            if action == "SHORT" and (tcn_prob > 0.48 or patch_prob > 0.48):
                return None

            # ── Entry engine: relaxed threshold (0.40 vs 0.65) ───────────────
            entry_data = _read_entry(self.r, pair)
            entry_score = float(entry_data.get("entry_score", 0.0))
            entry_type  = entry_data.get("entry_type", "WAIT")

            if entry_score < 0.40:
                logger.debug(
                    f"Aggregator | {pair} TC: entry_score {entry_score:.2f} < 0.40, skip"
                )
                return None

            # BLOCKED_SR is a hard stop even on Path B (entering into a wall is bad)
            if entry_type == "BLOCKED_SR":
                return None

            # ── Compute confluence confidence ─────────────────────────────────
            # Base 0.62, boosted by slope strength and A4 probability alignment
            slope_boost = min(0.05, abs(d1_slope) * 0.01 + abs(h1_slope) * 0.02)
            a4_boost    = (abs(tcn_prob - 0.5) + abs(patch_prob - 0.5)) * 0.1
            confidence  = round(min(0.80, 0.62 + slope_boost + a4_boost), 4)

            # APEX penalty applies to Path B too
            try:
                regime_data  = _read_regime(self.r, pair)
                regime_b     = regime_data.get("regime", "MR")

                # HMM tag — observability only on bypass paths (no gate)
                hmm_raw_b    = _read_regime_hmm(self.r, pair)
                hmm_regime_b = (hmm_raw_b or {}).get("regime", "")
                hmm_conf_b   = round((hmm_raw_b or {}).get("confidence", 0.0), 4)
                g1b_b        = _hmm_regime_tag(hmm_regime_b, regime_b, False)

                apex_ctx = {
                    "a2_fund_score": 0.0,   # no fundamental score in Path B
                    "a1_confidence": regime_data.get("confidence", 0.0),
                    "a1_regime":     regime_b,
                    "a5_sentiment":  0.0,
                    "direction":     "LONG" if tc_direction == "LONG_BASE" else "SHORT",
                }
                apex_mult, apex_reasons = apex_framework.get_penalty(pair, apex_ctx)
                if apex_mult < 1.0:
                    confidence = round(confidence * apex_mult, 4)
            except Exception:
                apex_mult    = 1.0
                apex_reasons = []
                regime_b     = "?"
                hmm_regime_b = ""
                hmm_conf_b   = 0.0
                g1b_b        = "SKIP"   # HMM tag on error

            now = datetime.now(timezone.utc).isoformat()
            regime = (json.loads(self.r.get(f"features:regime:{pair}") or "{}") or {}).get("regime", "?")

            logger.info(
                f"Aggregator | {pair} PATH-B CONFLUENCE: {action} "
                f"D1={d1_slope:+.3f}% H1={h1_slope:+.3f}% "
                f"A4={a4_dir} tcn={tcn_prob:.2f} patch={patch_prob:.2f} "
                f"entry={entry_score:.2f} conf={confidence:.2f}"
            )

            return {
                "pair":         pair,
                "action":       action,
                "direction":    tc_direction,
                "confidence":   confidence,
                "size_mult":    0.75,
                "gate1":        f"BYPASS:D1_TREND({d1_slope:+.3f}%)",
                "gate2":        f"BYPASS:H1_TREND({h1_slope:+.3f}%)",
                "gate3":        f"PASS:A4({a4_dir},tcn={tcn_prob:.2f},patch={patch_prob:.2f})",
                "gate4":        f"PASS:ENTRY({entry_score:.2f})",
                "entry_score":  round(entry_score, 4),
                "entry_type":   entry_type,
                "entry_reason": entry_data.get("reason", ""),
                "regime":       regime,
                "hmm_regime":   hmm_regime_b,
                "hmm_conf":     hmm_conf_b,
                "gate1b":       g1b_b,
                "a2_score":     0.0,
                "a2_bias":      "NEUTRAL",
                "a2_conf":      0.0,
                "opus_bias":    "NEUTRAL",
                "opus_conv":    0.0,
                "sent_bias":    "NEUTRAL",
                "sent_score":   0.0,
                "tcn_prob":     round(tcn_prob,   4),
                "patch_prob":   round(patch_prob, 4),
                "a4_conf":      round(tech_data.get("confidence", 0.0), 4),
                "conflict":     "",
                "apex_mult":    apex_mult,
                "apex_reasons": apex_reasons,
                "path":         "TECH_CONFLUENCE",
                "reason": (
                    f"PATH-B | D1={d1_slope:+.3f}% H1={h1_slope:+.3f}% "
                    f"A4={a4_dir} entry={entry_score:.2f}"
                ),
                "updated_at":   now,
            }

        except Exception as exc:
            logger.warning(f"Aggregator | {pair} TC path failed: {exc}")
            return None

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
        self._persist_to_db(results)
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


    def _persist_to_db(self, results: Dict[str, Dict]) -> None:
        """Persist all pair signals to analyst_signals and regime_history tables."""
        try:
            import json as _json
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            signal_rows = []
            regime_rows = []

            for pair, d in results.items():
                action   = d.get("action", "FLAT")
                conf     = float(d.get("confidence", 0.0))
                _regime_map = {"THV": "TRENDING_HV", "TLV": "TRENDING_LV", "MR": "MEAN_REVERTING", "PNS": "PRE_NEWS_STAGNATION"}
                regime   = _regime_map.get(d.get("regime", "PNS"), "PRE_NEWS_STAGNATION")
                size     = float(d.get("size_multiplier", 1.0))
                sig_val  = 1.0 if action == "LONG" else -1.0 if action == "SHORT" else 0.0
                meta     = _json.dumps({
                    "regime": regime,
                    "size":   size,
                    "reason": d.get("reason", ""),
                    "a2_bias": d.get("a2_bias", ""),
                })
                signal_rows.append({
                    "time": now, "analyst_id": 1,
                    "analyst_name": "aggregator",
                    "pair": pair, "currency": pair[:3],
                    "signal_type": action,
                    "signal_value": sig_val,
                    "confidence": conf,
                    "metadata": meta,
                    "model_version": "v1.0",
                })
                regime_rows.append({
                    "time": now, "pair": pair,
                    "regime": regime, "confidence": conf,
                    "prob_thv": 0.0, "prob_tlv": 0.0,
                    "prob_mr": 0.0, "prob_pns": 0.0,
                    "model_version": "v1.0",
                })

            with db.connection() as conn:
                for row in signal_rows:
                    conn.execute(text("""
                        INSERT INTO analyst_signals
                            (time, analyst_id, analyst_name, pair, currency,
                             signal_type, signal_value, confidence, metadata, model_version)
                        VALUES
                            (:time, :analyst_id, :analyst_name, :pair, :currency,
                             :signal_type, :signal_value, :confidence, CAST(:metadata AS jsonb), :model_version)
                    """), row)
                for row in regime_rows:
                    conn.execute(text("""
                        INSERT INTO regime_history
                            (time, pair, regime, confidence,
                             prob_thv, prob_tlv, prob_mr, prob_pns, model_version)
                        VALUES
                            (:time, :pair, :regime, :confidence,
                             :prob_thv, :prob_tlv, :prob_mr, :prob_pns, :model_version)
                        ON CONFLICT DO NOTHING
                    """), row)
                conn.commit()
            logger.success(f"Aggregator | Persisted {len(signal_rows)} signals to DB")
        except Exception as exc:
            logger.warning(f"Aggregator | DB persist failed: {exc}")

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
        hmm_regime: str = "",
        gate1b: str = "SKIP",
        gate4: str = "",
        entry_score: float = 0.0,
        entry_type: str = "",
    ) -> Dict:
        """Build a FLAT decision dict."""
        return {
            "pair":        pair,
            "action":      "FLAT",
            "direction":   "FLAT",
            "confidence":  0.0,
            "size_mult":   0.0,
            "gate1":       g1,
            "gate1b":      gate1b,
            "gate2":       g2,
            "gate3":       g3,
            "gate4":       gate4,
            "entry_score": round(entry_score, 4),
            "entry_type":  entry_type,
            "regime":      regime_data.get("regime", "?"),
            "hmm_regime":  hmm_regime,
            "a2_score":    round(a2_data.get("score", 0.0), 4),
            "a2_bias":     a2_data.get("bias", "NEUTRAL"),
            "a2_conf":     round(a2_data.get("confidence", 0.0), 4),
            "opus_bias":   opus_data.get("bias", "NEUTRAL"),
            "opus_conv":   round(opus_data.get("conviction", 0.0), 4),
            "sent_bias":   sentiment_data.get("bias", "NEUTRAL"),
            "sent_score":  round(sentiment_data.get("sentiment", 0.0), 4),
            "conflict":    conflict,
            "reason":      f"FLAT | g1={g1} g1b={gate1b} g2={g2} g3={g3}" + (f" g4={gate4}" if gate4 else ""),
            "updated_at":  now,
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
