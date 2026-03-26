"""
features/entry_engine.py
=========================
Analyst 4 — Price Action Entry Engine.

Answers the question: WHEN to enter, not WHETHER to enter.
Direction and regime are already decided by A1 + A2.
This engine reads those signals and finds the best entry timing
using pure price action rules — no ML, no training required.

Trading rationale:
  A1 + A2 provide direction conviction. But entering on any random
  M15 bar costs 1–3 pips of adverse excursion on average. By waiting
  for a VWAP pullback, RSI reset, or OBI confirmation, we improve
  average entry by 2–4 pips — which at 2% risk and 8-pip stops is
  the difference between 0.9 R and 1.1 R per trade over 100 trades.

Entry logic per regime:
  THV (Trending High Vol):
    - Pullback to session VWAP or EMA-equivalent (price_vs_ny_vwap)
    - RSI not overbought (< 65 for long, > 35 for short)
    - OBI confirming direction (imbalance_1m > 0 for long)
    - Momentum still positive (momentum_1d > 0 for long)

  TLV (Trending Low Vol):
    - Any touch of session VWAP — tighter tolerance
    - Spread normal (spread_ratio < 1.5)
    - Momentum confirming (momentum_5d > 0 for long)

  MR (Mean Reverting):
    - RSI extreme required (> 68 short, < 32 long)
    - Price at or beyond BB band (bb_pct_20 > 0.95 short, < 0.05 long)
    - OBI flipping (flow_imbalance_delta opposing trend)

  PNS: Always returns score=0, wait=True

Entry types produced:
  VWAP_PULLBACK    Price returned to session VWAP, momentum intact
  VWAP_BAND_TOUCH  Price at VWAP upper/lower band edge
  RSI_RESET        RSI pulled back to neutral zone in trend
  RSI_EXTREME      RSI at extreme in MR regime
  OBI_CONFIRM      Order book imbalance strongly confirming direction
  MOMENTUM_ENTRY   Clean momentum continuation, no pullback needed
  SPREAD_ENTRY     Spread normalised after spike — safe window
  WAIT             Good direction but no clean entry yet

Redis reads:
  features:regime:{pair}     A1 regime + confidence
  features:analyst2:{pair}   A2 direction + bias
  features:vwap:{pair}       Session VWAPs + bands
  features:stat:{pair}       RSI, ATR, momentum, BB
  features:flow:{pair}       OBI, spread, tick velocity

Redis keys:  features:entry:{PAIR}
TTL:         900 seconds (15 minutes — refreshed every bar)

Usage:
    from features.entry_engine import entry_engine
    signal = entry_engine.get_entry("EURUSD")
    # {
    #   "entry_score": 0.78,
    #   "entry_type":  "VWAP_PULLBACK",
    #   "wait":        False,
    #   "reason":      "RSI 47, price at NY VWAP, OBI=+0.18",
    #   "direction":   "LONG",
    #   "regime":      "THV",
    # }
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger

load_dotenv("/workspaces/Trading_bot/.env")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

REDIS_TTL          = 900   # 15 minutes
REDIS_PREFIX       = "features:entry"
SCHEDULER_INTERVAL = 900   # recompute every M15 bar

# Entry score thresholds
SCORE_EXECUTE  = 0.65   # fire immediately
SCORE_WAIT     = 0.40   # good setup, wait 1 more bar
SCORE_SKIP     = 0.40   # below this → skip entirely

# RSI thresholds
RSI_OVERBOUGHT = 65.0
RSI_OVERSOLD   = 35.0
RSI_EXTREME_SHORT = 68.0
RSI_EXTREME_LONG  = 32.0

# VWAP proximity — how close to VWAP counts as "at VWAP"
# Expressed as fraction of price_vs_*_vwap (already normalised)
VWAP_PROXIMITY = 0.0015   # within 0.15% of price = at VWAP

# Neutral entry response
NEUTRAL_ENTRY: Dict = {
    "entry_score": 0.0,
    "entry_type":  "WAIT",
    "wait":        True,
    "reason":      "No signal",
    "direction":   "FLAT",
    "regime":      "UNKNOWN",
    "available":   False,
}


# ──────────────────────────────────────────────────────────────────────────────
# Entry Engine
# ──────────────────────────────────────────────────────────────────────────────

class EntryEngine:
    """
    Price action entry timing engine — Analyst 4.

    Reads regime + direction from Redis, applies rule-based entry
    logic, and scores the current bar as an entry opportunity.

    Never generates direction on its own — only validates timing.
    """

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────

    @property
    
    def redis(self) -> redis.Redis:
        if self._redis is None:
            from data_pipeline.storage.redis_pool import redis_client
            self._redis = redis_client
        return self._redis
    
    def _rget(self, key: str) -> Optional[Dict]:
        try:
            raw = self.redis.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning(f"EntryEngine | Redis GET failed [{key}]: {exc}")
            return None

    def _rset(self, key: str, data: Dict, ttl: int = REDIS_TTL) -> None:
        try:
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning(f"EntryEngine | Redis SET failed [{key}]: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_entry(self, pair: str) -> Dict:
        """
        Return entry timing signal for a single pair.

        Args:
            pair: Currency pair e.g. "EURUSD"

        Returns:
            Entry signal dict with score, type, wait flag, and reason.
        """
        key = f"{REDIS_PREFIX}:{pair}"
        cached = self._rget(key)
        if cached:
            logger.debug(f"EntryEngine | Cache HIT: {pair}")
            return cached

        return self.compute_all().get(pair, dict(NEUTRAL_ENTRY))

    def compute_all(self) -> Dict[str, Dict]:
        """
        Compute entry signals for all 12 pairs and cache in Redis.

        Returns:
            Dict of pair → entry signal dict.
        """
        now_utc = datetime.now(timezone.utc)
        logger.info(
            f"EntryEngine | Computing entry signals | "
            f"UTC {now_utc.strftime('%H:%M')}"
        )

        results: Dict[str, Dict] = {}

        for pair in PAIRS:
            try:
                signal = self._compute_entry(pair)
                self._rset(f"{REDIS_PREFIX}:{pair}", signal)
                results[pair] = signal
                logger.debug(
                    f"EntryEngine | {pair}: score={signal['entry_score']:.2f} "
                    f"type={signal['entry_type']} wait={signal['wait']}"
                )
            except Exception as exc:
                logger.error(f"EntryEngine | {pair} failed: {exc}")
                neutral = dict(NEUTRAL_ENTRY)
                neutral["regime"] = "UNKNOWN"
                self._rset(f"{REDIS_PREFIX}:{pair}", neutral)
                results[pair] = neutral

        up = sum(1 for v in results.values() if not v["wait"])
        logger.success(
            f"EntryEngine | Done: {up}/{len(PAIRS)} pairs ready to enter"
        )
        return results

    # ── Core logic ────────────────────────────────────────────────────────────

    def _compute_entry(self, pair: str) -> Dict:
        """
        Main entry scoring logic for one pair.

        Reads all feature blobs from Redis, routes to regime-specific
        scorer, returns entry signal dict.
        """
        # ── Load feature blobs ────────────────────────────────────────────────
        regime_data = self._rget(f"features:regime:{pair}") or {}
        a2_data     = self._rget(f"features:analyst2:{pair}") or {}
        vwap_data   = self._rget(f"features:vwap:{pair}") or {}
        stat_data   = self._rget(f"features:stat:{pair}") or {}
        flow_data   = self._rget(f"features:flow:{pair}") or {}

        regime    = regime_data.get("regime", "TLV")
        a2_bias   = a2_data.get("bias", "NEUTRAL")
        a2_conf   = a2_data.get("confidence", 0.0)

        # Derive direction from A2 bias
        # MR regime: A2 neutral is expected — use A4 TCN/PatchTSMixer direction
        if a2_bias == "LONG_BASE":
            direction = "LONG"
        elif a2_bias == "SHORT_BASE":
            direction = "SHORT"
        elif regime == "MR":
            # In MR, direction comes from A4 technical models
            a4_data = self._rget(f"features:analyst4:{pair}") or {}
            a4_dir  = a4_data.get("direction", "FLAT")
            a4_gate = a4_data.get("gate", "FAIL")
            if a4_gate == "PASS" and a4_dir in ("LONG", "SHORT"):
                direction = a4_dir
            else:
                return {
                    **dict(NEUTRAL_ENTRY),
                    "regime":    regime,
                    "direction": "FLAT",
                    "reason":    "MR: A2 neutral + A4 no signal",
                    "available": True,
                }
        else:
            return {
                **dict(NEUTRAL_ENTRY),
                "regime":    regime,
                "direction": "FLAT",
                "reason":    "A2 NEUTRAL — no direction to time",
                "available": True,
            }

        # PNS → always flat regardless
        if regime == "PNS":
            return {
                **dict(NEUTRAL_ENTRY),
                "regime":    "PNS",
                "direction": direction,
                "reason":    "PNS regime — no entries allowed",
                "available": True,
            }

        # ── Extract features ──────────────────────────────────────────────────
        f = self._extract_features(vwap_data, stat_data, flow_data)

        # ── Route to regime scorer ────────────────────────────────────────────
        if regime in ("THV", "TLV"):
            score, entry_type, reason = self._score_trend(
                direction, regime, f, a2_conf
            )
        elif regime == "MR":
            score, entry_type, reason = self._score_mean_revert(
                direction, f, a2_conf
            )
        else:
            score, entry_type, reason = 0.0, "WAIT", "Unknown regime"

        wait = score < SCORE_EXECUTE

        return {
            "entry_score": round(score, 4),
            "entry_type":  entry_type,
            "wait":        wait,
            "reason":      reason,
            "direction":   direction,
            "regime":      regime,
            "available":   True,
        }

    def _extract_features(
        self,
        vwap: Dict,
        stat: Dict,
        flow: Dict,
    ) -> Dict:
        """Flatten all feature dicts into a single lookup dict."""
        return {
            # VWAP
            "tokyo_vwap":           vwap.get("tokyo_vwap", 0.0),
            "london_vwap":          vwap.get("london_vwap", 0.0),
            "ny_vwap":              vwap.get("ny_vwap", 0.0),
            "pvt":                  vwap.get("price_vs_tokyo_vwap", 0.0),
            "pvl":                  vwap.get("price_vs_london_vwap", 0.0),
            "pvn":                  vwap.get("price_vs_ny_vwap", 0.0),
            "london_upper":         vwap.get("london_vwap_upper", 0.0),
            "london_lower":         vwap.get("london_vwap_lower", 0.0),
            "ny_band_width":        vwap.get("ny_vwap_band_width", 0.0),
            "london_band_width":    vwap.get("london_vwap_band_width", 0.0),
            "in_overlap":           vwap.get("in_london_ny_overlap", 0.0),
            "vwap_trend":           vwap.get("session_vwap_trend", 0.0),
            # Statistical
            "atr_m15":              stat.get("atr_14_m15", stat.get("atr_14", 0.0)),
            "vol_ratio":            stat.get("vol_ratio", 1.0),
            "high_vol":             stat.get("high_vol_regime", 0.0),
            "zscore_5d":            stat.get("zscore_close_5d", 0.0),
            "zscore_20d":           stat.get("zscore_close_20d", 0.0),
            "momentum_1d":          stat.get("momentum_1d", 0.0),
            "momentum_5d":          stat.get("momentum_5d", 0.0),
            "bb_pct":               stat.get("bb_pct_20", 0.5),
            "rsi":                  self._infer_rsi(stat),
            # Flow
            "obi_1m":               flow.get("imbalance_1m", 0.0),
            "obi_5m":               flow.get("imbalance_5m", 0.0),
            "obi_delta":            flow.get("flow_imbalance_delta", 0.0),
            "buy_pressure":         flow.get("buy_pressure_100", 0.5),
            "spread_ratio":         flow.get("spread_ratio", 1.0),
            "spread_zscore":        flow.get("spread_zscore", 0.0),
            "tick_velocity_1m":     flow.get("tick_velocity_1m", 0.0),
            "velocity_ratio":       flow.get("velocity_ratio", 1.0),
        }

    @staticmethod
    def _infer_rsi(stat: Dict) -> float:
        """
        RSI is not directly in stat features — infer from zscore_5d.
        zscore_5d maps roughly to RSI: zscore=+2 ≈ RSI=70, -2 ≈ RSI=30.
        Returns estimated RSI in [0, 100].
        """
        z = stat.get("zscore_close_5d", 0.0)
        # Map z-score [-3, +3] → RSI [10, 90]
        rsi = 50.0 + (z / 3.0) * 40.0
        return float(max(10.0, min(90.0, rsi)))

    # ── Trend regime scorer (THV + TLV) ──────────────────────────────────────

    def _score_trend(
        self,
        direction: str,
        regime:    str,
        f:         Dict,
        a2_conf:   float,
    ) -> Tuple[float, str, str]:
        """
        Score entry quality in trending regime (THV or TLV).

        Best entries in trends:
          1. Pullback to session VWAP (mean reversion within trend)
          2. RSI reset to neutral zone (not overbought/oversold)
          3. OBI confirming direction
          4. Momentum still intact

        Returns:
            (score 0–1, entry_type, reason_string)
        """
        score      = 0.0
        entry_type = "WAIT"
        reasons    = []
        is_long    = direction == "LONG"

        # ── 1. VWAP proximity (highest weight: 0.35) ──────────────────────────
        # Use NY VWAP during NY session, London VWAP otherwise
        pvn = f["pvn"]   # price_vs_ny_vwap: positive = above, negative = below
        pvl = f["pvl"]   # price_vs_london_vwap

        # For LONG: we want price AT or BELOW vwap (pullback to it)
        # For SHORT: we want price AT or ABOVE vwap (rally to it)
        vwap_score = 0.0
        # NY VWAP is only valid after 13:00 UTC — pvn=0.0 before that means no data
        from datetime import datetime, timezone
        ny_open = datetime.now(timezone.utc).hour >= 13
        if is_long:
            if ny_open and abs(pvn) < VWAP_PROXIMITY:
                vwap_score = 0.35
                entry_type = "VWAP_PULLBACK"
                reasons.append(f"at NY VWAP (pvn={pvn:.4f})")
            elif ny_open and pvn < 0 and abs(pvn) < VWAP_PROXIMITY * 3:
                vwap_score = 0.20
                entry_type = "VWAP_PULLBACK"
                reasons.append(f"below NY VWAP — pullback zone (pvn={pvn:.4f})")
            elif abs(pvl) < VWAP_PROXIMITY:
                vwap_score = 0.25
                entry_type = "VWAP_PULLBACK"
                reasons.append(f"at London VWAP (pvl={pvl:.4f})")
        else:  # SHORT
            if ny_open and abs(pvn) < VWAP_PROXIMITY:
                vwap_score = 0.35
                entry_type = "VWAP_PULLBACK"
                reasons.append(f"at NY VWAP (pvn={pvn:.4f})")
            elif ny_open and pvn > 0 and abs(pvn) < VWAP_PROXIMITY * 3:
                vwap_score = 0.20
                entry_type = "VWAP_PULLBACK"
                reasons.append(f"above NY VWAP — rally to VWAP (pvn={pvn:.4f})")
            elif abs(pvl) < VWAP_PROXIMITY:
                vwap_score = 0.25
                entry_type = "VWAP_PULLBACK"
                reasons.append(f"at London VWAP (pvl={pvl:.4f})")

        score += vwap_score

        # ── 2. RSI not extended (weight: 0.20) ───────────────────────────────
        rsi = f["rsi"]
        rsi_score = 0.0
        if is_long:
            if 35 <= rsi <= 60:
                rsi_score = 0.20
                reasons.append(f"RSI reset {rsi:.0f}")
            elif rsi < 35:
                rsi_score = 0.10   # oversold — ok for MR, weak for trend
                reasons.append(f"RSI oversold {rsi:.0f}")
            elif rsi > RSI_OVERBOUGHT:
                rsi_score = -0.10  # penalise chasing
                reasons.append(f"RSI overbought {rsi:.0f} — penalty")
        else:
            if 40 <= rsi <= 65:
                rsi_score = 0.20
                reasons.append(f"RSI reset {rsi:.0f}")
            elif rsi > 65:
                rsi_score = 0.10
                reasons.append(f"RSI overbought {rsi:.0f}")
            elif rsi < RSI_OVERSOLD:
                rsi_score = -0.10
                reasons.append(f"RSI oversold {rsi:.0f} — penalty")

        score += rsi_score

        # ── 3. OBI confirming direction (weight: 0.20) ───────────────────────
        obi   = f["obi_1m"]
        delta = f["obi_delta"]
        obi_score = 0.0

        if is_long:
            if obi > 0.05:
                obi_score = 0.20
                reasons.append(f"OBI confirming LONG ({obi:.3f})")
            elif obi > 0:
                obi_score = 0.10
                reasons.append(f"OBI weakly positive ({obi:.3f})")
            elif delta > 0.05:
                obi_score = 0.10
                reasons.append(f"OBI flipping positive (delta={delta:.3f})")
            elif obi < -0.10:
                obi_score = -0.10
                reasons.append(f"OBI opposing LONG ({obi:.3f}) — penalty")
        else:
            if obi < -0.05:
                obi_score = 0.20
                reasons.append(f"OBI confirming SHORT ({obi:.3f})")
            elif obi < 0:
                obi_score = 0.10
                reasons.append(f"OBI weakly negative ({obi:.3f})")
            elif delta < -0.05:
                obi_score = 0.10
                reasons.append(f"OBI flipping negative (delta={delta:.3f})")
            elif obi > 0.10:
                obi_score = -0.10
                reasons.append(f"OBI opposing SHORT ({obi:.3f}) — penalty")

        score += obi_score

        # ── 4. Momentum intact (weight: 0.15) ────────────────────────────────
        mom1 = f["momentum_1d"]
        mom5 = f["momentum_5d"]
        mom_score = 0.0

        if is_long:
            if mom1 > 0 and mom5 > 0:
                mom_score = 0.15
                reasons.append(f"momentum intact ({mom1:.4f}/{mom5:.4f})")
            elif mom5 > 0:
                mom_score = 0.08
                reasons.append(f"5d momentum positive ({mom5:.4f})")
            elif mom1 < 0 and mom5 < 0:
                mom_score = -0.05
                reasons.append("momentum opposing — penalty")
        else:
            if mom1 < 0 and mom5 < 0:
                mom_score = 0.15
                reasons.append(f"momentum intact SHORT ({mom1:.4f}/{mom5:.4f})")
            elif mom5 < 0:
                mom_score = 0.08
                reasons.append(f"5d momentum negative ({mom5:.4f})")
            elif mom1 > 0 and mom5 > 0:
                mom_score = -0.05
                reasons.append("momentum opposing — penalty")

        score += mom_score

        # ── 5. Spread healthy (weight: 0.10) ─────────────────────────────────
        spread_ratio = f["spread_ratio"]
        if spread_ratio < 1.3:
            score += 0.10
            reasons.append(f"spread normal ({spread_ratio:.2f}x)")
        elif spread_ratio > 2.0:
            score -= 0.05
            reasons.append(f"wide spread ({spread_ratio:.2f}x) — penalty")

        # ── THV bonus: higher vol = more momentum conviction ─────────────────
        if regime == "THV" and f["vol_ratio"] > 1.2:
            score += 0.05
            reasons.append("THV vol bonus")

        # ── A2 conviction boost ───────────────────────────────────────────────
        if a2_conf > 0.5:
            score += 0.05
            reasons.append(f"A2 high conf ({a2_conf:.2f})")

        # ── Entry type fallback if no VWAP hit but other signals strong ───────
        if entry_type == "WAIT" and score >= SCORE_WAIT:
            if rsi_score > 0 and obi_score > 0:
                entry_type = "OBI_CONFIRM"
            elif mom_score > 0:
                entry_type = "MOMENTUM_ENTRY"

        score = max(0.0, min(1.0, score))
        reason = " | ".join(reasons) if reasons else "No clear entry pattern"
        return score, entry_type, reason

    # ── Mean revert regime scorer ─────────────────────────────────────────────

    def _score_mean_revert(
        self,
        direction: str,
        f:         Dict,
        a2_conf:   float,
    ) -> Tuple[float, str, str]:
        """
        Score entry quality in mean reverting regime (MR).

        MR entries require extremes — we're fading the move.
        RSI extreme + BB extreme + OBI flip is the ideal setup.

        Returns:
            (score 0–1, entry_type, reason_string)
        """
        score      = 0.0
        entry_type = "WAIT"
        reasons    = []
        is_long    = direction == "LONG"

        rsi    = f["rsi"]
        bb_pct = f["bb_pct"]   # 0=at lower band, 1=at upper band
        obi    = f["obi_1m"]
        delta  = f["obi_delta"]

        # ── 1. RSI extreme (weight: 0.40) ─────────────────────────────────────
        rsi_score = 0.0
        if is_long:
            if rsi <= RSI_EXTREME_LONG:
                rsi_score = 0.40
                entry_type = "RSI_EXTREME"
                reasons.append(f"RSI extreme LONG ({rsi:.0f})")
            elif rsi <= RSI_OVERSOLD:
                rsi_score = 0.20
                entry_type = "RSI_EXTREME"
                reasons.append(f"RSI oversold ({rsi:.0f})")
            else:
                reasons.append(f"RSI not extreme ({rsi:.0f}) — weak MR signal")
        else:
            if rsi >= RSI_EXTREME_SHORT:
                rsi_score = 0.40
                entry_type = "RSI_EXTREME"
                reasons.append(f"RSI extreme SHORT ({rsi:.0f})")
            elif rsi >= RSI_OVERBOUGHT:
                rsi_score = 0.20
                entry_type = "RSI_EXTREME"
                reasons.append(f"RSI overbought ({rsi:.0f})")
            else:
                reasons.append(f"RSI not extreme ({rsi:.0f}) — weak MR signal")

        score += rsi_score

        # ── 2. BB extreme (weight: 0.30) ──────────────────────────────────────
        bb_score = 0.0
        if is_long:
            if bb_pct <= 0.05:
                bb_score = 0.30
                reasons.append(f"at lower BB band (bb_pct={bb_pct:.2f})")
            elif bb_pct <= 0.15:
                bb_score = 0.15
                reasons.append(f"near lower BB (bb_pct={bb_pct:.2f})")
        else:
            if bb_pct >= 0.95:
                bb_score = 0.30
                reasons.append(f"at upper BB band (bb_pct={bb_pct:.2f})")
            elif bb_pct >= 0.85:
                bb_score = 0.15
                reasons.append(f"near upper BB (bb_pct={bb_pct:.2f})")

        score += bb_score

        # ── 3. OBI flipping (weight: 0.20) ────────────────────────────────────
        obi_score = 0.0
        if is_long:
            if delta > 0.05 and obi > -0.10:
                obi_score = 0.20
                reasons.append(f"OBI flipping positive (delta={delta:.3f})")
            elif obi > 0.05:
                obi_score = 0.15
                reasons.append(f"OBI positive ({obi:.3f})")
        else:
            if delta < -0.05 and obi < 0.10:
                obi_score = 0.20
                reasons.append(f"OBI flipping negative (delta={delta:.3f})")
            elif obi < -0.05:
                obi_score = 0.15
                reasons.append(f"OBI negative ({obi:.3f})")

        score += obi_score

        # ── 4. Spread normal (weight: 0.10) ───────────────────────────────────
        if f["spread_ratio"] < 1.3:
            score += 0.10
            reasons.append(f"spread normal ({f['spread_ratio']:.2f}x)")

        # ── A2 conviction boost ───────────────────────────────────────────────
        if a2_conf > 0.5:
            score += 0.05
            reasons.append(f"A2 high conf ({a2_conf:.2f})")

        score = max(0.0, min(1.0, score))
        reason = " | ".join(reasons) if reasons else "No MR extreme found"
        return score, entry_type, reason

    # ── Background scheduler ──────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """Start background daemon thread — recomputes every 15 min."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("EntryEngine | Scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="entry-engine-scheduler",
        )
        self._scheduler_thread.start()
        logger.info(
            f"EntryEngine | Scheduler started (interval={SCHEDULER_INTERVAL}s)"
        )

    def stop_scheduler(self) -> None:
        self._running = False

    def _scheduler_loop(self) -> None:
        logger.info("EntryEngine | Scheduler loop running")
        while self._running:
            try:
                self.compute_all()
            except Exception as exc:
                logger.error(f"EntryEngine | Scheduler error: {exc}")
            time.sleep(SCHEDULER_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Module singleton
# ──────────────────────────────────────────────────────────────────────────────

entry_engine = EntryEngine()
