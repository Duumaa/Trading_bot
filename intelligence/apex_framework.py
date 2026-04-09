"""
intelligence/apex_framework.py
================================
APEX — Autonomous Pattern eXpansion Framework.

Derives dynamic confidence penalty rules from the trade_postmortems table
and applies them at signal evaluation time. Unlike static thresholds in
signal_aggregator.py, APEX rules are recomputed nightly from the actual
failure database — so they adapt as the bot accumulates more trade history.

APEX sits between _compute_confidence() and the final signal output:
    conf = _compute_confidence(...)
    conf = conf * PAIR_MODEL_QUALITY.get(pair, 1.0)
    conf = conf * apex_framework.get_penalty(pair, signal_ctx)   ← new

Rules currently derived:
    R1  Negative A2 score         a2_fund_score < 0        → 0.40×
    R2  Weak A2 + strong A1       a2 < 0.30, a1_conf > 0.55 → 0.65×
    R3  Very weak A2 (any)        a2 < 0.25                → 0.70×
    R4  Silent sentiment + weak A2  |a5| < 0.03, a2 < 0.35  → 0.80×
    R5  Dynamic pair quality      pairs with ≥3 MODEL_FAILUREs, avg_a2 < 0.40
                                  penalty proportional to failure rate
    R6  Opposing A5 sentiment     direction=LONG + a5 < -0.05 → 0.55×
                                  direction=SELL + a5 > +0.05 → 0.55×
                                  (27/51 MODEL_FAILUREs had opposing sentiment,
                                   avg -16.4 pips vs -5.4 when aligned)
    R7  MR regime                 a1_regime == "MR"        → 0.80×
                                  (22% WR in MR from backtest, worst regime)

Rules are applied multiplicatively; the final penalty is capped at 0.35×
(no signal can be zeroed out by APEX alone).

Refresh: nightly at 00:05 UTC (background thread).
Redis:   apex:rules:summary  — JSON snapshot of active rules (TTL 25h)
         apex:penalty:{PAIR}  — last penalty applied + reasons (TTL 1h)
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text

load_dotenv("/workspaces/Trading_bot/.env")

# ── constants ────────────────────────────────────────────────────────────────

# Minimum number of MODEL_FAILURE records needed before APEX applies R1–R4
MIN_MODEL_FAILURES_GLOBAL = 15

# R5: minimum pair-level MODEL_FAILUREs before pair penalty activates
MIN_PAIR_FAILURES = 3

# Floor — APEX cannot reduce confidence below this fraction of its input
APEX_FLOOR_MULT = 0.35

# Rule thresholds (static; derived from postmortem analysis Mar 2026)
A2_NEGATIVE_THRESHOLD     = 0.0    # R1
A2_WEAK_THRESHOLD         = 0.25   # R3
A2_TREND_MISS_THRESHOLD   = 0.30   # R2 (A2 side)
A1_OVERCONFIDENT          = 0.55   # R2 (A1 side)
A2_SILENT_THRESHOLD       = 0.35   # R4 (A2 side)
A5_SILENT_ABS             = 0.03   # R4 (A5 side)
A5_OPPOSING_THRESHOLD     = 0.05   # R6: sentiment opposes direction beyond this

# Refresh schedule: 00:05 UTC
REFRESH_HOUR   = 0
REFRESH_MINUTE = 5


# ── helpers ──────────────────────────────────────────────────────────────────

def _redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


def _db_connection():
    """Return a SQLAlchemy connection via the shared DB singleton."""
    from data_pipeline.storage.timeseries_db import db
    return db.connection()


# ══════════════════════════════════════════════════════════════════════════════
# APEX FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

class ApexFramework:
    """
    Dynamic confidence penalty engine derived from postmortem failure patterns.

    Call flow (in signal_aggregator.py after _compute_confidence):
        mult, reasons = apex_framework.get_penalty(pair, signal_ctx)
        confidence = round(confidence * mult, 4)

    signal_ctx keys expected:
        a2_fund_score  float   [-1, +1]
        a1_confidence  float   [0, 1]
        a1_regime      str     THV | TLV | MR | PNS
        a5_sentiment   float   [-1, +1]
        direction      str     LONG | SHORT | FLAT
    """

    def __init__(self) -> None:
        self._lock            = threading.Lock()
        self._rules_active    = False          # True once patterns loaded
        self._global_n        = 0              # total MODEL_FAILUREs seen
        # Dynamic per-pair quality multipliers (R5), keyed by symbol
        self._pair_penalty: Dict[str, float]  = {}
        # Pair → avg_a2 (informational, surfaced in summary)
        self._pair_avg_a2: Dict[str, float]   = {}
        self._last_refresh: Optional[datetime] = None

        self._scheduler_thread: Optional[threading.Thread] = None

    # ── pattern learning ──────────────────────────────────────────────────────

    def update_patterns(self) -> Dict[str, Any]:
        """
        Query trade_postmortems → recompute R5 pair penalties.

        Returns a summary dict for logging / Telegram.
        """
        logger.info("ApexFramework | Refreshing patterns from trade_postmortems ...")

        try:
            with _db_connection() as conn:
                # Global MODEL_FAILURE count (gates R1–R4)
                row = conn.execute(text(
                    "SELECT COUNT(*) FROM trade_postmortems WHERE failure_reason = 'MODEL_FAILURE'"
                )).fetchone()
                global_n = int(row[0]) if row else 0

                # Per-pair MODEL_FAILURE stats
                rows = conn.execute(text("""
                    SELECT
                        t.symbol,
                        COUNT(*)                                                    AS failures,
                        AVG((pm.analyst_conflict->>'a2_fund_score')::float)         AS avg_a2,
                        AVG((pm.analyst_conflict->>'a1_confidence')::float)         AS avg_a1,
                        SUM(CASE WHEN (pm.analyst_conflict->>'a2_fund_score')::float < 0 THEN 1 ELSE 0 END)
                                                                                    AS neg_a2_cnt
                    FROM trade_postmortems pm
                    JOIN trades t ON t.id = pm.trade_id
                    WHERE pm.failure_reason = 'MODEL_FAILURE'
                      AND pm.analyst_conflict IS NOT NULL
                    GROUP BY t.symbol
                    ORDER BY failures DESC
                """)).fetchall()

        except Exception as exc:
            logger.error(f"ApexFramework | DB query failed: {exc}")
            return {}

        pair_penalty: Dict[str, float] = {}
        pair_avg_a2:  Dict[str, float] = {}

        for row in rows:
            symbol, n_fail, avg_a2, avg_a1, neg_a2_cnt = (
                row[0], int(row[1]),
                float(row[2]) if row[2] is not None else 0.0,
                float(row[3]) if row[3] is not None else 0.0,
                int(row[4]) if row[4] is not None else 0,
            )

            pair_avg_a2[symbol] = round(avg_a2, 4)

            if n_fail < MIN_PAIR_FAILURES:
                continue

            # R5: penalty proportional to failure severity
            # Severity = how far avg_a2 is below the activation threshold (0.40)
            # Full penalty (-15%) at avg_a2 = 0.10, tapers to 0% at avg_a2 = 0.40
            ACTIVATION_A2 = 0.40
            if avg_a2 < ACTIVATION_A2:
                severity = (ACTIVATION_A2 - avg_a2) / ACTIVATION_A2   # 0.0–1.0
                penalty  = 1.0 - (severity * 0.15)                     # 0.85–1.0
                pair_penalty[symbol] = round(penalty, 4)
                logger.info(
                    f"ApexFramework | R5 {symbol}: {n_fail} failures, "
                    f"avg_a2={avg_a2:.3f} → pair penalty {penalty:.4f}"
                )

        with self._lock:
            self._global_n      = global_n
            self._rules_active  = global_n >= MIN_MODEL_FAILURES_GLOBAL
            self._pair_penalty  = pair_penalty
            self._pair_avg_a2   = pair_avg_a2
            self._last_refresh  = datetime.now(timezone.utc)

        # Persist summary to Redis
        summary = self._build_summary()
        try:
            r = _redis_client()
            r.set("apex:rules:summary", json.dumps(summary), ex=90_000)   # 25h
        except Exception as exc:
            logger.warning(f"ApexFramework | Redis write failed: {exc}")

        logger.success(
            f"ApexFramework | Patterns refreshed — {global_n} MODEL_FAILUREs, "
            f"R5 active for {len(pair_penalty)} pairs, "
            f"global rules {'ON' if self._rules_active else 'OFF (need ≥' + str(MIN_MODEL_FAILURES_GLOBAL) + ')'}"
        )
        return summary

    # ── penalty computation ───────────────────────────────────────────────────

    def get_penalty(
        self,
        pair: str,
        signal_ctx: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """
        Compute penalty multiplier for a signal about to be issued.

        Returns:
            (multiplier: float, reasons: List[str])
            multiplier is in [APEX_FLOOR_MULT, 1.0]
            reasons lists every rule that fired.
        """
        with self._lock:
            rules_active = self._rules_active
            pair_penalty = self._pair_penalty.get(pair, 1.0)

        a2           = float(signal_ctx.get("a2_fund_score", 1.0))
        a1_conf      = float(signal_ctx.get("a1_confidence", 0.0))
        a5_sent      = float(signal_ctx.get("a5_sentiment", 0.0))
        direction    = str(signal_ctx.get("direction", "FLAT"))
        regime       = str(signal_ctx.get("a1_regime", ""))

        mult    = 1.0
        reasons: List[str] = []

        # ── R1: Negative A2 score ─────────────────────────────────────────────
        if rules_active and a2 < A2_NEGATIVE_THRESHOLD:
            mult    *= 0.40
            reasons.append(
                f"R1:NegativeA2(a2={a2:.3f}→×0.40)"
            )

        # ── R2: Weak A2 + overconfident A1 ────────────────────────────────────
        elif rules_active and a2 < A2_TREND_MISS_THRESHOLD and a1_conf > A1_OVERCONFIDENT:
            mult    *= 0.65
            reasons.append(
                f"R2:WeakA2+StrongA1(a2={a2:.3f},a1={a1_conf:.3f}→×0.65)"
            )

        # ── R3: Very weak A2 (alone) ──────────────────────────────────────────
        elif rules_active and a2 < A2_WEAK_THRESHOLD:
            mult    *= 0.70
            reasons.append(
                f"R3:VeryWeakA2(a2={a2:.3f}→×0.70)"
            )

        # ── R4: Weak A2 + silent sentiment ───────────────────────────────────
        if (
            rules_active
            and a2 < A2_SILENT_THRESHOLD
            and abs(a5_sent) < A5_SILENT_ABS
            and len(reasons) == 0        # only if R1/R2/R3 didn't fire (avoid stacking)
        ):
            mult    *= 0.80
            reasons.append(
                f"R4:WeakA2+SilentSentiment(a2={a2:.3f},a5={a5_sent:.4f}→×0.80)"
            )

        # ── R6: Opposing A5 sentiment ─────────────────────────────────────────
        # Fires when sentiment actively opposes trade direction — independent of
        # R1–R4 (strength-based) so it stacks. Evidence: 27/51 MODEL_FAILUREs
        # had opposing sentiment, losing -16.4 pips avg vs -5.4 when aligned.
        if rules_active and a5_sent != 0.0 and direction != "FLAT":
            sentiment_opposing = (
                (direction == "LONG"  and a5_sent < -A5_OPPOSING_THRESHOLD) or
                (direction == "SHORT" and a5_sent >  A5_OPPOSING_THRESHOLD)
            )
            if sentiment_opposing:
                mult *= 0.55
                reasons.append(
                    f"R6:OpposingSentiment(dir={direction},a5={a5_sent:.4f}→×0.55)"
                )

        # ── R7: MR regime ─────────────────────────────────────────────────────
        # MR is the worst regime: 22% WR, -$98 from backtest, 12/51 MODEL_FAILUREs.
        # A 0.80× penalty makes MR entries harder to clear the confidence threshold.
        if rules_active and regime == "MR":
            mult *= 0.80
            reasons.append(f"R7:MRRegime(→×0.80)")

        # ── R5: Pair-specific quality penalty ─────────────────────────────────
        if pair_penalty < 1.0:
            mult    *= pair_penalty
            reasons.append(
                f"R5:PairQuality({pair}→×{pair_penalty:.4f})"
            )

        # Apply floor
        mult = max(mult, APEX_FLOOR_MULT)

        # Persist per-pair result to Redis (debug / Telegram)
        if reasons:
            try:
                r = _redis_client()
                payload = {
                    "pair":       pair,
                    "multiplier": round(mult, 4),
                    "reasons":    reasons,
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                    "a2":          round(a2, 4),
                    "a1_conf":     round(a1_conf, 4),
                    "a5_sent":     round(a5_sent, 4),
                }
                r.set(f"apex:penalty:{pair}", json.dumps(payload), ex=3_600)
            except Exception:
                pass   # non-blocking — diagnostics only

            logger.debug(
                f"ApexFramework | {pair} penalty={mult:.4f} rules={reasons}"
            )

        return round(mult, 4), reasons

    # ── summary / reporting ───────────────────────────────────────────────────

    def _build_summary(self) -> Dict[str, Any]:
        """Build a JSON-serialisable summary of active rules."""
        with self._lock:
            return {
                "global_model_failures": self._global_n,
                "rules_active":          self._rules_active,
                "last_refresh":          (
                    self._last_refresh.isoformat()
                    if self._last_refresh else None
                ),
                "global_rules": {
                    "R1_negative_a2":       {"threshold": A2_NEGATIVE_THRESHOLD, "mult": 0.40},
                    "R2_weak_a2_strong_a1": {"a2_threshold": A2_TREND_MISS_THRESHOLD,
                                             "a1_threshold": A1_OVERCONFIDENT, "mult": 0.65},
                    "R3_very_weak_a2":      {"threshold": A2_WEAK_THRESHOLD, "mult": 0.70},
                    "R4_silent_sentiment":  {"a2_threshold": A2_SILENT_THRESHOLD,
                                             "a5_abs_threshold": A5_SILENT_ABS, "mult": 0.80},
                    "R6_opposing_sentiment":{"a5_threshold": A5_OPPOSING_THRESHOLD, "mult": 0.55},
                    "R7_mr_regime":         {"regime": "MR", "mult": 0.80},
                },
                "pair_penalties_R5":     self._pair_penalty,
                "pair_avg_a2":           self._pair_avg_a2,
                "floor_mult":            APEX_FLOOR_MULT,
            }

    def get_summary_text(self) -> str:
        """Return a human-readable Telegram-friendly summary of active rules."""
        s = self._build_summary()
        lines = [
            "<b>APEX Framework Status</b>",
            f"Global MODEL_FAILUREs: {s['global_model_failures']}",
            f"Rules active: {'✅ YES' if s['rules_active'] else '❌ NO (need ≥' + str(MIN_MODEL_FAILURES_GLOBAL) + ')'}",
            f"Last refresh: {s['last_refresh'] or 'never'}",
            "",
            "<b>Global rules (R1–R4):</b>",
            f"  R1 Negative A2 (a2 &lt; 0)          → ×0.40",
            f"  R2 Weak A2 &lt; 0.30 + A1 &gt; 0.55  → ×0.65",
            f"  R3 Very weak A2 &lt; 0.25             → ×0.70",
            f"  R4 |A5| &lt; 0.03 + A2 &lt; 0.35     → ×0.80",
            f"  R6 Sentiment opposes direction   → ×0.55",
            f"  R7 MR regime                     → ×0.80",
        ]

        if s["pair_penalties_R5"]:
            lines.append("")
            lines.append("<b>Pair quality penalties (R5):</b>")
            for sym, mult in sorted(s["pair_penalties_R5"].items()):
                avg_a2 = s["pair_avg_a2"].get(sym, "?")
                lines.append(
                    f"  {sym:7s} ×{mult:.4f}  (avg_a2={avg_a2:.3f})"
                )
        else:
            lines.append("")
            lines.append("Pair penalties (R5): none active")

        return "\n".join(lines)

    # ── scheduler ─────────────────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """Start background thread that refreshes patterns nightly at 00:05 UTC."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.debug("ApexFramework | Scheduler already running")
            return

        # Initial load on startup
        self.update_patterns()

        def _loop() -> None:
            while True:
                try:
                    now = datetime.now(timezone.utc)
                    # Sleep until next 00:05 UTC
                    target = now.replace(
                        hour=REFRESH_HOUR,
                        minute=REFRESH_MINUTE,
                        second=0,
                        microsecond=0,
                    )
                    if target <= now:
                        # Already past today's target — schedule for tomorrow
                        from datetime import timedelta
                        target = target + timedelta(days=1)

                    sleep_secs = (target - now).total_seconds()
                    logger.info(
                        f"ApexFramework | Next pattern refresh in "
                        f"{sleep_secs/3600:.1f}h at {target.isoformat()}"
                    )
                    time.sleep(sleep_secs)
                    self.update_patterns()
                except Exception as exc:
                    logger.error(f"ApexFramework | Scheduler error: {exc}")
                    time.sleep(3_600)   # back off 1h on error

        self._scheduler_thread = threading.Thread(
            target=_loop, name="apex-scheduler", daemon=True
        )
        self._scheduler_thread.start()
        logger.success("ApexFramework | Scheduler started (nightly 00:05 UTC)")


# ── module singleton ──────────────────────────────────────────────────────────

apex_framework = ApexFramework()
