"""
execution/technical_exit_monitor.py
=====================================
Technical Invalidation Exit Monitor — G10 Autonomous Dominance Architecture

Monitors all open positions every M15 candle close and exits or tightens
stops when the TCN+PatchTSMixer (Analyst 4) signals invalidate the trade.

PHILOSOPHY:
    Entry used Analyst 4 to confirm direction. The open trade must remain
    continuously re-validated by those same models. If the technical picture
    flips or degrades while fundamentals are still intact, the timing was
    wrong — exit now, re-enter when technicals agree again.

    This is NOT ensembling or averaging. It is:
        "The technician who called the entry is now calling the exit."

THREE INVALIDATION CONDITIONS (checked in priority order):
    1. FULL_FLIP    — Both TCN+Patch now oppose trade direction.
                      Action: CLOSE IMMEDIATELY. No debate.

    2. MIXED_WEAK   — One model opposes OR combined confidence < CONF_CLOSE_THRESHOLD.
                      Action: Tighten SL to breakeven or last structure level.

    3. DECAY_TREND  — Confidence trending down for DECAY_CANDLES consecutive
                      readings without recovery.
                      Action: Close position (slow bleed = as bad as flip).

FUNDAMENTAL VETO (prevents premature exit):
    If A1+A2 fundamentals are still strongly aligned with trade direction
    AND a HIGH-impact event is within 2 hours → hold but tighten stop.
    Rationale: pre-event technical noise should not kill a good macro thesis.
    This veto only applies to MIXED_WEAK, NOT to FULL_FLIP.

REDIS KEYS:
    features:analyst4:{PAIR}            → Current TCN+Patch inference output
    signal:{PAIR}                       → Aggregated gate signal (for fund veto)
    tech:exit:history:{PAIR}:{pos_id}   → List[float] confidence readings, TTL=2h
    tech:exit:last_check:{PAIR}         → Timestamp of last check (dedup)

CLOSE REASONS (written to trades table):
    CLOSED_TECH_INVALIDATION_FLIP       — Full directional flip
    CLOSED_TECH_INVALIDATION_DECAY      — Sustained confidence decay
    CLOSED_TECH_TIGHTENED               — Stop tightened, position held

USAGE (in main_loop.py, after each M15 candle):
    from execution.technical_exit_monitor import tech_exit_monitor
    tech_exit_monitor.run_checks()
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import redis
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text

load_dotenv("/workspaces/Trading_bot/.env")

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — tune after postmortem data (target: April 2026)
# ─────────────────────────────────────────────────────────────────────────────

# Combined A4 confidence below this → CLOSE (with full flip confirmed)
CONF_CLOSE_THRESHOLD: float = 0.38

# Combined A4 confidence below this → TIGHTEN STOP (single model or soft drop)
CONF_TIGHTEN_THRESHOLD: float = 0.48

# Number of consecutive M15 candles with declining confidence → close on decay
DECAY_CANDLES: int = 4

# Minimum confidence drop per candle to count as "declining" (not just noise)
DECAY_MIN_DROP_PER_CANDLE: float = 0.03

# Fundamental veto: if fund score is this strong, allow tighten-not-close
FUNDAMENTAL_VETO_THRESHOLD: float = 0.60

# Hours before high-impact event within which veto is active
PRE_EVENT_VETO_HOURS: float = 2.0

# ATR multiplier for tightened stop (replaces current SL distance)
TIGHTEN_ATR_MULTIPLIER: float = 0.75

# Redis TTL for confidence history (2 hours = 8 M15 candles with buffer)
HISTORY_TTL_SECONDS: int = 7200


# ─────────────────────────────────────────────────────────────────────────────
# REDIS CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def _get_redis() -> redis.Redis:
    """Return authenticated Redis connection."""
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES (typed dicts, no external deps)
# ─────────────────────────────────────────────────────────────────────────────

class A4Signal:
    """Parsed output from Analyst 4 (TCN + PatchTSMixer) Redis cache."""

    def __init__(
        self,
        tcn_direction: str,       # "BUY" | "SELL" | "NEUTRAL"
        tcn_confidence: float,
        patch_direction: str,     # "BUY" | "SELL" | "NEUTRAL"
        patch_confidence: float,
    ) -> None:
        self.tcn_direction = tcn_direction
        self.tcn_confidence = tcn_confidence
        self.patch_direction = patch_direction
        self.patch_confidence = patch_confidence

    @property
    def combined_confidence(self) -> float:
        """Average confidence of both models."""
        return (self.tcn_confidence + self.patch_confidence) / 2.0

    @property
    def both_agree(self) -> bool:
        """Both models pointing the same direction (not NEUTRAL)."""
        return (
            self.tcn_direction == self.patch_direction
            and self.tcn_direction != "NEUTRAL"
        )

    def opposes(self, trade_direction: str) -> Tuple[bool, bool]:
        """
        Check how many models oppose the trade direction.

        Args:
            trade_direction: "BUY" or "SELL"

        Returns:
            (tcn_opposes, patch_opposes)
        """
        opposite = "SELL" if trade_direction == "BUY" else "BUY"
        return (
            self.tcn_direction == opposite,
            self.patch_direction == opposite,
        )


class OpenPosition:
    """Minimal representation of a live trade from the DB."""

    def __init__(
        self,
        position_id: int,
        symbol: str,
        direction: str,         # "BUY" | "SELL"
        entry_price: float,
        stop_loss: float,
        current_atr: float,
        entry_time: datetime,
        ctrader_position_id: str = "",
    ) -> None:
        self.position_id = position_id
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.current_atr = current_atr
        self.entry_time = entry_time
        self.ctrader_position_id = ctrader_position_id


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODULE
# ─────────────────────────────────────────────────────────────────────────────

class TechnicalExitMonitor:
    """
    Monitors open positions for technical invalidation every M15 candle.

    Reads A4 (TCN + PatchTSMixer) signals from Redis each candle close.
    Closes or tightens positions when technical conviction evaporates.
    Writes close reason to trades table for postmortem analysis.
    """

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────

    def run_checks(self) -> Dict[str, str]:
        """
        Run technical invalidation checks on all open positions.

        Called once per M15 candle close from main_loop.py.

        Returns:
            Dict mapping symbol → action taken, e.g. {"AUDJPY": "CLOSED_FLIP"}
        """
        logger.info("TechExitMonitor | Running M15 invalidation checks...")

        results: Dict[str, str] = {}

        try:
            positions = self._get_open_positions()
            if not positions:
                logger.debug("TechExitMonitor | No open positions — skipping.")
                return {}

            logger.info(
                f"TechExitMonitor | Checking {len(positions)} open position(s): "
                f"{[p.symbol for p in positions]}"
            )

            for pos in positions:
                action = self._check_position(pos)
                results[pos.symbol] = action
                if action != "HOLD":
                    logger.info(
                        f"TechExitMonitor | {pos.symbol} → {action} "
                        f"(id={pos.position_id})"
                    )

        except Exception as exc:
            logger.error(f"TechExitMonitor | run_checks failed: {exc}")

        return results

    # ──────────────────────────────────────────────────────────────────────
    # POSITION CHECKING LOGIC
    # ──────────────────────────────────────────────────────────────────────

    def _check_position(self, pos: OpenPosition) -> str:
        """
        Evaluate a single open position against current A4 signals.

        Returns:
            "HOLD"              — No action, thesis intact
            "CLOSED_FLIP"       — Closed: full directional flip
            "CLOSED_DECAY"      — Closed: sustained confidence decay
            "TIGHTENED"         — Stop moved to breakeven/structure
            "TIGHTENED_VETO"    — Tightened (fundamental veto blocked close)
        """
        a4 = self._get_a4_signal(pos.symbol)
        if a4 is None:
            logger.warning(
                f"TechExitMonitor | {pos.symbol} — A4 signal missing "
                f"from Redis. Skipping (fail-safe: hold)."
            )
            return "HOLD"

        # Store confidence reading for decay tracking
        combined_conf = a4.combined_confidence
        self._push_confidence_history(pos.symbol, pos.position_id, combined_conf)

        tcn_opp, patch_opp = a4.opposes(pos.direction)

        # ── CONDITION 1: FULL FLIP — both models now oppose ──────────────
        if tcn_opp and patch_opp:
            logger.warning(
                f"TechExitMonitor | {pos.symbol} FULL FLIP detected. "
                f"Trade={pos.direction} | "
                f"TCN={a4.tcn_direction}({a4.tcn_confidence:.2f}) "
                f"Patch={a4.patch_direction}({a4.patch_confidence:.2f})"
            )
            # No fundamental veto on full flip — close unconditionally
            self._execute_close(pos, "CLOSED_TECH_INVALIDATION_FLIP")
            return "CLOSED_FLIP"

        # ── CONDITION 2: MIXED WEAKNESS — one opposes or conf collapsed ──
        one_opposes = tcn_opp or patch_opp
        conf_below_close = combined_conf < CONF_CLOSE_THRESHOLD
        conf_below_tighten = combined_conf < CONF_TIGHTEN_THRESHOLD

        if one_opposes or conf_below_close:
            # Check fundamental veto before closing
            if self._fundamental_veto_active(pos.symbol, pos.direction):
                logger.info(
                    f"TechExitMonitor | {pos.symbol} MIXED_WEAK but "
                    f"fundamental veto active — tightening stop only."
                )
                self._tighten_stop(pos, reason="CLOSED_TECH_TIGHTENED_VETO")
                return "TIGHTENED_VETO"

            # No veto — close on mixed weakness
            logger.warning(
                f"TechExitMonitor | {pos.symbol} MIXED_WEAK — "
                f"one_opposes={one_opposes} conf={combined_conf:.2f} "
                f"(threshold={CONF_CLOSE_THRESHOLD}). Closing."
            )
            self._execute_close(pos, "CLOSED_TECH_INVALIDATION_FLIP")
            return "CLOSED_FLIP"

        # ── CONDITION 3: CONFIDENCE DECAY over multiple candles ──────────
        if conf_below_tighten and not one_opposes:
            # Soft zone: conf is declining but models not yet opposed
            if self._is_sustained_decay(pos.symbol, pos.position_id):
                logger.warning(
                    f"TechExitMonitor | {pos.symbol} DECAY_TREND — "
                    f"conf={combined_conf:.2f} declining for "
                    f"{DECAY_CANDLES}+ candles. Closing."
                )
                self._execute_close(pos, "CLOSED_TECH_INVALIDATION_DECAY")
                return "CLOSED_DECAY"

            # Decay starting but not sustained yet — tighten as precaution
            logger.info(
                f"TechExitMonitor | {pos.symbol} EARLY_DECAY — "
                f"conf={combined_conf:.2f} below tighten threshold. "
                f"Tightening stop."
            )
            self._tighten_stop(pos, reason="CLOSED_TECH_TIGHTENED")
            return "TIGHTENED"

        # ── ALL CLEAR — thesis intact ─────────────────────────────────────
        logger.debug(
            f"TechExitMonitor | {pos.symbol} HOLD — "
            f"TCN={a4.tcn_direction}({a4.tcn_confidence:.2f}) "
            f"Patch={a4.patch_direction}({a4.patch_confidence:.2f}) "
            f"combined={combined_conf:.2f}"
        )
        return "HOLD"

    # ──────────────────────────────────────────────────────────────────────
    # FUNDAMENTAL VETO
    # ──────────────────────────────────────────────────────────────────────

    def _fundamental_veto_active(self, symbol: str, trade_direction: str) -> bool:
        """
        Returns True if fundamental analysts + event proximity justify
        holding through technical weakness (tighten only, not close).

        Veto conditions (BOTH must be true):
            1. A1+A2 fundamental score still strongly supports trade direction
            2. A HIGH-impact event is within PRE_EVENT_VETO_HOURS

        The veto NEVER applies to FULL_FLIP (Condition 1).
        It only softens MIXED_WEAK (Condition 2).
        """
        try:
            r = self._get_redis_client()

            # Check fundamental alignment
            signal_raw = r.get(f"signal:{symbol}")
            if not signal_raw:
                return False

            signal = json.loads(signal_raw)
            fund_score = abs(float(signal.get("fund_score", 0.0)))
            fund_direction = signal.get("direction", "NEUTRAL")

            if fund_score < FUNDAMENTAL_VETO_THRESHOLD:
                return False  # Fundamentals weak — no veto
            if fund_direction != trade_direction:
                return False  # Fundamentals also flipped — no veto

            # Check event proximity
            from data_pipeline.storage.timeseries_db import db
            now_utc = datetime.now(timezone.utc)
            window_end = now_utc + timedelta(hours=PRE_EVENT_VETO_HOURS)

            with db.connection() as conn:
                result = conn.execute(
                    text("""
                        SELECT COUNT(*) FROM economic_events
                        WHERE event_time BETWEEN :now AND :window_end
                          AND impact = 'HIGH'
                          AND country IN (
                              SELECT unnest(string_to_array(:currencies, ','))
                          )
                    """),
                    {
                        "now": now_utc,
                        "window_end": window_end,
                        "currencies": self._get_pair_currencies(symbol),
                    },
                )
                row = result.fetchone()
                event_imminent = (row[0] > 0) if row else False

            if event_imminent:
                logger.debug(
                    f"TechExitMonitor | {symbol} — Fundamental veto active: "
                    f"fund_score={fund_score:.2f} + HIGH event within "
                    f"{PRE_EVENT_VETO_HOURS}h."
                )
                return True

        except Exception as exc:
            logger.warning(
                f"TechExitMonitor | Fundamental veto check failed for "
                f"{symbol}: {exc}. Defaulting to no veto."
            )

        return False

    # ──────────────────────────────────────────────────────────────────────
    # CONFIDENCE HISTORY (decay tracking)
    # ──────────────────────────────────────────────────────────────────────

    def _push_confidence_history(
        self, symbol: str, position_id: int, confidence: float
    ) -> None:
        """
        Push latest combined confidence to Redis history list for this position.
        Keeps only the last DECAY_CANDLES+1 readings.
        """
        try:
            r = self._get_redis_client()
            key = f"tech:exit:history:{symbol}:{position_id}"
            r.rpush(key, str(round(confidence, 4)))
            # Trim to last DECAY_CANDLES+1 values (1 extra for comparison)
            r.ltrim(key, -(DECAY_CANDLES + 1), -1)
            r.expire(key, HISTORY_TTL_SECONDS)
        except Exception as exc:
            logger.warning(
                f"TechExitMonitor | Failed to push confidence history "
                f"for {symbol}: {exc}"
            )

    def _is_sustained_decay(self, symbol: str, position_id: int) -> bool:
        """
        Returns True if confidence has declined monotonically for
        DECAY_CANDLES consecutive M15 candles, each dropping by at
        least DECAY_MIN_DROP_PER_CANDLE.

        Monotonic: every reading must be lower than the previous.
        This prevents noise spikes from triggering a decay close.
        """
        try:
            r = self._get_redis_client()
            key = f"tech:exit:history:{symbol}:{position_id}"
            raw_history = r.lrange(key, 0, -1)

            if len(raw_history) < DECAY_CANDLES:
                return False  # Not enough readings yet

            # Take the last DECAY_CANDLES readings
            history = [float(v) for v in raw_history[-DECAY_CANDLES:]]

            # Check monotonic decline with minimum drop per candle
            for i in range(1, len(history)):
                drop = history[i - 1] - history[i]
                if drop < DECAY_MIN_DROP_PER_CANDLE:
                    return False  # Not declining fast enough

            logger.debug(
                f"TechExitMonitor | {symbol} decay history: "
                f"{[round(h, 3) for h in history]}"
            )
            return True

        except Exception as exc:
            logger.warning(
                f"TechExitMonitor | Decay check failed for {symbol}: {exc}"
            )
            return False

    # ──────────────────────────────────────────────────────────────────────
    # EXECUTION ACTIONS
    # ──────────────────────────────────────────────────────────────────────

    def _execute_close(self, pos: OpenPosition, reason: str) -> None:
        """
        Close an open position via order_executor and update DB.

        Args:
            pos:    The open position to close.
            reason: Close reason string (written to trades table).
        """
        try:
            from execution.order_executor import order_executor

            logger.warning(
                f"TechExitMonitor | CLOSING {pos.symbol} id={pos.position_id} "
                f"reason={reason}"
            )

            # Send close order via cTrader
            close_result = order_executor.close_position(
                pair=pos.symbol,
                position_id=pos.ctrader_position_id or str(pos.position_id),
                reason=reason,
            )

            # Update DB regardless of result in PAPER mode
            # In LIVE mode close_result confirms the order was accepted
            self._update_trade_close_reason(pos.position_id, reason)
            self._cleanup_history(pos.symbol, pos.position_id)

            if close_result:
                logger.success(
                    f"TechExitMonitor | {pos.symbol} id={pos.position_id} "
                    f"closed successfully. Reason: {reason}"
                )
            else:
                logger.warning(
                    f"TechExitMonitor | {pos.symbol} id={pos.position_id} "
                    f"close_position returned falsy — DB updated anyway. "
                    f"Verify position status on cTrader manually."
                )

        except Exception as exc:
            logger.error(
                f"TechExitMonitor | _execute_close failed for "
                f"{pos.symbol} id={pos.position_id}: {exc}"
            )

    def _tighten_stop(self, pos: OpenPosition, reason: str) -> None:
        """
        Move stop-loss to breakeven or tighter ATR-based level.

        Logic:
            If current unrealised P&L > 0 → move SL to entry (breakeven)
            If unrealised P&L ≤ 0         → tighten to ATR × TIGHTEN_ATR_MULTIPLIER

        The tighter of the two is used.
        """
        try:
            from execution.order_executor import order_executor

            # Attempt to get current mid price
            r = self._get_redis_client()
            tick_raw = r.get(f"tick:{pos.symbol}")
            current_price: Optional[float] = None

            if tick_raw:
                tick = json.loads(tick_raw)
                current_price = (
                    float(tick.get("bid", 0)) + float(tick.get("ask", 0))
                ) / 2.0

            # Compute tighter SL candidates
            atr_tight = pos.current_atr * TIGHTEN_ATR_MULTIPLIER

            if pos.direction == "BUY":
                atr_sl = (current_price or pos.entry_price) - atr_tight
                breakeven_sl = pos.entry_price + 0.00002  # 0.2 pip above entry
                new_sl = max(atr_sl, breakeven_sl, pos.stop_loss)
                # Never move SL further away from price
                if new_sl <= pos.stop_loss:
                    logger.debug(
                        f"TechExitMonitor | {pos.symbol} tighten: "
                        f"new SL {new_sl:.5f} not better than "
                        f"current {pos.stop_loss:.5f} — skipping."
                    )
                    return
            else:  # SELL
                atr_sl = (current_price or pos.entry_price) + atr_tight
                breakeven_sl = pos.entry_price - 0.00002
                new_sl = min(atr_sl, breakeven_sl, pos.stop_loss)
                if new_sl >= pos.stop_loss:
                    logger.debug(
                        f"TechExitMonitor | {pos.symbol} tighten: "
                        f"new SL {new_sl:.5f} not better than "
                        f"current {pos.stop_loss:.5f} — skipping."
                    )
                    return

            logger.info(
                f"TechExitMonitor | {pos.symbol} TIGHTEN SL: "
                f"{pos.stop_loss:.5f} → {new_sl:.5f} "
                f"(reason={reason})"
            )

            modify_result = order_executor.modify_stop_loss(
                position_id=pos.position_id,
                symbol=pos.symbol,
                new_stop_loss=new_sl,
            )

            if modify_result:
                self._update_trade_stop_loss(pos.position_id, new_sl, reason)
                logger.success(
                    f"TechExitMonitor | {pos.symbol} id={pos.position_id} "
                    f"SL tightened to {new_sl:.5f}"
                )
            else:
                logger.error(
                    f"TechExitMonitor | modify_stop_loss failed for "
                    f"{pos.symbol} id={pos.position_id}"
                )

        except Exception as exc:
            logger.error(
                f"TechExitMonitor | _tighten_stop failed for "
                f"{pos.symbol} id={pos.position_id}: {exc}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # DB READS
    # ──────────────────────────────────────────────────────────────────────

    def _get_open_positions(self) -> List[OpenPosition]:
        """
        Fetch all open positions from the trades table.

        Returns list of OpenPosition objects with current ATR attached.
        """
        positions: List[OpenPosition] = []
        try:
            from data_pipeline.storage.timeseries_db import db

            with db.connection() as conn:
                result = conn.execute(text("""
                    SELECT
                        t.id,
                        t.symbol,
                        t.direction,
                        t.entry_price,
                        t.stop_loss,
                        t.entry_time,
                        t.ctrader_position_id
                    FROM trades t
                    WHERE t.status = 'OPEN'
                    ORDER BY t.entry_time ASC
                """))
                rows = result.fetchall()

            # Filter out hedge legs — hedge_manager owns those positions
            try:
                import redis as _redis_lib
                _r = _redis_lib.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    password=os.getenv("REDIS_PASSWORD") or None,
                    decode_responses=True,
                )
                rows = [row for row in rows if not _r.get(f"hedge:active:{row[1]}")]
            except Exception:
                pass  # if Redis fails, proceed with all rows

            for row in rows:
                pos_id, symbol, direction, entry, sl, entry_time, ctrader_id = row

                # Get current ATR for this pair from statistical features
                atr = self._get_current_atr(symbol)

                positions.append(
                    OpenPosition(
                        position_id=int(pos_id),
                        symbol=str(symbol),
                        direction=str(direction),
                        entry_price=float(entry),
                        stop_loss=float(sl),
                        current_atr=atr,
                        entry_time=entry_time,
                        ctrader_position_id=str(ctrader_id) if ctrader_id else "",
                    )
                )

        except Exception as exc:
            logger.error(
                f"TechExitMonitor | _get_open_positions failed: {exc}"
            )

        return positions

    def _get_current_atr(self, symbol: str) -> float:
        """
        Read current ATR for the symbol from Redis statistical features.
        Falls back to a conservative default if not found.
        """
        DEFAULT_ATR = 0.0010  # 10 pips fallback
        try:
            r = self._get_redis_client()
            raw = r.get(f"features:statistical:{symbol}")
            if not raw:
                return DEFAULT_ATR
            data = json.loads(raw)
            return float(data.get("atr_14", DEFAULT_ATR))
        except Exception:
            return DEFAULT_ATR

    # ──────────────────────────────────────────────────────────────────────
    # DB WRITES
    # ──────────────────────────────────────────────────────────────────────

    def _update_trade_close_reason(
        self, position_id: int, reason: str
    ) -> None:
        """
        Update trade record with invalidation close reason and closed_at time.
        Leaves the actual status update to order_executor (avoids double-write).
        Uses 'close_reason' column — add if not in schema:
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS close_reason VARCHAR(80);
        """
        try:
            from data_pipeline.storage.timeseries_db import db

            with db.connection() as conn:
                # Map long reason string to short allowed status value
                status_map = {
                    "CLOSED_TECH_INVALIDATION_FLIP":  "CLOSED_TECH_FLIP",
                    "CLOSED_TECH_INVALIDATION_DECAY": "CLOSED_TECH_DECAY",
                }
                status_value = status_map.get(reason, "CLOSED_MANUAL")

                conn.execute(
                    text("""
                        UPDATE trades
                        SET
                            status       = :status_val,
                            exit_time    = :now,
                            close_reason = :reason
                        WHERE id = :pos_id
                          AND status = 'OPEN'
                    """),
                    {
                        "status_val": status_value,
                        "reason": reason,
                        "now": datetime.now(timezone.utc),
                        "pos_id": position_id,
                    },
                )
                conn.commit()

        except Exception as exc:
            logger.error(
                f"TechExitMonitor | DB update failed for id={position_id}: {exc}"
            )

    def _update_trade_stop_loss(
        self, position_id: int, new_sl: float, reason: str
    ) -> None:
        """Update stop_loss and add note to close_reason for tighten events."""
        try:
            from data_pipeline.storage.timeseries_db import db

            with db.connection() as conn:
                conn.execute(
                    text("""
                        UPDATE trades
                        SET
                            stop_loss    = :new_sl,
                            close_reason = :reason,
                            updated_at   = :now
                        WHERE id = :pos_id
                          AND status = 'OPEN'
                    """),
                    {
                        "new_sl": new_sl,
                        "reason": reason,
                        "now": datetime.now(timezone.utc),
                        "pos_id": position_id,
                    },
                )
                conn.commit()

        except Exception as exc:
            logger.error(
                f"TechExitMonitor | SL DB update failed for "
                f"id={position_id}: {exc}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # REDIS READS
    # ──────────────────────────────────────────────────────────────────────

    def _get_a4_signal(self, symbol: str) -> Optional[A4Signal]:
        """
        Read current TCN + PatchTSMixer inference from Redis.

        Actual Redis structure (confirmed 2026-03-25):
        {
            "pair":            "USDCAD",
            "direction":       "LONG" | "SHORT",   # shared direction
            "confidence":      0.0-1.0,             # combined confidence
            "tcn_prob":        0.0-1.0,             # TCN raw probability
            "patch_prob":      0.0-1.0,             # PatchTSMixer raw probability
            "agreed":          true | false,        # both models agree
            "regime":          "THV" | "TLV" | "MR" | "PNS",
            "gate":            "PASS" | "FAIL",
            "updated_at":      "ISO timestamp"
        }

        Note: LONG/SHORT mapped to BUY/SELL internally.
        When agreed=False, one model is treated as NEUTRAL.
        """
        try:
            r = self._get_redis_client()
            raw = r.get(f"features:analyst4:{symbol}")
            if not raw:
                return None

            data = json.loads(raw)

            # Map LONG/SHORT → BUY/SELL
            direction_map = {"LONG": "BUY", "SHORT": "SELL"}
            raw_dir = str(data.get("direction", "NEUTRAL")).upper()
            direction = direction_map.get(raw_dir, "NEUTRAL")

            tcn_prob   = float(data.get("tcn_prob", 0.0))
            patch_prob = float(data.get("patch_prob", 0.0))
            agreed     = bool(data.get("agreed", False))

            # If models didn't agree, treat patch as NEUTRAL
            # (TCN is the stronger model at AUC 0.737)
            patch_direction = direction if agreed else "NEUTRAL"

            return A4Signal(
                tcn_direction=direction,
                tcn_confidence=tcn_prob,
                patch_direction=patch_direction,
                patch_confidence=patch_prob if agreed else 0.0,
            )

        except Exception as exc:
            logger.warning(
                f"TechExitMonitor | Failed to read A4 signal for "
                f"{symbol}: {exc}"
            )
            return None

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _get_redis_client(self) -> redis.Redis:
        """Lazy Redis connection with reuse."""
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    def _cleanup_history(self, symbol: str, position_id: int) -> None:
        """Delete confidence history after position closed."""
        try:
            r = self._get_redis_client()
            r.delete(f"tech:exit:history:{symbol}:{position_id}")
        except Exception:
            pass  # Non-critical cleanup

    @staticmethod
    def _get_pair_currencies(symbol: str) -> str:
        """
        Return comma-separated currency codes for a forex pair.

        Example: "AUDJPY" → "AUD,JPY"
        """
        # Standard 6-char forex symbols
        if len(symbol) == 6:
            return f"{symbol[:3]},{symbol[3:]}"
        return symbol  # Fallback for unexpected format

    # ──────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC / MANUAL OVERRIDE
    # ──────────────────────────────────────────────────────────────────────

    def get_position_status(self, symbol: str) -> Dict[str, Any]:
        """
        Human-readable status of current A4 signals vs any open position.
        Useful for /signals Telegram command and debug sessions.

        Returns:
            Dict with a4_signal, trade direction, invalidation risk level
        """
        result: Dict[str, Any] = {"symbol": symbol}

        try:
            a4 = self._get_a4_signal(symbol)
            if a4:
                result["tcn"] = f"{a4.tcn_direction} ({a4.tcn_confidence:.2f})"
                result["patch"] = f"{a4.patch_direction} ({a4.patch_confidence:.2f})"
                result["combined_conf"] = round(a4.combined_confidence, 3)
                result["models_agree"] = a4.both_agree
            else:
                result["a4"] = "NOT IN REDIS"

            # Find open position for this symbol
            positions = self._get_open_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    result["trade_direction"] = pos.direction
                    result["entry"] = pos.entry_price
                    result["stop_loss"] = pos.stop_loss

                    if a4:
                        tcn_opp, patch_opp = a4.opposes(pos.direction)
                        if tcn_opp and patch_opp:
                            result["risk"] = "🔴 FULL_FLIP — would close NOW"
                        elif tcn_opp or patch_opp or \
                                a4.combined_confidence < CONF_CLOSE_THRESHOLD:
                            result["risk"] = "🟠 MIXED_WEAK — would close/tighten"
                        elif a4.combined_confidence < CONF_TIGHTEN_THRESHOLD:
                            result["risk"] = "🟡 EARLY_DECAY — would tighten"
                        else:
                            result["risk"] = "🟢 HOLD — thesis intact"
                    break
            else:
                result["trade"] = "No open position"

        except Exception as exc:
            result["error"] = str(exc)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

tech_exit_monitor = TechnicalExitMonitor()
