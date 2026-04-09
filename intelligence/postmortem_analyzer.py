"""
intelligence/postmortem_analyzer.py
====================================
Automated loss classification engine for the G10 Autonomous Dominance Architecture.

Every CLOSED_SL trade is sent to Groq LLaMA with full trade context (regime,
analyst signals, session timing, nearby economic events, post-SL price action)
and classified into one of 7 failure categories. Results are stored in
trade_postmortems and surfaced via weekly Telegram summary.

Failure categories:
    BAD_TIMING      — Right direction, wrong session/time (thin liquidity, rollover)
    REGIME_WRONG    — A1 misidentified regime; price behaved opposite to expectation
    NEWS_SURPRISE   — High-impact event near entry reversed price unexpectedly
    MODEL_FAILURE   — All signals agreed but direction was simply wrong
    OVERTRADED_PAIR — Same pair repeatedly hitting SL (frequency/threshold issue)
    SPREAD_COST     — Loss within 1-2× spread; should have been filtered at gate
    ENTRY_LEVEL     — Direction correct (price recovered ≥1R after SL), but entry
                      price was wrong (near S/R, chased, or SL too tight for ATR)

Usage:
    # In main_loop.py after trade closes:
    from intelligence.postmortem_analyzer import postmortem_analyzer
    postmortem_analyzer.analyze_trade(trade_id)

    # Backfill all existing unanalyzed CLOSED_SL trades:
    postmortem_analyzer.analyze_all_unanalyzed()

    # Weekly Telegram summary:
    text = postmortem_analyzer.weekly_summary()

Redis keys: None (all persistence goes to TimescaleDB)
DB tables:  trades (read), trade_postmortems (write), economic_events (read)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text

load_dotenv("/workspaces/Trading_bot/.env")

# ── constants ────────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

FAILURE_TAGS = [
    "BAD_TIMING",
    "REGIME_WRONG",
    "NEWS_SURPRISE",
    "MODEL_FAILURE",
    "OVERTRADED_PAIR",
    "SPREAD_COST",
    "ENTRY_LEVEL",
]

# UTC hour ranges for forex sessions
SESSIONS = {
    "SYDNEY":  (21, 6),   # 21:00–06:00 UTC  (wraps midnight)
    "TOKYO":   (0,  9),   # 00:00–09:00 UTC
    "LONDON":  (7,  16),  # 07:00–16:00 UTC
    "NEW_YORK":(12, 21),  # 12:00–21:00 UTC
    "ROLLOVER":(21, 22),  # 21:00–22:00 UTC — always BAD_TIMING candidate
}

SYSTEM_PROMPT = """You are a senior FX trading risk analyst reviewing a losing trade.
Your job is to classify the PRIMARY reason for the stop-loss hit into exactly one tag.

Available tags and their definitions:
  BAD_TIMING      — The direction was correct but entry timing was poor (wrong session,
                    thin liquidity window, rollover hour, Sunday open, Monday pre-London).
  REGIME_WRONG    — The regime classifier (A1 LightGBM) assigned the wrong regime.
                    Price behaved as a different regime would predict.
  NEWS_SURPRISE   — A high-impact scheduled or unscheduled event near entry time
                    caused a sharp reversal that no model could have predicted.
  MODEL_FAILURE   — All analyst signals agreed but the directional call was simply wrong.
                    No timing, regime, or news excuse — the models were collectively wrong.
                    Use this ONLY when post-SL price action confirms price did NOT recover
                    in the predicted direction.
  OVERTRADED_PAIR — This pair has been hit for SL multiple times recently. The issue is
                    frequency or confidence threshold, not a single-trade model error.
  SPREAD_COST     — The loss in pips was within 1–2× typical spread for this pair.
                    The trade should have been filtered by a tighter entry gate.
  ENTRY_LEVEL     — The directional prediction was correct: post-SL price action shows
                    price moved ≥1R in the predicted direction within 4 hours of SL hit.
                    The loss was caused by a bad entry price — entered too close to
                    resistance (for LONG) or support (for SHORT), chased price too far
                    from the ideal entry zone, or the SL was too tight for the pair's
                    normal ATR-based volatility. This tag covers two sub-causes you
                    should explain in reasoning:
                      CHASE: entry_position_in_bar > 0.7 — bot entered after most of
                             the M15 bar had already moved; should have waited for
                             pullback.
                      LEVEL: entry_position_in_bar ≤ 0.7 but price recovered — entry
                             was at a structurally bad level (near S/R, poor R:R zone).

Key decision rules:
  - entry_position_in_bar > 0.7 AND did_price_recover=true → ENTRY_LEVEL (CHASE sub-cause)
  - entry_position_in_bar > 0.7 AND did_price_recover=false → MODEL_FAILURE (chased and wrong)
  - entry_position_in_bar ≤ 0.7 AND did_price_recover=true → ENTRY_LEVEL (LEVEL sub-cause)
  - entry_position_in_bar ≤ 0.7 AND did_price_recover=false → MODEL_FAILURE or REGIME_WRONG
  - BAD_TIMING overrides the above if entry was in rollover/thin liquidity session.
  - NEWS_SURPRISE overrides if a HIGH-impact event is within ±2h of entry.

Rules:
  - Output ONLY a JSON object. No preamble. No explanation outside the JSON.
  - primary_tag must be exactly one of the 7 tags above.
  - secondary_tag is optional — null if not applicable.
  - confidence is 0.0–1.0 reflecting how certain you are of the primary tag.
  - reasoning is 1–2 sentences maximum.
  - lessons is a list of 1–3 short actionable strings.

Output format (strict JSON, no markdown fences):
{
  "primary_tag": "ENTRY_LEVEL",
  "secondary_tag": "OVERTRADED_PAIR",
  "confidence": 0.87,
  "reasoning": "Price recovered +18 pips in BUY direction within 2 hours of SL hit, confirming directional call was correct. Entry was 8 pips from nearest resistance — SL was clipped by noise before trend resumed.",
  "lessons": ["Block LONG entries within 15 pips of D1 resistance", "Widen SL by 0.5× ATR on AUDJPY to survive retest noise"]
}"""


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_session(entry_time: datetime) -> str:
    """Return the primary forex session active at entry_time (UTC)."""
    h = entry_time.hour
    # Rollover check first (highest priority bad-timing signal)
    if 21 <= h < 22:
        return "ROLLOVER"
    if 7 <= h < 16:
        return "LONDON"
    if 12 <= h < 21:
        return "NEW_YORK"
    if 0 <= h < 9:
        return "TOKYO"
    return "SYDNEY"


def _pips_for_pair(symbol: str, price_diff: float) -> float:
    """Convert raw price difference to pips for any G10 pair."""
    jpy_pairs = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}
    if symbol.upper() in jpy_pairs:
        return round(price_diff * 100, 1)
    return round(price_diff * 10000, 1)


# ── main class ───────────────────────────────────────────────────────────────

class PostmortemAnalyzer:
    """
    Classifies every CLOSED_SL trade using Groq LLaMA.

    Deduplication: skips trades already present in trade_postmortems.
    Rate limiting: 1-second sleep between Groq calls to stay within free tier.
    """

    def __init__(self) -> None:
        self._groq_key: str = os.getenv("GROQ_API_KEY", "")
        if not self._groq_key:
            logger.warning("PostmortemAnalyzer | GROQ_API_KEY not set — Groq calls will fail")

    # ── public API ───────────────────────────────────────────────────────────

    # Minimum time to wait after trade close before analyzing (so post-SL bars exist)
    PENDING_HOURS = 4

    def analyze_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Analyze a single trade by ID.

        Skips if already analyzed or if < PENDING_HOURS have elapsed since
        exit_time (post-SL price action data would be incomplete). Deferred
        trades are picked up by the hourly _postmortem_pickup_thread.

        Args:
            trade_id: Primary key from trades table.

        Returns:
            Dict with classification result, or None on failure/skip/pending.
        """
        logger.info(f"PostmortemAnalyzer | Analyzing trade_id={trade_id}")

        if self._already_analyzed(trade_id):
            logger.debug(f"PostmortemAnalyzer | trade_id={trade_id} already analyzed — skip")
            return None

        trade = self._fetch_trade(trade_id)
        if not trade:
            logger.error(f"PostmortemAnalyzer | trade_id={trade_id} not found in DB")
            return None

        if trade["status"] not in ("CLOSED_SL", "CLOSED_TECH_FLIP", "CLOSED_MANUAL"):
            logger.debug(f"PostmortemAnalyzer | trade_id={trade_id} status={trade['status']} — skip")
            return None

        # Guard: need PENDING_HOURS of post-SL M15 bars for ENTRY_LEVEL detection
        exit_time = trade.get("exit_time")
        if exit_time:
            if exit_time.tzinfo is None:
                exit_time = exit_time.replace(tzinfo=timezone.utc)
            ready_at = exit_time + timedelta(hours=self.PENDING_HOURS)
            if datetime.now(timezone.utc) < ready_at:
                logger.info(
                    f"PostmortemAnalyzer | trade_id={trade_id} pending — "
                    f"not yet {self.PENDING_HOURS}h post-exit "
                    f"(ready at {ready_at.strftime('%H:%M UTC')})"
                )
                return None

        # Hedge legs are not directional bets — skip Groq, auto-tag instead
        if trade.get("signal_path") == "HEDGE_CYCLE":
            result = {
                "primary_tag":   "HEDGE_LEG_SL",
                "secondary_tag": None,
                "confidence":    1.0,
                "lessons":       [
                    "Hedge leg SL hit — price range exceeded ATR×2 before cycling back.",
                    "Not a directional model failure. Review ATR_STOP_MULT or pair volatility.",
                ],
            }
            self._write_postmortem(trade_id, trade, "auto-tagged:HEDGE_CYCLE", result)
            logger.info(
                f"PostmortemAnalyzer | trade_id={trade_id} {trade['symbol']} "
                f"→ HEDGE_LEG_SL (auto-tagged, no Groq)"
            )
            return result

        context = self._build_context(trade)
        result  = self._call_groq(context)

        if result:
            self._write_postmortem(trade_id, trade, context, result)
            logger.success(
                f"PostmortemAnalyzer | trade_id={trade_id} {trade['symbol']} "
                f"→ {result['primary_tag']} (conf={result['confidence']:.2f})"
            )

        return result

    def analyze_all_unanalyzed(self, status_filter: str = "CLOSED_SL") -> int:
        """
        Backfill all trades not yet in trade_postmortems.

        Args:
            status_filter: Which status to analyze. Default CLOSED_SL.
                           Pass 'ALL' to include CLOSED_TECH_FLIP + CLOSED_MANUAL.

        Returns:
            Number of trades successfully analyzed.
        """
        logger.info(f"PostmortemAnalyzer | Starting batch analysis (filter={status_filter})")

        trade_ids = self._fetch_unanalyzed_ids(status_filter)
        logger.info(f"PostmortemAnalyzer | Found {len(trade_ids)} unanalyzed trades")

        success = 0
        for i, trade_id in enumerate(trade_ids, 1):
            try:
                result = self.analyze_trade(trade_id)
                if result:
                    success += 1
                # Groq free tier: 30 req/min → 1 req/2s to be safe
                time.sleep(5)
            except Exception as exc:
                logger.error(f"PostmortemAnalyzer | trade_id={trade_id} failed: {exc}")
                continue

            if i % 10 == 0:
                logger.info(f"PostmortemAnalyzer | Progress: {i}/{len(trade_ids)} ({success} ok)")

        logger.success(f"PostmortemAnalyzer | Batch complete: {success}/{len(trade_ids)} analyzed")
        return success

    def reanalyze_all(self, status_filter: str = "CLOSED_SL") -> int:
        """
        Delete all existing postmortem records and rerun classification from scratch.

        Use this when the tag set or system prompt has changed and you want fresh
        classifications for all trades.

        Args:
            status_filter: Which trade status to reanalyze. Default CLOSED_SL.
                           Pass 'ALL' to include CLOSED_TECH_FLIP + CLOSED_MANUAL.

        Returns:
            Number of trades successfully analyzed.
        """
        logger.warning("PostmortemAnalyzer | reanalyze_all — deleting all existing postmortems")
        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                result = conn.execute(text("DELETE FROM trade_postmortems"))
                conn.commit()
                deleted = result.rowcount
            logger.info(f"PostmortemAnalyzer | Deleted {deleted} existing postmortem records")
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | reanalyze_all delete failed: {exc}")
            return 0

        return self.analyze_all_unanalyzed(status_filter=status_filter)

    def reanalyze_premature(self) -> int:
        """
        Find postmortems that were written within PENDING_HOURS of trade exit
        (before post-SL price action bars were available), delete them, then
        re-analyze with complete data.

        Run this once after deploying the 4h pending guard to fix existing
        misclassifications caused by immediate analysis on trade close.

        Returns:
            Number of trades successfully re-analyzed.
        """
        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                rows = conn.execute(text("""
                    SELECT pm.trade_id
                    FROM trade_postmortems pm
                    JOIN trades t ON t.id = pm.trade_id
                    WHERE pm.analyzed_at < t.exit_time + INTERVAL '4 hours'
                      AND t.exit_time IS NOT NULL
                      AND t.exit_time < NOW() - INTERVAL '4 hours'
                """)).fetchall()
                trade_ids = [r[0] for r in rows]

                if not trade_ids:
                    logger.info("PostmortemAnalyzer | reanalyze_premature: no premature records found")
                    return 0

                logger.warning(
                    f"PostmortemAnalyzer | reanalyze_premature: deleting {len(trade_ids)} "
                    f"premature postmortems — {trade_ids}"
                )
                for tid in trade_ids:
                    conn.execute(text(
                        "DELETE FROM trade_postmortems WHERE trade_id = :tid"
                    ), {"tid": tid})
                conn.commit()

        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | reanalyze_premature error: {exc}")
            return 0

        logger.info(f"PostmortemAnalyzer | reanalyze_premature: re-analyzing {len(trade_ids)} trades")
        success = 0
        for tid in trade_ids:
            try:
                result = self.analyze_trade(tid)
                if result:
                    success += 1
                time.sleep(5)
            except Exception as exc:
                logger.error(f"PostmortemAnalyzer | reanalyze_premature trade_id={tid}: {exc}")

        logger.success(f"PostmortemAnalyzer | reanalyze_premature: {success}/{len(trade_ids)} done")
        return success

    def weekly_summary(self, days_back: int = 7) -> str:
        """
        Generate a Telegram-ready text summary of loss patterns.

        Args:
            days_back: How many days of postmortems to include (default 7).

        Returns:
            Formatted string ready to send via Telegram.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days_back)

        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                rows = conn.execute(text("""
                    SELECT
                        pm.failure_reason,
                        pm.analyst_conflict,
                        pm.lessons,
                        pm.analyzed_at,
                        t.symbol,
                        t.direction,
                        t.pnl_pips,
                        t.pnl_usd,
                        t.regime,
                        t.entry_time
                    FROM trade_postmortems pm
                    JOIN trades t ON t.id = pm.trade_id
                    WHERE pm.analyzed_at >= :since
                    ORDER BY pm.analyzed_at DESC
                """), {"since": since}).fetchall()
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | weekly_summary DB error: {exc}")
            return "⚠️ Postmortem summary failed — DB error."

        if not rows:
            return f"📊 *Postmortem Summary* (last {days_back}d)\nNo analyzed trades yet."

        # ── aggregate by tag ─────────────────────────────────────────────────
        tag_counts: Dict[str, int] = {}
        tag_pnl:    Dict[str, float] = {}
        pair_counts: Dict[str, int] = {}
        all_lessons: List[str] = []

        for row in rows:
            tag = row.failure_reason or "UNKNOWN"
            tag_counts[tag]  = tag_counts.get(tag, 0) + 1
            tag_pnl[tag]     = tag_pnl.get(tag, 0.0) + (row.pnl_usd or 0.0)
            pair_counts[row.symbol] = pair_counts.get(row.symbol, 0) + 1

            if row.lessons:
                for lesson in row.lessons:
                    if lesson not in all_lessons:
                        all_lessons.append(lesson)

        total_trades = len(rows)
        total_usd    = sum(tag_pnl.values())

        # ── format (HTML — matches Telegram parse_mode=HTML) ─────────────────
        lines = [
            f"📊 <b>Postmortem Report — Last {days_back} Days</b>",
            f"Analyzed: {total_trades} losses | Total PnL: ${total_usd:+.2f}",
            "",
            "🏷 <b>Failure Tag Breakdown:</b>",
        ]

        for tag in sorted(tag_counts, key=tag_counts.get, reverse=True):
            count = tag_counts[tag]
            pnl   = tag_pnl[tag]
            pct   = count / total_trades * 100
            lines.append(f"  <code>{tag:<20}</code> {count:>3}× ({pct:.0f}%)  ${pnl:+.2f}")

        lines.append("")
        lines.append("📌 <b>Most Lost Pairs:</b>")
        for pair, cnt in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"  {pair}  {cnt}× losses")

        if all_lessons:
            lines.append("")
            lines.append("💡 <b>Top Lessons:</b>")
            for lesson in all_lessons[:5]:
                lines.append(f"  • {lesson}")

        return "\n".join(lines)

    def get_pair_pattern(self, symbol: str, days_back: int = 14) -> Dict[str, Any]:
        """
        Return failure pattern breakdown for a specific pair.

        Useful for deciding whether to raise thresholds or blacklist a pair.

        Args:
            symbol: e.g. 'AUDJPY'
            days_back: lookback window

        Returns:
            Dict with tag_counts, win_rate, total_pnl, top_lesson
        """
        since = datetime.now(timezone.utc) - timedelta(days=days_back)

        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                rows = conn.execute(text("""
                    SELECT pm.failure_reason, pm.lessons, t.pnl_usd
                    FROM trade_postmortems pm
                    JOIN trades t ON t.id = pm.trade_id
                    WHERE t.symbol = :sym AND pm.analyzed_at >= :since
                """), {"sym": symbol, "since": since}).fetchall()
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | get_pair_pattern error: {exc}")
            return {}

        if not rows:
            return {"symbol": symbol, "analyzed_trades": 0}

        tag_counts: Dict[str, int] = {}
        total_pnl  = 0.0
        all_lessons: List[str] = []

        for row in rows:
            tag = row.failure_reason or "UNKNOWN"
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            total_pnl += row.pnl_usd or 0.0
            if row.lessons:
                all_lessons.extend(row.lessons)

        top_tag = max(tag_counts, key=tag_counts.get) if tag_counts else "UNKNOWN"

        return {
            "symbol":          symbol,
            "analyzed_trades": len(rows),
            "tag_counts":      tag_counts,
            "dominant_tag":    top_tag,
            "total_pnl_usd":   round(total_pnl, 2),
            "top_lesson":      all_lessons[0] if all_lessons else None,
        }

    # ── private: data fetching ────────────────────────────────────────────────

    def _fetch_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Load a single trade row from DB."""
        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                row = conn.execute(text("""
                    SELECT id, symbol, direction, entry_price, exit_price,
                           stop_loss, take_profit, entry_time, exit_time,
                           status, pnl_pips, pnl_usd, regime, close_reason,
                           signal_1_regime, signal_1_confidence,
                           signal_2_fund_score, signal_4_tsmixer,
                           signal_5_sentiment, signal_6_bias, signal_path
                    FROM trades WHERE id = :tid
                """), {"tid": trade_id}).fetchone()
            if not row:
                return None
            return dict(row._mapping)
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | _fetch_trade error: {exc}")
            return None

    def _fetch_unanalyzed_ids(self, status_filter: str) -> List[int]:
        """Return trade IDs not yet in trade_postmortems."""
        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                if status_filter == "ALL":
                    status_clause = "status IN ('CLOSED_SL','CLOSED_TECH_FLIP','CLOSED_MANUAL')"
                    params: Dict[str, Any] = {}
                else:
                    status_clause = "status = :status"
                    params = {"status": status_filter}

                rows = conn.execute(text(f"""
                    SELECT id FROM trades
                    WHERE {status_clause}
                      AND id NOT IN (SELECT trade_id FROM trade_postmortems)
                      AND exit_time < NOW() - INTERVAL '{self.PENDING_HOURS} hours'
                    ORDER BY entry_time ASC
                """), params).fetchall()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | _fetch_unanalyzed_ids error: {exc}")
            return []

    def _already_analyzed(self, trade_id: int) -> bool:
        """Return True if this trade already has a postmortem record."""
        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                row = conn.execute(text(
                    "SELECT 1 FROM trade_postmortems WHERE trade_id = :tid"
                ), {"tid": trade_id}).fetchone()
            return row is not None
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | _already_analyzed error: {exc}")
            return False

    def _fetch_nearby_events(
        self, symbol: str, entry_time: datetime, window_hours: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Return HIGH-impact events within ±window_hours of entry_time.

        Extracts relevant currencies from the pair symbol.
        """
        base = symbol[:3].upper()
        quote = symbol[3:].upper()
        t_from = entry_time - timedelta(hours=window_hours)
        t_to   = entry_time + timedelta(hours=window_hours)

        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                rows = conn.execute(text("""
                    SELECT event_time, country, event_name, impact, actual, forecast
                    FROM economic_events
                    WHERE event_time BETWEEN :t_from AND :t_to
                      AND impact = 'HIGH'
                      AND country IN (:base, :quote)
                    ORDER BY event_time ASC
                """), {"t_from": t_from, "t_to": t_to,
                       "base": base, "quote": quote}).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | _fetch_nearby_events error: {exc}")
            return []

    def _fetch_post_sl_price_action(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check what price did in the 4 hours after the trade was closed.

        Fetches M15 bars after exit_time and checks whether price reached the
        1R target in the predicted direction. This is the key signal for
        distinguishing ENTRY_LEVEL (models right, entry wrong) from
        MODEL_FAILURE (direction was simply wrong).

        Returns:
            Dict with:
              did_price_recover  — True if price moved ≥1R in signal direction
              max_favorable_pips — Largest favorable move after SL (in pips)
              bars_checked       — Number of M15 bars available
        """
        empty = {"did_price_recover": False, "max_favorable_pips": 0.0, "bars_checked": 0}

        exit_time = trade.get("exit_time")
        if not exit_time:
            return empty

        if exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)

        symbol     = trade["symbol"]
        direction  = trade["direction"]    # "BUY" or "SELL"
        entry_price = float(trade["entry_price"])
        stop_loss   = float(trade["stop_loss"])
        sl_distance = abs(entry_price - stop_loss)

        if sl_distance == 0:
            return empty

        target_1r = (
            entry_price + sl_distance if direction == "BUY"
            else entry_price - sl_distance
        )
        end_time = exit_time + timedelta(hours=4)

        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                rows = conn.execute(text("""
                    SELECT time, high, low
                    FROM ohlcv
                    WHERE symbol    = :sym
                      AND timeframe = 'M15'
                      AND time      >  :exit_time
                      AND time      <= :end_time
                    ORDER BY time ASC
                """), {
                    "sym":       symbol,
                    "exit_time": exit_time,
                    "end_time":  end_time,
                }).fetchall()
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | _fetch_post_sl_price_action error: {exc}")
            return empty

        if not rows:
            return empty

        # ohlcv stores JPY prices ×100 — normalise to actual price before pip calc
        is_jpy = symbol.upper().endswith("JPY")

        did_recover    = False
        max_fav_pips   = 0.0

        for row in rows:
            high = float(row.high)
            low  = float(row.low)

            if is_jpy and high > 1000:
                high /= 100.0
                low  /= 100.0

            if direction == "BUY":
                if high >= target_1r:
                    did_recover = True
                favorable_move = max(high - entry_price, 0.0)
            else:
                if low <= target_1r:
                    did_recover = True
                favorable_move = max(entry_price - low, 0.0)

            fav_pips = _pips_for_pair(symbol, favorable_move)
            if fav_pips > max_fav_pips:
                max_fav_pips = fav_pips

        return {
            "did_price_recover":  did_recover,
            "max_favorable_pips": round(max_fav_pips, 1),
            "bars_checked":       len(rows),
        }

    def _fetch_entry_bar(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch the M15 bar that contained the entry and compute chase metrics.

        Key metric: entry_position_in_bar
          BUY  → 1.0 = entry at bar top    (chased the move up)
          SELL → 1.0 = entry at bar bottom  (chased the move down)
          < 0.3 = entered early / on a pullback (good)
          > 0.7 = likely chased price

        Returns:
            bar_open, bar_high, bar_low, bar_close (actual price),
            bar_range_pips, entry_position_in_bar (0.0–1.0),
            pips_moved_before_entry (bar already moved this far in entry direction)
        """
        empty = {
            "bar_open": None, "bar_high": None, "bar_low": None, "bar_close": None,
            "bar_range_pips": None, "entry_position_in_bar": None,
            "pips_moved_before_entry": None,
        }

        entry_time = trade.get("entry_time")
        if not entry_time:
            return empty
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        symbol      = trade["symbol"]
        direction   = trade["direction"]
        entry_price = float(trade["entry_price"])
        is_jpy      = symbol.upper().endswith("JPY")

        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                row = conn.execute(text("""
                    SELECT time, open, high, low, close
                    FROM ohlcv
                    WHERE symbol    = :sym
                      AND timeframe = 'M15'
                      AND time      <= :entry_time
                    ORDER BY time DESC
                    LIMIT 1
                """), {"sym": symbol, "entry_time": entry_time}).fetchone()
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | _fetch_entry_bar error: {exc}")
            return empty

        if not row:
            return empty

        bar_open  = float(row.open)
        bar_high  = float(row.high)
        bar_low   = float(row.low)
        bar_close = float(row.close)

        # Normalise JPY ×100 storage to actual price
        if is_jpy and bar_high > 1000:
            bar_open  /= 100.0
            bar_high  /= 100.0
            bar_low   /= 100.0
            bar_close /= 100.0

        bar_range = bar_high - bar_low
        if bar_range == 0:
            return empty

        bar_range_pips = _pips_for_pair(symbol, bar_range)

        # How far into the bar (in the entry direction) was the entry?
        # 1.0 = entered at the absolute extreme = fully chased
        if direction == "BUY":
            entry_position = (entry_price - bar_low) / bar_range
            # How many pips had price already moved up from bar open before entry
            pips_moved = _pips_for_pair(symbol, max(entry_price - bar_open, 0.0))
        else:
            entry_position = (bar_high - entry_price) / bar_range
            pips_moved = _pips_for_pair(symbol, max(bar_open - entry_price, 0.0))

        return {
            "bar_open":               round(bar_open, 5),
            "bar_high":               round(bar_high, 5),
            "bar_low":                round(bar_low, 5),
            "bar_close":              round(bar_close, 5),
            "bar_range_pips":         round(bar_range_pips, 1),
            "entry_position_in_bar":  round(min(max(entry_position, 0.0), 1.0), 2),
            "pips_moved_before_entry": round(pips_moved, 1),
        }

    def _count_recent_sl_same_pair(
        self, symbol: str, before_time: datetime, lookback_hours: int = 48
    ) -> int:
        """Count how many CLOSED_SL this pair had in the last lookback_hours."""
        since = before_time - timedelta(hours=lookback_hours)
        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                row = conn.execute(text("""
                    SELECT COUNT(*) FROM trades
                    WHERE symbol = :sym
                      AND status = 'CLOSED_SL'
                      AND entry_time BETWEEN :since AND :before
                """), {"sym": symbol, "since": since,
                       "before": before_time}).fetchone()
            return row[0] if row else 0
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | _count_recent_sl_same_pair error: {exc}")
            return 0

    # ── private: context building ─────────────────────────────────────────────

    def _build_context(self, trade: Dict[str, Any]) -> str:
        """
        Assemble a structured natural-language prompt context for Groq.

        Includes: trade basics, signal values at entry, session, nearby events,
        pair frequency signal.
        """
        entry_time: datetime = trade["entry_time"]
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        session         = _get_session(entry_time)
        nearby_events   = self._fetch_nearby_events(trade["symbol"], entry_time)
        recent_sl_count = self._count_recent_sl_same_pair(
            trade["symbol"], entry_time, lookback_hours=48
        )
        post_sl         = self._fetch_post_sl_price_action(trade)
        entry_bar       = self._fetch_entry_bar(trade)

        # Raw pips loss (entry → exit or SL)
        exit_price = trade.get("exit_price") or trade["stop_loss"]
        raw_diff   = abs(trade["entry_price"] - exit_price)
        pips_lost  = _pips_for_pair(trade["symbol"], raw_diff)

        events_text = "None"
        if nearby_events:
            parts = []
            for ev in nearby_events[:3]:
                delta_min = int(
                    (ev["event_time"] - entry_time).total_seconds() / 60
                )
                sign = "+" if delta_min >= 0 else ""
                parts.append(
                    f"{ev['country']} {ev['event_name']} "
                    f"({sign}{delta_min}min from entry) "
                    f"actual={ev.get('actual','?')} forecast={ev.get('forecast','?')}"
                )
            events_text = " | ".join(parts)

        context = f"""
TRADE DETAILS:
  Trade ID:      {trade['id']}
  Pair:          {trade['symbol']}
  Direction:     {trade['direction']}
  Entry price:   {trade['entry_price']}
  Exit price:    {exit_price}
  Stop loss:     {trade['stop_loss']}
  Pips lost:     {pips_lost}
  USD PnL:       {trade.get('pnl_usd', 'N/A')}
  Status:        {trade['status']}
  Close reason:  {trade.get('close_reason', 'N/A')}

TIMING:
  Entry time (UTC): {entry_time.strftime('%Y-%m-%d %H:%M')}
  Session at entry: {session}
  Day of week:      {entry_time.strftime('%A')}

ANALYST SIGNALS AT ENTRY:
  A1 Regime:         {trade.get('signal_1_regime') or trade.get('regime', 'N/A')}
  A1 Confidence:     {trade.get('signal_1_confidence', 'N/A')}
  A2 Fund score:     {trade.get('signal_2_fund_score', 'N/A')}  (range -1 to +1)
  A4 TSMixer prob:   {trade.get('signal_4_tsmixer', 'N/A')}
  A5 Sentiment:      {trade.get('signal_5_sentiment', 'N/A')}
  A6 Opus bias:      {trade.get('signal_6_bias', 'N/A')}

MARKET CONTEXT:
  HIGH-impact events within ±4h: {events_text}
  Same pair CLOSED_SL in prior 48h: {recent_sl_count}

ENTRY BAR ANALYSIS (M15 bar that contained the entry):
  bar_open:               {entry_bar['bar_open']}
  bar_high:               {entry_bar['bar_high']}
  bar_low:                {entry_bar['bar_low']}
  bar_close:              {entry_bar['bar_close']}
  bar_range_pips:         {entry_bar['bar_range_pips']} pips total range
  entry_position_in_bar:  {entry_bar['entry_position_in_bar']}  (0.0=bar extreme opposite direction, 1.0=bar extreme same direction = fully chased)
  pips_moved_before_entry:{entry_bar['pips_moved_before_entry']} pips already moved in {trade['direction']} direction from bar open
  note: entry_position_in_bar > 0.7 strongly suggests ENTRY_CHASE (bot entered after most of the move)

POST-SL PRICE ACTION (4h after close — key for ENTRY_LEVEL vs MODEL_FAILURE):
  did_price_recover:  {post_sl['did_price_recover']}
  max_favorable_pips: {post_sl['max_favorable_pips']} pips in {trade['direction']} direction
  bars_checked:       {post_sl['bars_checked']} M15 bars
  note: did_price_recover=true means price reached entry+1R in signal direction after SL hit

CLASSIFICATION TASK:
Classify the PRIMARY reason this trade hit its stop loss.
Return strict JSON as specified in your instructions.
""".strip()

        return context

    # ── private: Groq call ────────────────────────────────────────────────────

    def _call_groq(self, context: str) -> Optional[Dict[str, Any]]:
        """
        Send context to Groq LLaMA and parse JSON classification response.

        Returns:
            Parsed dict with primary_tag, secondary_tag, confidence,
            reasoning, lessons — or None on failure.
        """
        if not self._groq_key:
            logger.error("PostmortemAnalyzer | No GROQ_API_KEY — cannot call Groq")
            return None

        headers = {
            "Authorization": f"Bearer {self._groq_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "temperature": 0.1,   # low temp → deterministic classification
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
        }

        try:
            resp = requests.post(
                GROQ_API_URL, headers=headers,
                json=payload, timeout=30
            )
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if model adds them
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            parsed = json.loads(raw_text)

            # Validate primary_tag
            if parsed.get("primary_tag") not in FAILURE_TAGS:
                logger.warning(
                    f"PostmortemAnalyzer | Unknown tag '{parsed.get('primary_tag')}' "
                    f"— defaulting to MODEL_FAILURE"
                )
                parsed["primary_tag"] = "MODEL_FAILURE"

            return parsed

        except requests.exceptions.RequestException as exc:
            logger.error(f"PostmortemAnalyzer | Groq HTTP error: {exc}")
            return None
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(f"PostmortemAnalyzer | Groq parse error: {exc} | raw={raw_text[:200]}")
            return None

    # ── private: DB write ─────────────────────────────────────────────────────

    def _write_postmortem(
        self,
        trade_id: int,
        trade: Dict[str, Any],
        context: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Insert postmortem record into trade_postmortems.

        Skips silently if trade_id already exists (dedup).
        """
        analyst_conflict = {
            "a1_regime":     trade.get("signal_1_regime"),
            "a1_confidence": trade.get("signal_1_confidence"),
            "a2_fund_score": trade.get("signal_2_fund_score"),
            "a4_tsmixer":    trade.get("signal_4_tsmixer"),
            "a5_sentiment":  trade.get("signal_5_sentiment"),
            "a6_bias":       trade.get("signal_6_bias"),
            "secondary_tag": result.get("secondary_tag"),
            "groq_confidence": result.get("confidence"),
        }

        lessons  = result.get("lessons") or []
        raw_json = json.dumps(result)

        try:
            from data_pipeline.storage.timeseries_db import db
            with db.connection() as conn:
                conn.execute(text("""
                    INSERT INTO trade_postmortems
                        (trade_id, analyzed_at, failure_reason, market_context,
                         analyst_conflict, lessons, penalty_applied,
                         groq_raw_output, model_used)
                    VALUES
                        (:trade_id, NOW(), :failure_reason, :market_context,
                         CAST(:analyst_conflict AS jsonb), CAST(:lessons AS text[]),
                         '{}'::jsonb, :groq_raw_output, :model_used)
                    ON CONFLICT DO NOTHING
                """), {
                    "trade_id":        trade_id,
                    "failure_reason":  result["primary_tag"],
                    "market_context":  result.get("reasoning", ""),
                    "analyst_conflict": json.dumps(analyst_conflict),
                    "lessons":         "{" + ",".join(
                        f'"{l.replace(chr(34), chr(39))}"' for l in lessons
                    ) + "}",
                    "groq_raw_output": raw_json,
                    "model_used":      GROQ_MODEL,
                })
                conn.commit()
            logger.debug(f"PostmortemAnalyzer | Wrote postmortem for trade_id={trade_id}")
        except Exception as exc:
            logger.error(f"PostmortemAnalyzer | DB write failed trade_id={trade_id}: {exc}")


# ── module singleton ──────────────────────────────────────────────────────────

postmortem_analyzer = PostmortemAnalyzer()
