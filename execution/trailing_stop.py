"""
execution/trailing_stop.py
===========================
Trailing stop manager — moves SL as price moves in our favour.

Course rule: "Roll stop-loss to guarantee profit on winning trade"
This module implements that rule automatically in three stages.

Trailing logic (per position):
  Stage 0 — Initial stop (set at open, 1.5× ATR behind entry)
    No action until price moves 1× risk_distance in our favour.

  Stage 1 — Breakeven (price moves 1× risk_distance in favour)
    Move SL to entry + 0.5 pip buffer.
    Result: worst case = scratch trade, no loss.

  Stage 2 — Trailing (price moves 2× risk_distance in favour)
    SL trails 1× risk_distance behind current price.
    Updated every run if price moved higher (BUY) or lower (SELL).

  Stage 3 — Lock profits (price within 0.5× risk_distance of TP)
    SL moves to 2× risk_distance behind price (tighter trail).
    We are near target — protect most of the profit.

risk_distance = entry_price - stop_loss (for BUY)
             = stop_loss - entry_price (for SELL)
This is the initial ATR-based stop set by risk_manager at open.
We reuse it as our trailing unit — no separate ATR lookup needed.

Price source (in order of preference):
  1. features:flow:{pair} — OBI data has latest spread/price info
  2. orderbook:{pair}     — FIX stream L2 snapshot
  3. features:vwap:{pair} — session VWAP as price proxy (fallback)
  4. Skip this cycle      — never use stale data for SL moves

Redis:
  Reads:  execution:open:{pair}     — position data
  Reads:  orderbook:{pair}          — current bid/ask
  Writes: execution:open:{pair}     — updates trail_stage, stop_loss

Scheduler: background thread, runs every 60 seconds.

Usage:
    from execution.trailing_stop import trailing_stop_manager
    trailing_stop_manager.start()
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import redis
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

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

# Pip buffer added when moving SL to breakeven — avoids stop hunting
BREAKEVEN_BUFFER_PIPS = {
    p: (0.005 if p in JPY_PAIRS else 0.00005) for p in PAIRS
}

# How often to check and update trailing stops (seconds)
SCHEDULER_INTERVAL = 60   # every 1 minute

# Trail stages
STAGE_INITIAL    = 0   # not yet moved
STAGE_BREAKEVEN  = 1   # SL at entry + buffer
STAGE_TRAILING   = 2   # SL trailing 1× risk behind price
STAGE_LOCKING    = 3   # SL trailing 0.5× risk behind price (near TP)

# Redis cache TTL for position updates (24h — same as order_executor)
CACHE_TTL = 86400


# ──────────────────────────────────────────────────────────────────────────────
# Trailing Stop Manager
# ──────────────────────────────────────────────────────────────────────────────

class TrailingStopManager:
    """
    Monitors all open positions and adjusts SL as price moves favourably.

    Runs as a background thread. Reads positions from Redis, fetches
    current price, computes the correct stage, and calls
    order_executor.amend_sltp() when SL needs to move.

    Never moves SL against the position — only locks in profit.
    Never moves SL if price data is unavailable — safety first.
    """

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────

    @property
    def r(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )
        return self._redis

    def _rget(self, key: str) -> Optional[Dict]:
        try:
            raw = self.r.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _rset(self, key: str, data: Dict, ttl: int = CACHE_TTL) -> None:
        try:
            self.r.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning(f"TrailingStop | Redis SET failed: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_once(self) -> None:
        """
        Check all open positions and update trailing stops if needed.
        Called by scheduler loop every 60 seconds.
        """
        positions = self._get_open_positions()
        if not positions:
            logger.debug("TrailingStop | No open positions")
            return

        logger.info(
            f"TrailingStop | Checking {len(positions)} open positions"
        )

        for pos in positions:
            try:
                self._process_position(pos)
            except Exception as exc:
                logger.error(
                    f"TrailingStop | {pos.get('pair')} failed: {exc}"
                )

    # ── Position loading ──────────────────────────────────────────────────────

    def _get_open_positions(self) -> List[Dict]:
        """Load all open positions from Redis execution:open:* keys."""
        positions = []
        for pair in PAIRS:
            pos = self._rget(f"execution:open:{pair}")
            if pos and pos.get("position_id"):
                positions.append(pos)
        return positions

    # ── Price fetching ────────────────────────────────────────────────────────

    def _get_current_price(self, pair: str, direction: str) -> Optional[float]:
        """
        Get current market price for a pair.

        For BUY positions we care about bid (what we can sell at).
        For SELL positions we care about ask (what we can buy back at).

        Price sources tried in order:
          1. orderbook:{pair} — FIX L2 snapshot (best, real-time)
          2. features:flow:{pair} — has spread data, infer mid
          3. features:vwap:{pair} — NY VWAP as last resort proxy

        Returns None if no reliable price available.
        """
        # Source 0: Latest tick from DB (most reliable — FIX stream)
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text
            with db.connection() as conn:
                row = conn.execute(text("""
                    SELECT bid, ask FROM ticks
                    WHERE symbol = :symbol
                      AND time > NOW() - INTERVAL '2 minutes'
                    ORDER BY time DESC
                    LIMIT 1
                """), {"symbol": pair}).fetchone()
            if row:
                price = float(row[0]) if direction == "BUY" else float(row[1])
                # JPY descaling
                if pair in JPY_PAIRS and price > 1000:
                    price /= 100.0
                if price > 0:
                    return price
        except Exception as exc:
            logger.debug(f"TrailingStop | {pair}: tick DB read failed: {exc}")
        # Source 1: FIX order book
        ob = self._rget(f"orderbook:{pair}")
        if ob:
            try:
                if direction == "BUY":
                    bids = ob.get("bids", [])
                    if bids:
                        price = float(bids[0]["price"])
                        if price > 0:
                            return price
                else:
                    asks = ob.get("asks", [])
                    if asks:
                        price = float(asks[0]["price"])
                        if price > 0:
                            return price
            except Exception:
                pass

        # Source 2: infer from VWAP features (mid price proxy)
        vwap = self._rget(f"features:vwap:{pair}")
        if vwap:
            try:
                ny_vwap = float(vwap.get("ny_vwap", 0))
                if ny_vwap > 0:
                    logger.debug(
                        f"TrailingStop | {pair}: using NY VWAP as price proxy"
                    )
                    return ny_vwap
            except Exception:
                pass

        logger.warning(
            f"TrailingStop | {pair}: no price source available — skipping"
        )
        return None

    # ── Core trailing logic ───────────────────────────────────────────────────

    def _process_position(self, pos: Dict) -> None:
        """
        Evaluate one position and move SL if conditions are met.

        Args:
            pos: Position dict from Redis execution:open:{pair}
        """
        pair        = pos["pair"]
        direction   = pos["direction"]          # "BUY" or "SELL"
        entry       = float(pos["entry_price"])
        current_sl  = float(pos["stop_loss"])
        take_profit = float(pos["take_profit"])
        position_id = pos["position_id"]
        stage       = int(pos.get("trail_stage", STAGE_INITIAL))
        buffer      = BREAKEVEN_BUFFER_PIPS[pair]

        # risk_distance = initial stop size (set by risk_manager at open)
        if direction == "BUY":
            risk_dist = entry - current_sl if stage == STAGE_INITIAL else float(pos.get("initial_risk", entry - current_sl))
        else:
            risk_dist = current_sl - entry if stage == STAGE_INITIAL else float(pos.get("initial_risk", current_sl - entry))

        if risk_dist <= 0:
            logger.warning(f"TrailingStop | {pair}: invalid risk_dist={risk_dist} — skipping")
            return

        # Store initial_risk on first run so it doesn't change as SL moves
        if "initial_risk" not in pos:
            pos["initial_risk"] = risk_dist
            self._rset(f"execution:open:{pair}", pos)

        # Get current price
        current_price = self._get_current_price(pair, direction)
        if current_price is None:
            return

        # Compute profit in risk units
        if direction == "BUY":
            profit_dist = current_price - entry
            dist_to_tp  = take_profit - current_price
        else:
            profit_dist = entry - current_price
            dist_to_tp  = current_price - take_profit

        profit_r = profit_dist / risk_dist  # profit in R multiples

        logger.debug(
            f"TrailingStop | {pair} {direction}: "
            f"entry={entry} current={current_price:.5f} "
            f"sl={current_sl} profit_R={profit_r:.2f} stage={stage}"
        )

        # ── Stage logic ───────────────────────────────────────────────────────

        new_sl    = None
        new_stage = stage

        if direction == "BUY":
            # Stage 1: Breakeven + start trail at 0.5R behind when +1R reached
            # Changed 2026-03-27: previously trail only started at +2R
            # Data showed most trades peak at +10-15 pips (avg SL=13 pips)
            # and never reached +2R — sitting at breakeven then flipping flat
            # Now: trail immediately from +1R at 0.5R distance
            if profit_r >= 1.0 and stage == STAGE_INITIAL:
                be_sl        = round(entry + buffer, 5)
                candidate_sl = round(current_price - risk_dist * 0.5, 5)
                new_sl       = max(be_sl, candidate_sl)
                new_stage    = STAGE_BREAKEVEN
                logger.info(
                    f"TrailingStop | {pair} → BREAKEVEN+TRAIL "
                    f"(+{profit_r:.2f}R) new_sl={new_sl}"
                )

            # Stage 1 ongoing: keep trailing at 0.5R
            elif stage == STAGE_BREAKEVEN:
                candidate_sl = round(current_price - risk_dist * 0.5, 5)
                if candidate_sl > current_sl:
                    new_sl = candidate_sl
                    logger.info(
                        f"TrailingStop | {pair} trail(0.5R) up: "
                        f"{current_sl:.5f} → {new_sl:.5f}"
                    )

            # Stage 2: Tighten trail to 0.3R when +2R reached
            elif profit_r >= 2.0 and stage == STAGE_INITIAL:
                candidate_sl = round(current_price - risk_dist * 0.3, 5)
                if candidate_sl > current_sl:
                    new_sl    = candidate_sl
                    new_stage = STAGE_TRAILING
                    logger.info(
                        f"TrailingStop | {pair} → TRAIL_TIGHT "
                        f"(+{profit_r:.2f}R) new_sl={new_sl}"
                    )

            # Stage 2 ongoing: keep trailing at 0.3R
            elif stage == STAGE_TRAILING:
                candidate_sl = round(current_price - risk_dist * 0.3, 5)
                if candidate_sl > current_sl:
                    new_sl = candidate_sl
                    logger.info(
                        f"TrailingStop | {pair} trail(0.3R) up: "
                        f"{current_sl:.5f} → {new_sl:.5f}"
                    )

            # Stage 3: Lock at 0.2R when near TP (within 0.5R of TP)
            if dist_to_tp <= risk_dist * 0.5 and stage in (STAGE_TRAILING, STAGE_BREAKEVEN):
                candidate_sl = round(current_price - risk_dist * 0.2, 5)
                if candidate_sl > current_sl:
                    new_sl    = candidate_sl
                    new_stage = STAGE_LOCKING
                    logger.info(
                        f"TrailingStop | {pair} → LOCKING "
                        f"(near TP) new_sl={new_sl}"
                    )
            elif stage == STAGE_LOCKING:
                candidate_sl = round(current_price - risk_dist * 0.2, 5)
                if candidate_sl > current_sl:
                    new_sl = candidate_sl
                    logger.info(
                        f"TrailingStop | {pair} lock tightening: "
                        f"{current_sl:.5f} → {new_sl:.5f}"
                    )

        else:  # SELL — mirror of BUY logic with directions flipped
            # Stage 1: Breakeven + trail at 0.5R above price
            if profit_r >= 1.0 and stage == STAGE_INITIAL:
                be_sl        = round(entry - buffer, 5)
                candidate_sl = round(current_price + risk_dist * 0.5, 5)
                new_sl       = min(be_sl, candidate_sl)
                new_stage    = STAGE_BREAKEVEN
                logger.info(
                    f"TrailingStop | {pair} → BREAKEVEN+TRAIL "
                    f"(+{profit_r:.2f}R) new_sl={new_sl}"
                )

            # Stage 1 ongoing: keep trailing at 0.5R
            elif stage == STAGE_BREAKEVEN:
                candidate_sl = round(current_price + risk_dist * 0.5, 5)
                if candidate_sl < current_sl:
                    new_sl = candidate_sl
                    logger.info(
                        f"TrailingStop | {pair} trail(0.5R) down: "
                        f"{current_sl:.5f} → {new_sl:.5f}"
                    )

            # Stage 2: Tighten to 0.3R at +2R
            elif profit_r >= 2.0 and stage == STAGE_INITIAL:
                candidate_sl = round(current_price + risk_dist * 0.3, 5)
                if candidate_sl < current_sl:
                    new_sl    = candidate_sl
                    new_stage = STAGE_TRAILING
                    logger.info(
                        f"TrailingStop | {pair} → TRAIL_TIGHT "
                        f"(+{profit_r:.2f}R) new_sl={new_sl}"
                    )

            # Stage 2 ongoing: trail at 0.3R
            elif stage == STAGE_TRAILING:
                candidate_sl = round(current_price + risk_dist * 0.3, 5)
                if candidate_sl < current_sl:
                    new_sl = candidate_sl
                    logger.info(
                        f"TrailingStop | {pair} trail(0.3R) down: "
                        f"{current_sl:.5f} → {new_sl:.5f}"
                    )

            # Stage 3: Lock at 0.2R near TP
            if dist_to_tp <= risk_dist * 0.5 and stage in (STAGE_TRAILING, STAGE_BREAKEVEN):
                candidate_sl = round(current_price + risk_dist * 0.2, 5)
                if candidate_sl < current_sl:
                    new_sl    = candidate_sl
                    new_stage = STAGE_LOCKING
                    logger.info(
                        f"TrailingStop | {pair} → LOCKING "
                        f"(near TP) new_sl={new_sl}"
                    )
            elif stage == STAGE_LOCKING:
                candidate_sl = round(current_price + risk_dist * 0.2, 5)
                if candidate_sl < current_sl:
                    new_sl = candidate_sl
                    logger.info(
                        f"TrailingStop | {pair} lock tightening: "
                        f"{current_sl:.5f} → {new_sl:.5f}"
                    )

        # ── Apply SL amendment if needed ──────────────────────────────────────
        if new_sl is not None:
            self._amend_sl(
                pair, position_id, new_sl, take_profit, new_stage, pos
            )

    def _amend_sl(
        self,
        pair:        str,
        position_id: str,
        new_sl:      float,
        take_profit: float,
        new_stage:   int,
        pos:         Dict,
    ) -> None:
        """
        Call order_executor.amend_sltp() and update Redis cache.

        Args:
            pair:        Currency pair
            position_id: cTrader position ID
            new_sl:      New stop loss price
            take_profit: Keep existing TP unchanged
            new_stage:   Updated trail stage
            pos:         Current position dict (will be updated in Redis)
        """
        try:
            from execution.order_executor import order_executor
            decimals = 3 if pair in {"USDJPY","EURJPY","GBPJPY","AUDJPY","NZDJPY"} else 5
            new_sl      = round(new_sl, decimals)
            take_profit = round(take_profit, decimals)
            result = order_executor.amend_sltp(
                pair=pair,
                position_id=position_id,
                stop_loss=new_sl,
                take_profit=take_profit,
            )

            if result.get("success"):
                # Update Redis cache with new SL and stage
                pos["stop_loss"]   = new_sl
                pos["trail_stage"] = new_stage
                self._rset(f"execution:open:{pair}", pos)
                logger.success(
                    f"TrailingStop | {pair} SL updated: "
                    f"{new_sl:.5f} stage={new_stage}"
                )
            else:
                logger.warning(
                    f"TrailingStop | {pair} amend failed: "
                    f"{result.get('error', 'unknown')}"
                )

        except Exception as exc:
            logger.error(f"TrailingStop | {pair} amend exception: {exc}")

    # ── Background scheduler ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start background daemon thread — checks trails every 60s."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("TrailingStop | Already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="trailing-stop-manager",
        )
        self._scheduler_thread.start()
        logger.info(
            f"TrailingStop | Started (interval={SCHEDULER_INTERVAL}s)"
        )

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        logger.info("TrailingStop | Loop running")
        while self._running:
            try:
                self.run_once()
            except Exception as exc:
                logger.error(f"TrailingStop | Loop error: {exc}")
            time.sleep(SCHEDULER_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Module singleton
# ──────────────────────────────────────────────────────────────────────────────

trailing_stop_manager = TrailingStopManager()
