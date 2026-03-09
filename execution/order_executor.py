"""
execution/order_executor.py
============================
Order Executor — cTrader OpenAPI Order Placement

Translates risk_manager position parameters into real cTrader
orders via the ProtoOA WebSocket protocol.

In PAPER mode (TRADING_MODE=PAPER):
  - Connects to cTrader DEMO server (demo.ctraderapi.com:5035)
  - Places real orders on demo account
  - Real fills, real spreads, real slippage simulation
  - All trades recorded in local DB for postmortem analysis

In LIVE mode (TRADING_MODE=LIVE):
  - Connects to cTrader LIVE server (live.ctraderapi.com:5035)
  - Places real orders with real money
  - Only activated after 1 month paper trading + human approval

Order flow:
  1. signal_aggregator → LONG/SHORT/FLAT decision
  2. risk_manager      → pre_trade_check + calculate_position
  3. order_executor    → place_market_order (this file)
  4. cTrader demo      → fills order, returns position_id
  5. risk_manager      → record_trade_open (DB)
  6. order_executor    → monitors position (SL/TP)
  7. On close          → risk_manager.record_trade_close

Protobuf messages used:
  ProtoOANewOrderReq       → place market order with SL/TP
  ProtoOAClosePositionReq  → manual close
  ProtoOAAmendPositionSLTPReq → modify SL/TP (trailing stop / scale-in)
  ProtoOAReconcileReq      → sync open positions

cTrader volume units:
  1 lot = 100,000 units
  0.01 lot = 1,000 units (minimum)
  volume field in ProtoOA = units (not lots)

SL/TP in cTrader:
  Passed as absolute price levels (stopLoss, takeProfit fields)
  NOT as pip distances

Redis keys written:
  execution:open:{PAIR}    active position data    TTL=24h
  execution:last_order     last order result

Usage:
  from execution.order_executor import order_executor
  result = order_executor.place_market_order(
      pair="EURGBP", direction="SELL",
      volume_units=88000, stop_loss=0.867, take_profit=0.864,
      signal_data={...}
  )
  order_executor.close_position(position_id=123456)
  order_executor.sync_positions()
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

# cTrader symbol IDs (confirmed working from project brief)
SYMBOL_MAP: Dict[str, int] = {
    "EURUSD": 1,  "GBPUSD": 2,  "EURJPY": 3,
    "USDJPY": 4,  "AUDUSD": 5,  "USDCHF": 6,
    "GBPJPY": 7,  "USDCAD": 8,  "EURGBP": 9,
    "AUDJPY": 11, "NZDUSD": 12, "NZDJPY": 21,
}

# Price precision per pair (decimal places for SL/TP)
PRICE_PRECISION: Dict[str, int] = {
    "EURUSD": 5, "GBPUSD": 5, "AUDUSD": 5,
    "NZDUSD": 5, "USDCAD": 5, "USDCHF": 5,
    "EURGBP": 5, "USDJPY": 3, "EURJPY": 3,
    "GBPJPY": 3, "AUDJPY": 3, "NZDJPY": 3,
}

# cTrader price multiplier: prices sent as integers
# Non-JPY: price × 100000 (5 decimal places)
# JPY:     price × 1000   (3 decimal places)
JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

ORDER_COMMENT  = "G10Bot"
SLIPPAGE_PIPS  = 3        # max allowed slippage in points
CACHE_TTL      = 86400    # 24h for open position cache

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
# PRICE HELPERS
# ══════════════════════════════════════════════════════════════

def _price_to_ctrader(pair: str, price: float) -> int:
    """
    Convert decimal price to cTrader integer format.
    cTrader sends/receives prices as integers:
      Non-JPY: 1.08523 → 108523  (× 100000)
      JPY:     149.350 → 149350  (× 1000)
    """
    if pair in JPY_PAIRS:
        return int(round(price * 1000))
    return int(round(price * 100000))


def _ctrader_to_price(pair: str, value: int) -> float:
    """Convert cTrader integer price back to decimal."""
    if pair in JPY_PAIRS:
        return value / 1000.0
    return value / 100000.0


# ══════════════════════════════════════════════════════════════
# ORDER RESULT PARSER
# ══════════════════════════════════════════════════════════════

def _parse_order_response(response: Any, pair: str) -> Dict:
    """
    Parse ProtoOA response into a clean result dict.
    Handles both execution reports and error responses.
    """
    if response is None:
        return {"success": False, "error": "No response from cTrader"}

    resp_type = type(response).__name__
    logger.debug(f"OrderExecutor | Response type: {resp_type}")

    # Execution report — order filled
    if "ExecutionEvent" in resp_type or "OrderEvent" in resp_type:
        try:
            pos_id     = getattr(response, "position",    None)
            order_id   = getattr(response, "order",       None)
            exec_price = getattr(response, "executionPrice", 0)

            position_id = None
            order_id_str = None

            if pos_id:
                position_id  = str(getattr(pos_id, "positionId", ""))
            if order_id:
                order_id_str = str(getattr(order_id, "orderId", ""))
                if not exec_price:
                    exec_price = getattr(order_id, "executionPrice", 0)

            fill_price = _ctrader_to_price(pair, exec_price) if exec_price else 0.0

            return {
                "success":     True,
                "position_id": position_id,
                "order_id":    order_id_str,
                "fill_price":  fill_price,
                "response_type": resp_type,
            }
        except Exception as exc:
            logger.warning(f"OrderExecutor | Parse warning: {exc}")
            return {
                "success":       True,
                "position_id":   None,
                "order_id":      None,
                "fill_price":    0.0,
                "response_type": resp_type,
                "raw":           str(response)[:200],
            }

    # Error response
    if "Error" in resp_type:
        error_code = getattr(response, "errorCode", "UNKNOWN")
        description = getattr(response, "description", "")
        return {
            "success": False,
            "error":   f"{error_code}: {description}",
        }

    # Reconcile response (position sync)
    if "Reconcile" in resp_type:
        positions = getattr(response, "position", [])
        return {
            "success":   True,
            "positions": [
                {
                    "position_id": str(p.positionId),
                    "symbol_id":   p.tradeData.symbolId,
                    "volume":      p.tradeData.volume,
                    "side":        p.tradeData.tradeSide,
                    "open_price":  _ctrader_to_price(pair, p.price),
                    "sl":          _ctrader_to_price(pair, p.stopLoss) if p.stopLoss else None,
                    "tp":          _ctrader_to_price(pair, p.takeProfit) if p.takeProfit else None,
                }
                for p in positions
            ],
        }

    # Unknown — return raw
    logger.warning(f"OrderExecutor | Unknown response type: {resp_type}")
    return {
        "success":       True,
        "response_type": resp_type,
        "raw":           str(response)[:300],
    }


# ══════════════════════════════════════════════════════════════
# ORDER EXECUTOR
# ══════════════════════════════════════════════════════════════

class OrderExecutor:
    """
    Order Executor — Places and manages cTrader orders.

    Wraps the existing CTraderOpenAPIConnector with order
    placement logic using ProtoOA protobuf messages.

    Safety rules:
      - TRADING_MODE=PAPER → demo server only
      - TRADING_MODE=LIVE  → live server (requires explicit env var)
      - All orders have mandatory SL and TP
      - Position ID stored in Redis + DB for monitoring
      - Never places order without prior risk_manager approval
    """

    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv(str(ROOT / ".env"))
        self._trading_mode = os.getenv("TRADING_MODE", "PAPER")
        self._account_id   = int(os.getenv("CTRADER_ACCOUNT_ID", "0"))
        self._connector    = None
        self._redis        = None

    @property
    def r(self):
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    @property
    def connector(self):
        """Lazy-load and auto-connect cTrader connector."""
        if self._connector is None:
            from data_pipeline.connectors.ctrader_rest_api import connector
            self._connector = connector
        if not self._connector._account_authed.is_set():
            logger.info("OrderExecutor | Connecting to cTrader...")
            self._connector.connect()
        return self._connector

    # ── Public API ────────────────────────────────────────────

    def place_market_order(
        self,
        pair:         str,
        direction:    str,        # "BUY" or "SELL"
        volume_units: int,
        stop_loss:    float,
        take_profit:  float,
        signal_data:  Dict,
        regime:       str = "TLV",
        risk_pct:     float = 0.01,
    ) -> Dict:
        """
        Place a market order on cTrader demo/live account.

        Args:
            pair:         Currency pair e.g. "EURGBP"
            direction:    "BUY" or "SELL"
            volume_units: Position size in units (1000 = 0.01 lot)
            stop_loss:    Stop loss price (absolute level)
            take_profit:  Take profit price (absolute level)
            signal_data:  Dict of analyst signals for DB recording
            regime:       Current regime (for DB recording)
            risk_pct:     Risk percentage used (for DB recording)

        Returns:
            Dict with success, position_id, fill_price, trade_db_id
        """
        logger.info(
            f"OrderExecutor | Placing {direction} {pair} | "
            f"{volume_units:,}u | SL={stop_loss} TP={take_profit}"
        )

        # Safety check: never execute without SL
        if stop_loss <= 0 or take_profit <= 0:
            return {
                "success": False,
                "error":   "MISSING_SL_TP: SL and TP required",
            }

        symbol_id = SYMBOL_MAP.get(pair)
        if not symbol_id:
            return {
                "success": False,
                "error":   f"UNKNOWN_PAIR: {pair} not in SYMBOL_MAP",
            }

        # Build ProtoOA order request
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOANewOrderReq,
            )
            from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
                ProtoOAOrderType,
                ProtoOATradeSide,
                ProtoOATimeInForce,
            )

            req = ProtoOANewOrderReq()
            req.ctidTraderAccountId = self._account_id
            req.symbolId            = symbol_id
            req.orderType           = ProtoOAOrderType.Value("MARKET")
            req.tradeSide           = ProtoOATradeSide.Value(direction)
            req.volume              = volume_units
            req.timeInForce         = ProtoOATimeInForce.Value("GOOD_TILL_CANCEL")
            req.stopLoss            = _price_to_ctrader(pair, stop_loss)
            req.takeProfit          = _price_to_ctrader(pair, take_profit)
            req.comment             = ORDER_COMMENT
            req.label               = f"G10-{pair}-{direction[:1]}"
            req.slippageInPoints    = SLIPPAGE_PIPS * 10  # points = pips × 10

        except Exception as exc:
            logger.error(f"OrderExecutor | Request build failed: {exc}")
            return {"success": False, "error": str(exc)}

        # Send to cTrader
        try:
            response = self.connector._send_request(req)
            result   = _parse_order_response(response, pair)
        except Exception as exc:
            logger.error(f"OrderExecutor | Send failed: {exc}")
            return {"success": False, "error": str(exc)}

        if not result["success"]:
            logger.error(
                f"OrderExecutor | Order rejected: {result.get('error')}"
            )
            return result

        # Extract fill details
        position_id  = result.get("position_id")
        order_id     = result.get("order_id")
        fill_price   = result.get("fill_price") or 0.0

        # If fill price not in response, use current mid
        if fill_price == 0.0:
            from execution.risk_manager import _get_current_price
            fill_price = _get_current_price(pair)

        logger.success(
            f"OrderExecutor | ✅ Filled: {pair} {direction} "
            f"{volume_units:,}u @ {fill_price} | "
            f"pos_id={position_id}"
        )

        # Record in DB via risk_manager
        from execution.risk_manager import risk_manager
        trade_id = risk_manager.record_trade_open(
            pair          = pair,
            direction     = direction,
            volume_units  = volume_units,
            entry_price   = fill_price,
            stop_loss     = stop_loss,
            take_profit   = take_profit,
            regime        = regime,
            risk_pct      = risk_pct,
            signal_data   = signal_data,
            fix_order_id  = order_id,
            position_id   = position_id,
        )

        # Cache open position in Redis
        position_data = {
            "pair":        pair,
            "direction":   direction,
            "volume":      volume_units,
            "entry_price": fill_price,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "position_id": position_id,
            "trade_db_id": trade_id,
            "regime":      regime,
            "opened_at":   datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.r.setex(
                f"execution:open:{pair}",
                CACHE_TTL,
                json.dumps(position_data)
            )
            self.r.setex(
                "execution:last_order",
                3600,
                json.dumps({**position_data, "order_id": order_id})
            )
        except Exception as exc:
            logger.warning(f"OrderExecutor | Redis cache failed: {exc}")

        return {
            "success":     True,
            "pair":        pair,
            "direction":   direction,
            "position_id": position_id,
            "order_id":    order_id,
            "fill_price":  fill_price,
            "trade_db_id": trade_id,
            "volume":      volume_units,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
        }

    def close_position(
        self,
        pair:        str,
        position_id: str,
        volume:      Optional[int] = None,
        reason:      str = "CLOSED_MANUAL",
    ) -> Dict:
        """
        Close an open position.

        Args:
            pair:        Currency pair
            position_id: cTrader position ID (string)
            volume:      Units to close (None = close all)
            reason:      Close reason for DB record

        Returns:
            Dict with success, fill_price, pnl
        """
        logger.info(
            f"OrderExecutor | Closing {pair} pos={position_id} "
            f"reason={reason}"
        )

        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAClosePositionReq,
            )

            # Get position volume if not specified
            if volume is None:
                cached = self._get_cached_position(pair)
                volume = cached.get("volume", 1000) if cached else 1000

            req = ProtoOAClosePositionReq()
            req.ctidTraderAccountId = self._account_id
            req.positionId          = int(position_id)
            req.volume              = volume

            response = self.connector._send_request(req)
            result   = _parse_order_response(response, pair)

        except Exception as exc:
            logger.error(f"OrderExecutor | Close failed: {exc}")
            return {"success": False, "error": str(exc)}

        if not result["success"]:
            logger.error(f"OrderExecutor | Close rejected: {result.get('error')}")
            return result

        fill_price = result.get("fill_price", 0.0)
        if fill_price == 0.0:
            from execution.risk_manager import _get_current_price
            fill_price = _get_current_price(pair)

        # Update DB
        from execution.risk_manager import risk_manager
        cached   = self._get_cached_position(pair)
        trade_id = cached.get("trade_db_id") if cached else None

        pnl = None
        if trade_id:
            pnl = risk_manager.record_trade_close(trade_id, fill_price, reason)

        # Remove from Redis
        try:
            self.r.delete(f"execution:open:{pair}")
        except Exception:
            pass

        logger.success(
            f"OrderExecutor | ✅ Closed {pair} @ {fill_price} "
            f"pnl=${pnl:+.2f}" if pnl else
            f"OrderExecutor | ✅ Closed {pair} @ {fill_price}"
        )

        return {
            "success":     True,
            "pair":        pair,
            "fill_price":  fill_price,
            "pnl_usd":     pnl,
            "trade_db_id": trade_id,
        }

    def amend_sltp(
        self,
        pair:        str,
        position_id: str,
        stop_loss:   Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict:
        """
        Modify SL/TP on an open position.
        Used for: trailing stop, rolling SL after soft target hit.

        Course rule: "Roll stop-loss to guarantee profit on winning trade"
        """
        logger.info(
            f"OrderExecutor | Amending {pair} pos={position_id} "
            f"SL={stop_loss} TP={take_profit}"
        )
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAAmendPositionSLTPReq,
            )

            req = ProtoOAAmendPositionSLTPReq()
            req.ctidTraderAccountId = self._account_id
            req.positionId          = int(position_id)

            if stop_loss is not None:
                req.stopLoss = _price_to_ctrader(pair, stop_loss)
            if take_profit is not None:
                req.takeProfit = _price_to_ctrader(pair, take_profit)

            response = self.connector._send_request(req)
            result   = _parse_order_response(response, pair)

            if result["success"]:
                # Update Redis cache
                cached = self._get_cached_position(pair)
                if cached:
                    if stop_loss:
                        cached["stop_loss"]   = stop_loss
                    if take_profit:
                        cached["take_profit"] = take_profit
                    self.r.setex(
                        f"execution:open:{pair}",
                        CACHE_TTL,
                        json.dumps(cached)
                    )
                logger.success(
                    f"OrderExecutor | SL/TP amended: {pair} "
                    f"SL={stop_loss} TP={take_profit}"
                )

            return result

        except Exception as exc:
            logger.error(f"OrderExecutor | Amend failed: {exc}")
            return {"success": False, "error": str(exc)}

    def sync_positions(self) -> List[Dict]:
        """
        Sync open positions from cTrader via ProtoOAReconcileReq.
        Called on startup to reconcile DB with broker state.
        """
        logger.info("OrderExecutor | Syncing positions from cTrader...")
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAReconcileReq,
            )

            req = ProtoOAReconcileReq()
            req.ctidTraderAccountId = self._account_id

            response = self.connector._send_request(req)
            result   = _parse_order_response(response, "EURUSD")

            positions = result.get("positions", [])
            logger.info(f"OrderExecutor | Synced {len(positions)} positions")
            return positions

        except Exception as exc:
            logger.error(f"OrderExecutor | Sync failed: {exc}")
            return []

    def get_open_positions_local(self) -> List[Dict]:
        """Get open positions from Redis cache (no API call)."""
        positions = []
        try:
            from execution.signal_aggregator import PAIRS
            for pair in PAIRS:
                val = self.r.get(f"execution:open:{pair}")
                if val:
                    positions.append(json.loads(val))
        except Exception as exc:
            logger.warning(f"OrderExecutor | Local positions failed: {exc}")
        return positions

    def execute_signal(self, decision: Dict) -> Optional[Dict]:
        """
        Full execution pipeline for a single signal decision.

        This is the main entry point called by main_loop.py:
          1. Validate decision is actionable
          2. Run risk_manager pre-trade checks
          3. Calculate position sizing
          4. Place order
          5. Return result

        Args:
            decision: Dict from signal_aggregator.evaluate_pair()

        Returns:
            Order result dict or None if skipped/rejected
        """
        pair      = decision.get("pair", "")
        action    = decision.get("action", "FLAT")
        regime    = decision.get("regime", "TLV")
        size_mult = decision.get("size_mult", 1.0)
        conf      = decision.get("confidence", 0.0)

        # Skip flat signals
        if action == "FLAT":
            return None

        direction = "BUY" if action == "LONG" else "SELL"

        # Risk manager pre-trade check
        from execution.risk_manager import risk_manager, _get_current_price
        check = risk_manager.pre_trade_check(pair, direction, regime, size_mult)
        if not check["approved"]:
            logger.debug(
                f"OrderExecutor | {pair} skipped: {check['reason']}"
            )
            return None

        # Calculate position
        entry_price = _get_current_price(pair)
        sizing      = risk_manager.calculate_position(
            pair        = pair,
            direction   = direction,
            entry_price = entry_price,
            regime      = regime,
            size_mult   = size_mult,
            starter     = True,
        )

        # Build signal data for DB
        signal_data = {
            "regime":      regime,
            "regime_conf": decision.get("a2_conf", 0.0),
            "a2_score":    decision.get("a2_score", 0.0),
            "tft_prob":    None,
            "tsmixer":     None,
            "sentiment":   decision.get("sent_score", 0.0),
            "opus_conv":   decision.get("opus_conv", 0.0),
        }

        # Place order
        result = self.place_market_order(
            pair         = pair,
            direction    = direction,
            volume_units = sizing["volume_units"],
            stop_loss    = sizing["stop_loss"],
            take_profit  = sizing["take_profit"],
            signal_data  = signal_data,
            regime       = regime,
            risk_pct     = sizing["risk_pct"],
        )

        if result["success"]:
            logger.success(
                f"OrderExecutor | Signal executed: "
                f"{pair} {direction} conf={conf:.2f} | "
                f"pos={result.get('position_id')} "
                f"fill={result.get('fill_price')}"
            )
        return result

    # ── Internal helpers ──────────────────────────────────────

    def _get_cached_position(self, pair: str) -> Optional[Dict]:
        """Get cached position data from Redis."""
        try:
            val = self.r.get(f"execution:open:{pair}")
            if val:
                return json.loads(val)
        except Exception:
            pass
        return None

    def print_status(self) -> None:
        """Print current open positions from local cache."""
        positions = self.get_open_positions_local()

        print("\n" + "=" * 65)
        print("ORDER EXECUTOR — OPEN POSITIONS")
        print("=" * 65)
        print(f"  Mode: {self._trading_mode} | Account: {self._account_id}")

        if not positions:
            print("  No open positions")
        else:
            print(
                f"\n  {'PAIR':<10} {'DIR':<5} {'UNITS':>8} "
                f"{'ENTRY':>9} {'SL':>9} {'TP':>9}"
            )
            print("  " + "-" * 55)
            for p in positions:
                print(
                    f"  {p['pair']:<10} {p['direction']:<5} "
                    f"{p['volume']:>8,} "
                    f"{p['entry_price']:>9.5f} "
                    f"{p['stop_loss']:>9.5f} "
                    f"{p['take_profit']:>9.5f}"
                )

        print("=" * 65 + "\n")


# ── Module singleton ──────────────────────────────────────────
order_executor = OrderExecutor()


# ── Direct run — test connection + dry run ────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")

    logger.info("OrderExecutor | Testing cTrader connection...")

    # Test connection
    try:
        conn = order_executor.connector
        logger.success("OrderExecutor | Connected to cTrader ✅")
    except Exception as exc:
        logger.error(f"OrderExecutor | Connection failed: {exc}")
        sys.exit(1)

    # Sync positions
    positions = order_executor.sync_positions()
    logger.info(f"OrderExecutor | Open positions on account: {len(positions)}")

    # Print status
    order_executor.print_status()

    # Show what would happen for today's 3 signals (DRY RUN — no actual orders)
    print("\nDRY RUN — Signal execution preview:")
    print("=" * 65)

    from execution.signal_aggregator import signal_aggregator
    tradeable = signal_aggregator.get_tradeable()

    for decision in tradeable:
        pair      = decision["pair"]
        action    = decision["action"]
        regime    = decision["regime"]
        size_mult = decision["size_mult"]
        direction = "BUY" if action == "LONG" else "SELL"

        from execution.risk_manager import risk_manager, _get_current_price
        entry   = _get_current_price(pair)
        sizing  = risk_manager.calculate_position(
            pair, direction, entry, regime, size_mult
        )

        print(f"\n  {pair} {action} (conf={decision['confidence']:.2f})")
        print(f"    Volume:  {sizing['volume_units']:,} units")
        print(f"    Entry:   ~{entry:.5f}")
        print(f"    SL:      {sizing['stop_loss']:.5f} ({sizing['stop_pips']:.1f} pips)")
        print(f"    TP:      {sizing['take_profit']:.5f}")
        print(f"    Risk:    ${sizing['risk_amount_usd']:.2f}")
        print(f"    Regime:  {regime} size={size_mult}x")

    print("\n" + "=" * 65)
    print("DRY RUN complete — no orders placed")
    print("To execute for real: order_executor.execute_signal(decision)")
    print("=" * 65)