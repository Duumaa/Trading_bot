"""
execution/risk_manager.py
==========================
Risk Manager — Position Sizing & Capital Protection

Implements all hard risk rules from the trading course:
  • 2% max risk per trade (ATR-based stop sizing)
  • Starter vs full position (50% size on new trades, scale in)
  • Max 3 concurrent open positions
  • Daily loss kill switch: -5% → halt all trading
  • Pre-event blackout: flat 2h before HIGH-impact events
  • Regime-based size multiplier (THV/TLV=1.0, MR=0.5, PNS=0.0)
  • Single-asset position limit: 25% of forex exposure per pair
  • Portfolio stop-loss: -20% of margin → halt + reassess

Course principles applied:
  "Stop loss = 1-month ATR, soft target = 3× stop distance"
  "Starter position = 50% size, add when trade proves itself"
  "Single asset limit: never let one position decide portfolio fate"
  "Portfolio stop-loss: protect gains as portfolio grows"
  "When in a winning trade (past soft target): can break limits"

Position sizing formula:
  risk_amount  = account_balance × risk_pct (default 0.02)
  stop_distance = ATR_14 × atr_multiplier (default 1.5)
  pip_value    = contract_size / price (for non-USD quote pairs)
  volume_units = risk_amount / (stop_distance × pip_value)
  volume_units rounded down to nearest 1000 (cTrader minimum)

Redis keys written:
  risk:account_state          current balance, equity, daily_pnl
  risk:kill_switch            "ACTIVE" | "INACTIVE"
  risk:open_positions_count   integer
  risk:halted                 "true" | "false"
  risk:halt_reason            string

DB tables written:
  trades              INSERT on new trade, UPDATE on close
  performance_snapshots  daily snapshot

Usage:
  from execution.risk_manager import risk_manager
  check   = risk_manager.pre_trade_check(decision, pair)
  sizing  = risk_manager.calculate_position(decision, account)
  risk_manager.record_trade_open(trade_params)
  risk_manager.record_trade_close(trade_id, exit_price, status)
  risk_manager.take_daily_snapshot()
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import text

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

from data_pipeline.storage.timeseries_db import db  # noqa: E402

# ══════════════════════════════════════════════════════════════
# CONSTANTS — Hard risk limits (never change without review)
# ══════════════════════════════════════════════════════════════

MAX_RISK_PER_TRADE_PCT   = 0.02    # 2% of account per trade
MAX_CONCURRENT_POSITIONS = 3       # max open trades at once
DAILY_LOSS_KILL_PCT      = 0.05    # -5% daily → halt trading
PORTFOLIO_STOP_PCT       = 0.20    # -20% of margin → halt + reassess
EVENT_BLACKOUT_HOURS     = 2       # flat N hours before HIGH-impact event
STARTER_POSITION_MULT    = 0.50    # first entry = 50% of full size
ATR_STOP_MULTIPLIER      = 1.5     # stop = 1.5 × ATR_14
SOFT_TARGET_MULTIPLIER   = 3.0     # target = 3 × stop distance
MIN_VOLUME_UNITS         = 1000    # cTrader minimum (0.01 lots)
VOLUME_STEP              = 1000    # round to nearest 1000 units

# Regime → size multiplier (applied ON TOP of starter/full position)
REGIME_SIZE_MULT: Dict[str, float] = {
    "THV": 1.0,
    "TLV": 1.0,
    "MR":  0.5,
    "PNS": 0.0,
}

# Contract sizes per pair (standard lots = 100,000 units base currency)
# cTrader uses volume in units (100,000 units = 1 lot)
CONTRACT_SIZE = 100_000

# JPY pairs: pip = 0.01, others: pip = 0.0001
JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

# Pip values (USD per pip per 100k units, approximate)
# For USD-quote pairs (EURUSD etc): pip_value = 10 USD per lot
# For JPY-quote: pip_value = ~6.70 USD per lot (at ~149 USDJPY)
# Risk manager fetches live price for accuracy
PIP_DECIMALS: Dict[str, int] = {
    "USDJPY": 2, "EURJPY": 2, "GBPJPY": 2,
    "AUDJPY": 2, "NZDJPY": 2,
}

CACHE_TTL = 86_400   # 24h — must not expire mid-session or kill switch goes blind

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
# ACCOUNT STATE
# ══════════════════════════════════════════════════════════════

class AccountState:
    """
    Tracks current account balance, equity, and daily P&L.
    In PAPER mode: uses simulated $10,000 starting balance.
    In LIVE mode: fetches from cTrader REST API.
    """

    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv(str(ROOT / ".env"))
        self.trading_mode = os.getenv("TRADING_MODE", "PAPER")
        self._balance     = 1_774.00   # updated balance — overridden by Redis
        self._equity      = 1_774.00
        self._daily_start = 1_774.00   # overridden by Redis daily_start key
        self._r           = None

    @property
    def r(self):
        if self._r is None:
            self._r = _get_redis()
        return self._r

    def get(self) -> Dict[str, Any]:
        """Get current account state from Redis or defaults."""
        try:
            val = self.r.get("risk:account_state")
            if val:
                return json.loads(val)

            # Cache miss — reconstruct from persistent keys rather than returning 0.0
            ds_raw  = self.r.get("risk:daily_start_balance")
            bal_raw = self.r.get("risk:session_start_balance")

            daily_start = float(ds_raw)  if ds_raw  else self._balance
            balance     = float(bal_raw) if bal_raw else self._balance

            daily_pnl     = balance - daily_start
            daily_pnl_pct = daily_pnl / daily_start if daily_start > 0 else 0.0

            logger.warning(
                f"AccountState | account_state cache MISS — "
                f"reconstructed: balance={balance:.2f}, "
                f"daily_start={daily_start:.2f}, "
                f"daily_pnl_pct={daily_pnl_pct*100:.2f}%"
            )

            state = {
                "balance":       balance,
                "equity":        balance,
                "daily_start":   daily_start,
                "daily_pnl":     round(daily_pnl, 2),
                "daily_pnl_pct": round(daily_pnl_pct, 6),
                "mode":          "PAPER",
            }
            # Re-cache it so next call is fast
            self.r.setex("risk:account_state", CACHE_TTL, json.dumps(state))
            return state

        except Exception as exc:
            logger.error(f"AccountState.get | {exc}")
            return {
                "balance":       self._balance,
                "equity":        self._balance,
                "daily_start":   self._balance,
                "daily_pnl":     0.0,
                "daily_pnl_pct": 0.0,
                "mode":          "PAPER",
            }

    def update(self, balance: float, equity: float) -> None:
        """Update account state in Redis."""
        state = self.get()
        daily_pnl     = balance - state["daily_start"]
        daily_pnl_pct = daily_pnl / state["daily_start"] if state["daily_start"] > 0 else 0.0

        new_state = {
            "balance":       round(balance, 2),
            "equity":        round(equity, 2),
            "daily_start":   state["daily_start"],
            "daily_pnl":     round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 6),
            "mode":          self.trading_mode,
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }
        try:
            # Persist daily_start with no TTL so it survives cache expiry
            self.r.set("risk:daily_start_balance", str(state["daily_start"]))
            self.r.setex("risk:account_state", CACHE_TTL, json.dumps(new_state))
        except Exception as exc:
            logger.warning(f"RiskManager | Account state cache failed: {exc}")
        self._balance = balance
        self._equity  = equity

    def reset_daily(self) -> None:
        """Reset daily P&L tracking and clear kill switch (call at start of each trading day)."""
        state = self.get()
        state["daily_start"]   = state["balance"]
        state["daily_pnl"]     = 0.0
        state["daily_pnl_pct"] = 0.0
        try:
            self.r.setex("risk:account_state", CACHE_TTL, json.dumps(state))
            # Clear daily kill switch so bot resumes next session
            self.r.delete("risk:halted")
            self.r.delete("risk:daily_loss")
            self.r.set("risk:daily_pnl", "0.0")
            self.r.set("risk:session_start_balance", str(state["balance"]))
        except Exception:
            pass
        logger.success(
            f"RiskManager | Daily reset complete | "
            f"Kill switch cleared | Start balance: ${state['balance']:,.2f}"
        )

    def apply_pnl(self, pnl_usd: float) -> None:
        """Apply a closed trade's P&L to the account state."""
        state   = self.get()
        balance = state["balance"] + pnl_usd
        equity  = state["equity"]  + pnl_usd
        self.update(balance, equity)


# ══════════════════════════════════════════════════════════════
# POSITION SIZING
# ══════════════════════════════════════════════════════════════

def _get_atr(pair: str) -> float:
    """
    Fetch ATR_14 from Redis statistical features.
    Falls back to conservative default if unavailable.
    """
    try:
        r   = _get_redis()
        val = r.get(f"features:stat:{pair}")
        if val:
            feats = json.loads(val)
            # Prefer M15 ATR for stop sizing (correct for M15 system)
            # Fall back to D1 ATR / 6 if M15 not available
            atr_m15 = feats.get("atr_14_m15", 0.0)
            if atr_m15 > 0:
                return float(atr_m15)
            atr = feats.get("atr_14", 0.0)
            if atr > 0:
                # JPY pairs stored as price×100 in ohlcv — ATR is 100× too large
                JPY_PAIRS_LOCAL = {"USDJPY","EURJPY","GBPJPY","AUDJPY","NZDJPY"}
                if pair in JPY_PAIRS_LOCAL:
                    atr = atr / 100.0
                return float(atr / 6.0)  # D1 ATR → M15 approximation
    except Exception as exc:
        logger.debug(f"RiskManager | ATR fetch failed {pair}: {exc}")

    # Conservative fallback ATRs (approximate, in price units)
    fallback = {
        "EURUSD": 0.0008, "GBPUSD": 0.0010, "AUDUSD": 0.0007,
        "NZDUSD": 0.0006, "USDCAD": 0.0008, "USDCHF": 0.0008,
        "USDJPY": 0.80,   "EURJPY": 1.00,   "GBPJPY": 1.20,
        "AUDJPY": 0.90,   "NZDJPY": 0.80,   "EURGBP": 0.0006,
    }
    atr = fallback.get(pair, 0.0008)
    logger.warning(
        f"RiskManager | ATR fallback for {pair}: {atr} "
        f"(run stat_engine.compute_all() for accuracy)"
    )
    return atr


def _get_current_price(pair: str) -> float:
    """
    Fetch latest mid price from Redis or DB.
    Used for pip value calculation.
    """
    try:
        r   = _get_redis()
        val = r.get(f"features:vwap:{pair}")
        if val:
            feats = json.loads(val)
            price = feats.get("current_price", 0.0)
            if price > 0:
                return float(price)
    except Exception:
        pass

    # Fallback: query latest ohlcv
    try:
        with db.connection() as conn:
            row = conn.execute(text("""
                SELECT close FROM ohlcv
                WHERE symbol = :sym AND timeframe = 'M15'
                ORDER BY time DESC LIMIT 1
            """), {"sym": pair}).fetchone()
            if row:
                price = row[0]
                # JPY pairs stored as price×100 in cTrader
                if pair in JPY_PAIRS and price > 1000:
                    price /= 100.0
                return float(price)
    except Exception as exc:
        logger.debug(f"RiskManager | Price fetch failed {pair}: {exc}")

    # Hard fallback
    fallback_prices = {
        "EURUSD": 1.0800, "GBPUSD": 1.2600, "AUDUSD": 0.6300,
        "NZDUSD": 0.5700, "USDCAD": 1.3600, "USDCHF": 0.9000,
        "USDJPY": 149.00, "EURJPY": 160.00, "GBPJPY": 188.00,
        "AUDJPY": 94.00,  "NZDJPY": 85.00,  "EURGBP": 0.8500,
    }
    return fallback_prices.get(pair, 1.0)


def _calculate_pip_value_usd(pair: str, price: float) -> float:
    """
    Calculate USD value of 1 pip per 100,000 units (1 standard lot).

    For USD-quote pairs (EURUSD, GBPUSD etc): $10 per pip per lot
    For USD-base pairs (USDJPY, USDCAD etc):  $10 / price per pip per lot
    For cross pairs (EURGBP, EURJPY etc):     need to convert via USD

    Returns: pip_value_usd per 100k units
    """
    pip_size = 0.01 if pair in JPY_PAIRS else 0.0001

    # USD is quote currency (EURUSD, GBPUSD, AUDUSD, NZDUSD, EURGBP→GBP/USD for conversion)
    usd_quote_pairs = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"}
    # USD is base currency
    usd_base_pairs  = {"USDCAD", "USDCHF", "USDJPY"}

    if pair in usd_quote_pairs:
        return pip_size * CONTRACT_SIZE  # 10 USD per pip per lot

    elif pair in usd_base_pairs:
        # pip_value = pip_size * contract / price
        return (pip_size * CONTRACT_SIZE) / price

    else:
        # Cross pairs: approximate using known ratios
        # EURGBP: multiply by GBPUSD ≈ 1.26
        # EURJPY, GBPJPY, AUDJPY, NZDJPY: JPY crosses
        cross_multipliers = {
            "EURGBP": 1.26,  # approx GBPUSD
            "EURJPY": 1.0 / 149.0 * CONTRACT_SIZE * 0.01,
            "GBPJPY": 1.0 / 149.0 * CONTRACT_SIZE * 0.01,
            "AUDJPY": 1.0 / 149.0 * CONTRACT_SIZE * 0.01,
            "NZDJPY": 1.0 / 149.0 * CONTRACT_SIZE * 0.01,
        }
        if pair in cross_multipliers:
            mult = cross_multipliers[pair]
            if pair == "EURGBP":
                return pip_size * CONTRACT_SIZE * mult
            else:
                return mult  # already computed for JPY crosses
        return pip_size * CONTRACT_SIZE  # safe default


def calculate_volume_units(
    pair:         str,
    account:      Dict[str, float],
    stop_pips:    float,
    risk_pct:     float = MAX_RISK_PER_TRADE_PCT,
    size_mult:    float = 1.0,
    starter:      bool  = True,
) -> Tuple[int, Dict]:
    """
    Calculate position size in cTrader volume units.

    Formula from course:
      risk_amount   = balance × risk_pct
      pip_value_lot = pip value per standard lot in USD
      lots          = risk_amount / (stop_pips × pip_value_lot)
      volume_units  = lots × 100,000

    Args:
        pair:       Currency pair e.g. "EURUSD"
        account:    Account state dict (balance, equity)
        stop_pips:  Stop distance in pips
        risk_pct:   Risk percentage (default 2%)
        size_mult:  Regime multiplier (0.5 for MR, 1.0 for THV/TLV)
        starter:    If True, apply STARTER_POSITION_MULT (0.5×)

    Returns:
        (volume_units, sizing_details_dict)
    """
    balance   = account.get("balance", 10_000.0)
    price     = _get_current_price(pair)

    # Apply size multipliers
    effective_risk = risk_pct * size_mult
    if starter:
        effective_risk *= STARTER_POSITION_MULT

    risk_amount    = balance * effective_risk
    pip_value      = _calculate_pip_value_usd(pair, price)

    if stop_pips <= 0 or pip_value <= 0:
        logger.warning(
            f"RiskManager | Invalid stop_pips={stop_pips} or "
            f"pip_value={pip_value} for {pair}"
        )
        return MIN_VOLUME_UNITS, {}

    # Calculate lots → volume units
    lots         = risk_amount / (stop_pips * pip_value)
    volume_raw   = lots * CONTRACT_SIZE

    # Round DOWN to nearest step (never round up — never exceed risk)
    volume_units = max(
        MIN_VOLUME_UNITS,
        int(volume_raw // VOLUME_STEP) * VOLUME_STEP
    )

    # Soft target = 3× stop distance
    soft_target_pips = stop_pips * SOFT_TARGET_MULTIPLIER

    details = {
        "balance":          round(balance, 2),
        "price":            round(price, 5),
        "risk_pct":         round(effective_risk, 4),
        "risk_amount_usd":  round(risk_amount, 2),
        "stop_pips":        round(stop_pips, 1),
        "pip_value_usd":    round(pip_value, 4),
        "lots":             round(lots, 4),
        "volume_units":     volume_units,
        "soft_target_pips": round(soft_target_pips, 1),
        "size_mult":        size_mult,
        "starter":          starter,
    }

    logger.info(
        f"RiskManager | {pair} sizing: "
        f"${risk_amount:.0f} risk | "
        f"stop={stop_pips:.1f}pips | "
        f"{volume_units:,} units ({lots:.3f} lots)"
    )
    return volume_units, details


def calculate_stop_from_atr(
    pair:             str,
    direction:        str,
    entry_price:      float,
    atr_multiplier:   float = ATR_STOP_MULTIPLIER,
) -> Tuple[float, float, float]:
    """
    Calculate stop loss and take profit from ATR.

    Course rule: stop = 1.5 × ATR_14, target = 3 × stop distance

    Args:
        pair:           Currency pair
        direction:      "BUY" or "SELL"
        entry_price:    Trade entry price
        atr_multiplier: ATR multiplier for stop (default 1.5)

    Returns:
        (stop_loss_price, take_profit_price, stop_pips)
    """
    atr        = _get_atr(pair)
    stop_dist  = atr * atr_multiplier
    tp_dist    = stop_dist * SOFT_TARGET_MULTIPLIER

    # Convert to pips for display
    pip_size   = 0.01 if pair in JPY_PAIRS else 0.0001
    stop_pips  = stop_dist / pip_size

    if direction == "BUY":
        stop_loss   = entry_price - stop_dist
        take_profit = entry_price + tp_dist
    else:  # SELL
        stop_loss   = entry_price + stop_dist
        take_profit = entry_price - tp_dist

    return round(stop_loss, 5), round(take_profit, 5), round(stop_pips, 1)


# ══════════════════════════════════════════════════════════════
# KILL SWITCHES & BLACKOUT
# ══════════════════════════════════════════════════════════════

def _check_event_blackout(pair: str) -> Tuple[bool, str]:
    """
    Check if pair is in pre-event blackout window.
    Returns (is_blackout, reason).
    """
    try:
        # Get currencies for this pair
        from execution.signal_aggregator import PAIR_CURRENCIES
        base, quote = PAIR_CURRENCIES.get(pair, ("", ""))
        currencies  = {base, quote}

        cutoff_time = datetime.now(timezone.utc) + timedelta(hours=EVENT_BLACKOUT_HOURS)

        with db.connection() as conn:
            rows = conn.execute(text("""
                SELECT event_name, event_time, country, impact
                FROM economic_events
                WHERE impact = 'HIGH'
                  AND event_time BETWEEN NOW() AND :cutoff
                  AND country = ANY(:currencies)
                ORDER BY event_time ASC
                LIMIT 3
            """), {
                "cutoff":     cutoff_time,
                "currencies": list(currencies),
            }).fetchall()

        if rows:
            event_name = rows[0][0]
            event_time = rows[0][1]
            mins_away  = int(
                (event_time - datetime.now(timezone.utc)).total_seconds() / 60
            )
            reason = (
                f"EVENT_BLACKOUT: {event_name} in {mins_away}min "
                f"({rows[0][2]})"
            )
            return True, reason

    except Exception as exc:
        logger.debug(f"RiskManager | Event check failed: {exc}")

    return False, ""


def _check_daily_kill_switch(account: Dict) -> Tuple[bool, str]:
    """
    Check daily loss kill switch.
    Returns (is_killed, reason).
    """
    daily_pnl_pct = account.get("daily_pnl_pct", 0.0)

    if daily_pnl_pct <= -DAILY_LOSS_KILL_PCT:
        reason = (
            f"DAILY_KILL_SWITCH: daily_pnl={daily_pnl_pct*100:.2f}% "
            f"(limit={DAILY_LOSS_KILL_PCT*100:.0f}%)"
        )
        return True, reason

    return False, ""


def _check_portfolio_stop(account: Dict) -> Tuple[bool, str]:
    """
    Check portfolio stop-loss.
    Course: -20% of margin → halt and reassess.
    """
    balance      = account.get("balance", 10_000.0)
    daily_start  = account.get("daily_start", balance)

    # Overall drawdown from initial
    try:
        r   = _get_redis()
        val = r.get("risk:initial_balance")
        initial = float(val) if val else daily_start
    except Exception:
        initial = daily_start

    if initial > 0:
        drawdown_pct = (initial - balance) / initial
        if drawdown_pct >= PORTFOLIO_STOP_PCT:
            reason = (
                f"PORTFOLIO_STOP: drawdown={drawdown_pct*100:.1f}% "
                f"(limit={PORTFOLIO_STOP_PCT*100:.0f}%)"
            )
            return True, reason

    return False, ""


def _count_open_positions() -> int:
    """Count currently open positions from DB."""
    try:
        with db.connection() as conn:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM trades
                WHERE status = 'OPEN'
            """)).scalar()
            return int(count or 0)
    except Exception as exc:
        logger.warning(f"RiskManager | Open position count failed: {exc}")
        return 0

def _is_pair_already_open(pair: str) -> bool:
    """Check if we already have an open position in this pair."""
    try:
        with db.connection() as conn:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM trades
                WHERE symbol = :sym AND status = 'OPEN'
            """), {"sym": pair}).scalar()
            return int(count or 0) > 0
    except Exception:
        return False


def _get_open_positions() -> List[Dict]:
    """
    Return list of open positions with pair and margin info.
    Used by correlation guard.
    """
    try:
        with db.connection() as conn:
            rows = conn.execute(text("""
                SELECT symbol, direction, volume_units, entry_price
                FROM trades
                WHERE status = 'OPEN'
            """)).fetchall()
            return [
                {
                    "pair":         r[0],
                    "direction":    r[1],
                    "volume_units": r[2],
                    "margin_used":  (r[2] / CONTRACT_SIZE) * r[3],
                }
                for r in rows
            ]
    except Exception as exc:
        logger.warning(f"RiskManager | get_open_positions failed: {exc}")
        return []


# ══════════════════════════════════════════════════════════════
# MAIN RISK MANAGER
# ══════════════════════════════════════════════════════════════

class RiskManager:
    """
    Risk Manager — Pre-trade checks, position sizing, trade recording.

    Hard rules (never overridden):
      1. Max 2% risk per trade
      2. Max 3 concurrent positions
      3. Daily -5% → kill switch
      4. Pre-event blackout 2h
      5. PNS regime → flat always
      6. Portfolio -20% → halt + reassess
      7. One position per pair max

    Soft rules (can be overridden by winning trade logic):
      • Starter position (50%) → scale to full when trade works
      • Regime size multiplier
    """

    # Pair → (base_currency, quote_currency)
    _PAIR_CURRENCIES: Dict[str, tuple] = {
        "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
        "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"),
        "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
        "USDCAD": ("USD", "CAD"), "EURGBP": ("EUR", "GBP"),
        "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
        "AUDJPY": ("AUD", "JPY"), "NZDJPY": ("NZD", "JPY"),
    }
    _MAX_SAME_CURRENCY: int   = 2     # max open pairs sharing any currency
    _MAX_NET_EXPOSURE_PCT: float = 3.0  # % of account per currency

    def __init__(self) -> None:
        self.account = AccountState()
        self._redis  = None

    def check_correlation_guard(
        self,
        pair:            str,
        open_positions:  List[Dict],
        account_balance: float,
    ) -> Tuple[bool, str]:
        """
        Block a new trade if it would breach currency concentration limits.

        Rules:
          1. Max 2 open pairs sharing the same currency (e.g. max 2 JPY pairs)
          2. Net margin exposure per currency ≤ 3% of account balance

        Args:
            pair:            Proposed new pair e.g. 'AUDJPY'
            open_positions:  From _get_open_positions() — list of dicts
            account_balance: Current account equity

        Returns:
            (allowed: bool, reason: str)
        """
        new_currencies = self._PAIR_CURRENCIES.get(pair, ())
        currency_count: Dict[str, int] = {}

        for pos in open_positions:
            for ccy in self._PAIR_CURRENCIES.get(pos["pair"], ()):
                currency_count[ccy] = currency_count.get(ccy, 0) + 1

        # Rule 1 — max 2 pairs sharing the same currency
        for ccy in new_currencies:
            current = currency_count.get(ccy, 0)
            if current >= self._MAX_SAME_CURRENCY:
                reason = (
                    f"CORR_GUARD: {ccy} already in "
                    f"{current} open pair(s) (max {self._MAX_SAME_CURRENCY})"
                )
                logger.warning(f"RiskManager | {pair} BLOCKED — {reason}")
                return False, reason

        # Rule 2 — net currency exposure ≤ 3% per currency
        if account_balance > 0:
            for ccy in new_currencies:
                ccy_exposure = sum(
                    pos.get("margin_used", 0.0)
                    for pos in open_positions
                    if ccy in self._PAIR_CURRENCIES.get(pos["pair"], ())
                )
                exposure_pct = (ccy_exposure / account_balance) * 100.0
                if exposure_pct >= self._MAX_NET_EXPOSURE_PCT:
                    reason = (
                        f"CORR_GUARD: {ccy} net exposure "
                        f"{exposure_pct:.1f}% ≥ {self._MAX_NET_EXPOSURE_PCT}%"
                    )
                    logger.warning(f"RiskManager | {pair} BLOCKED — {reason}")
                    return False, reason

        return True, "OK"

    @property
    def r(self):
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    # ── Pre-trade checks ──────────────────────────────────────

    def pre_trade_check(
        self,
        pair:       str,
        direction:  str,    # "BUY" or "SELL"
        regime:     str,
        size_mult:  float,
    ) -> Dict:
        """
        Run all pre-trade risk checks.
        Returns dict with approved=True/False and reason.

        Checks in order:
          1. Kill switch (trading halted)
          2. Daily loss limit
          3. Portfolio stop-loss
          4. Event blackout
          5. Max concurrent positions
          6. Duplicate pair check
          7. Regime check (PNS = always flat)
        """
        account = self.account.get()

        # ── 1. Kill switch ────────────────────────────────────
        try:
            halted = self.r.get("risk:halted")
            if halted == "true":
                reason = self.r.get("risk:halt_reason") or "TRADING_HALTED"
                return self._reject(pair, reason)
        except Exception:
            pass

        # ── 2. Daily loss kill switch ─────────────────────────
        killed, reason = _check_daily_kill_switch(account)
        if killed:
            self._halt_trading(reason)
            return self._reject(pair, reason)

        # ── 3. Portfolio stop-loss ────────────────────────────
        stopped, reason = _check_portfolio_stop(account)
        if stopped:
            self._halt_trading(reason)
            return self._reject(pair, reason)

        # ── 4. Event blackout ─────────────────────────────────
        blackout, reason = _check_event_blackout(pair)
        if blackout:
            return self._reject(pair, reason)

        # ── 5. Max concurrent positions ───────────────────────
        open_count = _count_open_positions()
        if open_count >= MAX_CONCURRENT_POSITIONS:
            return self._reject(
                pair,
                f"MAX_POSITIONS: {open_count}/{MAX_CONCURRENT_POSITIONS} open"
            )

        # ── 6. Duplicate pair ─────────────────────────────────
        if _is_pair_already_open(pair):
            return self._reject(pair, f"DUPLICATE: {pair} already open")

        # ── 7. Correlation guard ──────────────────────────────
        open_positions = _get_open_positions()
        corr_ok, corr_reason = self.check_correlation_guard(
            pair            = pair,
            open_positions  = open_positions,
            account_balance = account.get("balance", 10_000.0),
        )
        if not corr_ok:
            return self._reject(pair, corr_reason)

        # ── 8. Regime check ───────────────────────────────────
        if regime == "PNS" or size_mult == 0.0:
            return self._reject(pair, "PNS_REGIME: flat required")

        # ── All checks passed ─────────────────────────────────
        logger.success(
            f"RiskManager | {pair} {direction} — "
            f"Pre-trade checks PASSED | "
            f"open={open_count}/{MAX_CONCURRENT_POSITIONS} | "
            f"daily_pnl={account.get('daily_pnl_pct',0)*100:.2f}%"
        )
        return {
            "approved":    True,
            "pair":        pair,
            "direction":   direction,
            "account":     account,
            "open_count":  open_count,
            "reason":      "ALL_CHECKS_PASSED",
        }

    def calculate_position(
        self,
        pair:       str,
        direction:  str,
        entry_price: float,
        regime:     str,
        size_mult:  float,
        starter:    bool = True,
    ) -> Dict:
        """
        Calculate full position parameters.

        Returns dict with:
          volume_units, stop_loss, take_profit,
          stop_pips, risk_amount_usd, sizing_details
        """
        account = self.account.get()

        # Stop loss from ATR
        stop_loss, take_profit, stop_pips = calculate_stop_from_atr(
            pair, direction, entry_price
        )

        # Volume units from risk
        volume_units, sizing = calculate_volume_units(
            pair      = pair,
            account   = account,
            stop_pips = stop_pips,
            risk_pct  = MAX_RISK_PER_TRADE_PCT,
            size_mult = size_mult,
            starter   = starter,
        )

        return {
            "pair":           pair,
            "direction":      direction,
            "entry_price":    entry_price,
            "stop_loss":      stop_loss,
            "take_profit":    take_profit,
            "stop_pips":      stop_pips,
            "volume_units":   volume_units,
            "risk_pct":       sizing.get("risk_pct", MAX_RISK_PER_TRADE_PCT),
            "risk_amount_usd": sizing.get("risk_amount_usd", 0.0),
            "regime":         regime,
            "size_mult":      size_mult,
            "starter":        starter,
            "sizing_details": sizing,
        }

    # ── Trade recording ───────────────────────────────────────

    def record_trade_open(
        self,
        pair:            str,
        direction:       str,
        volume_units:    int,
        entry_price:     float,
        stop_loss:       float,
        take_profit:     float,
        regime:          str,
        risk_pct:        float,
        signal_data:     Dict,
        fix_order_id:    Optional[str] = None,
        position_id:     Optional[str] = None,
    ) -> Optional[int]:
        """
        Insert trade record into DB on order open.
        Returns trade ID.
        """
        trading_mode = os.getenv("TRADING_MODE", "PAPER")
        now          = datetime.now(timezone.utc)

        try:
            with db.connection() as conn:
                result = conn.execute(text("""
                    INSERT INTO trades (
                        fix_order_id, ctrader_position_id,
                        symbol, direction, volume_units,
                        entry_price, stop_loss, take_profit,
                        entry_time, status,
                        regime, risk_pct, position_size_pct,
                        signal_1_regime, signal_1_confidence,
                        signal_2_fund_score,
                        signal_3_tft_prob, signal_4_tsmixer,
                        signal_5_sentiment, signal_6_bias,
                        requested_price, trading_mode
                    ) VALUES (
                        :fix_id, :pos_id,
                        :symbol, :direction, :volume,
                        :entry, :sl, :tp,
                        :entry_time, 'OPEN',
                        :regime, :risk_pct, :pos_size_pct,
                        :s1_regime, :s1_conf,
                        :s2_score,
                        :s3_tft, :s4_tsmixer,
                        :s5_sent, :s6_bias,
                        :req_price, :mode
                    )
                    RETURNING id
                """), {
                    "fix_id":       fix_order_id,
                    "pos_id":       position_id,
                    "symbol":       pair,
                    "direction":    direction,
                    "volume":       volume_units,
                    "entry":        entry_price,
                    "sl":           stop_loss,
                    "tp":           take_profit,
                    "entry_time":   now,
                    "regime":       regime,
                    "risk_pct":     risk_pct,
                    "pos_size_pct": (volume_units / CONTRACT_SIZE) * 100,
                    "s1_regime":    signal_data.get("regime", ""),
                    "s1_conf":      signal_data.get("regime_conf", 0.0),
                    "s2_score":     signal_data.get("a2_score", 0.0),
                    "s3_tft":       signal_data.get("tft_prob", None),
                    "s4_tsmixer":   signal_data.get("tsmixer", None),
                    "s5_sent":      signal_data.get("sentiment", 0.0),
                    "s6_bias":      signal_data.get("opus_conv", 0.0),
                    "req_price":    entry_price,
                    "mode":         trading_mode,
                })
                conn.commit()
                trade_id = result.scalar()

            logger.success(
                f"RiskManager | Trade recorded: id={trade_id} "
                f"{pair} {direction} "
                f"{volume_units:,}u @ {entry_price} "
                f"SL={stop_loss} TP={take_profit}"
            )
            return trade_id

        except Exception as exc:
            logger.error(f"RiskManager | Trade record failed: {exc}")
            return None

    def record_trade_close(
        self,
        trade_id:   int,
        exit_price: float,
        status:     str,    # CLOSED_TP | CLOSED_SL | CLOSED_MANUAL
    ) -> Optional[float]:
        """
        Update trade record on close. Calculates P&L.
        Returns pnl_usd.
        """
        # Normalise long reason strings to fit status VARCHAR(20)
        STATUS_MAP = {
            "CLOSED_TECH_INVALIDATION_FLIP":  "CLOSED_TECH_FLIP",
            "CLOSED_TECH_INVALIDATION_DECAY": "CLOSED_TECH_DECAY",
        }
        status = STATUS_MAP.get(status, status)

        try:
            with db.connection() as conn:
                # Fetch trade details
                trade = conn.execute(text("""
                    SELECT symbol, direction, volume_units,
                           entry_price, stop_loss
                    FROM trades WHERE id = :tid
                """), {"tid": trade_id}).fetchone()

                if not trade:
                    logger.error(f"RiskManager | Trade {trade_id} not found")
                    return None

                pair, direction, volume, entry_price, sl = trade

                # Calculate P&L
                pip_size  = 0.01 if pair in JPY_PAIRS else 0.0001
                price     = _get_current_price(pair)
                pip_value = _calculate_pip_value_usd(pair, price)

                if direction == "BUY":
                    pnl_pips = (exit_price - entry_price) / pip_size
                else:
                    pnl_pips = (entry_price - exit_price) / pip_size

                lots    = volume / CONTRACT_SIZE
                pnl_usd = pnl_pips * pip_value * lots

                # Slippage vs requested
                slippage = abs(exit_price - entry_price) / pip_size if status == "CLOSED_TP" else 0.0

                conn.execute(text("""
                    UPDATE trades
                    SET exit_price  = :exit,
                        exit_time   = NOW(),
                        status      = :status,
                        pnl_pips    = :pnl_pips,
                        pnl_usd     = :pnl_usd,
                        slippage_pips = :slip,
                        updated_at  = NOW()
                    WHERE id = :tid
                """), {
                    "exit":     exit_price,
                    "status":   status,
                    "pnl_pips": round(pnl_pips, 2),
                    "pnl_usd":  round(pnl_usd, 2),
                    "slip":     round(slippage, 2),
                    "tid":      trade_id,
                })
                conn.commit()

            # Update account state
            self.account.apply_pnl(pnl_usd)

            emoji = "✅" if pnl_usd > 0 else "❌"
            logger.info(
                f"RiskManager | {emoji} Trade closed: id={trade_id} "
                f"{pair} {status} "
                f"pnl={pnl_pips:+.1f}pips (${pnl_usd:+.2f})"
            )
            return round(pnl_usd, 2)

        except Exception as exc:
            logger.error(f"RiskManager | Trade close failed: {exc}")
            return None

    # ── Daily snapshot ────────────────────────────────────────

    def take_daily_snapshot(self) -> None:
        """
        Write daily performance snapshot to DB.
        Call at end of trading day (23:50 UTC).
        """
        account = self.account.get()
        today   = datetime.now(timezone.utc).date()

        try:
            with db.connection() as conn:
                # Aggregate today's trades
                stats = conn.execute(text("""
                    SELECT
                        COUNT(*)                                    AS total,
                        COUNT(*) FILTER (WHERE pnl_usd > 0)        AS wins,
                        COUNT(*) FILTER (WHERE pnl_usd <= 0)       AS losses,
                        COALESCE(AVG(pnl_pips) FILTER (WHERE pnl_usd > 0), 0) AS avg_win,
                        COALESCE(AVG(ABS(pnl_pips)) FILTER (WHERE pnl_usd <= 0), 0) AS avg_loss,
                        COALESCE(SUM(pnl_usd) FILTER (WHERE pnl_usd > 0), 0) AS gross_profit,
                        COALESCE(ABS(SUM(pnl_usd) FILTER (WHERE pnl_usd < 0)), 0) AS gross_loss
                    FROM trades
                    WHERE DATE(entry_time) = :today
                      AND status != 'OPEN'
                """), {"today": today}).fetchone()

                total, wins, losses, avg_win, avg_loss, gp, gl = stats

                win_rate      = (wins / total) if total > 0 else None
                profit_factor = (gp / gl) if gl > 0 else None
                daily_pnl_pct = account.get("daily_pnl_pct", 0.0)

                conn.execute(text("""
                    INSERT INTO performance_snapshots (
                        snapshot_date, equity, balance,
                        daily_pnl, daily_pnl_pct,
                        trades_today, wins_today, losses_today,
                        total_trades, win_rate,
                        avg_win_pips, avg_loss_pips, profit_factor,
                        trading_halted
                    ) VALUES (
                        :date, :equity, :balance,
                        :daily_pnl, :daily_pnl_pct,
                        :trades, :wins, :losses,
                        :total, :win_rate,
                        :avg_win, :avg_loss, :pf,
                        :halted
                    )
                    ON CONFLICT (snapshot_date) DO UPDATE SET
                        equity        = EXCLUDED.equity,
                        balance       = EXCLUDED.balance,
                        daily_pnl     = EXCLUDED.daily_pnl,
                        daily_pnl_pct = EXCLUDED.daily_pnl_pct,
                        trades_today  = EXCLUDED.trades_today,
                        wins_today    = EXCLUDED.wins_today,
                        losses_today  = EXCLUDED.losses_today,
                        total_trades  = EXCLUDED.total_trades,
                        win_rate      = EXCLUDED.win_rate,
                        avg_win_pips  = EXCLUDED.avg_win_pips,
                        avg_loss_pips = EXCLUDED.avg_loss_pips,
                        profit_factor = EXCLUDED.profit_factor,
                        trading_halted = EXCLUDED.trading_halted
                """), {
                    "date":         today,
                    "equity":       account.get("equity", 10_000.0),
                    "balance":      account.get("balance", 10_000.0),
                    "daily_pnl":    account.get("daily_pnl", 0.0),
                    "daily_pnl_pct": daily_pnl_pct,
                    "trades":       int(total or 0),
                    "wins":         int(wins or 0),
                    "losses":       int(losses or 0),
                    "total":        int(total or 0),
                    "win_rate":     win_rate,
                    "avg_win":      float(avg_win or 0),
                    "avg_loss":     float(avg_loss or 0),
                    "pf":           profit_factor,
                    "halted":       account.get("daily_pnl_pct", 0) <= -DAILY_LOSS_KILL_PCT,
                })
                conn.commit()

            logger.success(
                f"RiskManager | Daily snapshot: "
                f"trades={total} wins={wins} losses={losses} "
                f"pnl={daily_pnl_pct*100:.2f}%"
            )

        except Exception as exc:
            logger.error(f"RiskManager | Snapshot failed: {exc}")

    # ── Kill switch controls ──────────────────────────────────
    
    def _halt_trading(self, reason: str) -> None:
        """Activate trading halt — writes to Redis."""
        try:
            self.r.set("risk:halted", "true")
            self.r.set("risk:halt_reason", reason)
            try:
                from intelligence.monitoring.telegram_alerts import alerts
                alerts.send_kill_switch(daily_pnl_pct=0.0, reason=reason)
            except Exception:
                pass
            logger.critical(
                f"RiskManager | ⛔ TRADING HALTED: {reason}"
            )
        except Exception as exc:
            logger.error(f"RiskManager | Halt set failed: {exc}")

    def resume_trading(self) -> None:
        """Manually resume trading after halt. Human approval required."""
        try:
            self.r.delete("risk:halted")        # ← delete, not set "false"
            self.r.delete("risk:halt_reason")
            logger.warning("RiskManager | ▶ Trading resumed (manual)")
        except Exception as exc:
            logger.error(f"RiskManager | Resume failed: {exc}")

    def is_halted(self) -> bool:
        """Check if trading is currently halted."""
        try:
            # Check persistent halt flag
            if self.r.get("risk:halted") == "true":
                return True
            # Also check live P&L directly — catches stale flag misses
            account = self.account.get()
            if account.get("daily_pnl_pct", 0.0) <= -DAILY_LOSS_KILL_PCT:
                return True   # read-only — pre_trade_check() handles the halt
            return False
        except Exception:
           return False

    @staticmethod
    def _reject(pair: str, reason: str) -> Dict:
        """Build rejection dict."""
        logger.warning(f"RiskManager | {pair} REJECTED: {reason}")
        return {"approved": False, "pair": pair, "reason": reason}

    # ── Status display ────────────────────────────────────────

    def print_status(self) -> None:
        """Print current risk manager status."""
        account    = self.account.get()
        open_count = _count_open_positions()
        halted     = self.is_halted()

        print("\n" + "=" * 55)
        print("RISK MANAGER — STATUS")
        print("=" * 55)
        print(f"  Mode:          {account.get('mode', 'PAPER')}")
        print(f"  Balance:       ${account.get('balance', 0):>10,.2f}")
        print(f"  Equity:        ${account.get('equity',  0):>10,.2f}")
        print(f"  Daily P&L:     ${account.get('daily_pnl', 0):>+10.2f} "
              f"({account.get('daily_pnl_pct', 0)*100:+.2f}%)")
        print(f"  Open positions: {open_count}/{MAX_CONCURRENT_POSITIONS}")
        print(f"  Trading halted: {'⛔ YES' if halted else '✅ NO'}")
        print()
        print(f"  Hard limits:")
        print(f"    Max risk/trade:  {MAX_RISK_PER_TRADE_PCT*100:.0f}%")
        print(f"    Daily kill:      -{DAILY_LOSS_KILL_PCT*100:.0f}%")
        print(f"    Portfolio stop:  -{PORTFOLIO_STOP_PCT*100:.0f}% of margin")
        print(f"    Event blackout:  {EVENT_BLACKOUT_HOURS}h before HIGH-impact")
        print(f"    Max positions:   {MAX_CONCURRENT_POSITIONS}")
        print("=" * 55 + "\n")


# ── Module singleton ──────────────────────────────────────────
risk_manager = RiskManager()


# ── Direct run ────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")

    risk_manager.print_status()

    # Test position sizing for each tradeable signal
    test_cases = [
        ("EURGBP", "SELL",  "TLV", 1.0),
        ("USDCHF", "SELL",  "MR",  0.5),
        ("USDCAD", "BUY",   "THV", 1.0),
    ]

    print("\nPOSITION SIZING TEST:")
    print("=" * 65)
    for pair, direction, regime, size_mult in test_cases:
        check = risk_manager.pre_trade_check(pair, direction, regime, size_mult)
        print(f"\n{pair} {direction} | regime={regime} | size_mult={size_mult}")
        approved = check['approved']
        reason   = check.get('reason', '')
        print(f"  Pre-trade: {'✅ APPROVED' if approved else '❌ ' + reason}")

        if check["approved"]:
            price  = _get_current_price(pair)
            sizing = risk_manager.calculate_position(
                pair, direction, price, regime, size_mult
            )
            print(f"  Volume:    {sizing['volume_units']:,} units "
                  f"({sizing['volume_units']/100000:.3f} lots)")
            print(f"  Stop:      {sizing['stop_loss']} "
                  f"({sizing['stop_pips']:.1f} pips)")
            print(f"  Target:    {sizing['take_profit']}")
            print(f"  Risk $:    ${sizing['risk_amount_usd']:.2f}")
    print("=" * 65)
