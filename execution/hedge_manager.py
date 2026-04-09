"""
execution/hedge_manager.py
==========================
Perpetual Hedge Manager — MR Regime Range Strategy

Philosophy
----------
In Mean Reversion (MR) regime, price oscillates between S/R walls — it does not
trend. Opening a single-direction trade in MR is the primary cause of 22% WR in
that regime. This module opens BOTH a BUY and SELL leg simultaneously, capturing
the oscillation.

Cycle mechanic (every M15 bar):
  Price approaches resistance → bank BUY leg profit, immediately re-open BUY at
  resistance price. SELL leg continues. Net: BUY re-anchored at a better price,
  never 1-sided, profit banked.

  Price approaches support → bank SELL leg profit, immediately re-open SELL at
  support price. BUY leg continues.

Regime conversion:
  A1 shifts to THV or TLV (2 consecutive M15 bars confirmed):
    - Determine keep direction from A2 bias (LONG_BASE → keep BUY, else keep SELL)
    - If nearest S/R wall is within 30 pips: wait for price to test wall then pull
      back before closing opposing leg (better exit price)
    - Close opposing leg → surviving leg handed to trailing stop via execution:open:{pair}

Safety rules:
  - Both legs always open in ACTIVE state — never 1-sided
  - SL on each leg: ATR × 2.0 (wider, protects extreme moves)
  - Max 6 cycles per hedge position (prevents runaway re-entries)
  - News gate: no cycling within 2h of HIGH-impact events
  - Max 3 simultaneous hedged pairs

Redis keys:
  hedge:active:{PAIR}   JSON — full hedge state, TTL=24h

DB table: hedge_positions (auto-created on first open)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis
from loguru import logger
from sqlalchemy import text

ROOT_PATH = "/workspaces/Trading_bot"
sys.path.insert(0, ROOT_PATH)

from dotenv import load_dotenv
load_dotenv(f"{ROOT_PATH}/.env")

from data_pipeline.storage.timeseries_db import db

# ── Constants ──────────────────────────────────────────────────────────────────

MR_CONFIDENCE_MIN      = 0.60   # A1 MR probability threshold to open hedge
SR_RANGE_MIN_PIPS      = 25     # minimum resistance+support range (spread viability)
SR_OPEN_MIN_PIPS       = 10     # don't open hedge if already within N pips of a wall
LEG_ORDER_SLEEP        = 1.5    # seconds between leg orders (prevent WebSocket response mixing)
CYCLE_SKIP_BARS        = 1      # skip cycling for N M15 bars after hedge opens (15 min)
CYCLE_PIPS             = 5      # close leg when price within N pips of S/R wall
CONVERSION_SR_PIPS     = 30     # if S/R within N pips → wait for wall test
CONVERSION_PULLBACK    = 8      # pips of retreat from S/R before converting
REGIME_CONFIRM_BARS    = 2      # consecutive non-MR bars before converting
MAX_CYCLE_COUNT        = 6      # max leg cycles per hedge position
ATR_STOP_MULT          = 2.0    # SL = entry ± ATR × 2.0
ATR_TP_SAFETY_MULT     = 12.0   # wide safety-net TP (normally not hit)
RISK_PER_LEG_PCT       = 0.01   # 1% risk per leg → total 2% = same as normal trade
MAX_CONCURRENT_HEDGES  = 3
CACHE_TTL              = 86_400  # 24h

# ── Profit preservation ────────────────────────────────────────────────────────
PROFIT_TARGET_MULT     = 2.0    # close both legs when realized_pnl ≥ 2× initial risk
HEDGE_MAX_HOURS        = 48     # time stop: close if open > 48h AND in profit
SL_WALK_FRACTION       = 0.33   # after each cycle, walk other leg's SL 1/3 toward breakeven

# ── Asymmetric exit (WINNER_RUNNING) ──────────────────────────────────────────
WINNER_THRESHOLD_MULT  = 1.0    # |price − entry| ≥ ATR × 1.0 → switch to WINNER_RUNNING
LOSER_INVALIDATE_MULT  = 1.5    # move ≥ ATR × 1.5 against loser → close + deferred re-entry
LOSER_WALK_FRAC        = 0.50   # aggressive SL walk per check in WINNER_RUNNING mode
REENTRY_WATCH_PIPS     = 3      # enter when price extends N more pips in extending direction
REENTRY_TTL_MIN        = 45     # minutes: force market re-entry if watch not reached by then
WINNER_TRAIL_PIPS      = 5      # trail winning solo leg at 5 pips behind price

# ── Wash hedge constants ───────────────────────────────────────────────────────
MIN_CYCLE_PROFIT_PIPS  = 5      # min pip gain from leg's entry before cycling is allowed
MAX_WASH_LOSS_USD      = 25.0   # wash mode hard stop — close if net hedge loss > $25

# ── Cascade pyramid constants ─────────────────────────────────────────────────
MAX_PYRAMID_TIERS      = 2      # max extra tiers beyond the base pair (0 = disabled)
PYRAMID_SIZE_FRACTION  = 0.50   # each tier volume = previous × 0.50 (Tier1=50%, Tier2=25%)
PYRAMID_MIN_PROFIT_USD = 1.0    # min cycle profit to trigger a new tier
PYRAMID_CLUSTER_STOP   = 40.0   # hard stop: close ALL tiers if net cluster loss > $40

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

def _round_price(pair: str, price: float) -> float:
    """Round price to cTrader-accepted decimal places (JPY=3, others=5)."""
    decimals = 3 if pair in JPY_PAIRS else 5
    return round(price, decimals)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else 0.0001


def _redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


def _get_current_price(pair: str, r: redis.Redis) -> Optional[float]:
    raw = r.get(f"tick:{pair}")
    if not raw:
        return None
    try:
        d   = json.loads(raw)
        bid = float(d.get("bid", 0))
        ask = float(d.get("ask", 0))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return bid or None
    except Exception:
        return None


def _get_atr(pair: str, r: redis.Redis) -> Optional[float]:
    raw = r.get(f"features:stat:{pair}")
    if not raw:
        return None
    try:
        return float(json.loads(raw).get("atr_14_m15", 0)) or None
    except Exception:
        return None


def _get_sr(pair: str, r: redis.Redis) -> Optional[Dict]:
    raw = r.get(f"features:sr:{pair}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _get_account_balance(r: redis.Redis) -> float:
    raw = r.get("risk:account_state")
    if raw:
        try:
            return float(json.loads(raw).get("balance", 2000.0))
        except Exception:
            pass
    return 2000.0


def _calc_net_hedge_pnl(pair: str, state: Dict, price: float) -> float:
    """
    Total hedge PnL = realized (banked from cycles) + unrealized (both open legs).
    Used by wash mode to decide whether to close or keep cycling.
    """
    pip     = _pip_size(pair)
    vol     = state.get("volume_per_leg", 1000)
    pip_val = _pip_value_per_unit(pair, price)
    realized = float(state.get("realized_pnl", 0.0))

    buy_upnl = sell_upnl = 0.0
    if state.get("buy_position_id"):
        b_entry  = float(state.get("buy_entry",  state.get("entry_price", price)))
        buy_upnl = (price - b_entry) / pip * pip_val * vol
    if state.get("sell_position_id"):
        s_entry   = float(state.get("sell_entry", state.get("entry_price", price)))
        sell_upnl = (s_entry - price) / pip * pip_val * vol

    return round(realized + buy_upnl + sell_upnl, 2)


def _pip_value_per_unit(pair: str, price: float) -> float:
    """USD value of 1 pip movement for 1 unit of the pair."""
    pip = _pip_size(pair)
    usd_quote = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURGBP"}
    usd_base  = {"USDJPY", "USDCAD", "USDCHF"}
    if pair in usd_quote:
        return pip
    elif pair in usd_base or pair in JPY_PAIRS:
        return pip / price if price > 0 else pip
    return pip


def _calc_cluster_net_pnl(pair: str, state: Dict, price: float) -> float:
    """
    Net PnL across the base hedge pair + all pyramid tiers.
    = base realized + base unrealized + sum of all tier unrealized.
    """
    base_pnl = _calc_net_hedge_pnl(pair, state, price)
    pip      = _pip_size(pair)
    pip_val  = _pip_value_per_unit(pair, price)

    tier_upnl = 0.0
    for tier in state.get("pyramid_tiers", []):
        vol   = tier.get("volume", 0)
        entry = float(tier.get("entry_price", price))
        if tier.get("buy_position_id"):
            tier_upnl += (price - entry) / pip * pip_val * vol
        if tier.get("sell_position_id"):
            tier_upnl += (entry - price) / pip * pip_val * vol

    return round(base_pnl + tier_upnl, 2)


def _calc_volume(pair: str, price: float, atr: float, balance: float) -> int:
    """
    Volume per hedge leg = 1% risk / (stop_pips × pip_value_per_unit).
    Stop = ATR × ATR_STOP_MULT.  Minimum 1000 units.
    """
    pip          = _pip_size(pair)
    risk_amount  = balance * RISK_PER_LEG_PCT
    stop_pips    = (atr * ATR_STOP_MULT) / pip

    # pip value per unit in USD (approximate)
    usd_quote = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURGBP"}
    usd_base  = {"USDJPY", "USDCAD", "USDCHF"}
    if pair in usd_quote:
        pip_val_per_unit = pip
    elif pair in usd_base:
        pip_val_per_unit = pip / price if price > 0 else pip
    elif pair in JPY_PAIRS:
        pip_val_per_unit = pip / price if price > 0 else pip
    else:
        pip_val_per_unit = pip

    if stop_pips <= 0 or pip_val_per_unit <= 0:
        return 1000

    raw = risk_amount / (stop_pips * pip_val_per_unit)
    return max(1000, int(raw / 1000) * 1000)


# ══════════════════════════════════════════════════════════════════════════════
# HEDGE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class HedgeManager:
    """
    Perpetual hedge position manager.

    Singleton — import and call:
        from execution.hedge_manager import hedge_manager
        hedge_manager.consider_opening("EURUSD")
        hedge_manager.check_all()
    """

    def __init__(self) -> None:
        self._table_ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def consider_opening(self, pair: str) -> bool:
        """
        Check all conditions and open hedge if appropriate.
        Called from main_loop every M15 bar for each pair.
        Returns True if hedge was opened.
        """
        r = _redis_client()

        # Already hedging this pair?
        if r.get(f"hedge:active:{pair}"):
            return False

        # Directional position already open?
        if r.get(f"execution:open:{pair}"):
            return False

        # Kill switch
        if r.get("risk:halted") == "true":
            return False

        # Regime check — must be MR with sufficient confidence
        regime_raw = r.get(f"features:regime:{pair}")
        if not regime_raw:
            return False
        try:
            regime_data = json.loads(regime_raw)
            regime      = regime_data.get("regime", "")
            mr_conf     = float(regime_data.get("probabilities", {}).get("MR", 0))
        except Exception:
            return False

        if regime != "MR" or mr_conf < MR_CONFIDENCE_MIN:
            return False

        # Event blackout
        ev_raw = r.get(f"features:event:{pair}")
        if ev_raw:
            try:
                if json.loads(ev_raw).get("high_impact_within_2h"):
                    return False
            except Exception:
                pass

        # S/R range must be large enough to be worth hedging.
        # Use STRUCTURAL walls (swing/round/pivot) not cycling walls (EMAs) —
        # EMAs can be 2-5 pips away and would produce absurdly tight ranges.
        sr = _get_sr(pair, r)
        if not sr:
            return False
        dist_res = float(sr.get("structural_distance_to_resistance",
                                sr.get("distance_to_resistance", 0)))
        dist_sup = float(sr.get("structural_distance_to_support",
                                sr.get("distance_to_support", 0)))
        if (dist_res + dist_sup) < SR_RANGE_MIN_PIPS:
            logger.debug(
                f"HedgeManager | {pair}: structural S/R range "
                f"{dist_res+dist_sup:.1f}pip < {SR_RANGE_MIN_PIPS}pip, skip"
            )
            return False

        # Don't open if price is already sitting on a structural wall
        if dist_res < SR_OPEN_MIN_PIPS or dist_sup < SR_OPEN_MIN_PIPS:
            logger.debug(
                f"HedgeManager | {pair}: too close to structural S/R wall "
                f"(res={dist_res:.1f}pip sup={dist_sup:.1f}pip) — wait for retracement"
            )
            return False

        price = _get_current_price(pair, r)
        atr   = _get_atr(pair, r)
        if not price or not atr:
            return False

        # Ensure SL covers at least 1.5× the structural range (M15 ATR alone can be inside range)
        pip           = _pip_size(pair)
        sr_range_pips = dist_res + dist_sup
        min_atr       = (sr_range_pips * pip * 1.5) / ATR_STOP_MULT
        atr           = max(atr, min_atr)

        # Max concurrent hedge count
        active_count = sum(1 for _ in r.scan_iter("hedge:active:*"))
        if active_count >= MAX_CONCURRENT_HEDGES:
            logger.debug(f"HedgeManager | Max hedges ({active_count}) reached, skip {pair}")
            return False

        balance = _get_account_balance(r)
        volume  = _calc_volume(pair, price, atr, balance)

        logger.info(
            f"HedgeManager | Opening hedge {pair} | "
            f"MR_conf={mr_conf:.2f} | structural range={dist_res+dist_sup:.1f}pip | "
            f"vol={volume:,}u per leg"
        )
        return self._open_hedge(pair, price, atr, volume, mr_conf, r)

    def check_all(self) -> None:
        """
        Full hedge check: S/R cycling + regime conversion + liveness.
        Runs every M1 via hedge background thread (was M15).
        """
        r = _redis_client()
        for key in list(r.scan_iter("hedge:active:*")):
            pair = key.split(":")[-1]
            try:
                self._check_hedge(pair, r)
            except Exception as exc:
                logger.error(f"HedgeManager | check_all {pair} failed: {exc}")

    def check_liveness_all(self) -> None:
        """
        Lightweight liveness check only — no cycling, no API calls.
        Runs every 30s to detect SL hits and PARTIAL states quickly.
        Only reads Redis + DB; closes survivors if needed.
        """
        r = _redis_client()
        for key in list(r.scan_iter("hedge:active:*")):
            pair = key.split(":")[-1]
            try:
                raw = r.get(f"hedge:active:{pair}")
                if not raw:
                    continue
                state  = json.loads(raw)
                status = state.get("status", "ACTIVE")

                # Clean up dead hedges
                if status in ("CONVERTED", "CLOSED"):
                    r.delete(f"hedge:active:{pair}")
                    continue

                # Handle PARTIAL (survivor needs immediate close)
                if status == "PARTIAL":
                    self._check_hedge(pair, r)
                    continue

                if status != "ACTIVE":
                    continue

                # Check if either leg hit SL (DB status changed without us knowing)
                buy_trade_id  = state.get("buy_trade_id")
                sell_trade_id = state.get("sell_trade_id")
                if not buy_trade_id or not sell_trade_id:
                    continue

                with db.connection() as conn:
                    rows = conn.execute(text(
                        "SELECT id, status FROM trades WHERE id = ANY(:ids)"
                    ), {"ids": [buy_trade_id, sell_trade_id]}).fetchall()
                status_map = {r_[0]: r_[1] for r_ in rows}
                buy_alive  = status_map.get(buy_trade_id,  "OPEN") == "OPEN"
                sell_alive = status_map.get(sell_trade_id, "OPEN") == "OPEN"

                if not buy_alive or not sell_alive:
                    logger.warning(
                        f"HedgeManager | {pair}: liveness check (30s) — "
                        f"{'BUY' if not buy_alive else 'SELL'} leg SL hit, handling now"
                    )
                    self._check_hedge(pair, r)  # ACTIVE liveness block will close survivor

            except Exception as exc:
                logger.error(f"HedgeManager | check_liveness {pair} failed: {exc}")

    def get_active_hedges(self) -> List[Dict]:
        """Return all active hedge states."""
        r = _redis_client()
        result = []
        for key in r.scan_iter("hedge:active:*"):
            raw = r.get(key)
            if raw:
                try:
                    result.append(json.loads(raw))
                except Exception:
                    pass
        return result

    def close_all(self, reason: str = "MANUAL") -> None:
        """Emergency: close all active hedge positions."""
        for hedge in self.get_active_hedges():
            pair = hedge.get("pair")
            if pair:
                self._close_both_legs(pair, reason)

    # ── Core: open hedge ──────────────────────────────────────────────────────

    def _open_hedge(
        self,
        pair:       str,
        price:      float,
        atr:        float,
        volume:     int,
        confidence: float,
        r:          redis.Redis,
    ) -> bool:
        """Place both legs and store hedge state in Redis + DB."""
        self._ensure_table()

        from execution.order_executor import order_executor

        pip         = _pip_size(pair)
        sl_distance = atr * ATR_STOP_MULT
        tp_distance = atr * ATR_TP_SAFETY_MULT

        signal_data = {
            "strategy":   "HEDGE",
            "path":       "HEDGE_CYCLE",
            "regime":     "MR",
            "confidence": confidence,
            "source":     "hedge_manager",
        }

        # ── BUY leg ───────────────────────────────────────────────────────────
        buy_sl = _round_price(pair, price - sl_distance)
        buy_tp = _round_price(pair, price + tp_distance)

        buy_res = order_executor.place_market_order(
            pair=pair, direction="BUY",
            volume_units=volume,
            stop_loss=buy_sl, take_profit=buy_tp,
            signal_data=signal_data,
            regime="MR", risk_pct=RISK_PER_LEG_PCT,
        )
        if not buy_res.get("success"):
            logger.error(f"HedgeManager | {pair} BUY leg rejected: {buy_res.get('error')}")
            return False

        # Remove execution:open key — hedge_manager owns tracking, not execution
        r.delete(f"execution:open:{pair}")
        time.sleep(LEG_ORDER_SLEEP)  # wait for WebSocket to flush response before next order

        # ── SELL leg ──────────────────────────────────────────────────────────
        sell_sl = _round_price(pair, price + sl_distance)
        sell_tp = _round_price(pair, price - tp_distance)

        sell_res = order_executor.place_market_order(
            pair=pair, direction="SELL",
            volume_units=volume,
            stop_loss=sell_sl, take_profit=sell_tp,
            signal_data=signal_data,
            regime="MR", risk_pct=RISK_PER_LEG_PCT,
        )
        if not sell_res.get("success"):
            logger.error(f"HedgeManager | {pair} SELL leg rejected: {sell_res.get('error')}")
            # Roll back BUY leg
            buy_pid = buy_res.get("position_id")
            if buy_pid:
                order_executor.close_position(pair, str(buy_pid), volume, "HEDGE_OPEN_FAIL")
            return False

        r.delete(f"execution:open:{pair}")

        # Extra sleep: prevents WebSocket fill responses from the SELL leg
        # being cross-assigned to the next hedge's BUY order when multiple
        # hedges open in the same M15 cycle (race condition on shared listener).
        time.sleep(LEG_ORDER_SLEEP * 2)

        # ── Store hedge state ─────────────────────────────────────────────────
        # initial_risk_usd: used for profit target (2× initial risk = close)
        pip_val = _pip_value_per_unit(pair, price)
        sl_pips = (sl_distance / pip) if pip > 0 else 0
        initial_risk_usd = round(sl_pips * pip_val * volume, 4)

        state: Dict[str, Any] = {
            "pair":              pair,
            "status":            "ACTIVE",
            "buy_position_id":   buy_res.get("position_id"),
            "sell_position_id":  sell_res.get("position_id"),
            "buy_trade_id":      buy_res.get("trade_db_id"),
            "sell_trade_id":     sell_res.get("trade_db_id"),
            "volume_per_leg":    volume,
            "entry_price":       round(price, 6),
            "atr_at_open":       round(atr, 6),
            "buy_sl":            buy_sl,
            "sell_sl":           sell_sl,
            "initial_risk_usd":  initial_risk_usd,
            "opened_at":         datetime.now(timezone.utc).isoformat(),
            "cycle_count":       0,
            "realized_pnl":      0.0,
            "hedge_mode":        "CYCLING",   # "CYCLING" | "WINNER_RUNNING"
            "winning_leg":       None,        # "BUY" | "SELL" — set in WINNER_RUNNING
            "pending_reentry":   None,        # {direction,watch_price,expires_at,volume}
            "regime_shift_bars": 0,
            "converting_to":     None,
            # Per-leg entry prices (updated after each cycle re-open)
            "buy_entry":         round(price, 6),
            "sell_entry":        round(price, 6),
            # Wash mode (activated when MAX_CYCLES reached with net negative PnL)
            "wash_mode":         False,
            # Cascade pyramid tiers (list of extra BUY+SELL pairs opened at each TP cycle)
            "pyramid_tiers":     [],
        }
        r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)

        # DB record
        try:
            with db.connection() as conn:
                conn.execute(text("""
                    INSERT INTO hedge_positions
                        (pair, status, buy_trade_id, sell_trade_id,
                         buy_position_id, sell_position_id, volume_per_leg)
                    VALUES
                        (:pair, 'ACTIVE', :bt, :st, :bp, :sp, :vol)
                """), {
                    "pair": pair,
                    "bt":   buy_res.get("trade_db_id"),
                    "st":   sell_res.get("trade_db_id"),
                    "bp":   buy_res.get("position_id"),
                    "sp":   sell_res.get("position_id"),
                    "vol":  volume,
                })
                conn.execute(text("COMMIT"))
        except Exception as exc:
            logger.warning(f"HedgeManager | {pair} DB insert failed: {exc}")

        logger.success(
            f"HedgeManager | ✅ {pair} hedge open | "
            f"BUY pos={buy_res.get('position_id')} | "
            f"SELL pos={sell_res.get('position_id')} | "
            f"vol={volume:,}u each | SL={sl_distance/pip:.0f}pip"
        )
        return True

    # ── Core: M15 check ───────────────────────────────────────────────────────

    def _check_hedge(self, pair: str, r: redis.Redis) -> None:
        """Full M15 check for one hedged pair."""
        raw = r.get(f"hedge:active:{pair}")
        if not raw:
            return

        state  = json.loads(raw)
        status = state.get("status", "ACTIVE")

        # Clean up fully converted/closed hedges
        if status in ("CONVERTED", "CLOSED"):
            r.delete(f"hedge:active:{pair}")
            return

        # ── PARTIAL: one leg gone (re-open failed after cycle) ───────────────────
        # Close the surviving leg immediately — an unhedged leg is just a naked bet.
        if status == "PARTIAL":
            buy_pos  = state.get("buy_position_id")
            sell_pos = state.get("sell_position_id")
            alive_leg = "SELL" if buy_pos is None else "BUY"
            alive_pos = sell_pos if buy_pos is None else buy_pos
            alive_vol = state.get("volume_per_leg", state.get("volume", 1000))
            logger.warning(
                f"HedgeManager | {pair}: PARTIAL state — {alive_leg} leg "
                f"(pos={alive_pos}) is naked, closing immediately"
            )
            if alive_pos:
                from execution.order_executor import order_executor
                # Set execution:open cache so close_position can update the DB trade record
                alive_tid = state.get("sell_trade_id") if alive_leg == "SELL" else state.get("buy_trade_id")
                if alive_tid:
                    import json as _json
                    r.setex(f"execution:open:{pair}", 30, _json.dumps({
                        "pair":        pair,
                        "direction":   alive_leg,
                        "volume":      alive_vol,
                        "entry_price": state.get("entry_price", 0),
                        "stop_loss":   0,
                        "take_profit": 0,
                        "position_id": alive_pos,
                        "trade_db_id": alive_tid,
                    }))
                order_executor.close_position(
                    pair=pair,
                    position_id=str(alive_pos),
                    volume=alive_vol,
                    reason="HEDGE_LEG_SL",
                )
            state["status"] = "CLOSED"
            r.set(f"hedge:active:{pair}", json.dumps(state), ex=3600)
            return

        price = _get_current_price(pair, r)
        sr    = _get_sr(pair, r)
        if not price or not sr:
            return

        # ── Regime shift check ────────────────────────────────────────────────
        regime_raw = r.get(f"features:regime:{pair}")
        if regime_raw:
            try:
                regime_data    = json.loads(regime_raw)
                current_regime = regime_data.get("regime", "MR")
            except Exception:
                current_regime = "MR"

            if current_regime in ("THV", "TLV") and status == "ACTIVE":
                a2_bias = "NEUTRAL"
                a2_raw  = r.get(f"features:analyst2:{pair}")
                if a2_raw:
                    try:
                        a2_bias = json.loads(a2_raw).get("bias", "NEUTRAL")
                    except Exception:
                        pass

                self._handle_regime_shift(pair, state, current_regime, a2_bias, price, sr, r)
                return  # don't cycle during regime shift assessment

        if status != "ACTIVE":
            return

        # ── Liveness check: verify both legs are still open in DB ────────────────
        # If cTrader hit a SL externally, the DB will show CLOSED_SL before
        # our Redis state knows — detect this and close the survivor immediately.
        buy_trade_id  = state.get("buy_trade_id")
        sell_trade_id = state.get("sell_trade_id")
        if buy_trade_id and sell_trade_id:
            try:
                with db.connection() as conn:
                    rows = conn.execute(text(
                        "SELECT id, status FROM trades WHERE id = ANY(:ids)"
                    ), {"ids": [buy_trade_id, sell_trade_id]}).fetchall()
                status_map = {r_[0]: r_[1] for r_ in rows}
                buy_alive  = status_map.get(buy_trade_id,  "OPEN") == "OPEN"
                sell_alive = status_map.get(sell_trade_id, "OPEN") == "OPEN"

                if not buy_alive or not sell_alive:
                    dead_leg     = "BUY"  if not buy_alive  else "SELL"
                    dead_trade   = buy_trade_id  if not buy_alive  else sell_trade_id
                    alive_leg    = "SELL" if not buy_alive  else "BUY"
                    alive_pos_id = state.get("sell_position_id") if not buy_alive else state.get("buy_position_id")
                    alive_vol    = state.get("volume_per_leg", state.get("volume", 1000))

                    logger.warning(
                        f"HedgeManager | {pair}: {dead_leg} leg (trade {dead_trade}) "
                        f"hit SL externally — closing {alive_leg} survivor"
                    )
                    if alive_pos_id:
                        from execution.order_executor import order_executor
                        order_executor.close_position(
                            pair=pair,
                            position_id=str(alive_pos_id),
                            volume=alive_vol,
                            reason="HEDGE_LEG_SL",
                        )
                    state["status"] = "CLOSED"
                    r.set(f"hedge:active:{pair}", json.dumps(state), ex=3600)
                    return
            except Exception as exc:
                logger.warning(f"HedgeManager | {pair}: liveness check failed: {exc}")

        # ── Skip cycling on the same bar the hedge was opened ────────────────────
        opened_at_str = state.get("opened_at", "")
        if opened_at_str:
            try:
                from datetime import timedelta
                opened_at = datetime.fromisoformat(opened_at_str)
                age_minutes = (datetime.now(timezone.utc) - opened_at).total_seconds() / 60
                if age_minutes < CYCLE_SKIP_BARS * 15:
                    logger.debug(f"HedgeManager | {pair}: opened {age_minutes:.0f}min ago — skip cycle (wait 1 bar)")
                    return
            except Exception:
                pass

        # ── Profit target: close both legs when banked ≥ 2× initial risk ────────
        realized     = state.get("realized_pnl", 0.0)
        initial_risk = state.get("initial_risk_usd", 0.0)
        if initial_risk > 0 and realized >= initial_risk * PROFIT_TARGET_MULT:
            logger.info(
                f"HedgeManager | {pair}: profit target hit "
                f"(${realized:+.2f} ≥ {PROFIT_TARGET_MULT}× ${initial_risk:.2f}) → closing"
            )
            self._close_both_legs(pair, "PROFIT_TARGET")
            return

        # ── Time stop: close if open > 48h AND in profit ──────────────────────
        opened_str = state.get("opened_at", "")
        if opened_str and realized > 0:
            try:
                from datetime import timedelta
                opened_dt = datetime.fromisoformat(opened_str)
                age_hours = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
                if age_hours > HEDGE_MAX_HOURS:
                    logger.info(
                        f"HedgeManager | {pair}: time stop ({age_hours:.0f}h > {HEDGE_MAX_HOURS}h) "
                        f"in profit ${realized:+.2f} → closing"
                    )
                    self._close_both_legs(pair, "TIME_STOP")
                    return
            except Exception:
                pass

        # ── Shared atr/pip for mode routing and cycling ───────────────────────
        atr = state.get("atr_at_open") or _get_atr(pair, r) or 0.001
        pip = _pip_size(pair)

        # ── Pyramid tier management (wash / hard stop / liveness) ────────────
        if self._check_pyramid_tiers(pair, state, price, r):
            return  # cluster was closed — nothing more to do

        # ── Pending re-entry (fires before event blackout — protective) ───────
        if state.get("pending_reentry"):
            self._manage_pending_reentry(pair, state, price, pip, r)
            r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
            if (state.get("hedge_mode") or "CYCLING") == "CYCLING":
                return  # re-entry fired and reset to CYCLING; next bar handles cycling

        # ── WINNER_RUNNING: SL walk + invalidation (not blocked by news) ─────
        hedge_mode = state.get("hedge_mode") or "CYCLING"  # None (pre-session4 hedges) → CYCLING
        if hedge_mode == "WINNER_RUNNING":
            self._handle_winner_running(pair, state, price, atr, pip, r)
            return

        # ── CYCLING → WINNER_RUNNING transition check ─────────────────────────
        entry_price = state.get("entry_price", price)
        move        = abs(price - entry_price)
        if move >= atr * WINNER_THRESHOLD_MULT:
            winning_leg          = "BUY" if price > entry_price else "SELL"
            state["hedge_mode"]  = "WINNER_RUNNING"
            state["winning_leg"] = winning_leg
            r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
            logger.info(
                f"HedgeManager | {pair}: CYCLING → WINNER_RUNNING | "
                f"{winning_leg} winning | {move/pip:.1f}pip from entry"
            )
            self._handle_winner_running(pair, state, price, atr, pip, r)
            return

        # ── Cycle count guard — with wash mode ───────────────────────────────
        if state.get("cycle_count", 0) >= MAX_CYCLE_COUNT:
            net_pnl = _calc_net_hedge_pnl(pair, state, price)
            if net_pnl >= 0:
                # Net positive or breakeven — clean wash close
                logger.info(
                    f"HedgeManager | {pair}: MAX_CYCLES net=${net_pnl:+.2f} ≥ 0 → WASH CLOSE"
                )
                self._close_both_legs(pair, "WASH_COMPLETE")
                return
            if net_pnl <= -MAX_WASH_LOSS_USD:
                # Loss too deep — hard stop regardless
                logger.warning(
                    f"HedgeManager | {pair}: WASH STOP — net=${net_pnl:+.2f} "
                    f"≤ -${MAX_WASH_LOSS_USD:.0f} → forced close"
                )
                self._close_both_legs(pair, "WASH_STOPPED")
                return
            # Net negative but within wash tolerance → extend cycling
            state["cycle_count"] = 0
            state["wash_mode"]   = True
            r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
            logger.info(
                f"HedgeManager | {pair}: WASH MODE — net=${net_pnl:+.2f}, "
                f"resetting cycles (hard stop ≤ -${MAX_WASH_LOSS_USD:.0f})"
            )
            return

        # ── Event blackout: skip S/R cycling near news ────────────────────────
        ev_raw = r.get(f"features:event:{pair}")
        if ev_raw:
            try:
                if json.loads(ev_raw).get("high_impact_within_2h"):
                    logger.debug(f"HedgeManager | {pair}: near high-impact event, skip cycle")
                    return
            except Exception:
                pass

        # ── S/R cycling with minimum pip profit guard ─────────────────────────
        # Recompute distances from current price + wall levels (not cached stale values)
        wall_res = sr.get("nearest_resistance")
        wall_sup = sr.get("nearest_support")
        dist_res = abs(wall_res - price) / pip if wall_res else 999
        dist_sup = abs(price - wall_sup)  / pip if wall_sup else 999

        if dist_res <= CYCLE_PIPS:
            # Guard: only cycle if BUY has gained ≥ MIN_CYCLE_PROFIT_PIPS from its entry.
            # Prevents pinned-price churn (close+reopen at same price = pure spread loss).
            buy_entry     = float(state.get("buy_entry", state.get("entry_price", price)))
            buy_gain_pips = (price - buy_entry) / pip
            if buy_gain_pips < MIN_CYCLE_PROFIT_PIPS:
                logger.debug(
                    f"HedgeManager | {pair}: BUY at resistance but gain "
                    f"{buy_gain_pips:.1f}pip < {MIN_CYCLE_PROFIT_PIPS}pip min — wait"
                )
            else:
                logger.info(f"HedgeManager | {pair}: BUY at resistance ({dist_res:.1f}pip) → cycle")
                pnl = self._cycle_leg(pair, "BUY", state, price, r)
                if pnl is not None:
                    state["realized_pnl"] = round(state.get("realized_pnl", 0) + pnl, 2)
                    state["cycle_count"]  = state.get("cycle_count", 0) + 1
                    r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
                    logger.info(
                        f"HedgeManager | {pair}: cycle #{state['cycle_count']} | "
                        f"banked ${pnl:+.2f} | total realized ${state['realized_pnl']:+.2f}"
                    )
                    # Pyramid: open new tier at this resistance level
                    self._maybe_add_pyramid_tier(pair, state, "BUY", pnl, price, r)
                    # Wash mode: check if net PnL recovered to breakeven
                    if state.get("wash_mode"):
                        net_pnl = _calc_net_hedge_pnl(pair, state, price)
                        if net_pnl >= 0:
                            logger.info(
                                f"HedgeManager | {pair}: WASH COMPLETE — "
                                f"net=${net_pnl:+.2f} → closing both legs"
                            )
                            self._close_both_legs(pair, "WASH_COMPLETE")
                            return

        elif dist_sup <= CYCLE_PIPS:
            # Guard: only cycle if SELL has gained ≥ MIN_CYCLE_PROFIT_PIPS from its entry.
            sell_entry     = float(state.get("sell_entry", state.get("entry_price", price)))
            sell_gain_pips = (sell_entry - price) / pip
            if sell_gain_pips < MIN_CYCLE_PROFIT_PIPS:
                logger.debug(
                    f"HedgeManager | {pair}: SELL at support but gain "
                    f"{sell_gain_pips:.1f}pip < {MIN_CYCLE_PROFIT_PIPS}pip min — wait"
                )
            else:
                logger.info(f"HedgeManager | {pair}: SELL at support ({dist_sup:.1f}pip) → cycle")
                pnl = self._cycle_leg(pair, "SELL", state, price, r)
                if pnl is not None:
                    state["realized_pnl"] = round(state.get("realized_pnl", 0) + pnl, 2)
                    state["cycle_count"]  = state.get("cycle_count", 0) + 1
                    r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
                    logger.info(
                        f"HedgeManager | {pair}: cycle #{state['cycle_count']} | "
                        f"banked ${pnl:+.2f} | total realized ${state['realized_pnl']:+.2f}"
                    )
                    # Pyramid: open new tier at this support level
                    self._maybe_add_pyramid_tier(pair, state, "SELL", pnl, price, r)
                    # Wash mode: check if net PnL recovered to breakeven
                    if state.get("wash_mode"):
                        net_pnl = _calc_net_hedge_pnl(pair, state, price)
                        if net_pnl >= 0:
                            logger.info(
                                f"HedgeManager | {pair}: WASH COMPLETE — "
                                f"net=${net_pnl:+.2f} → closing both legs"
                            )
                            self._close_both_legs(pair, "WASH_COMPLETE")
                            return

    # ── Core: cycle one leg ───────────────────────────────────────────────────

    def _cycle_leg(
        self,
        pair:      str,
        leg:       str,   # "BUY" or "SELL"
        state:     Dict,
        price:     float,
        r:         redis.Redis,
    ) -> Optional[float]:
        """
        Close one leg (bank profit), immediately re-open at same price.
        Returns realized PnL from closed leg, or None on failure.
        Both legs remain open at all times — account is never 1-sided.
        """
        from execution.order_executor import order_executor

        pid_key = "buy_position_id"  if leg == "BUY"  else "sell_position_id"
        tid_key = "buy_trade_id"     if leg == "BUY"  else "sell_trade_id"

        position_id = state.get(pid_key)
        trade_id    = state.get(tid_key)
        volume      = state.get("volume_per_leg", 1000)
        atr         = state.get("atr_at_open", 0.0010)

        if not position_id:
            logger.warning(f"HedgeManager | {pair}: {leg} position_id missing, skip cycle")
            return None

        # ── Close: temporarily expose cached position so close_position can
        #    update the DB trade record via record_trade_close ─────────────────
        leg_cache = {
            "pair":        pair,
            "direction":   leg,
            "volume":      volume,
            "entry_price": state.get("entry_price", price),
            "stop_loss":   0,
            "take_profit": 0,
            "position_id": position_id,
            "trade_db_id": trade_id,
        }
        r.setex(f"execution:open:{pair}", 30, json.dumps(leg_cache))  # 30s temp

        close_res = order_executor.close_position(
            pair=pair,
            position_id=str(position_id),
            volume=volume,
            reason=f"HEDGE_CYCLE_{leg}",
        )

        # close_position deletes execution:open:{pair} — that's fine
        if not close_res.get("success"):
            r.delete(f"execution:open:{pair}")
            err = close_res.get("error", "")
            logger.error(f"HedgeManager | {pair}: {leg} close failed: {err}")
            if "POSITION_NOT_FOUND" in err:
                # Ghost position — null it in state and mark PARTIAL so the
                # surviving leg can be closed cleanly on next liveness check.
                state[pid_key] = None
                state["status"] = "PARTIAL"
                r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
                logger.warning(
                    f"HedgeManager | {pair}: {leg} ghost position cleared → PARTIAL"
                )
            return None

        fill_price = close_res.get("fill_price", price)
        pnl        = close_res.get("pnl_usd") or 0.0

        # ── Re-open at the same price (S/R extreme = good reversal level) ────
        sl_dist = atr * ATR_STOP_MULT
        tp_dist = atr * ATR_TP_SAFETY_MULT

        if leg == "BUY":
            new_sl = _round_price(pair, fill_price - sl_dist)
            new_tp = _round_price(pair, fill_price + tp_dist)
        else:
            new_sl = _round_price(pair, fill_price + sl_dist)
            new_tp = _round_price(pair, fill_price - tp_dist)

        time.sleep(LEG_ORDER_SLEEP)

        new_res = order_executor.place_market_order(
            pair=pair, direction=leg,
            volume_units=volume,
            stop_loss=new_sl, take_profit=new_tp,
            signal_data={
                "strategy": "HEDGE_CYCLE",
                "path":     "HEDGE_CYCLE",
                "cycle":    state.get("cycle_count", 0) + 1,
            },
            regime="MR", risk_pct=RISK_PER_LEG_PCT,
        )

        # Always delete execution:open after re-entry (hedge owns tracking)
        r.delete(f"execution:open:{pair}")

        if not new_res.get("success"):
            logger.error(
                f"HedgeManager | ❌ {pair}: {leg} re-open FAILED — hedge is now 1-sided! "
                f"error={new_res.get('error')}"
            )
            state[pid_key] = None
            state[tid_key] = None
            state["status"] = "PARTIAL"
            r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
            return pnl

        # Update hedge state with new leg IDs, new SL, and new per-leg entry
        state[pid_key] = new_res.get("position_id")
        state[tid_key] = new_res.get("trade_db_id")
        sl_key    = "buy_sl"    if leg == "BUY" else "sell_sl"
        entry_key = "buy_entry" if leg == "BUY" else "sell_entry"
        state[sl_key]    = new_sl
        state[entry_key] = round(fill_price, 6)   # track this leg's new entry

        # Walk the OTHER leg's SL toward breakeven
        self._walk_sl(pair, leg, state, price, r)

        logger.success(
            f"HedgeManager | {pair}: {leg} cycled @ {fill_price} | "
            f"pnl=${pnl:+.2f} | new pos={new_res.get('position_id')}"
        )
        return pnl

    # ── Profit preservation: walk SL after cycle ─────────────────────────────

    def _walk_sl(
        self,
        pair:  str,
        cycled_leg: str,   # the leg that just cycled ("BUY" or "SELL")
        state: Dict,
        price: float,
        r:     redis.Redis,
    ) -> None:
        """
        After leg X cycles, tighten the OTHER leg's SL by SL_WALK_FRACTION
        of the remaining distance to breakeven.

        BUY SL walks UP toward entry (locks profit as price oscillates up).
        SELL SL walks DOWN toward entry (locks profit as price oscillates down).

        Formula (same for both directions):
          new_sl = current_sl + (entry_price - current_sl) × SL_WALK_FRACTION
          BUY:  current_sl < entry → (entry - sl) > 0 → sl walks up ✓
          SELL: current_sl > entry → (entry - sl) < 0 → sl walks down ✓
        """
        from execution.order_executor import order_executor

        other_leg    = "SELL" if cycled_leg == "BUY" else "BUY"
        sl_key       = "buy_sl"      if other_leg == "BUY" else "sell_sl"
        pid_key      = "buy_position_id"  if other_leg == "BUY" else "sell_position_id"
        tp_key_alt   = "sell_sl"     if other_leg == "BUY" else "buy_sl"  # unused but for clarity

        current_sl   = state.get(sl_key)
        entry_price  = state.get("entry_price")
        position_id  = state.get(pid_key)
        atr          = state.get("atr_at_open", 0.001)
        pip          = _pip_size(pair)

        if current_sl is None or entry_price is None or not position_id:
            return

        new_sl = _round_price(pair, current_sl + (entry_price - current_sl) * SL_WALK_FRACTION)

        # Safety: don't walk BUY SL above entry, or SELL SL below entry
        if other_leg == "BUY"  and new_sl >= entry_price:
            new_sl = _round_price(pair, entry_price - atr * 0.1)
        if other_leg == "SELL" and new_sl <= entry_price:
            new_sl = _round_price(pair, entry_price + atr * 0.1)

        # Market validity check: SL must be on the correct side of current price.
        # cTrader rejects SELL SL below ask and BUY SL above bid (TRADING_BAD_STOPS).
        if other_leg == "SELL" and new_sl < price + pip:
            logger.warning(
                f"HedgeManager | {pair}: SELL SL walk skip — "
                f"new_sl={new_sl:.5f} would cross market ({price:.5f})"
            )
            return
        if other_leg == "BUY" and new_sl > price - pip:
            logger.warning(
                f"HedgeManager | {pair}: BUY SL walk skip — "
                f"new_sl={new_sl:.5f} would cross market ({price:.5f})"
            )
            return

        # Wide safety TP unchanged
        tp_dist = atr * ATR_TP_SAFETY_MULT
        if other_leg == "BUY":
            new_tp = _round_price(pair, entry_price + tp_dist)
        else:
            new_tp = _round_price(pair, entry_price - tp_dist)

        try:
            order_executor.amend_sltp(pair, str(position_id), new_sl, new_tp)
            state[sl_key] = new_sl
            logger.info(
                f"HedgeManager | {pair}: {other_leg} SL walked "
                f"{current_sl:.5f} → {new_sl:.5f} (toward breakeven)"
            )
        except Exception as exc:
            logger.warning(f"HedgeManager | {pair}: SL walk failed: {exc}")

    # ── Asymmetric exit: WINNER_RUNNING stage 2/3 ────────────────────────────

    def _handle_winner_running(
        self,
        pair:  str,
        state: Dict,
        price: float,
        atr:   float,
        pip:   float,
        r:     redis.Redis,
    ) -> None:
        """
        Stage 2 — winning leg runs freely toward its TP.
        Losing leg receives aggressive 50% SL walk each check.
        Stage 3 — if losing leg's drawdown ≥ ATR × 1.5, close it + set deferred re-entry.
        """
        winning_leg = state.get("winning_leg", "BUY")
        losing_leg  = "SELL" if winning_leg == "BUY" else "BUY"

        # If the losing leg is already gone — trail winner SL to protect gains
        pid_key = "buy_position_id" if losing_leg == "BUY" else "sell_position_id"
        if not state.get(pid_key):
            self._trail_winner_sl(pair, winning_leg, state, price, atr, r)
            return

        entry_price = state.get("entry_price", price)

        # Stage 3 trigger: price moved ≥ LOSER_INVALIDATE_MULT×ATR against the loser
        if winning_leg == "BUY":
            move_against_loser = price - entry_price
        else:
            move_against_loser = entry_price - price

        if move_against_loser >= atr * LOSER_INVALIDATE_MULT:
            logger.info(
                f"HedgeManager | {pair}: {losing_leg} INVALIDATED — "
                f"{move_against_loser/pip:.1f}pip ≥ {LOSER_INVALIDATE_MULT}×ATR "
                f"→ closing + deferred re-entry"
            )
            self._invalidate_losing_leg(pair, losing_leg, state, price, atr, pip, r)
            return

        # Stage 2: walk losing leg SL aggressively and persist
        self._walk_sl_winner_mode(pair, losing_leg, state, price, r)
        r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)

    def _trail_winner_sl(
        self,
        pair:        str,
        winning_leg: str,
        state:       Dict,
        price:       float,
        atr:         float,
        r:           redis.Redis,
    ) -> None:
        """
        After the loser leg is gone, trail the winner's SL at WINNER_TRAIL_PIPS
        (5 pips) behind price — tight trail to capture hedge profits quickly.
        Never allows SL to go below breakeven (entry price).
        """
        from execution.order_executor import order_executor

        pid_key = "buy_position_id"  if winning_leg == "BUY"  else "sell_position_id"
        sl_key  = "buy_sl"           if winning_leg == "BUY"  else "sell_sl"

        position_id = state.get(pid_key)
        entry_price = state.get("entry_price", price)
        current_sl  = state.get(sl_key)
        if not position_id or current_sl is None:
            return

        # Trail at WINNER_TRAIL_PIPS behind price (tight for hedge profit capture)
        trail_dist = WINNER_TRAIL_PIPS * _pip_size(pair)
        if winning_leg == "BUY":
            trail_sl = _round_price(pair, price - trail_dist)
            # Floor: never below entry (breakeven protection)
            trail_sl = max(trail_sl, _round_price(pair, entry_price))
            if trail_sl <= current_sl:
                return  # SL already at or above trail level — no move needed
        else:
            trail_sl = _round_price(pair, price + trail_dist)
            # Ceiling: never above entry (breakeven protection)
            trail_sl = min(trail_sl, _round_price(pair, entry_price))
            if trail_sl >= current_sl:
                return

        tp_dist = atr * ATR_TP_SAFETY_MULT
        new_tp = _round_price(pair, price + tp_dist) if winning_leg == "BUY" else _round_price(pair, price - tp_dist)

        result = order_executor.amend_sltp(pair, str(position_id), trail_sl, new_tp)
        if not result.get("success"):
            err = result.get("error", "")
            if "POSITION_NOT_FOUND" in err:
                # Position was externally closed (SL/TP hit) — clean up state + DB
                logger.warning(
                    f"HedgeManager | {pair}: {winning_leg} winner gone externally "
                    f"(POSITION_NOT_FOUND) — cleaning state + DB"
                )
                tid_key  = "buy_trade_id" if winning_leg == "BUY" else "sell_trade_id"
                trade_id = state.get(tid_key)
                if trade_id:
                    try:
                        from execution.risk_manager import risk_manager
                        risk_manager.record_trade_close(trade_id, price, "CLOSED_SL")
                    except Exception as _e:
                        logger.warning(f"HedgeManager | {pair}: DB close failed: {_e}")
                state[pid_key] = None
                state[sl_key]  = None
                r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
            else:
                logger.warning(f"HedgeManager | {pair}: winner trail amend failed: {err}")
            return
        state[sl_key] = trail_sl
        r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
        logger.info(
            f"HedgeManager | {pair}: {winning_leg} winner trail "
            f"{current_sl:.5f} → {trail_sl:.5f} (solo leg)"
        )

    def _invalidate_losing_leg(
        self,
        pair:       str,
        losing_leg: str,
        state:      Dict,
        price:      float,
        atr:        float,
        pip:        float,
        r:          redis.Redis,
    ) -> None:
        """Close the losing leg and store a deferred market re-entry watch."""
        from execution.order_executor import order_executor

        pid_key = "buy_position_id"  if losing_leg == "BUY"  else "sell_position_id"
        tid_key = "buy_trade_id"     if losing_leg == "BUY"  else "sell_trade_id"

        position_id = state.get(pid_key)
        trade_id    = state.get(tid_key)
        volume      = state.get("volume_per_leg", 1000)

        if not position_id:
            logger.warning(f"HedgeManager | {pair}: {losing_leg} pos_id missing for invalidation")
            return

        leg_cache = {
            "pair": pair, "direction": losing_leg, "volume": volume,
            "entry_price": state.get("entry_price", 0),
            "stop_loss": 0, "take_profit": 0,
            "position_id": position_id, "trade_db_id": trade_id,
        }
        r.setex(f"execution:open:{pair}", 30, json.dumps(leg_cache))

        close_res = order_executor.close_position(
            pair=pair, position_id=str(position_id),
            volume=volume, reason="HEDGE_LOSER_INVALIDATED",
        )

        if not close_res.get("success"):
            r.delete(f"execution:open:{pair}")
            err = close_res.get("error", "")
            if "POSITION_NOT_FOUND" in err:
                # Position was externally stopped by cTrader (SL hit before invalidation fired).
                # Update DB directly so the trade record doesn't stay stuck as OPEN.
                logger.warning(
                    f"HedgeManager | {pair}: {losing_leg} externally stopped "
                    f"(POSITION_NOT_FOUND) — cleaning state + DB"
                )
                if trade_id:
                    try:
                        from execution.risk_manager import risk_manager
                        risk_manager.record_trade_close(trade_id, price, "CLOSED_SL")
                    except Exception as _db_exc:
                        logger.warning(f"HedgeManager | {pair}: DB close failed: {_db_exc}")
                state[pid_key] = None
                state[tid_key] = None
                state["pending_reentry"] = None  # don't re-enter — exit was forced by SL
                r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
            else:
                logger.error(
                    f"HedgeManager | {pair}: {losing_leg} invalidation close failed: {err}"
                )
            return

        pnl = close_res.get("pnl_usd") or 0.0
        state["realized_pnl"] = round(state.get("realized_pnl", 0) + pnl, 2)
        state[pid_key] = None
        state[tid_key] = None

        # Watch price: N pips further in the extending direction = better re-entry level
        # Losing SELL → price moved UP → re-enter SELL when price pushes 3 pips higher
        # Losing BUY  → price moved DOWN → re-enter BUY when price pushes 3 pips lower
        if losing_leg == "SELL":
            watch_price = _round_price(pair, price + REENTRY_WATCH_PIPS * pip)
        else:
            watch_price = _round_price(pair, price - REENTRY_WATCH_PIPS * pip)

        expires_at = datetime.now(timezone.utc).timestamp() + REENTRY_TTL_MIN * 60

        state["pending_reentry"] = {
            "direction":   losing_leg,
            "watch_price": watch_price,
            "expires_at":  expires_at,
            "volume":      volume,
        }

        r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
        logger.info(
            f"HedgeManager | {pair}: {losing_leg} closed (invalidated) | "
            f"pnl=${pnl:+.2f} | watch={watch_price:.5f} | TTL={REENTRY_TTL_MIN}min"
        )

    def _manage_pending_reentry(
        self,
        pair:  str,
        state: Dict,
        price: float,
        pip:   float,
        r:     redis.Redis,
    ) -> None:
        """Fire deferred re-entry when price reaches watch level or TTL expires."""
        reentry = state.get("pending_reentry")
        if not reentry:
            return

        direction   = reentry["direction"]
        watch_price = reentry["watch_price"]
        expires_at  = reentry["expires_at"]
        volume      = reentry.get("volume", state.get("volume_per_leg", 1000))

        price_hit = (
            (direction == "SELL" and price >= watch_price) or
            (direction == "BUY"  and price <= watch_price)
        )
        ttl_expired = datetime.now(timezone.utc).timestamp() >= expires_at

        if not price_hit and not ttl_expired:
            return

        trigger = "price_watch" if price_hit else "TTL_expired"
        logger.info(
            f"HedgeManager | {pair}: pending re-entry {direction} @ {price:.5f} "
            f"({trigger}) | watch={watch_price:.5f}"
        )

        atr     = state.get("atr_at_open", 0.001)
        sl_dist = atr * ATR_STOP_MULT
        tp_dist = atr * ATR_TP_SAFETY_MULT

        if direction == "BUY":
            new_sl = _round_price(pair, price - sl_dist)
            new_tp = _round_price(pair, price + tp_dist)
        else:
            new_sl = _round_price(pair, price + sl_dist)
            new_tp = _round_price(pair, price - tp_dist)

        from execution.order_executor import order_executor
        new_res = order_executor.place_market_order(
            pair=pair, direction=direction,
            volume_units=volume,
            stop_loss=new_sl, take_profit=new_tp,
            signal_data={"strategy": "HEDGE_REENTRY", "path": "HEDGE_CYCLE"},
            regime="MR", risk_pct=RISK_PER_LEG_PCT,
        )
        r.delete(f"execution:open:{pair}")

        if not new_res.get("success"):
            logger.error(
                f"HedgeManager | {pair}: pending re-entry {direction} FAILED: "
                f"{new_res.get('error')} — clearing watch"
            )
            state["pending_reentry"] = None
            return

        pid_key = "buy_position_id"  if direction == "BUY"  else "sell_position_id"
        tid_key = "buy_trade_id"     if direction == "BUY"  else "sell_trade_id"
        sl_key  = "buy_sl"           if direction == "BUY"  else "sell_sl"

        state[pid_key]           = new_res.get("position_id")
        state[tid_key]           = new_res.get("trade_db_id")
        state[sl_key]            = new_sl
        state["pending_reentry"] = None
        state["hedge_mode"]      = "CYCLING"   # both legs open again — reset to CYCLING
        state["winning_leg"]     = None

        logger.success(
            f"HedgeManager | {pair}: ✅ re-entry {direction} pos={new_res.get('position_id')} "
            f"@ {price:.5f} | trigger={trigger} | mode → CYCLING"
        )

    def _walk_sl_winner_mode(
        self,
        pair:       str,
        losing_leg: str,
        state:      Dict,
        price:      float,
        r:          redis.Redis,
    ) -> None:
        """Aggressive 50% SL walk on the losing leg in WINNER_RUNNING mode."""
        from execution.order_executor import order_executor

        sl_key  = "buy_sl"           if losing_leg == "BUY"  else "sell_sl"
        pid_key = "buy_position_id"  if losing_leg == "BUY"  else "sell_position_id"

        current_sl  = state.get(sl_key)
        entry_price = state.get("entry_price")
        position_id = state.get(pid_key)
        atr         = state.get("atr_at_open", 0.001)
        pair_name   = state.get("pair", pair)
        pip         = _pip_size(pair)

        if current_sl is None or entry_price is None or not position_id:
            return

        new_sl = _round_price(pair_name, current_sl + (entry_price - current_sl) * LOSER_WALK_FRAC)

        # Cap: don't push SL past entry (that would flip to a winning SL on the loser)
        if losing_leg == "BUY"  and new_sl >= entry_price:
            new_sl = _round_price(pair_name, entry_price - atr * 0.1)
        if losing_leg == "SELL" and new_sl <= entry_price:
            new_sl = _round_price(pair_name, entry_price + atr * 0.1)

        # Market validity check: SL must be on the correct side of the current price.
        # cTrader rejects SELL SL below ask and BUY SL above bid (TRADING_BAD_STOPS).
        if losing_leg == "SELL" and new_sl < price + pip:
            logger.warning(
                f"HedgeManager | {pair}: SELL SL walk skip — "
                f"new_sl={new_sl:.5f} would cross market ({price:.5f}), holding current SL"
            )
            return
        if losing_leg == "BUY" and new_sl > price - pip:
            logger.warning(
                f"HedgeManager | {pair}: BUY SL walk skip — "
                f"new_sl={new_sl:.5f} would cross market ({price:.5f}), holding current SL"
            )
            return

        tp_dist = atr * ATR_TP_SAFETY_MULT
        if losing_leg == "BUY":
            new_tp = _round_price(pair_name, entry_price + tp_dist)
        else:
            new_tp = _round_price(pair_name, entry_price - tp_dist)

        try:
            order_executor.amend_sltp(pair, str(position_id), new_sl, new_tp)
            state[sl_key] = new_sl
            logger.info(
                f"HedgeManager | {pair}: {losing_leg} SL walked (winner-mode 50%) "
                f"{current_sl:.5f} → {new_sl:.5f}"
            )
        except Exception as exc:
            logger.warning(f"HedgeManager | {pair}: winner-mode SL walk failed: {exc}")

    # ── Core: regime conversion ───────────────────────────────────────────────

    def _handle_regime_shift(
        self,
        pair:    str,
        state:   Dict,
        regime:  str,
        a2_bias: str,
        price:   float,
        sr:      Dict,
        r:       redis.Redis,
    ) -> None:
        """
        Manage MR → THV/TLV transition.
        Waits for S/R wall test + pullback before closing opposing leg.
        """
        # Determine which direction to keep
        if a2_bias == "LONG_BASE":
            keep_dir  = "BUY"
            close_dir = "SELL"
        elif a2_bias == "SHORT_BASE":
            keep_dir  = "SELL"
            close_dir = "BUY"
        else:
            # A2 neutral — can't pick direction yet, wait
            logger.debug(f"HedgeManager | {pair}: {regime} confirmed but A2=NEUTRAL, waiting")
            state["regime_shift_bars"] = 0
            r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
            return

        shift_bars = state.get("regime_shift_bars", 0) + 1
        state["regime_shift_bars"] = shift_bars
        state["converting_to"]     = keep_dir
        state["status"]            = "CONVERTING"
        r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)

        logger.info(
            f"HedgeManager | {pair}: regime shift → {regime} | A2={a2_bias} | "
            f"keep={keep_dir} close={close_dir} | "
            f"bar {shift_bars}/{REGIME_CONFIRM_BARS}"
        )

        if shift_bars < REGIME_CONFIRM_BARS:
            return  # wait for confirmation

        # ── Confirmed regime shift: apply S/R timing ──────────────────────────
        dist_res = float(sr.get("distance_to_resistance", 999))
        dist_sup = float(sr.get("distance_to_support", 999))

        if close_dir == "SELL":
            # Keeping BUY (bullish). Check resistance above.
            # If resistance close: price might test it → wait for pullback before closing SELL
            # (SELL profits from resistance touch, close at the pullback = better price)
            if dist_res < CONVERSION_SR_PIPS:
                if dist_res > CONVERSION_PULLBACK:
                    # Price has pulled back from resistance (it already touched it)
                    logger.info(
                        f"HedgeManager | {pair}: post-resistance pullback "
                        f"({dist_res:.1f}pip) confirmed → converting now"
                    )
                    self._execute_conversion(pair, state, close_dir, r)
                else:
                    logger.debug(
                        f"HedgeManager | {pair}: resistance {dist_res:.1f}pip — "
                        f"waiting for test + pullback > {CONVERSION_PULLBACK}pip"
                    )
            else:
                # Resistance too far — convert immediately
                logger.info(f"HedgeManager | {pair}: no nearby resistance → converting now")
                self._execute_conversion(pair, state, close_dir, r)

        else:
            # Keeping SELL (bearish). Check support below.
            if dist_sup < CONVERSION_SR_PIPS:
                if dist_sup > CONVERSION_PULLBACK:
                    logger.info(
                        f"HedgeManager | {pair}: post-support bounce "
                        f"({dist_sup:.1f}pip) confirmed → converting now"
                    )
                    self._execute_conversion(pair, state, close_dir, r)
                else:
                    logger.debug(
                        f"HedgeManager | {pair}: support {dist_sup:.1f}pip — "
                        f"waiting for test + bounce > {CONVERSION_PULLBACK}pip"
                    )
            else:
                logger.info(f"HedgeManager | {pair}: no nearby support → converting now")
                self._execute_conversion(pair, state, close_dir, r)

    def _execute_conversion(
        self,
        pair:      str,
        state:     Dict,
        close_dir: str,   # the leg to close
        r:         redis.Redis,
    ) -> None:
        """
        Close the opposing leg.
        Hand surviving leg to trailing stop via execution:open:{pair}.
        """
        from execution.order_executor import order_executor

        pid_key  = "buy_position_id"  if close_dir == "BUY"  else "sell_position_id"
        tid_key  = "buy_trade_id"     if close_dir == "BUY"  else "sell_trade_id"
        keep_pid = "sell_position_id" if close_dir == "BUY"  else "buy_position_id"
        keep_tid = "sell_trade_id"    if close_dir == "BUY"  else "buy_trade_id"

        close_pos_id = state.get(pid_key)
        close_tid    = state.get(tid_key)
        volume       = state.get("volume_per_leg", 1000)
        keep_dir     = state.get("converting_to", "BUY")

        if not close_pos_id:
            logger.error(f"HedgeManager | {pair}: {close_dir} position_id missing for conversion")
            return

        # Set temp cache so close_position can update DB record
        leg_cache = {
            "pair":        pair,
            "direction":   close_dir,
            "volume":      volume,
            "entry_price": state.get("entry_price", 0),
            "stop_loss":   0,
            "take_profit": 0,
            "position_id": close_pos_id,
            "trade_db_id": close_tid,
        }
        r.setex(f"execution:open:{pair}", 30, json.dumps(leg_cache))

        close_res = order_executor.close_position(
            pair=pair,
            position_id=str(close_pos_id),
            volume=volume,
            reason=f"HEDGE_CONVERT_{close_dir}",
        )

        if not close_res.get("success"):
            r.delete(f"execution:open:{pair}")
            logger.error(f"HedgeManager | {pair}: conversion close failed: {close_res.get('error')}")
            return

        # close_position deleted execution:open:{pair} — now restore it for surviving leg
        surviving_pos_id = state.get(keep_pid)
        surviving_tid    = state.get(keep_tid)
        pnl              = close_res.get("pnl_usd") or 0.0
        fill_price       = close_res.get("fill_price", state.get("entry_price", 0))

        total_pnl = round(state.get("realized_pnl", 0) + pnl, 2)

        # Hand surviving leg to normal position management (trailing stop, tech exit)
        if surviving_pos_id:
            from execution.risk_manager import _get_current_price as _gcp
            cur_price = _gcp(pair) or state.get("entry_price", 0)
            atr       = state.get("atr_at_open", 0.0010)

            if keep_dir == "BUY":
                new_sl = _round_price(pair, cur_price - atr * 1.5)
                new_tp = _round_price(pair, cur_price + atr * 4.5)
            else:
                new_sl = _round_price(pair, cur_price + atr * 1.5)
                new_tp = _round_price(pair, cur_price - atr * 4.5)

            surviving_cache = {
                "pair":        pair,
                "direction":   keep_dir,
                "volume":      volume,
                "entry_price": state.get("entry_price", cur_price),
                "stop_loss":   new_sl,
                "take_profit": new_tp,
                "position_id": surviving_pos_id,
                "trade_db_id": surviving_tid,
                "regime":      state.get("converting_to_regime", "THV"),
                "opened_at":   datetime.now(timezone.utc).isoformat(),
            }
            r.setex(f"execution:open:{pair}", 86400, json.dumps(surviving_cache))

            # Tighten SL on surviving leg to ATR × 1.5 (from hedge's wide ATR × 2.0)
            try:
                order_executor.amend_sltp(pair, str(surviving_pos_id), new_sl, new_tp)
                logger.info(f"HedgeManager | {pair}: surviving {keep_dir} SL→{new_sl} TP→{new_tp}")
            except Exception as exc:
                logger.warning(f"HedgeManager | {pair}: SL/TP amend failed: {exc}")

        # Mark hedge as converted and clean up Redis
        state["status"]             = "CONVERTED"
        state["realized_pnl"]       = total_pnl
        state["conversion_direction"] = keep_dir
        r.set(f"hedge:active:{pair}", json.dumps(state), ex=3600)  # 1h then auto-expire

        # DB update
        try:
            with db.connection() as conn:
                conn.execute(text("""
                    UPDATE hedge_positions
                    SET status='CONVERTED', converted_at=NOW(),
                        conversion_direction=:dir, total_realized_pnl=:pnl
                    WHERE id = (
                        SELECT id FROM hedge_positions
                        WHERE pair=:pair AND status IN ('ACTIVE','CONVERTING')
                        ORDER BY id DESC LIMIT 1
                    )
                """), {"pair": pair, "dir": keep_dir, "pnl": total_pnl})
                conn.execute(text("COMMIT"))
        except Exception as exc:
            logger.warning(f"HedgeManager | {pair}: DB conversion update failed: {exc}")

        logger.success(
            f"HedgeManager | ✅ {pair} converted: "
            f"closed {close_dir} @ {fill_price:.5f} | "
            f"keeping {keep_dir} pos={surviving_pos_id} | "
            f"total P&L ${total_pnl:+.2f}"
        )

    # ── Cascade pyramid ───────────────────────────────────────────────────────

    def _maybe_add_pyramid_tier(
        self,
        pair:       str,
        state:      Dict,
        cycled_leg: str,   # "BUY" or "SELL" — which leg just profitably cycled
        cycle_pnl:  float,
        price:      float,
        r:          redis.Redis,
    ) -> None:
        """
        After a profitable S/R cycle, open a new pyramid tier (BUY + SELL) at the
        current price.  Each tier is half the volume of the previous one.

        Pyramid logic:
          Price goes UP → BUY leg keeps cycling and banking profit at resistance.
          Each cycle adds a new tier: a fresh BUY+SELL at the resistance level.
          Accumulated SELL legs (going OTM as price rises) form a natural hedge.
          When price reverses, those SELL legs profit — washing the BUY losses.
        """
        if MAX_PYRAMID_TIERS == 0:
            return

        # Only in MR regime
        regime_raw = r.get(f"features:regime:{pair}")
        if regime_raw:
            try:
                if json.loads(regime_raw).get("regime", "MR") != "MR":
                    return
            except Exception:
                pass

        if cycle_pnl < PYRAMID_MIN_PROFIT_USD:
            return
        if state.get("wash_mode"):
            return

        existing_tiers = state.get("pyramid_tiers", [])
        if len(existing_tiers) >= MAX_PYRAMID_TIERS:
            return

        from execution.order_executor import order_executor

        tier_num  = len(existing_tiers) + 1
        base_vol  = state.get("volume_per_leg", 1000)
        tier_vol  = max(1000, int(base_vol * (PYRAMID_SIZE_FRACTION ** tier_num)))
        atr       = state.get("atr_at_open", 0.001)
        sl_dist   = atr * ATR_STOP_MULT
        tp_dist   = atr * ATR_TP_SAFETY_MULT

        buy_sl  = _round_price(pair, price - sl_dist)
        buy_tp  = _round_price(pair, price + tp_dist)
        sell_sl = _round_price(pair, price + sl_dist)
        sell_tp = _round_price(pair, price - tp_dist)

        sig = {"strategy": "HEDGE_PYRAMID", "path": "HEDGE_CYCLE", "tier": tier_num}

        buy_res = order_executor.place_market_order(
            pair=pair, direction="BUY", volume_units=tier_vol,
            stop_loss=buy_sl, take_profit=buy_tp,
            signal_data=sig, regime="MR",
            risk_pct=RISK_PER_LEG_PCT * (PYRAMID_SIZE_FRACTION ** tier_num),
        )
        r.delete(f"execution:open:{pair}")

        if not buy_res.get("success"):
            logger.error(
                f"HedgeManager | {pair}: pyramid tier {tier_num} BUY failed: "
                f"{buy_res.get('error')}"
            )
            return

        time.sleep(LEG_ORDER_SLEEP)

        sell_res = order_executor.place_market_order(
            pair=pair, direction="SELL", volume_units=tier_vol,
            stop_loss=sell_sl, take_profit=sell_tp,
            signal_data=sig, regime="MR",
            risk_pct=RISK_PER_LEG_PCT * (PYRAMID_SIZE_FRACTION ** tier_num),
        )
        r.delete(f"execution:open:{pair}")

        if not sell_res.get("success"):
            logger.error(
                f"HedgeManager | {pair}: pyramid tier {tier_num} SELL failed — "
                f"closing orphan BUY"
            )
            order_executor.close_position(
                pair, str(buy_res.get("position_id")), tier_vol, "PYRAMID_ABORT"
            )
            return

        new_tier = {
            "tier_num":         tier_num,
            "triggered_by":     cycled_leg,
            "entry_price":      round(price, 6),
            "volume":           tier_vol,
            "buy_position_id":  buy_res.get("position_id"),
            "sell_position_id": sell_res.get("position_id"),
            "buy_trade_id":     buy_res.get("trade_db_id"),
            "sell_trade_id":    sell_res.get("trade_db_id"),
            "buy_sl":           buy_sl,
            "sell_sl":          sell_sl,
            "opened_at":        datetime.now(timezone.utc).isoformat(),
        }
        existing_tiers.append(new_tier)
        state["pyramid_tiers"] = existing_tiers
        r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)

        logger.success(
            f"HedgeManager | {pair}: ✅ pyramid tier {tier_num} @ {price:.5f} | "
            f"vol={tier_vol} BUY={buy_res.get('position_id')} "
            f"SELL={sell_res.get('position_id')} | triggered by {cycled_leg} cycle"
        )

    def _check_pyramid_tiers(
        self,
        pair:  str,
        state: Dict,
        price: float,
        r:     redis.Redis,
    ) -> bool:
        """
        Manage all open pyramid tiers every M15 bar.

        1. Liveness — if a tier leg's SL price has been breached, close it + update DB.
        2. Wash close — if cluster net PnL ≥ 0 AND some profit was already banked,
           price has reversed enough to cover the losses → close everything.
        3. Hard stop — if cluster net < −PYRAMID_CLUSTER_STOP → force-close all.

        Returns True if the whole cluster was closed (caller must return immediately).
        """
        tiers = state.get("pyramid_tiers", [])
        if not tiers:
            return False

        from execution.order_executor import order_executor
        from execution.risk_manager   import risk_manager

        pip     = _pip_size(pair)
        pip_val = _pip_value_per_unit(pair, price)
        changed = False

        for tier in tiers:
            entry = float(tier.get("entry_price", price))
            vol   = tier.get("volume", 1000)

            # ── BUY leg: SL hit when price falls to/below buy_sl ─────────────
            buy_pos = tier.get("buy_position_id")
            buy_sl  = tier.get("buy_sl")
            if buy_pos and buy_sl is not None and price <= buy_sl:
                logger.info(
                    f"HedgeManager | {pair}: tier {tier['tier_num']} BUY SL hit @ {price:.5f}"
                )
                r.setex(f"execution:open:{pair}", 30, json.dumps({
                    "pair": pair, "direction": "BUY", "volume": vol,
                    "entry_price": entry, "stop_loss": 0, "take_profit": 0,
                    "position_id": buy_pos, "trade_db_id": tier.get("buy_trade_id"),
                }))
                res = order_executor.close_position(pair, str(buy_pos), vol, "CLOSED_SL")
                if res.get("success") or "POSITION_NOT_FOUND" in res.get("error", ""):
                    if not res.get("success") and tier.get("buy_trade_id"):
                        try:
                            risk_manager.record_trade_close(
                                tier["buy_trade_id"], price, "CLOSED_SL"
                            )
                        except Exception:
                            pass
                    pnl = res.get("pnl_usd") or round(
                        (price - entry) / pip * pip_val * vol, 2
                    )
                    state["realized_pnl"] = round(
                        state.get("realized_pnl", 0) + pnl, 2
                    )
                    tier["buy_position_id"] = None
                    tier["buy_trade_id"]    = None
                    changed = True

            # ── SELL leg: SL hit when price rises to/above sell_sl ───────────
            sell_pos = tier.get("sell_position_id")
            sell_sl  = tier.get("sell_sl")
            if sell_pos and sell_sl is not None and price >= sell_sl:
                logger.info(
                    f"HedgeManager | {pair}: tier {tier['tier_num']} SELL SL hit @ {price:.5f}"
                )
                r.setex(f"execution:open:{pair}", 30, json.dumps({
                    "pair": pair, "direction": "SELL", "volume": vol,
                    "entry_price": entry, "stop_loss": 0, "take_profit": 0,
                    "position_id": sell_pos, "trade_db_id": tier.get("sell_trade_id"),
                }))
                res = order_executor.close_position(pair, str(sell_pos), vol, "CLOSED_SL")
                if res.get("success") or "POSITION_NOT_FOUND" in res.get("error", ""):
                    if not res.get("success") and tier.get("sell_trade_id"):
                        try:
                            risk_manager.record_trade_close(
                                tier["sell_trade_id"], price, "CLOSED_SL"
                            )
                        except Exception:
                            pass
                    pnl = res.get("pnl_usd") or round(
                        (entry - price) / pip * pip_val * vol, 2
                    )
                    state["realized_pnl"] = round(
                        state.get("realized_pnl", 0) + pnl, 2
                    )
                    tier["sell_position_id"] = None
                    tier["sell_trade_id"]    = None
                    changed = True

        if changed:
            r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)

        # ── Cluster-level checks ──────────────────────────────────────────────
        cluster_pnl = _calc_cluster_net_pnl(pair, state, price)

        if cluster_pnl <= -PYRAMID_CLUSTER_STOP:
            logger.warning(
                f"HedgeManager | {pair}: PYRAMID HARD STOP — "
                f"cluster net=${cluster_pnl:+.2f} ≤ −${PYRAMID_CLUSTER_STOP:.0f}"
            )
            self._close_all_pyramid_tiers(pair, state, r, "PYRAMID_STOP")
            self._close_both_legs(pair, "PYRAMID_STOP")
            return True

        # Wash: cluster recovered to ≥ 0 AND some profit was banked (confirms reversal)
        if cluster_pnl >= 0 and tiers and state.get("realized_pnl", 0) > 0:
            logger.info(
                f"HedgeManager | {pair}: PYRAMID WASH COMPLETE — "
                f"cluster net=${cluster_pnl:+.2f} ≥ 0 → closing all"
            )
            self._close_all_pyramid_tiers(pair, state, r, "PYRAMID_WASH")
            self._close_both_legs(pair, "PYRAMID_WASH")
            return True

        return False

    def _close_all_pyramid_tiers(
        self,
        pair:   str,
        state:  Dict,
        r:      redis.Redis,
        reason: str = "MANUAL",
    ) -> None:
        """Close every open leg across all pyramid tiers."""
        from execution.order_executor import order_executor

        for tier in state.get("pyramid_tiers", []):
            vol = tier.get("volume", 1000)
            for leg, pid_key, tid_key in [
                ("BUY",  "buy_position_id",  "buy_trade_id"),
                ("SELL", "sell_position_id", "sell_trade_id"),
            ]:
                pid = tier.get(pid_key)
                tid = tier.get(tid_key)
                if not pid:
                    continue
                try:
                    r.setex(f"execution:open:{pair}", 30, json.dumps({
                        "pair": pair, "direction": leg, "volume": vol,
                        "entry_price": tier.get("entry_price", 0),
                        "stop_loss": 0, "take_profit": 0,
                        "position_id": pid, "trade_db_id": tid,
                    }))
                    order_executor.close_position(pair, str(pid), vol, reason)
                except Exception as exc:
                    logger.error(
                        f"HedgeManager | {pair}: close tier {tier.get('tier_num')} "
                        f"{leg} failed: {exc}"
                    )
                    r.delete(f"execution:open:{pair}")

        state["pyramid_tiers"] = []
        r.set(f"hedge:active:{pair}", json.dumps(state), ex=CACHE_TTL)
        logger.info(f"HedgeManager | {pair}: all pyramid tiers closed | reason={reason}")

    # ── Close both legs ───────────────────────────────────────────────────────

    def _close_both_legs(self, pair: str, reason: str = "MANUAL") -> None:
        """Close both legs. Used for emergency / max-cycle situations."""
        r = _redis_client()
        raw = r.get(f"hedge:active:{pair}")
        if not raw:
            return

        state  = json.loads(raw)
        volume = state.get("volume_per_leg", 1000)

        from execution.order_executor import order_executor

        for leg, pid_key, tid_key in [
            ("BUY",  "buy_position_id",  "buy_trade_id"),
            ("SELL", "sell_position_id", "sell_trade_id"),
        ]:
            pid = state.get(pid_key)
            tid = state.get(tid_key)
            if not pid:
                continue
            try:
                leg_cache = {
                    "pair": pair, "direction": leg, "volume": volume,
                    "entry_price": state.get("entry_price", 0),
                    "stop_loss": 0, "take_profit": 0,
                    "position_id": pid, "trade_db_id": tid,
                }
                r.setex(f"execution:open:{pair}", 30, json.dumps(leg_cache))
                order_executor.close_position(pair, str(pid), volume, reason)
            except Exception as exc:
                logger.error(f"HedgeManager | {pair}: close {leg} failed: {exc}")
                r.delete(f"execution:open:{pair}")

        r.delete(f"hedge:active:{pair}")

        try:
            with db.connection() as conn:
                conn.execute(text("""
                    UPDATE hedge_positions
                    SET status='CLOSED', converted_at=NOW(), notes=:reason
                    WHERE id = (
                        SELECT id FROM hedge_positions
                        WHERE pair=:pair AND status IN ('ACTIVE','CONVERTING','PARTIAL')
                        ORDER BY id DESC LIMIT 1
                    )
                """), {"pair": pair, "reason": reason})
                conn.execute(text("COMMIT"))
        except Exception as exc:
            logger.warning(f"HedgeManager | {pair}: DB close failed: {exc}")

        logger.info(f"HedgeManager | {pair}: both legs closed | reason={reason}")

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        try:
            with db.connection() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS hedge_positions (
                        id                   SERIAL PRIMARY KEY,
                        pair                 VARCHAR(10)  NOT NULL,
                        status               VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',
                        buy_trade_id         INT,
                        sell_trade_id        INT,
                        buy_position_id      VARCHAR(50),
                        sell_position_id     VARCHAR(50),
                        volume_per_leg       INT          NOT NULL,
                        opened_at            TIMESTAMPTZ  DEFAULT NOW(),
                        converted_at         TIMESTAMPTZ,
                        conversion_direction VARCHAR(10),
                        cycle_count          INT          DEFAULT 0,
                        total_realized_pnl   FLOAT        DEFAULT 0.0,
                        notes                TEXT
                    )
                """))
                conn.execute(text("COMMIT"))
            self._table_ready = True
            logger.debug("HedgeManager | hedge_positions table ready")
        except Exception as exc:
            logger.error(f"HedgeManager | Table setup failed: {exc}")


# ── Module singleton ──────────────────────────────────────────────────────────

hedge_manager = HedgeManager()
