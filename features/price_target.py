"""
features/price_target.py
========================
Price Target Engine — ATR-based TP with S/R snap.

Computes the final take-profit price by snapping the raw ATR×3
target to the nearest S/R wall when one sits between entry and target.

Rules
-----
  SL  = ATR × 1.5  (computed in risk_manager, passed in)
  TP  = ATR × 4.5  (3× the stop distance — raw target)
  Snap: if the nearest S/R wall is between (entry + 1.5R) and raw_TP,
        snap TP 3 pips below (LONG) or above (SHORT) that wall.
  Guard: snapped TP must be ≥ 1.5R from entry — below that it is not
         worth taking; the raw TP is returned unchanged.

Rationale
---------
Without snapping, the bot often runs a trade straight into a wall and
gets a CLOSED_SL because resistance absorbed the move before 3R.
Snapping just below the wall locks in a realistic winner instead of
holding for a target that gets blocked.

Output
------
  tp_price  (float) — final take-profit price (may equal raw_tp)
  r_ratio   (float) — realised R:R = tp_distance / sl_distance
  tp_source (str)   — "ATR_RAW" | "SR_SNAPPED"
"""
from __future__ import annotations

from typing import Tuple

from loguru import logger

# Lazy import to avoid circular dependency at module load time
# (sr_features → timeseries_db → settings; risk_manager also imports both)

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

# Minimum R multiple the snapped TP must achieve to be worth using
MIN_SNAP_R = 1.5

# Buffer placed between snapped TP and the S/R wall (pips)
SNAP_BUFFER_PIPS = 3


def _pip(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else 0.0001


class PriceTargetEngine:
    """
    Price Target Engine — singleton at module bottom.

    Usage::

        from features.price_target import price_target_engine
        tp, r_ratio, source = price_target_engine.compute(
            pair        = "EURUSD",
            direction   = "BUY",
            entry_price = 1.0850,
            sl_price    = 1.0835,
            raw_tp      = 1.0917,   # entry + ATR×4.5
        )
    """

    # ── public API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        pair:        str,
        direction:   str,    # "BUY" | "SELL"
        entry_price: float,
        sl_price:    float,
        raw_tp:      float,
    ) -> Tuple[float, float, str]:
        """
        Return (tp_price, r_ratio, tp_source).

        Args:
            pair:        G10 currency pair, e.g. "EURUSD"
            direction:   "BUY" or "SELL"
            entry_price: Anticipated fill price
            sl_price:    Stop loss price (ATR-based, already computed)
            raw_tp:      Raw ATR take-profit (entry ± ATR × 4.5)

        Returns:
            tp_price  — final take-profit price
            r_ratio   — distance-to-TP / distance-to-SL
            tp_source — "ATR_RAW" or "SR_SNAPPED"
        """
        sl_dist = abs(entry_price - sl_price)

        if sl_dist <= 0:
            logger.warning(f"PriceTarget | {pair}: sl_dist=0, returning raw TP")
            return raw_tp, 3.0, "ATR_RAW"

        raw_r = abs(raw_tp - entry_price) / sl_dist

        # Attempt S/R snap
        try:
            from features.sr_features import sr_engine
            sr = sr_engine.get_features(pair)
        except Exception as exc:
            logger.warning(f"PriceTarget | {pair}: S/R unavailable ({exc})")
            return raw_tp, round(raw_r, 2), "ATR_RAW"

        if not sr:
            return raw_tp, round(raw_r, 2), "ATR_RAW"

        buf       = SNAP_BUFFER_PIPS * _pip(pair)
        min_dist  = sl_dist * MIN_SNAP_R   # minimum TP distance to accept a snap

        if direction == "BUY":
            wall = sr.get("nearest_resistance", 0.0)
            if wall and wall > entry_price:
                # Wall sits between (entry + min_dist) and raw_tp → snap below it.
                # Outer condition guarantees wall >= entry + min_dist, so snapped
                # (= wall - buf) is close enough — no redundant inner check needed.
                if (entry_price + min_dist) < wall <= raw_tp:
                    snapped = wall - buf
                    r = (snapped - entry_price) / sl_dist
                    logger.info(
                        f"PriceTarget | {pair} BUY snap: "
                        f"TP {raw_tp:.5f}→{snapped:.5f} "
                        f"(wall={wall:.5f} R={r:.2f})"
                    )
                    return round(snapped, 5), round(r, 2), "SR_SNAPPED"

        else:  # SELL
            wall = sr.get("nearest_support", 0.0)
            if wall and wall > 0:
                # Wall sits between raw_tp and (entry - min_dist) → snap above it.
                if raw_tp <= wall < (entry_price - min_dist):
                    snapped = wall + buf
                    r = (entry_price - snapped) / sl_dist
                    logger.info(
                        f"PriceTarget | {pair} SELL snap: "
                        f"TP {raw_tp:.5f}→{snapped:.5f} "
                        f"(wall={wall:.5f} R={r:.2f})"
                    )
                    return round(snapped, 5), round(r, 2), "SR_SNAPPED"

        # No snap applicable — return raw TP
        logger.debug(
            f"PriceTarget | {pair} {direction}: raw TP={raw_tp:.5f} "
            f"R={raw_r:.2f} (no snap)"
        )
        return raw_tp, round(raw_r, 2), "ATR_RAW"


# ── module singleton ──────────────────────────────────────────────────────────

price_target_engine = PriceTargetEngine()
