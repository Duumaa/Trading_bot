"""
models/analyst7_smc/score.py
=============================
A7 SMC model inference wrapper.

Loads the trained LightGBM model once at import time and exposes a single
function: `smc_score(pair, direction, smc_feats, context_feats)`.

Returns a float 0.0–1.0 representing the probability that the current
SMC setup leads to price moving ≥ 1× ATR in `direction` within 4h.

Integration with Gate 4 (entry_engine.py):
  score >= 0.65 → entry_score *= 1.15   (+15% boost)
  score >= 0.55 → entry_score *= 1.05   (+5% boost)
  score <= 0.35 → entry_score *= 0.90   (-10% penalty)
  otherwise     → no change

Only applied when the model file exists and loaded successfully.
If model unavailable, returns 0.5 (neutral — no effect on entry_score).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

MODEL_PATH = Path(__file__).parent / "a7_smc_lgbm.pkl"

FEATURE_COLS = [
    "fvg_aligned_dist", "fvg_opposing_dist",
    "fvg_inside_aligned", "fvg_inside_opposing",
    "liq_sweep_aligned", "liq_sweep_opposing",
    "ob_aligned_dist", "ob_opposing_dist",
    "msb_aligned_bars", "msb_opposing_bars",
    "atr_pips", "adx", "bb_pct", "price_vs_session",
    "hour_of_day", "day_of_week",
]

# SMC entry score multipliers
BOOST_HIGH  = (0.65, 1.15)   # score >= 0.65 → +15%
BOOST_LOW   = (0.55, 1.05)   # score >= 0.55 → +5%
PENALTY     = (0.35, 0.90)   # score <= 0.35 → -10%


class _A7Scorer:
    def __init__(self) -> None:
        self._model  = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if not MODEL_PATH.exists():
            logger.warning(f"A7 SMC | Model not found at {MODEL_PATH} — scoring disabled")
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            self._model  = bundle["model"]
            self._loaded = True
            auc = bundle.get("test_auc", "?")
            logger.info(f"A7 SMC | Model loaded | test_auc={auc:.4f}")
        except Exception as exc:
            logger.error(f"A7 SMC | Failed to load model: {exc}")

    def predict(
        self,
        direction: str,       # "LONG" or "SHORT"
        smc:       Dict,      # from smc_engine.get_features(pair)
        atr_pips:  float,
        adx:       float,
        bb_pct:    float,
        price_vs_session: float,
        hour:      int,
        dow:       int,
    ) -> float:
        """Return probability 0.0–1.0. Falls back to 0.5 if model unavailable."""
        if not self._loaded or self._model is None:
            return 0.5

        # Map LONG/SHORT → BUY/SELL for direction-aware remapping
        is_buy = direction in ("LONG", "BUY")

        if is_buy:
            row = [
                smc.get("fvg_bull_dist",        0.0),
                smc.get("fvg_bear_dist",         0.0),
                smc.get("fvg_inside_bull",       0.0),
                smc.get("fvg_inside_bear",       0.0),
                smc.get("liq_sweep_bull_bars", 200.0),
                smc.get("liq_sweep_bear_bars", 200.0),
                smc.get("ob_bull_dist",          0.0),
                smc.get("ob_bear_dist",          0.0),
                smc.get("msb_bull_bars",       200.0),
                smc.get("msb_bear_bars",       200.0),
            ]
        else:
            row = [
                smc.get("fvg_bear_dist",         0.0),
                smc.get("fvg_bull_dist",         0.0),
                smc.get("fvg_inside_bear",       0.0),
                smc.get("fvg_inside_bull",       0.0),
                smc.get("liq_sweep_bear_bars", 200.0),
                smc.get("liq_sweep_bull_bars", 200.0),
                smc.get("ob_bear_dist",          0.0),
                smc.get("ob_bull_dist",          0.0),
                smc.get("msb_bear_bars",       200.0),
                smc.get("msb_bull_bars",       200.0),
            ]

        row += [atr_pips, adx, bb_pct, price_vs_session, float(hour), float(dow)]

        try:
            X = np.array([row], dtype=float)
            return float(self._model.predict_proba(X)[0, 1])
        except Exception as exc:
            logger.debug(f"A7 SMC | Predict failed: {exc}")
            return 0.5

    def entry_multiplier(self, smc_prob: float) -> tuple[float, str]:
        """
        Convert raw probability to (multiplier, reason_string).
        multiplier=1.0 means no change.
        """
        if smc_prob >= BOOST_HIGH[0]:
            return BOOST_HIGH[1], f"SMC_BOOST_HIGH({smc_prob:.2f})"
        if smc_prob >= BOOST_LOW[0]:
            return BOOST_LOW[1], f"SMC_BOOST({smc_prob:.2f})"
        if smc_prob <= PENALTY[0]:
            return PENALTY[1], f"SMC_PENALTY({smc_prob:.2f})"
        return 1.0, ""


# Module singleton — loaded once at import
a7_scorer = _A7Scorer()
