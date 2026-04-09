"""
models/analyst1_hmm/hmm_regime_classifier.py
=============================================
Analyst 1 — HMM Regime Classifier (parallel to LightGBM A1)

Uses a per-pair Gaussian Hidden Markov Model to classify market regimes into
the same 4 states as A1: THV / TLV / MR / PNS.

Why HMM alongside LightGBM:
  LightGBM treats each bar independently (stateless).
  HMM explicitly models *regime persistence* and *transition probabilities*,
  capturing the fact that THV rarely snaps to PNS in a single bar — there is
  a lag structure the sequence model encodes and LightGBM ignores.

  When A1 (LightGBM) and HMM *disagree* on the trend/range category:
    → size_mult × 0.25  (same signal as regime instability)
  When they *agree*:
    → no change (HMM acts as silent validator, not an extra hurdle)

Architecture:
  Model:    hmmlearn.GaussianHMM(n_components=4, covariance_type="diag")
  Features: 5 per bar (log_return, atr_ratio, vol_ratio, momentum_5d,
            bb_pct_centered) — fewer is better for HMM
  Training: per-pair on M15 OHLCV from TimescaleDB (same source as A1)
            Minimum 5,000 bars required (~52 days M15)
  Labelling: Post-hoc — fit states, then map each hidden state to a regime
             using the mean atr_ratio and |momentum| of its assigned samples.

Redis:
  Key: features:regime_hmm:{PAIR}   TTL: 900s (15 min)
  Value: {regime, hmm_state, confidence, category, probabilities, updated_at}

Usage:
  from models.analyst1_hmm.hmm_regime_classifier import hmm_regime_classifier

  # Train all pairs (one-time or retrain):
  hmm_regime_classifier.train_all()

  # Inference (called by signal_aggregator scheduler):
  hmm_regime_classifier.compute_all()

  # Single pair:
  result = hmm_regime_classifier.get_regime("AUDUSD")
  # → {"regime": "THV", "category": "TRENDING", "confidence": 0.82, ...}
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

from data_pipeline.storage.timeseries_db import db  # noqa: E402

# ── optional hmmlearn import ──────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("RegimeHMM | hmmlearn not installed — run: pip install hmmlearn")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

# Regime definitions (same encoding as A1)
REGIME_LABELS   = {0: "THV", 1: "TLV", 2: "MR", 3: "PNS"}
REGIME_CATEGORY = {
    "THV": "TRENDING",
    "TLV": "TRENDING",
    "MR":  "RANGING",
    "PNS": "RANGING",
}

# JPY pairs: price stored ×100 in DB — normalise before computing returns
JPY_PAIRS = {"EURJPY", "GBPJPY", "USDJPY", "AUDJPY", "NZDJPY"}

# HMM training parameters
HMM_N_COMPONENTS  = 4       # One hidden state per regime
HMM_COVAR_TYPE    = "diag"  # More stable than "full" on limited FX data
HMM_N_ITER        = 300
HMM_TOL           = 1e-5
HMM_RANDOM_STATE  = 42
MIN_TRAIN_BARS    = 5_000   # ~52 days at M15

# Inference: how many recent bars to feed to the HMM
INFERENCE_LOOKBACK = 500    # ~5 trading days at M15

# Persistence-biased transition matrix prior
# Encourages the HMM to stay in each regime rather than thrashing every bar.
# Off-diagonal entries are low but non-zero to allow transitions.
TRANSMAT_PRIOR_DIAG = 0.90

# Cache
CACHE_TTL          = 900    # 15 minutes
SCHEDULER_INTERVAL = 900

# Model storage
MODEL_DIR = ROOT / "models" / "analyst1_hmm" / "saved_models"

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class HMMFeatureBuilder:
    """
    Builds a 5-column observation matrix from raw OHLCV data.

    Features (chosen to be informative and low-redundancy for HMM):
      1. log_return_1bar  — single-bar log return (directional signal, ~ N(0,σ))
      2. atr_ratio        — ATR₁₄ / MA₅₀(ATR₁₄)  (volatility level)
      3. vol_ratio        — std(returns, 32) / std(returns, 128)  (vol regime)
      4. momentum_5d      — (close - close₋₉₆) / close₋₉₆  (signed trend)
      5. bb_pct_centered  — Bollinger %B mapped to [-0.5, +0.5]  (mean-reversion)

    Fewer features = better conditioned Gaussian emissions = better HMM convergence.
    """

    def build(self, df: pd.DataFrame, pair: str = "") -> np.ndarray:
        """
        Args:
            df:   DataFrame with columns [open, high, low, close].
                  Index must be DatetimeIndex, sorted ascending.
            pair: Pair name (used for JPY ×100 normalisation).

        Returns:
            np.ndarray of shape (n, 5) — NaN rows at start dropped.
        """
        close = df["close"].values.copy().astype(float)
        high  = df["high"].values.copy().astype(float)
        low   = df["low"].values.copy().astype(float)

        # Normalise JPY ×100 storage
        if pair in JPY_PAIRS:
            mask = close > 1000
            close[mask] /= 100.0
            high[mask]  /= 100.0
            low[mask]   /= 100.0

        n = len(close)

        # ── 1. Log return (1 bar) ─────────────────────────────────────────────
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(np.maximum(close[1:] / np.maximum(close[:-1], 1e-10), 1e-10))

        # ── 2. ATR ratio ──────────────────────────────────────────────────────
        tr = np.zeros(n)
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        atr_ma50 = pd.Series(atr14).rolling(50, min_periods=20).mean().values
        atr_ratio = np.where(atr_ma50 > 1e-10, atr14 / atr_ma50, 1.0)

        # ── 3. Vol ratio (short/long) ─────────────────────────────────────────
        ret_s = pd.Series(log_ret)
        vol5d  = ret_s.rolling(32,  min_periods=10).std().values   # ~8h
        vol20d = ret_s.rolling(128, min_periods=32).std().values   # ~2 days
        vol_ratio = np.where(vol20d > 1e-10, vol5d / vol20d, 1.0)

        # ── 4. Momentum 5d (96 M15 bars) ─────────────────────────────────────
        momentum5d = np.zeros(n)
        momentum5d[96:] = (close[96:] - close[:-96]) / np.maximum(close[:-96], 1e-10)

        # ── 5. Bollinger Band %B (centered) ──────────────────────────────────
        close_s  = pd.Series(close)
        bb_mid   = close_s.rolling(20, min_periods=10).mean()
        bb_std   = close_s.rolling(20, min_periods=10).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = (bb_upper - bb_lower).values
        bb_pct_raw = np.where(
            bb_range > 1e-10,
            (close - bb_lower.values) / bb_range,
            0.5,
        )
        bb_pct_centered = np.clip(bb_pct_raw, 0.0, 1.0) - 0.5   # [-0.5, 0.5]

        # ── Stack and drop burn-in NaN rows ──────────────────────────────────
        X = np.column_stack([
            log_ret,
            atr_ratio,
            vol_ratio,
            momentum5d,
            bb_pct_centered,
        ])

        # Mark rows with invalid data as NaN in atr_ratio (burn-in period)
        # Drop them for training; at inference keep all rows (HMM needs sequence)
        return X


_feature_builder = HMMFeatureBuilder()

# ══════════════════════════════════════════════════════════════════════════════
# STATE → REGIME MAPPER
# ══════════════════════════════════════════════════════════════════════════════

class HMMStateMapper:
    """
    Maps the fitted HMM's hidden states (0–3, arbitrary order) to regime labels.

    Strategy:
      After training, run Viterbi on the training data → state assignments.
      For each state, compute:
        - mean_atr_ratio    (volatility level)
        - mean_abs_momentum (trend strength)
        - mean_vol_ratio    (vol short vs long)

      Score each state for each regime and assign by best match (no ties allowed):
        THV:  high atr + high |momentum|
        TLV:  low  atr + high |momentum|
        PNS:  low  atr + low  |momentum| + vol_ratio ≤ 1 (compressed)
        MR:   residual (oscillating, moderate atr)

      If a regime has no assigned state (data imbalance), the lowest-score
      unassigned state is used as fallback.
    """

    def build_map(
        self,
        model: "GaussianHMM",
        X: np.ndarray,
        lengths: List[int],
    ) -> Dict[int, str]:
        """
        Args:
            model:   Fitted GaussianHMM.
            X:       Training feature matrix (n, 5).
            lengths: Sequence lengths for predict().

        Returns:
            Dict mapping {hmm_state_int → regime_str}.
        """
        states = model.predict(X, lengths=lengths)

        # Compute per-state statistics from training assignments
        state_stats: Dict[int, Dict[str, float]] = {}
        for s in range(model.n_components):
            mask = states == s
            if mask.sum() < 10:
                state_stats[s] = {
                    "mean_atr_ratio": 1.0,
                    "mean_abs_momentum": 0.0,
                    "mean_vol_ratio": 1.0,
                    "count": int(mask.sum()),
                }
                continue
            state_stats[s] = {
                "mean_atr_ratio":    float(np.mean(X[mask, 1])),  # col 1 = atr_ratio
                "mean_abs_momentum": float(np.mean(np.abs(X[mask, 3]))),  # col 3 = momentum
                "mean_vol_ratio":    float(np.mean(X[mask, 2])),  # col 2 = vol_ratio
                "count":             int(mask.sum()),
            }

        logger.debug(f"RegimeHMM | State stats: {state_stats}")

        # Normalise across states to [0, 1]
        atr_vals = np.array([state_stats[s]["mean_atr_ratio"] for s in range(model.n_components)])
        mom_vals = np.array([state_stats[s]["mean_abs_momentum"] for s in range(model.n_components)])
        vol_vals = np.array([state_stats[s]["mean_vol_ratio"] for s in range(model.n_components)])

        def _norm(arr: np.ndarray) -> np.ndarray:
            span = arr.max() - arr.min()
            return (arr - arr.min()) / span if span > 1e-10 else arr * 0.0 + 0.5

        atr_n = _norm(atr_vals)
        mom_n = _norm(mom_vals)
        vol_n = _norm(vol_vals)   # lower = more compressed (PNS-like)

        # Score each state for each regime
        scores: Dict[str, np.ndarray] = {
            "THV": atr_n * mom_n,                           # high vol + trend
            "TLV": (1.0 - atr_n) * mom_n,                  # low vol + trend
            "PNS": (1.0 - atr_n) * (1.0 - mom_n) * (1.0 - vol_n),  # stagnant + quiet
            "MR":  (1.0 - mom_n) * ((atr_n + (1.0 - vol_n)) / 2),  # ranging
        }

        # Greedy assignment: best state per regime (without replacement)
        assigned_regimes: Dict[int, str] = {}
        assigned_states: set = set()
        # Sort regimes by max score (most distinguishable first)
        for regime in sorted(scores, key=lambda r: scores[r].max(), reverse=True):
            ranked = np.argsort(-scores[regime])
            for s in ranked:
                if s not in assigned_states:
                    assigned_regimes[s] = regime
                    assigned_states.add(s)
                    break

        # Assign any leftover states (shouldn't happen with n_components=4)
        fallback_label = "MR"
        for s in range(model.n_components):
            if s not in assigned_regimes:
                logger.warning(f"RegimeHMM | State {s} unassigned — defaulting to {fallback_label}")
                assigned_regimes[s] = fallback_label

        logger.info(
            f"RegimeHMM | State map: "
            + " | ".join(f"State{s}→{assigned_regimes[s]}" for s in range(model.n_components))
        )
        return assigned_regimes


_state_mapper = HMMStateMapper()

# ══════════════════════════════════════════════════════════════════════════════
# REDIS HELPER
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class RegimeHMMClassifier:
    """
    Per-pair Gaussian HMM regime classifier.

    Designed as a *parallel* signal alongside the LightGBM A1 classifier.
    It does not replace A1 — it validates it.

    Disagreement protocol:
      TRENDING = {THV, TLV}   RANGING = {MR, PNS}
      If A1_category ≠ HMM_category → size_mult × 0.25
      (same as the regime instability check — but the two penalties do NOT stack)

    Key methods:
      train_all()     — train per-pair models, save to disk
      compute_all()   — run inference, cache to Redis
      get_regime(pair) — read cached result for one pair
    """

    def __init__(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._models:    Dict[str, "GaussianHMM"]  = {}
        self._state_maps: Dict[str, Dict[int, str]] = {}
        self._load_models()

    # ── Model persistence ─────────────────────────────────────────────────────

    def _model_path(self, pair: str) -> Path:
        return MODEL_DIR / f"hmm_{pair}.pkl"

    def _load_models(self) -> None:
        """Load all saved per-pair models from disk at startup."""
        if not HMM_AVAILABLE:
            return
        loaded = 0
        for pair in PAIRS:
            path = self._model_path(pair)
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        bundle = pickle.load(f)
                    self._models[pair]    = bundle["model"]
                    self._state_maps[pair] = bundle["state_map"]
                    loaded += 1
                except Exception as exc:
                    logger.warning(f"RegimeHMM | Failed to load {pair} model: {exc}")
        if loaded:
            logger.info(f"RegimeHMM | Loaded {loaded}/{len(PAIRS)} saved models")

    def _save_model(self, pair: str, model: "GaussianHMM", state_map: Dict[int, str]) -> None:
        path = self._model_path(pair)
        with open(path, "wb") as f:
            pickle.dump({"model": model, "state_map": state_map}, f)
        logger.success(f"RegimeHMM | Saved {pair} model → {path}")

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_ohlcv(self, pair: str, limit: int = 50_000) -> pd.DataFrame:
        """Fetch M15 OHLCV bars from TimescaleDB."""
        with db.connection() as conn:
            rows = conn.execute(
                text("""
                    SELECT time, open, high, low, close
                    FROM ohlcv
                    WHERE symbol = :symbol
                      AND time > '2020-01-01'
                    ORDER BY time ASC
                    LIMIT :limit
                """),
                {"symbol": pair, "limit": limit},
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        return df

    # ── Training ──────────────────────────────────────────────────────────────

    def train_pair(self, pair: str) -> bool:
        """
        Train HMM for a single pair.

        Returns True on success, False on failure.
        """
        if not HMM_AVAILABLE:
            logger.error("RegimeHMM | hmmlearn not available — cannot train")
            return False

        logger.info(f"RegimeHMM | Training {pair}...")

        df = self._fetch_ohlcv(pair, limit=100_000)
        if len(df) < MIN_TRAIN_BARS:
            logger.warning(
                f"RegimeHMM | {pair}: only {len(df)} bars — need {MIN_TRAIN_BARS}. Skipping."
            )
            return False

        X_raw = _feature_builder.build(df, pair=pair)

        # Drop burn-in (first ~130 bars where rolling features are NaN/unreliable)
        valid_start = 130
        X = X_raw[valid_start:]

        # Clip extreme outliers (flash crashes, data errors)
        for col_idx in range(X.shape[1]):
            p1, p99 = np.nanpercentile(X[:, col_idx], [1, 99])
            X[:, col_idx] = np.clip(X[:, col_idx], p1, p99)

        # Replace any remaining NaN/inf with column median
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            bad = ~np.isfinite(col)
            if bad.any():
                X[bad, col_idx] = np.nanmedian(col)

        # Build persistence-biased transition matrix
        n = HMM_N_COMPONENTS
        transmat_init = np.full((n, n), (1.0 - TRANSMAT_PRIOR_DIAG) / (n - 1))
        np.fill_diagonal(transmat_init, TRANSMAT_PRIOR_DIAG)

        model = GaussianHMM(
            n_components=HMM_N_COMPONENTS,
            covariance_type=HMM_COVAR_TYPE,
            n_iter=HMM_N_ITER,
            tol=HMM_TOL,
            random_state=HMM_RANDOM_STATE,
            init_params="mc",   # K-means init means + covars only (not startprob/transmat)
            params="mcst",      # EM learns means, covars, startprob, transmat
        )
        # Set persistence-biased starting points for startprob and transmat.
        # These will be refined by EM — they just give the optimiser a good head start.
        model.startprob_ = np.ones(n) / n
        model.transmat_  = transmat_init

        try:
            model.fit(X, lengths=[len(X)])
        except Exception as exc:
            logger.error(f"RegimeHMM | {pair} training failed: {exc}")
            return False

        if not model.monitor_.converged:
            logger.warning(f"RegimeHMM | {pair} did not fully converge — using current params")

        # Map states to regime labels
        state_map = _state_mapper.build_map(model, X, lengths=[len(X)])

        # Store and save
        with self._lock:
            self._models[pair]     = model
            self._state_maps[pair] = state_map
        self._save_model(pair, model, state_map)

        # Log transition matrix (regime switching statistics)
        tm = np.exp(model.transmat_) if hasattr(model, "transmat_") else model.transmat_
        logger.info(
            f"RegimeHMM | {pair} trained on {len(X):,} bars | "
            f"State map: {state_map}"
        )
        return True

    def train_all(self, pairs: Optional[List[str]] = None) -> Dict[str, bool]:
        """Train HMM for all (or specified) pairs. Returns {pair: success}."""
        pairs = pairs or PAIRS
        results = {}
        for pair in pairs:
            results[pair] = self.train_pair(pair)
        success = sum(results.values())
        logger.success(f"RegimeHMM | Training complete: {success}/{len(pairs)} pairs succeeded")
        return results

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer_pair(self, pair: str) -> Optional[Dict]:
        """
        Run HMM inference on the most recent INFERENCE_LOOKBACK bars.

        Returns dict suitable for Redis caching, or None on failure.
        """
        if pair not in self._models or pair not in self._state_maps:
            return None

        df = self._fetch_ohlcv(pair, limit=INFERENCE_LOOKBACK + 200)
        if len(df) < 150:
            return None

        X = _feature_builder.build(df.tail(INFERENCE_LOOKBACK + 150), pair=pair)

        # Clip and clean (same as training)
        for col_idx in range(X.shape[1]):
            p1, p99 = np.nanpercentile(X[:, col_idx], [1, 99])
            X[:, col_idx] = np.clip(X[:, col_idx], p1, p99)
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            bad = ~np.isfinite(col)
            if bad.any():
                X[bad, col_idx] = float(np.nanmedian(col[~bad])) if (~bad).any() else 0.0

        model     = self._models[pair]
        state_map = self._state_maps[pair]

        try:
            # Get state sequence via Viterbi
            state_seq = model.predict(X, lengths=[len(X)])
            current_state = int(state_seq[-1])

            # Posterior probabilities for the last observation
            log_posteriors = model.predict_proba(X, lengths=[len(X)])
            posteriors      = log_posteriors[-1]   # shape (n_components,)

            # Map to regime probabilities
            regime_probs: Dict[str, float] = {}
            for s, prob in enumerate(posteriors):
                regime = state_map.get(s, "MR")
                regime_probs[regime] = regime_probs.get(regime, 0.0) + float(prob)

            current_regime = state_map.get(current_state, "MR")
            confidence     = float(regime_probs.get(current_regime, 0.0))
            category       = REGIME_CATEGORY.get(current_regime, "RANGING")

        except Exception as exc:
            logger.error(f"RegimeHMM | {pair} inference failed: {exc}")
            return None

        return {
            "pair":        pair,
            "regime":      current_regime,
            "hmm_state":   current_state,
            "category":    category,
            "confidence":  round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in regime_probs.items()},
            "updated_at":  datetime.now(timezone.utc).isoformat(),
        }

    def compute_all(self) -> Dict[str, Optional[Dict]]:
        """
        Run inference for all pairs and cache results to Redis.

        Called by the background scheduler in main_loop.py (15-min interval).
        """
        if not HMM_AVAILABLE:
            return {}
        if not self._models:
            logger.warning("RegimeHMM | No models loaded — run train_all() first")
            return {}

        r = _get_redis()
        results = {}
        for pair in PAIRS:
            try:
                result = self._infer_pair(pair)
                if result:
                    r.setex(
                        f"features:regime_hmm:{pair}",
                        CACHE_TTL,
                        json.dumps(result),
                    )
                    results[pair] = result
                    logger.debug(
                        f"RegimeHMM | {pair}: {result['regime']} "
                        f"conf={result['confidence']:.2f}"
                    )
                else:
                    results[pair] = None
            except Exception as exc:
                logger.error(f"RegimeHMM | compute {pair} failed: {exc}")
                results[pair] = None

        success = sum(1 for v in results.values() if v is not None)
        logger.info(f"RegimeHMM | compute_all done: {success}/{len(PAIRS)} cached")
        return results

    def get_regime(self, pair: str) -> Optional[Dict]:
        """
        Read cached HMM regime for a pair from Redis.

        Returns None if no model trained or cache empty.
        """
        try:
            r = _get_redis()
            val = r.get(f"features:regime_hmm:{pair}")
            if val:
                return json.loads(val)
        except Exception as exc:
            logger.error(f"RegimeHMM | get_regime {pair} failed: {exc}")
        return None

    def start_scheduler(self) -> None:
        """Start background daemon thread to recompute HMM regimes every 15 min."""
        def _loop():
            while True:
                try:
                    self.compute_all()
                except Exception as exc:
                    logger.error(f"RegimeHMM | Scheduler error: {exc}")
                import time
                time.sleep(SCHEDULER_INTERVAL)

        t = threading.Thread(target=_loop, daemon=True, name="RegimeHMMScheduler")
        t.start()
        logger.info("RegimeHMM | Scheduler started (15-min interval)")

    def summary(self) -> None:
        """Print current HMM regime summary for all pairs (debugging/monitoring)."""
        r = _get_redis()
        print("\n── HMM Regime Summary ────────────────────────────────────")
        for pair in PAIRS:
            val = r.get(f"features:regime_hmm:{pair}")
            if val:
                d = json.loads(val)
                probs_str = " ".join(
                    f"{k}={v:.2f}" for k, v in sorted(d.get("probabilities", {}).items())
                )
                print(
                    f"  {pair:<8}  {d['regime']:<4}  "
                    f"conf={d['confidence']:.2f}  [{probs_str}]  "
                    f"cat={d['category']}"
                )
            else:
                print(f"  {pair:<8}  — no cache")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

hmm_regime_classifier = RegimeHMMClassifier()
