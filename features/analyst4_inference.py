"""
features/analyst4_inference.py
================================
Analyst 4 — TCN + PatchTSMixer ONNX Inference Engine

Loads trained ONNX models and runs directional inference on all 12 G10
pairs every 15 minutes. Results are cached in Redis and consumed by
signal_aggregator.py as Gate 3 (Technical ML).

Gate logic (co-gate — both models must agree):
  LONG:  TCN_prob > threshold AND PatchTSMixer_prob > threshold
  SHORT: TCN_prob < (1-threshold) AND PatchTSMixer_prob < (1-threshold)
  FLAT:  Models disagree OR both uncertain

Regime-adaptive thresholds (from holdout analysis 2024-2026):
  THV: patch=0.55  tcn=0.54  (precision 65.75%, coverage 73.6%)
  TLV: patch=0.58  tcn=0.57  (precision 62.33%, coverage 63.2%)
  MR:  patch=0.54  tcn=0.54  (precision 66.18%, coverage 74.4%)
  PNS: disabled              (near random, 58.7%)

Redis output key: features:analyst4:{pair}  TTL=900s (15min)
Output schema:
  {
    "pair":        "AUDJPY",
    "direction":   "LONG",        # LONG | SHORT | FLAT
    "confidence":  0.71,          # 0.0-1.0, geometric mean of both probs
    "tcn_prob":    0.68,          # raw TCN output (sigmoid)
    "patch_prob":  0.63,          # raw PatchTSMixer output (sigmoid)
    "agreed":      true,          # both models agree above threshold
    "regime":      "THV",         # regime at inference time
    "gate":        "PASS",        # PASS | FAIL | FLAT
    "updated_at":  "2026-03-19T..."
  }

Model files (must exist in models/analyst4/):
  analyst4_tcn.onnx
  analyst4_patch_tsmixer.onnx
  analyst4_thresholds.json
  analyst4_scaler.pkl
  analyst4_feature_list.json
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis
from loguru import logger
from sqlalchemy import text

# ── Optional imports with graceful fallback ────────────────────────────────────
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("Analyst4 | onnxruntime not installed — inference disabled")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("Analyst4 | ta-lib not installed — features disabled")

# ── Constants ──────────────────────────────────────────────────────────────────
SEQ_LEN      = 96          # 24h of M15 bars lookback
HORIZON      = 8           # 8 bars = 2h ahead (not used at inference)
N_FEATURES   = 16
CACHE_TTL    = 900         # 15 min Redis TTL
REFRESH_SEC  = 900         # background scheduler interval
MODEL_DIR    = Path('/workspaces/Trading_bot/models/analyst4')

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD',
    'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY',
]

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY'}

# Regime-adaptive thresholds from holdout analysis
REGIME_THRESHOLDS: Dict[str, Dict[str, float]] = {
    'THV': {'patch': 0.55, 'tcn': 0.54},
    'TLV': {'patch': 0.58, 'tcn': 0.57},
    'MR':  {'patch': 0.54, 'tcn': 0.54},
    'PNS': {'patch': 0.99, 'tcn': 0.99},  # effectively disabled
}

# Confidence boost/dampener based on agreement strength
AGREE_BOOST    = 1.10   # both models strongly agree
AGREE_NEUTRAL  = 1.00   # both agree at threshold
DISAGREE_DAMP  = 0.90   # models disagree (shouldn't reach gate but just in case)
SINGLE_PENALTY = 0.85   # only one model fires — OR logic, reduced conviction


class Analyst4Inference:
    """
    TCN + PatchTSMixer ONNX inference engine.

    Loads models once at startup, runs every 15 minutes via background
    scheduler. Each pair gets 96 M15 bars from DB, 16 features computed,
    scaled, and fed through both models. Results written to Redis.
    """

    def __init__(self) -> None:
        self._sess_tcn: Optional[object]   = None
        self._sess_patch: Optional[object] = None
        self._scaler: Optional[object]     = None
        self._feature_cols: List[str]      = []
        self._thresholds: Dict             = {}
        self._ready: bool                  = False
        self._lock = threading.Lock()

        # Redis connection
        from dotenv import load_dotenv
        load_dotenv('/workspaces/Trading_bot/.env')
        self._r = redis.Redis(
            host     = os.getenv('REDIS_HOST', 'localhost'),
            port     = int(os.getenv('REDIS_PORT', '6379')),
            password = os.getenv('REDIS_PASSWORD') or None,
            decode_responses = True,
        )

        self._load_models()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load ONNX sessions, scaler, feature list, thresholds."""
        if not ONNX_AVAILABLE:
            logger.error("Analyst4 | onnxruntime missing — run: pip3 install onnxruntime --break-system-packages")
            return
        if not TA_AVAILABLE:
            logger.error("Analyst4 | ta library missing — run: pip3 install ta --break-system-packages")
            return

        try:
            tcn_path   = MODEL_DIR / 'analyst4_tcn.onnx'
            patch_path = MODEL_DIR / 'analyst4_patch_tsmixer.onnx'
            scaler_path= MODEL_DIR / 'analyst4_scaler.pkl'
            feat_path  = MODEL_DIR / 'analyst4_feature_list.json'
            thresh_path= MODEL_DIR / 'analyst4_thresholds.json'

            for p in [tcn_path, patch_path, scaler_path, feat_path, thresh_path]:
                if not p.exists():
                    logger.error(f"Analyst4 | Missing file: {p}")
                    return

            # ONNX sessions — CPU only, no CUDA dependency
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            self._sess_tcn   = ort.InferenceSession(str(tcn_path),   sess_options=opts)
            self._sess_patch = ort.InferenceSession(str(patch_path), sess_options=opts)

            with open(scaler_path, 'rb') as f:
                self._scaler = pickle.load(f)

            with open(feat_path) as f:
                self._feature_cols = json.load(f)

            with open(thresh_path) as f:
                self._thresholds = json.load(f)

            self._ready = True
            logger.success(
                f"Analyst4 | Models loaded ✅ | "
                f"TCN + PatchTSMixer ready | {len(self._feature_cols)} features"
            )

        except Exception as exc:
            logger.error(f"Analyst4 | Model load failed: {exc}")
            self._ready = False

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_bars(self, pair: str) -> Optional[pd.DataFrame]:
        """
        Load last 96+50 M15 bars from DB for a pair.
        +50 buffer for feature warmup (EMA200 needs at least 200 but
        we use 96+200 to be safe — load 300 bars total).
        """
        try:
            from data_pipeline.storage.timeseries_db import db

            with db.connection() as conn:
                rows = conn.execute(text("""
                    SELECT time, open, high, low, close,
                           COALESCE(volume, 1.0) AS volume
                    FROM ohlcv
                    WHERE symbol    = :pair
                      AND timeframe = 'M15'
                      AND time      > NOW() - INTERVAL '5 days'
                    ORDER BY time DESC
                    LIMIT 300
                """), {'pair': pair}).fetchall()

            if not rows or len(rows) < SEQ_LEN + 50:
                logger.warning(f"Analyst4 | {pair}: only {len(rows) if rows else 0} bars — need {SEQ_LEN + 50}")
                return None

            df = pd.DataFrame(rows, columns=['time','open','high','low','close','volume'])
            df = df.sort_values('time').reset_index(drop=True)

            # JPY scaling fix — cTrader stores JPY pairs ×100
            if pair in JPY_PAIRS and df['close'].median() > 500:
                for col in ['open','high','low','close']:
                    df[col] = df[col] / 100.0

            return df

        except Exception as exc:
            logger.error(f"Analyst4 | {pair}: DB load failed: {exc}")
            return None

    # ── Feature engineering ────────────────────────────────────────────────────

    def _compute_features(self, df: pd.DataFrame, pair: str) -> Optional[pd.DataFrame]:
        """
        Compute 16 technical features — must exactly match training notebook.
        All features are stationary (returns/ratios) — no raw prices.
        """
        try:
            d = df.copy()
            c, h, lo = d['close'], d['high'], d['low']

            d['f_return']          = np.log(c / c.shift(1))
            atr                    = ta.volatility.AverageTrueRange(h, lo, c, window=14).average_true_range()
            d['f_atr_pct']         = atr / c
            atr5                   = ta.volatility.AverageTrueRange(h, lo, c, window=5).average_true_range()
            atr50                  = ta.volatility.AverageTrueRange(h, lo, c, window=50).average_true_range()
            d['f_vol_ratio']       = atr5 / (atr50 + 1e-10)
            rsi                    = ta.momentum.RSIIndicator(c, window=14).rsi()
            d['f_rsi']             = (rsi - 50) / 50.0
            ema20                  = c.ewm(span=20, adjust=False).mean()
            d['f_ema_slope']       = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-10)
            ema50                  = c.ewm(span=50, adjust=False).mean()
            d['f_price_vs_ema50']  = (c - ema50) / (ema50 + 1e-10)
            ema200                 = c.ewm(span=200, adjust=False).mean()
            d['f_price_vs_ema200'] = (c - ema200) / (ema200 + 1e-10)
            d['f_momentum_8']      = (c - c.shift(8))  / (c.shift(8)  + 1e-10)
            d['f_momentum_32']     = (c - c.shift(32)) / (c.shift(32) + 1e-10)
            bb                     = ta.volatility.BollingerBands(c, window=20, window_dev=2)
            d['f_bb_width']        = (bb.bollinger_hband() - bb.bollinger_lband()) / (c + 1e-10)
            adx                    = ta.trend.ADXIndicator(h, lo, c, window=14).adx()
            d['f_adx']             = adx / 100.0
            d['f_higher_highs']    = (h > h.shift(1)).astype(float)
            d['f_hl_range_pct']    = (h - lo) / (c + 1e-10)
            typical                = (h + lo + c) / 3
            vwap_96                = (typical * d['volume']).rolling(96).sum() / (d['volume'].rolling(96).sum() + 1e-10)
            d['f_vwap_dist']       = (c - vwap_96) / (vwap_96 + 1e-10)
            body                   = (c - d['open']).abs()
            rng                    = (h - lo).replace(0, np.nan)
            d['f_body_ratio']      = (body / rng).fillna(0)
            upper_wick             = h - d[['open','close']].max(axis=1)
            d['f_wick_ratio']      = (upper_wick / rng).fillna(0)

            d = d.dropna(subset=self._feature_cols)
            if len(d) < SEQ_LEN:
                logger.warning(f"Analyst4 | {pair}: after dropna only {len(d)} bars < {SEQ_LEN}")
                return None

            return d.reset_index(drop=True)

        except Exception as exc:
            logger.error(f"Analyst4 | {pair}: feature computation failed: {exc}")
            return None

    # ── ONNX inference ─────────────────────────────────────────────────────────

    def _run_inference(
        self,
        df: pd.DataFrame,
        pair: str,
        regime: str,
    ) -> Optional[Dict]:
        """
        Scale last 96 bars, run both ONNX models, apply co-gate logic.

        Returns dict with direction, confidence, probs, gate result.
        """
        try:
            # Take last SEQ_LEN rows
            feat_arr = df[self._feature_cols].values[-SEQ_LEN:]   # (96, 16)

            # Scale — use transform (scaler fitted on training data)
            
            feat_df     = pd.DataFrame(feat_arr, columns=self._feature_cols)
            feat_scaled = self._scaler.transform(feat_df)
            # Add batch dimension → (1, 96, 16) float32
            x = feat_scaled[np.newaxis, :, :].astype(np.float32)

            # Run both models
            tcn_logit   = self._sess_tcn.run(None,   {'input': x})[0][0]
            patch_logit = self._sess_patch.run(None, {'input': x})[0][0]

            # Sigmoid → probability
            tcn_prob   = float(1 / (1 + np.exp(-tcn_logit)))
            patch_prob = float(1 / (1 + np.exp(-patch_logit)))

            # Get regime-specific thresholds
            thresholds = REGIME_THRESHOLDS.get(regime, REGIME_THRESHOLDS['MR'])
            t_tcn   = thresholds['tcn']
            t_patch = thresholds['patch']

            # Per-model direction flags
            tcn_long    = tcn_prob   > t_tcn
            tcn_short   = tcn_prob   < (1 - t_tcn)
            patch_long  = patch_prob > t_patch
            patch_short = patch_prob < (1 - t_patch)

            both_long  = tcn_long  and patch_long
            both_short = tcn_short and patch_short

            # Conflict: models point opposite directions — always block
            conflict = (tcn_long and patch_short) or (tcn_short and patch_long)

            if both_long:
                direction = 'LONG'
                agreed    = True
                # Confidence = geometric mean of how far above threshold each model is
                tcn_strength   = (tcn_prob   - t_tcn)   / (1.0 - t_tcn)
                patch_strength = (patch_prob - t_patch) / (1.0 - t_patch)
                raw_conf       = float(np.sqrt(tcn_strength * patch_strength))
                confidence     = min(round(raw_conf * AGREE_BOOST, 4), 1.0)
                gate           = 'PASS'

            elif both_short:
                direction = 'SHORT'
                agreed    = True
                tcn_strength   = ((1 - t_tcn)   - tcn_prob)   / (1 - t_tcn)
                patch_strength = ((1 - t_patch) - patch_prob) / (1 - t_patch)
                raw_conf       = float(np.sqrt(tcn_strength * patch_strength))
                confidence     = min(round(raw_conf * AGREE_BOOST, 4), 1.0)
                gate           = 'PASS'

            elif not conflict and (tcn_long or patch_long):
                # Only one model fires LONG, other is neutral (not actively short)
                direction = 'LONG'
                agreed    = False
                if tcn_long:
                    strength = (tcn_prob - t_tcn) / (1.0 - t_tcn)
                else:
                    strength = (patch_prob - t_patch) / (1.0 - t_patch)
                confidence = min(round(strength * SINGLE_PENALTY, 4), 1.0)
                gate       = 'PASS_SINGLE'

            elif not conflict and (tcn_short or patch_short):
                # Only one model fires SHORT, other is neutral
                direction = 'SHORT'
                agreed    = False
                if tcn_short:
                    strength = ((1 - t_tcn) - tcn_prob) / (1 - t_tcn)
                else:
                    strength = ((1 - t_patch) - patch_prob) / (1 - t_patch)
                confidence = min(round(strength * SINGLE_PENALTY, 4), 1.0)
                gate       = 'PASS_SINGLE'

            else:
                direction  = 'FLAT'
                agreed     = False
                confidence = 0.0
                gate       = 'FAIL'

            return {
                'pair':       pair,
                'direction':  direction,
                'confidence': confidence,
                'tcn_prob':   round(tcn_prob,   4),
                'patch_prob': round(patch_prob, 4),
                'agreed':     agreed,
                'regime':     regime,
                'gate':       gate,
                'tcn_threshold':   t_tcn,
                'patch_threshold': t_patch,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            logger.error(f"Analyst4 | {pair}: inference failed: {exc}")
            return None

    # ── Regime reader ──────────────────────────────────────────────────────────

    def _get_regime(self, pair: str) -> str:
        """Read current regime from Redis (set by Analyst 1 LightGBM)."""
        try:
            val = self._r.get(f'features:regime:{pair}')
            if val:
                d = json.loads(val)
                return d.get('regime', 'MR')
        except Exception:
            pass
        return 'MR'  # safe default — MR thresholds are middle ground

    # ── Main compute ───────────────────────────────────────────────────────────

    def compute_pair(self, pair: str) -> Optional[Dict]:
        """
        Full pipeline for one pair:
        DB → features → scale → ONNX → Redis

        Returns result dict or None on failure.
        """
        if not self._ready:
            return None

        regime = self._get_regime(pair)

        # PNS — skip entirely, gate will never pass anyway
        if regime == 'PNS':
            result = {
                'pair': pair, 'direction': 'FLAT', 'confidence': 0.0,
                'tcn_prob': 0.5, 'patch_prob': 0.5, 'agreed': False,
                'regime': 'PNS', 'gate': 'FAIL:PNS',
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            self._r.setex(f'features:analyst4:{pair}', CACHE_TTL, json.dumps(result))
            return result

        df = self._load_bars(pair)
        if df is None:
            return None

        df = self._compute_features(df, pair)
        if df is None:
            return None

        result = self._run_inference(df, pair, regime)
        if result is None:
            return None

        # Write to Redis
        self._r.setex(f'features:analyst4:{pair}', CACHE_TTL, json.dumps(result))
        return result

    def compute_all(self) -> Dict[str, Dict]:
        """
        Run inference for all 12 pairs.
        Called by background scheduler every 15 minutes.
        """
        if not self._ready:
            logger.warning("Analyst4 | Not ready — models not loaded")
            return {}

        logger.info("Analyst4 | Starting compute_all for 12 pairs...")
        results = {}
        passes  = 0
        fails   = 0

        for pair in PAIRS:
            result = self.compute_pair(pair)
            if result:
                results[pair] = result
                gate = result.get('gate', 'FAIL')
                if gate in ('PASS', 'PASS_SINGLE'):
                    passes += 1
                    single_tag = " (single)" if gate == 'PASS_SINGLE' else ""
                    logger.info(
                        f"Analyst4 | {pair}: {result['direction']}{single_tag} "
                        f"conf={result['confidence']:.3f} "
                        f"tcn={result['tcn_prob']:.3f} "
                        f"patch={result['patch_prob']:.3f} "
                        f"regime={result['regime']} ✅"
                    )
            else:
                fails += 1
                logger.warning(f"Analyst4 | {pair}: failed")

        logger.success(
            f"Analyst4 | compute_all done | "
            f"PASS={passes} FLAT={len(results)-passes} FAIL={fails}"
        )
        return results

    def get_features(self, pair: str) -> Dict:
        """
        Cache-first get for signal_aggregator.
        Returns cached value if fresh, recomputes if stale.
        """
        cached = self._r.get(f'features:analyst4:{pair}')
        if cached:
            return json.loads(cached)

        logger.debug(f"Analyst4 | {pair}: cache miss — recomputing...")
        result = self.compute_pair(pair)
        if result:
            return result

        # Return neutral FLAT on failure — never block other gates
        return {
            'pair': pair, 'direction': 'FLAT', 'confidence': 0.0,
            'tcn_prob': 0.5, 'patch_prob': 0.5, 'agreed': False,
            'regime': 'UNKNOWN', 'gate': 'FAIL:NO_DATA',
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }

    # ── Background scheduler ───────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """
        Start background daemon thread.
        Runs compute_all() every 15 minutes aligned to M15 bar closes.
        """
        def _run() -> None:
            logger.info("Analyst4 | Scheduler started — running every 15 min")
            # Initial run immediately
            self.compute_all()
            while True:
                time.sleep(REFRESH_SEC)
                try:
                    self.compute_all()
                except Exception as exc:
                    logger.error(f"Analyst4 | Scheduler error: {exc}")

        t = threading.Thread(target=_run, daemon=True, name='analyst4-scheduler')
        t.start()
        logger.info("Analyst4 | Background scheduler thread started")


# ── Module singleton ───────────────────────────────────────────────────────────
analyst4 = Analyst4Inference()


# ── Quick test (run directly) ──────────────────────────────────────────────────
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('/workspaces/Trading_bot/.env')

    print("\n" + "="*60)
    print("ANALYST 4 — TCN + PatchTSMixer Inference Test")
    print("="*60)

    if not analyst4._ready:
        print("❌ Models not loaded — check model files and onnxruntime install")
    else:
        print(f"✅ Models loaded | Features: {len(analyst4._feature_cols)}")
        print(f"\nRunning inference on all 12 pairs...\n")
        print(f"{'Pair':<8} {'Gate':<8} {'Direction':<8} {'Conf':>6} {'TCN':>6} {'Patch':>7} {'Regime'}")
        print("─" * 65)

        results = analyst4.compute_all()
        for pair in PAIRS:
            d = results.get(pair, {})
            gate  = d.get('gate', 'NO_DATA')
            dirn  = d.get('direction', '?')
            conf  = d.get('confidence', 0)
            tcn   = d.get('tcn_prob', 0)
            patch = d.get('patch_prob', 0)
            reg   = d.get('regime', '?')
            icon  = '✅' if gate == 'PASS' else ('⚡' if gate == 'PASS_SINGLE' else '❌')
            print(f"{icon} {pair:<8} {gate:<8} {dirn:<8} {conf:>6.3f} {tcn:>6.3f} {patch:>7.3f} {reg}")
