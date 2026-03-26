"""
models/analyst3_tft/inference.py
=================================
Analyst 3 — TFT inference (DEFERRED).

TFT v1 achieved 52.5% directional accuracy — not sufficient for
deployment. Slot reserved for v2 retraining on Colab Pro+ with
larger dataset and better hyperparameters.

Signal aggregator handles available=False gracefully — Gate 2
auto-passes when technical analysts are unavailable.
"""
from __future__ import annotations
from typing import Dict

NEUTRAL_SIGNAL: Dict = {
    "bias": "NEUTRAL", "confidence": 0.0, "direction": "FLAT",
    "raw_up": 0.5, "raw_down": 0.5, "available": False,
}

class TFTAnalyst:
    def get_signal(self, pair: str) -> Dict:
        return dict(NEUTRAL_SIGNAL)
    def compute_all(self) -> Dict:
        return {p: dict(NEUTRAL_SIGNAL) for p in [
            "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","NZDUSD",
            "USDCAD","EURGBP","EURJPY","GBPJPY","AUDJPY","NZDJPY",
        ]}
    def start_scheduler(self) -> None:
        pass

tft_analyst = TFTAnalyst()
