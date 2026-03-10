"""
intelligence/monitoring/telegram_alerts.py
==========================================
Telegram alert system for G10 Dominance Trading Bot.

Sends real-time notifications for:
  - Bot startup / shutdown
  - New trade signals generated
  - Trade opened / closed with P&L
  - Daily performance summary
  - Kill switch triggered
  - Errors and warnings
  - Regime changes
  - Data feed failures

Usage:
    from intelligence.monitoring.telegram_alerts import alerts
    alerts.send_signal(pair="EURUSD", action="LONG", conf=0.72, size=1.0, regime="THV")
    alerts.send_trade_opened(pair="EURUSD", action="LONG", entry=1.0850, sl=1.0820, tp=1.0900)
    alerts.send_daily_summary(trades=5, wins=3, pnl_pct=1.2, balance=1961.51)
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from loguru import logger
import requests


class TelegramAlerts:
    """
    Telegram notification system.

    All methods are fire-and-forget — they run in a background thread
    so they never block the main trading loop.
    """

    def __init__(self) -> None:
        self._token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled: bool = bool(self._token and self._chat_id)
        self._base_url: str = f"https://api.telegram.org/bot{self._token}/sendMessage"

        if not self._enabled:
            logger.warning("TelegramAlerts | TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing — alerts disabled")
        else:
            logger.info("TelegramAlerts | Initialized ✅")

    # ── Core send ──────────────────────────────────────────────────────────

    def _send(self, text: str, parse_mode: str = "HTML") -> None:
        """Send message in background thread — never blocks."""
        if not self._enabled:
            return
        threading.Thread(
            target=self._send_sync,
            args=(text, parse_mode),
            daemon=True
        ).start()

    def _send_sync(self, text: str, parse_mode: str = "HTML") -> None:
        """Synchronous send with retry."""
        for attempt in range(3):
            try:
                r = requests.post(
                    self._base_url,
                    json={
                        "chat_id": self._chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )
                if r.status_code == 200:
                    return
                logger.warning(f"TelegramAlerts | HTTP {r.status_code}: {r.text[:100]}")
            except Exception as exc:
                logger.warning(f"TelegramAlerts | Attempt {attempt+1} failed: {exc}")
                time.sleep(2)

    # ── Bot lifecycle ──────────────────────────────────────────────────────

    def send_startup(self, mode: str = "PAPER", balance: float = 0.0) -> None:
        """Bot started successfully."""
        self._send(
            f"🟢 <b>G10 Bot — ONLINE</b>\n"
            f"Mode: <code>{mode}</code>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_shutdown(self, reason: str = "Manual stop") -> None:
        """Bot stopped."""
        self._send(
            f"🔴 <b>G10 Bot — OFFLINE</b>\n"
            f"Reason: <code>{reason}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_heartbeat(self, pairs_watching: int = 12, open_trades: int = 0) -> None:
        """Hourly heartbeat so you know the bot is alive."""
        self._send(
            f"💓 <b>Heartbeat</b>\n"
            f"Watching: <code>{pairs_watching} pairs</code>\n"
            f"Open trades: <code>{open_trades}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    # ── Signals ───────────────────────────────────────────────────────────

    def send_signal(
        self,
        pair: str,
        action: str,
        conf: float,
        size: float,
        regime: str,
        a2_score: float = 0.0,
        opus_bias: str = "NEUTRAL",
        sentiment: str = "NEUTRAL",
    ) -> None:
        """New trade signal generated (may or may not execute)."""
        arrow = "🟢 LONG" if action == "LONG" else "🔴 SHORT"
        self._send(
            f"📡 <b>Signal: {pair}</b>\n"
            f"Direction: <b>{arrow}</b>\n"
            f"Confidence: <code>{conf:.0%}</code>\n"
            f"Size: <code>{size}x</code>  Regime: <code>{regime}</code>\n"
            f"A2 Score: <code>{a2_score:+.3f}</code>\n"
            f"Opus: <code>{opus_bias}</code>  Sentiment: <code>{sentiment}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    # ── Trades ────────────────────────────────────────────────────────────

    def send_trade_opened(
        self,
        pair: str,
        action: str,
        entry: float,
        sl: float,
        tp: float,
        size_lots: float = 0.01,
        account_risk_pct: float = 2.0,
    ) -> None:
        """Trade order placed."""
        arrow = "🟢 LONG" if action == "LONG" else "🔴 SHORT"
        self._send(
            f"✅ <b>Trade OPENED: {pair}</b>\n"
            f"Direction: <b>{arrow}</b>\n"
            f"Entry: <code>{entry:.5f}</code>\n"
            f"SL: <code>{sl:.5f}</code>  TP: <code>{tp:.5f}</code>\n"
            f"Size: <code>{size_lots} lots</code>  Risk: <code>{account_risk_pct:.1f}%</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_trade_closed(
        self,
        pair: str,
        action: str,
        entry: float,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        reason: str = "TP/SL",
    ) -> None:
        """Trade closed — win or loss."""
        if pnl_usd >= 0:
            emoji = "💰"
            result = "WIN"
        else:
            emoji = "💸"
            result = "LOSS"

        self._send(
            f"{emoji} <b>Trade CLOSED: {pair} — {result}</b>\n"
            f"Direction: <code>{action}</code>\n"
            f"Entry: <code>{entry:.5f}</code> → Exit: <code>{exit_price:.5f}</code>\n"
            f"P&L: <code>{pnl_usd:+.2f} USD ({pnl_pct:+.2f}%)</code>\n"
            f"Reason: <code>{reason}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    # ── Risk alerts ───────────────────────────────────────────────────────

    def send_kill_switch(self, daily_pnl_pct: float, reason: str = "Daily loss limit") -> None:
        """Kill switch triggered — all trading stopped."""
        self._send(
            f"🚨 <b>KILL SWITCH TRIGGERED</b>\n"
            f"Reason: <code>{reason}</code>\n"
            f"Daily P&L: <code>{daily_pnl_pct:+.2f}%</code>\n"
            f"All trading HALTED until next session.\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_regime_change(self, pair: str, old_regime: str, new_regime: str) -> None:
        """Market regime changed for a pair."""
        self._send(
            f"🔄 <b>Regime Change: {pair}</b>\n"
            f"{old_regime} → <b>{new_regime}</b>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_event_blackout(self, event_name: str, currency: str, minutes: int) -> None:
        """Pre-event blackout activated."""
        self._send(
            f"⏸ <b>Event Blackout Active</b>\n"
            f"Event: <code>{event_name}</code> ({currency})\n"
            f"Trading paused for <code>{minutes} minutes</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    # ── Daily summary ─────────────────────────────────────────────────────

    def send_daily_summary(
        self,
        trades: int,
        wins: int,
        losses: int,
        pnl_usd: float,
        pnl_pct: float,
        balance: float,
        best_pair: str = "N/A",
        worst_pair: str = "N/A",
    ) -> None:
        """End of day performance summary."""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        emoji = "📈" if pnl_usd >= 0 else "📉"

        self._send(
            f"{emoji} <b>Daily Summary</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Trades: <code>{trades}</code>  W/L: <code>{wins}/{losses}</code>  WR: <code>{win_rate:.0f}%</code>\n"
            f"P&L: <code>{pnl_usd:+.2f} USD ({pnl_pct:+.2f}%)</code>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Best: <code>{best_pair}</code>  Worst: <code>{worst_pair}</code>\n"
            f"Date: <code>{self._now()}</code>"
        )

    # ── Errors ────────────────────────────────────────────────────────────

    def send_error(self, component: str, error: str) -> None:
        """Critical error in a component."""
        self._send(
            f"⚠️ <b>Error: {component}</b>\n"
            f"<code>{error[:200]}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_data_feed_warning(self, feed: str, last_update: str) -> None:
        """Data feed stale or failing."""
        self._send(
            f"📡 <b>Data Feed Warning</b>\n"
            f"Feed: <code>{feed}</code>\n"
            f"Last update: <code>{last_update}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# Module singleton
alerts = TelegramAlerts()
