"""
intelligence/monitoring/telegram_alerts.py
==========================================
Telegram alert system for G10 Dominance Trading Bot.

TWO modes in one module:
  1. PUSH alerts  — bot sends notifications automatically
  2. PULL commands — you send commands, bot replies with live data

Push alerts (called from main_loop / order_executor / risk_manager):
    alerts.send_startup()
    alerts.send_shutdown()
    alerts.send_signal()
    alerts.send_trade_opened()
    alerts.send_trade_closed()
    alerts.send_kill_switch()
    alerts.send_daily_summary()
    alerts.send_error()
    alerts.send_heartbeat()

Pull commands (you type in Telegram, bot replies):
    /status    — uptime, Redis health, active signal count
    /signals   — all 12 pairs: direction, confidence, regime
    /regime    — regime for all 12 pairs
    /positions — open positions from Redis
    /pnl       — today's P&L stats
    /pair XX   — full breakdown for one pair
    /price     — live bid/ask/spread for all 12 pairs
    /pause     — halt new trade execution
    /resume    — resume trade execution
    /restart   — graceful restart via SIGTERM → systemd auto-restart
    /help      — command list

Usage:
    from intelligence.monitoring.telegram_alerts import alerts
    alerts.start_command_bot()   # call once in main_loop startup
    alerts.send_startup(mode="PAPER", balance=1961.51)
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import redis
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv("/workspaces/Trading_bot/.env")

# ──────────────────────────────────────────────────────────────────
PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY",
    # "USDCHF" — SUSPENDED: SNB intervention risk, no signal_aggregator coverage
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

REGIME_EMOJI = {"THV": "🔥", "TLV": "📈", "MR": "↔️", "PNS": "😴"}
DIR_EMOJI    = {"LONG": "🟢▲", "SHORT": "🔴▼", "FLAT": "⬜"}


# ══════════════════════════════════════════════════════════════════
class TelegramAlerts:
    """
    Unified Telegram push-alerts + command-bot for G10 trading system.

    Singleton — import and use `alerts` at module bottom.
    Call alerts.start_command_bot() once during main_loop startup
    to enable the /command polling loop.
    """

    def __init__(self) -> None:
        self._token    = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id  = os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled  = bool(self._token and self._chat_id)
        self._base_url = f"https://api.telegram.org/bot{self._token}"

        self._paused           = False
        self._start_time       = datetime.now(timezone.utc)
        self._last_update_id   = 0
        self._polling_running  = False

        # Redis (lazy)
        self._redis_conn: Optional[redis.Redis] = None

        if not self._enabled:
            logger.warning(
                "TelegramAlerts | Token or Chat ID missing — alerts disabled"
            )
        else:
            logger.info("TelegramAlerts | Initialized ✅")

    # ── Redis ──────────────────────────────────────────────────────

    def _get_redis(self) -> redis.Redis:
        if self._redis_conn is None:
            self._redis_conn = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )
        return self._redis_conn

    def _rget(self, key: str) -> Optional[dict]:
        try:
            raw = self._get_redis().get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    # ── HTTP ───────────────────────────────────────────────────────

    def _api(self, method: str, payload: dict, timeout: int = 10) -> Optional[dict]:
        try:
            r = requests.post(
                f"{self._base_url}/{method}",
                json=payload,
                timeout=timeout,
            )
            data = r.json()
            return data if data.get("ok") else None
        except Exception as exc:
            logger.debug(f"TelegramAlerts | API {method} error: {exc}")
            return None

    def _send(self, text: str) -> None:
        """Fire-and-forget push — never blocks the trading loop."""
        if not self._enabled:
            return
        threading.Thread(
            target=self._send_sync,
            args=(text,),
            daemon=True,
        ).start()

    def _send_sync(self, text: str) -> None:
        """Synchronous send with 3 retries."""
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{self._base_url}/sendMessage",
                    json={
                        "chat_id":                  self._chat_id,
                        "text":                     text,
                        "parse_mode":               "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )
                if r.status_code == 200:
                    return
                logger.warning(
                    f"TelegramAlerts | HTTP {r.status_code}: {r.text[:100]}"
                )
            except Exception as exc:
                logger.warning(
                    f"TelegramAlerts | Send attempt {attempt+1} failed: {exc}"
                )
                time.sleep(2)

    def _reply(self, text: str) -> None:
        """Reply to a command (same as _send but logs it)."""
        logger.debug(f"TelegramAlerts | Reply: {text[:60]}...")
        self._send_sync(text)

    # ══════════════════════════════════════════════════════════════
    # COMMAND BOT — polling loop
    # ══════════════════════════════════════════════════════════════

    def start_command_bot(self) -> None:
        """
        Start the Telegram command polling loop as a daemon thread.
        Call once from main_loop.py startup sequence.
        """
        if not self._enabled:
            logger.warning(
                "TelegramAlerts | Command bot disabled — token/chat_id missing"
            )
            return
        if self._polling_running:
            return

        self._polling_running = True
        t = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="telegram-cmd-bot",
        )
        t.start()
        logger.success("TelegramAlerts | Command bot started ✅")

    def _poll_loop(self) -> None:
        """Long-poll Telegram for incoming commands."""
        logger.info("TelegramAlerts | Command bot polling...")
        while self._polling_running:
            try:
                data = self._api("getUpdates", {
                    "offset":  self._last_update_id + 1,
                    "timeout": 25,  # long-poll seconds (server-side wait)
                }, timeout=35)       # requests read timeout > long-poll
                if data and data.get("result"):
                    for update in data["result"]:
                        self._last_update_id = update["update_id"]
                        self._dispatch(update)
            except Exception as exc:
                logger.warning(f"TelegramAlerts | Poll error: {exc}")
            time.sleep(1)

    def _dispatch(self, update: dict) -> None:
        """Route an incoming update to the correct command handler."""
        message = update.get("message", {})
        text    = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))

        # Security: only respond to your own chat
        if chat_id != self._chat_id:
            logger.warning(
                f"TelegramAlerts | Rejected message from chat {chat_id}"
            )
            return

        if not text:
            return

        parts = text.split(maxsplit=1)
        cmd   = parts[0].lower().split("@")[0]
        arg   = parts[1].strip() if len(parts) > 1 else ""

        logger.info(f"TelegramAlerts | Command: {cmd!r} arg={arg!r}")

        handlers = {
            "/start":     self._cmd_start,
            "/help":      self._cmd_help,
            "/status":    self._cmd_status,
            "/signals":   self._cmd_signals,
            "/groq":      self._cmd_groq,
            "/regime":    self._cmd_regime,
            "/positions": self._cmd_positions,
            "/pnl":       self._cmd_pnl,
            "/pair":      self._cmd_pair,
            "/price":     self._cmd_price,
            "/pause":     self._cmd_pause,
            "/resume":    self._cmd_resume,
            "/restart":   self._cmd_restart,
            "/postmortem":   self._cmd_postmortem,
            "/apex":      self._cmd_apex,
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                handler(arg)
            except Exception as exc:
                logger.error(f"TelegramAlerts | Command {cmd} failed: {exc}")
                self._reply(f"❌ Error running {cmd}:\n<code>{exc}</code>")
        else:
            self._reply(
                f"❓ Unknown command: <code>{cmd}</code>\n"
                f"Type /help for the command list."
            )

    # ── Command handlers ───────────────────────────────────────────

    def _cmd_start(self, _: str) -> None:
        self._reply(
            "🤖 <b>G10 Dominance Bot</b>\n\n"
            "Autonomous G10 FX trading system.\n"
            "Type /help for all commands."
        )

    def _cmd_help(self, _: str) -> None:
        self._reply(
            "📋 <b>Commands</b>\n\n"
            "/status — Bot health and uptime\n"
            "/signals — All 12 pair signals\n"
            "/groq — Latest Groq sentiment analysis\n"
            "/regime — Market regimes\n"
            "/positions — Open positions\n"
            "/pnl — Today's P&amp;L\n"
            "/pair EURUSD — Full pair detail\n"
            "/price — Live bid/ask for all pairs\n"
            "/pause — Halt new trades\n"
            "/resume — Resume trading\n"
            "/restart — Restart bot service\n"
            "/postmortem — Weekly loss analysis\n"
            "/apex — APEX penalty rules status\n"
            "/help — This menu"
        )

    def _cmd_status(self, _: str) -> None:
        uptime  = datetime.now(timezone.utc) - self._start_time
        hours   = int(uptime.total_seconds()) // 3600
        mins    = (int(uptime.total_seconds()) % 3600) // 60
        state   = "⏸ PAUSED" if self._paused else "✅ RUNNING"

        try:
            self._get_redis().ping()
            redis_ok = "✅"
        except Exception:
            redis_ok = "❌"

        active = sum(
            1 for p in PAIRS
            if (self._rget(f"signal:{p}") or {}).get("action") not in ("FLAT", None)
        )

        self._reply(
            f"🤖 <b>Bot Status</b>\n\n"
            f"State: {state}\n"
            f"Uptime: <code>{hours}h {mins}m</code>\n"
            f"Redis: {redis_ok}\n"
            f"Active signals: <code>{active}/{len(PAIRS)}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def _cmd_signals(self, _: str) -> None:
        lines = ["📡 <b>Current Signals</b>\n"]
        for pair in PAIRS:
            sig  = self._rget(f"signal:{pair}") or {}
            act  = sig.get("action", "FLAT")
            conf = sig.get("confidence", 0.0)
            reg  = sig.get("regime", "???")
            size = sig.get("size_mult", sig.get("size_multiplier", 1.0))  # fix key name
            g1   = sig.get("gate1", "?")[:12]
            g2   = sig.get("gate2", "?")[:12]
            g3   = sig.get("gate3", "?")[:28]
            g4   = sig.get("gate4", "?")[:20]
            a2s  = sig.get("a2_score", 0.0)
            de   = DIR_EMOJI.get(act, "⬜")
            re   = REGIME_EMOJI.get(reg, "")

            # HMM indicator: show when HMM disagrees with A1
            hmm_reg  = sig.get("hmm_regime", "")
            gate1b   = sig.get("gate1b", "")
            hmm_tag  = ""
            if "HMM_UPGRADE" in gate1b and hmm_reg:
                hmm_tag = f"🔮{hmm_reg}"   # HMM saw trend A1 missed
            elif "A1_LEADS" in gate1b and hmm_reg:
                hmm_tag = f"⚡{hmm_reg}"   # A1 saw trend HMM missed

            path  = sig.get("path", "FUNDAMENTAL")
            is_tc = path == "TECH_CONFLUENCE"

            if act in ("LONG", "SHORT"):
                hmm_str = f"  {hmm_tag}" if hmm_tag else ""
                if is_tc:
                    d1s = sig.get("gate1", "")
                    h1s = sig.get("gate2", "")
                    lines.append(
                        f"{de} <b>{pair}</b>  conf=<b>{conf:.2f}</b>  {re}{reg}  {size:.2f}x  🔵<b>TC</b>{hmm_str}\n"
                        f"   {d1s[:24]}  {h1s[:24]}\n"
                        f"   A4:{g3[:28]}\n"
                        f"   Entry:{g4[:20]}"
                    )
                else:
                    lines.append(
                        f"{de} <b>{pair}</b>  conf=<b>{conf:.2f}</b>  {re}{reg}  {size:.2f}x{hmm_str}\n"
                        f"   G1:{g1}  G2:{g2}\n"
                        f"   G3-ML:{g3}\n"
                        f"   G4:{g4}  A2={a2s:.3f}"
                    )
            elif sig.get("reason") == "HEDGE_ACTIVE":
                hedge = self._rget(f"hedge:active:{pair}") or {}
                cycles  = hedge.get("cycle_count", 0)
                banked  = hedge.get("realized_pnl", 0.0)
                status  = hedge.get("status", "ACTIVE")
                lines.append(
                    f"🔀 <b>{pair}</b>  ↔️MR  🔁HEDGE [{status}]  "
                    f"cycles={cycles}  banked=${banked:+.2f}"
                )
            else:
                # FLAT — show why it was blocked + HMM tag if relevant
                block = g1 if not g1.startswith("PASS") else (
                        g2 if not g2.startswith("PASS") else (
                        g3 if not g3.startswith("PASS") else g4))
                hmm_str = f"|{hmm_tag}" if hmm_tag else ""
                lines.append(
                    f"{de} <b>{pair}</b>  {re}{reg}  ✗{block}{hmm_str}"
                )
        self._reply("\n".join(lines))

    def _cmd_groq(self, _: str) -> None:
        """Send latest Groq sentiment analysis for all pairs."""
        lines = ["🤖 <b>Groq Sentiment Analysis</b>\n"]
        any_data = False
        for pair in PAIRS:
            val = self._rget(f"sentiment:groq:{pair}") or self._rget(f"features:sentiment:{pair}")
            if not val:
                continue
            any_data  = True
            score     = val.get("sentiment_score", val.get("sentiment", 0.0))
            conf      = val.get("confidence", 0.0)
            stance    = val.get("stance", val.get("bias", "NEUTRAL"))
            updated   = val.get("updated_at", "?")[:16]
            bar       = "▲" if score > 0.1 else ("▼" if score < -0.1 else "➡️")
            lines.append(
                f"{bar} <b>{pair}</b>  score={score:+.3f}  conf={conf:.2f}\n"
                f"   {stance}  <i>{updated}</i>"
            )
        if not any_data:
            lines.append("No Groq data available — runs daily at market open")
        # Add Opus weekly summary
        opus = self._rget("features:opus:weekly")
        if opus:
            regime    = opus.get("market_regime", "?")
            summary   = opus.get("strategist_summary", "")[:200]
            week_of   = opus.get("week_of", "?")
            top       = opus.get("top_trades", [])
            risks     = opus.get("key_risks", [])
            biases    = opus.get("currency_biases", {})
            # Currency bias summary
            bias_str  = "  ".join([
                f"{cur}:{v.get('bias','?')[:4]}"
                for cur, v in biases.items()
            ]) if biases else "?"
            # Top trades
            trades_str = "\n   ".join([f"• {t}" for t in top[:3]]) if top else "?"
            # Key risks
            risks_str  = "\n   ".join([f"⚠️ {r}" for r in risks[:2]]) if risks else "?"
            lines.append(
                f"\n🔭 <b>Opus Weekly</b>  <i>w/o {week_of}</i>\n"
                f"   Regime: <b>{regime}</b>\n"
                f"   {summary}\n"
                f"\n   <b>Currency biases:</b>\n   {bias_str}\n"
                f"\n   <b>Top trades:</b>\n   {trades_str}\n"
                f"\n   <b>Risks:</b>\n   {risks_str}"
            )
        self._reply("\n".join(lines))

    def _cmd_regime(self, _: str) -> None:
        lines = ["🌡 <b>Market Regimes</b>\n"]
        for pair in PAIRS:
            sig = self._rget(f"signal:{pair}") or {}
            reg = sig.get("regime", "???")
            re  = REGIME_EMOJI.get(reg, "❓")
            lines.append(f"  {re} <b>{pair}</b>: {reg}")
        self._reply("\n".join(lines))

    def _cmd_positions(self, _: str) -> None:
        # Read execution:open:{pair} keys written by order_executor
        import redis as _redis, os as _os, json as _json
        try:
            _r = _redis.Redis(host="localhost", port=6379,
                password=_os.getenv("REDIS_PASSWORD"), decode_responses=True)
            pairs = ["EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD",
                     "USDCAD","EURGBP","EURJPY","GBPJPY","AUDJPY","NZDJPY"]
            positions = {}
            for p in pairs:
                val = _r.get(f"execution:open:{p}")
                if val:
                    positions[p] = _json.loads(val)
        except Exception:
            positions = {}
        if not positions:
            self._reply("🔭 No open positions")
            return

        lines = ["📂 <b>Open Positions</b>\n"]
        for pos_id, pos in positions.items():
            pair  = pos.get("pair",           pos.get("symbol", "???"))
            side  = pos.get("action",         pos.get("direction", "???"))
            entry = pos.get("entry_price",    0.0)
            pnl   = pos.get("unrealized_pnl", 0.0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            pnl_s = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
            lines.append(
                f"{emoji} <b>{pair}</b> {side}  "
                f"@{entry:.5f}  P&amp;L: ${pnl_s}"
            )
        self._reply("\n".join(lines))

    def _cmd_pnl(self, _: str) -> None:
        state       = self._rget("risk:account_state") or {}
        balance     = state.get("balance",       0.0)
        daily_start = state.get("daily_start",   balance)

        trades, wins, losses, max_dd, realized = 0, 0, 0, 0.0, 0.0
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text
            today = datetime.now(timezone.utc).date()
            with db.connection() as conn:
                row = conn.execute(text("""
                    SELECT
                        COUNT(*)                                        AS trades,
                        SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END)  AS wins,
                        SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END)  AS losses,
                        MIN(pnl_usd / NULLIF(:start, 0) * 100)         AS max_dd,
                        COALESCE(SUM(pnl_usd), 0.0)                    AS realized
                    FROM trades
                    WHERE exit_time::date = :today
                      AND status LIKE 'CLOSED%'
                """), {"today": today, "start": daily_start}).fetchone()
                if row and row[0]:
                    trades  = int(row[0]  or 0)
                    wins    = int(row[1]  or 0)
                    losses  = int(row[2]  or 0)
                    max_dd  = abs(float(row[3] or 0.0))
                    realized = float(row[4] or 0.0)
        except Exception as exc:
            logger.warning(f"Telegram | /pnl DB query failed: {exc}")

        wr    = (wins / trades * 100) if trades > 0 else 0.0
        emoji = "🟢" if realized >= 0 else "🔴"
        pnl_s = f"+${realized:.2f}" if realized >= 0 else f"-${abs(realized):.2f}"
        self._reply(
            f"💰 <b>Today's P&amp;L</b>\n\n"
            f"{emoji} Realized: <code>{pnl_s}</code>\n"
            f"Trades: <code>{trades}</code>  W/L: <code>{wins}/{losses}</code>\n"
            f"Win rate: <code>{wr:.1f}%</code>\n"
            f"Max drawdown: <code>{max_dd:.2f}%</code>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def _cmd_pair(self, arg: str) -> None:
        pair = arg.upper()
        if pair not in PAIRS:
            self._reply(
                f"❌ Unknown pair: <code>{pair}</code>\n"
                f"Valid: {', '.join(PAIRS)}"
            )
            return

        sig = self._rget(f"signal:{pair}")
        if not sig:
            self._reply(f"❌ No signal data for <b>{pair}</b>")
            return

        act  = sig.get("action", "FLAT")
        conf = sig.get("confidence", 0.0)
        reg  = sig.get("regime", "???")
        size = sig.get("size_multiplier", 1.0)
        de   = DIR_EMOJI.get(act, "⬜")
        re   = REGIME_EMOJI.get(reg, "")

        lines = [f"🔍 <b>{pair} Detail</b>\n"]
        lines.append(f"Signal: {de} <b>{act}</b>")
        lines.append(f"Confidence: <code>{conf:.2f}</code>")
        lines.append(f"Regime: {re} <code>{reg}</code>")
        lines.append(f"Size: <code>{size}x</code>")

        scores = sig.get("analyst_scores", {})
        if scores:
            lines.append("\n🧠 <b>Analyst Scores</b>")
            for k, v in scores.items():
                lines.append(f"  {k}: <code>{v}</code>")

        yf = self._rget(f"features:yield:{pair}")
        if yf:
            spread = yf.get("yield_spread_10y")
            carry  = yf.get("carry_score")
            lines.append("\n📈 <b>Yield</b>")
            if spread is not None:
                lines.append(f"  Spread 10Y: <code>{spread:.3f}%</code>")
            if carry is not None:
                lines.append(f"  Carry score: <code>{carry:.3f}</code>")

        sf = self._rget(f"features:statistical:{pair}")
        if sf:
            atr    = sf.get("atr_14")
            zscore = sf.get("price_zscore_20")
            lines.append("\n📊 <b>Stats</b>")
            if atr is not None:
                lines.append(f"  ATR-14: <code>{atr:.5f}</code>")
            if zscore is not None:
                lines.append(f"  Z-score 20: <code>{zscore:.2f}</code>")

        self._reply("\n".join(lines))

    def _cmd_price(self, _: str) -> None:
        """Fetch latest bid/ask/spread for all 12 pairs."""
        from sqlalchemy import text as sa_text
        from data_pipeline.storage.timeseries_db import db

        prices = {}

        # ── DB ticks first (FIX collector flushes here) ──────────
        try:
            with db.connection() as conn:
                for pair in PAIRS:
                    row = conn.execute(sa_text("""
                        SELECT bid, ask
                        FROM ticks
                        WHERE symbol = :sym
                          AND time > NOW() - INTERVAL '10 minutes'
                        ORDER BY time DESC
                        LIMIT 1
                    """), {"sym": pair}).fetchone()

                    if row:
                        bid, ask = float(row[0]), float(row[1])
                        if pair in JPY_PAIRS and bid > 1000:
                            bid /= 100.0
                            ask /= 100.0
                        prices[pair] = {
                            "bid":    bid,
                            "ask":    ask,
                            "spread": round(
                                (ask - bid) * (100 if pair in JPY_PAIRS else 10000), 1
                            ),
                            "source": "live",
                        }
        except Exception as exc:
            logger.warning(f"TelegramAlerts | /price DB query failed: {exc}")

        # ── Fallback: VWAP from Redis for missing pairs ───────────
        for pair in PAIRS:
            if pair in prices:
                continue
            try:
                raw = self._get_redis().get(f"features:vwap:{pair}")
                if not raw:
                    continue
                vwap_data = json.loads(raw)
                vwap = vwap_data.get("ny_vwap") or vwap_data.get("london_vwap")
                if not vwap or float(vwap) == 0:
                    continue
                stat_raw = self._get_redis().get(f"features:stat:{pair}")
                atr  = json.loads(stat_raw).get("atr_14", 0.0001) if stat_raw else 0.0001
                half = float(atr) * 0.05
                bid  = round(float(vwap) - half, 5)
                ask  = round(float(vwap) + half, 5)
                prices[pair] = {
                    "bid":    bid,
                    "ask":    ask,
                    "spread": round(
                        half * 2 * (100 if pair in JPY_PAIRS else 10000), 1
                    ),
                    "source": "vwap",
                }
            except Exception:
                pass

        # ── Format ────────────────────────────────────────────────
        now   = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        lines = [f"💹 <b>Live Prices</b> — {now}\n"]

        for pair in PAIRS:
            if pair not in prices:
                lines.append(f"<code>{pair:<8}</code> ❌ no data")
                continue
            p   = prices[pair]
            dp  = 3 if pair in JPY_PAIRS else 5
            tag = "~" if p["source"] == "vwap" else ""
            lines.append(
                f"<code>{pair:<8}</code> "
                f"B:<code>{p['bid']:.{dp}f}</code> "
                f"A:<code>{p['ask']:.{dp}f}</code> "
                f"Spd:<code>{p['spread']}</code>{tag}"
            )

        lines.append("\n<i>~ = VWAP estimate (no recent tick)</i>")
        self._reply("\n".join(lines))

    def _cmd_pause(self, _: str) -> None:
        self._paused = True
        logger.warning("TelegramAlerts | Trading PAUSED via Telegram command")
        self._reply(
            "⏸ <b>Trading PAUSED</b>\n\n"
            "No new orders will be placed.\n"
            "Open positions continue to be monitored.\n"
            "Send /resume to restart trading."
        )

    def _cmd_resume(self, _: str) -> None:
        self._paused = False
        logger.info("TelegramAlerts | Trading RESUMED via Telegram command")
        self._reply("▶️ <b>Trading RESUMED</b>\nNew signals will be executed normally.")

    def _cmd_restart(self, _: str) -> None:
        """Gracefully restart the bot via SIGTERM → systemd auto-restart."""
        logger.warning("TelegramAlerts | Restart requested via Telegram")
        self._reply(
            "🔄 <b>Restarting bot...</b>\n\n"
            "Sending graceful shutdown signal.\n"
            "systemd will restart the service automatically.\n"
            "Bot will be back online in ~15 seconds."
        )
        # Small delay so the reply is sent before the process exits
        threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

    def _cmd_postmortem(self, _: str) -> None:
        """Send weekly postmortem loss analysis summary."""
        self._reply("🔍 Generating postmortem report...")
        try:
            from intelligence.postmortem_analyzer import postmortem_analyzer
            msg = postmortem_analyzer.weekly_summary(days_back=7)
            self._reply(msg)
        except Exception as exc:
            logger.error(f"TelegramAlerts | /postmortem failed: {exc}")
            self._reply(f"⚠️ Postmortem failed: {exc}")

    def _cmd_apex(self, _: str) -> None:
        """Show APEX framework status and active penalty rules."""
        try:
            from intelligence.apex_framework import apex_framework
            self._reply(apex_framework.get_summary_text())
        except Exception as exc:
            logger.error(f"TelegramAlerts | /apex failed: {exc}")
            self._reply(f"⚠️ APEX status failed: {exc}")

    # ══════════════════════════════════════════════════════════════
    # PUSH ALERT METHODS (called from main_loop / executors)
    # ══════════════════════════════════════════════════════════════

    @property
    def is_paused(self) -> bool:
        """Check from main_loop before executing any order."""
        return self._paused

    def send_startup(self, mode: str = "PAPER", balance: float = 0.0) -> None:
        self._send(
            f"🟢 <b>G10 Bot — ONLINE</b>\n"
            f"Mode: <code>{mode}</code>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Commands: /help\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_shutdown(self, reason: str = "Manual stop") -> None:
        self._send(
            f"🔴 <b>G10 Bot — OFFLINE</b>\n"
            f"Reason: <code>{reason}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_heartbeat(
        self, pairs_watching: int = 12, open_trades: int = 0
    ) -> None:
        self._send(
            f"💓 <b>Heartbeat</b>\n"
            f"Watching: <code>{pairs_watching} pairs</code>\n"
            f"Open trades: <code>{open_trades}</code>\n"
            f"Paused: <code>{self._paused}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_signal(
        self,
        pair:       str,
        action:     str,
        conf:       float,
        size:       float,
        regime:     str,
        a2_score:   float = 0.0,
        opus_bias:  str   = "NEUTRAL",
        sentiment:  str   = "NEUTRAL",
    ) -> None:
        arrow = "🟢 LONG" if action == "LONG" else "🔴 SHORT"
        self._send(
            f"📡 <b>Signal: {pair}</b>\n"
            f"Direction: <b>{arrow}</b>\n"
            f"Confidence: <code>{conf:.0%}</code>\n"
            f"Size: <code>{size}x</code>  Regime: <code>{regime}</code>\n"
            f"A2 Score: <code>{a2_score:+.3f}</code>\n"
            f"Opus: <code>{opus_bias}</code>  "
            f"Sentiment: <code>{sentiment}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_trade_opened(
        self,
        pair:             str,
        action:           str,
        entry:            float,
        sl:               float,
        tp:               float,
        size_lots:        float = 0.01,
        account_risk_pct: float = 2.0,
    ) -> None:
        arrow = "🟢 LONG" if action == "LONG" else "🔴 SHORT"
        self._send(
            f"✅ <b>Trade OPENED: {pair}</b>\n"
            f"Direction: <b>{arrow}</b>\n"
            f"Entry: <code>{entry:.5f}</code>\n"
            f"SL: <code>{sl:.5f}</code>  "
            f"TP: <code>{tp:.5f}</code>\n"
            f"Size: <code>{size_lots} lots</code>  "
            f"Risk: <code>{account_risk_pct:.1f}%</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_trade_closed(
        self,
        pair:       str,
        action:     str,
        entry:      float,
        exit_price: float,
        pnl_usd:    float,
        pnl_pct:    float,
        reason:     str = "TP/SL",
    ) -> None:
        emoji  = "💰" if pnl_usd >= 0 else "💸"
        result = "WIN" if pnl_usd >= 0 else "LOSS"
        self._send(
            f"{emoji} <b>Trade CLOSED: {pair} — {result}</b>\n"
            f"Direction: <code>{action}</code>\n"
            f"Entry: <code>{entry:.5f}</code> → "
            f"Exit: <code>{exit_price:.5f}</code>\n"
            f"P&amp;L: <code>{pnl_usd:+.2f} USD ({pnl_pct:+.2f}%)</code>\n"
            f"Reason: <code>{reason}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_kill_switch(
        self, daily_pnl_pct: float, reason: str = "Daily loss limit"
    ) -> None:
        self._send(
            f"🚨 <b>KILL SWITCH TRIGGERED</b>\n"
            f"Reason: <code>{reason}</code>\n"
            f"Daily P&amp;L: <code>{daily_pnl_pct:+.2f}%</code>\n"
            f"All trading HALTED until next session.\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_regime_change(
        self, pair: str, old_regime: str, new_regime: str
    ) -> None:
        self._send(
            f"🔄 <b>Regime Change: {pair}</b>\n"
            f"{old_regime} → <b>{new_regime}</b>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_event_blackout(
        self, event_name: str, currency: str, minutes: int
    ) -> None:
        self._send(
            f"⏸ <b>Event Blackout Active</b>\n"
            f"Event: <code>{event_name}</code> ({currency})\n"
            f"Trading paused for <code>{minutes} minutes</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_daily_summary(
        self,
        trades:     int,
        wins:       int,
        losses:     int,
        pnl_usd:    float,
        pnl_pct:    float,
        balance:    float,
        best_pair:  str = "N/A",
        worst_pair: str = "N/A",
    ) -> None:
        wr    = (wins / trades * 100) if trades > 0 else 0
        emoji = "📈" if pnl_usd >= 0 else "📉"
        self._send(
            f"{emoji} <b>Daily Summary</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Trades: <code>{trades}</code>  "
            f"W/L: <code>{wins}/{losses}</code>  "
            f"WR: <code>{wr:.0f}%</code>\n"
            f"P&amp;L: <code>{pnl_usd:+.2f} USD ({pnl_pct:+.2f}%)</code>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Best: <code>{best_pair}</code>  "
            f"Worst: <code>{worst_pair}</code>\n"
            f"Date: <code>{self._now()}</code>"
        )

    def send_error(self, component: str, error: str) -> None:
        self._send(
            f"⚠️ <b>Error: {component}</b>\n"
            f"<code>{error[:300]}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    def send_data_feed_warning(self, feed: str, last_update: str) -> None:
        self._send(
            f"📡 <b>Data Feed Warning</b>\n"
            f"Feed: <code>{feed}</code>\n"
            f"Last update: <code>{last_update}</code>\n"
            f"Time: <code>{self._now()}</code>"
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ──────────────────────────────────────────────────────────────────
# Module singleton
# ──────────────────────────────────────────────────────────────────
alerts = TelegramAlerts()
