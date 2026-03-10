"""
data_pipeline/connectors/ctrader_fix.py
========================================
Dedicated FIX 4.4 price session collector for G10 tick + L2 data.

Uses the official cTraderFixPy library (pip install ctrader-fix).
Runs as a background thread — completely separate from the OpenAPI
REST and live stream connectors, eliminating the ALREADY_LOGGED_IN
conflict entirely.

What this collects:
  - Every bid/ask tick for all 12 G10 pairs → ticks table
  - L2 order book snapshots (top 10 levels) → order_book_snapshots table
  - Redis cache for real-time feature engine access

FIX session config (Pepperstone demo):
  Host:          demo-uk-eqx-01.p.c-trader.com
  Port:          5201
  SenderCompID:  demo.pepperstone.XXXXXXXX
  TargetCompID:  cServer
  SenderSubID:   QUOTE
  TargetSubID:   QUOTE
  BeginString:   FIX.4.4

Redis keys written:
  tick:EURUSD        → {"bid": 1.0850, "ask": 1.0852, "time": "..."}
  orderbook:EURUSD   → {"bids": [...], "asks": [...], "time": "..."}

Usage:
    from data_pipeline.connectors.ctrader_fix import fix_collector
    fix_collector.start()   # starts background thread
    fix_collector.stop()    # graceful shutdown
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path("/workspaces/Trading_bot")
import sys
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(str(ROOT / ".env"))


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ── G10 pairs to subscribe ────────────────────────────────────────────────────
G10_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

# ── FIX message type constants ────────────────────────────────────────────────
MSG_LOGON                        = "A"
MSG_LOGOUT                       = "5"
MSG_HEARTBEAT                    = "0"
MSG_TEST_REQUEST                 = "1"
MSG_MARKET_DATA_REQUEST          = "V"
MSG_MARKET_DATA_SNAPSHOT         = "W"
MSG_MARKET_DATA_INCREMENTAL      = "X"
MSG_MARKET_DATA_REJECT           = "Y"

# MDEntryType values
MD_BID   = "0"
MD_ASK   = "1"
MD_TRADE = "2"

# ── DB batch config ───────────────────────────────────────────────────────────
TICK_BATCH_SIZE    = 100   # flush ticks to DB every N ticks
TICK_FLUSH_SECS    = 5     # or every N seconds, whichever comes first


class FIXTickCollector:
    """
    Dedicated FIX 4.4 price session for tick + L2 collection.

    Runs in a background daemon thread. Automatically reconnects
    on disconnect. Stores data to TimescaleDB and Redis.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._connected = threading.Event()
        self._logged_in = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Tick batch buffer
        self._tick_buffer: List[Dict] = []
        self._last_flush = time.time()

        # Request ID counter
        self._req_id = 0

        # Redis (lazy init)
        self._redis = None

        # Config from .env
        self._host     = _get("CTRADER_FIX_HOST", "demo-uk-eqx-01.p.c-trader.com")
        self._port     = int(_get("CTRADER_FIX_PORT", "5201"))
        self._username = _get("CTRADER_FIX_USERNAME")
        self._password = _get("CTRADER_FIX_PASSWORD")
        self._sender   = _get("CTRADER_SENDER_COMP_ID")
        self._target   = _get("CTRADER_TARGET_COMP_ID", "cServer")

        logger.info(
            f"FIXCollector | host={self._host}:{self._port} "
            f"sender={self._sender[:20]}..."
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start FIX collector in background daemon thread."""
        t = threading.Thread(
            target=self._run_loop,
            name="fix-collector",
            daemon=True
        )
        t.start()
        logger.info("FIXCollector | Background thread started")

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("FIXCollector | Stopping...")
        self._stop_event.set()
        self._flush_ticks()  # flush remaining buffer
        if self._client:
            try:
                from ctrader_fix import LogoutRequest
                logout = LogoutRequest(self._build_config())
                self._client.send(logout)
                time.sleep(1)
            except Exception:
                pass

    def get_latest_tick(self, pair: str) -> Optional[Dict]:
        """Get latest tick from Redis cache."""
        try:
            r = self._get_redis()
            raw = r.get(f"tick:{pair}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def get_orderbook(self, pair: str) -> Optional[Dict]:
        """Get latest order book from Redis cache."""
        try:
            r = self._get_redis()
            raw = r.get(f"orderbook:{pair}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Reconnect loop — keeps collector alive forever."""
        while not self._stop_event.is_set():
            try:
                logger.info("FIXCollector | Connecting...")
                self._connect()

                # Wait for logon
                if not self._logged_in.wait(timeout=30):
                    logger.error("FIXCollector | Logon timeout — retrying in 30s")
                    time.sleep(30)
                    continue

                logger.success("FIXCollector | Logged in ✅ — subscribing to 12 pairs")
                self._subscribe_all()

                # Keep alive — periodic heartbeat check
                while not self._stop_event.is_set() and self._logged_in.is_set():
                    time.sleep(5)
                    self._maybe_flush_ticks()

            except Exception as exc:
                logger.error(f"FIXCollector | Error: {exc}")

            if not self._stop_event.is_set():
                logger.info("FIXCollector | Reconnecting in 30s...")
                self._logged_in.clear()
                self._connected.clear()
                time.sleep(30)

    def _connect(self) -> None:
        """Create FIX client and start service."""
        from ctrader_fix import Client
        from twisted.internet import reactor

        config = self._build_config()
        self._client = Client(self._host, self._port, ssl=False)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        if reactor.running:
            reactor.callFromThread(self._client.startService)
        else:
            ready = threading.Event()
            def _run():
                reactor.callWhenRunning(ready.set)
                reactor.run(installSignalHandlers=False)
            t = threading.Thread(target=_run, name="fix-reactor", daemon=True)
            t.start()
            if not ready.wait(timeout=10):
                raise RuntimeError("FIX reactor failed to start")
            reactor.callFromThread(self._client.startService)

        self._connected.wait(timeout=15)

    # ── FIX callbacks ──────────────────────────────────────────────────────────

    def _on_connected(self, client: Any) -> None:
        logger.info("FIXCollector | TCP connected — sending logon")
        self._connected.set()
        self._send_logon()

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        logger.warning(f"FIXCollector | Disconnected: {reason}")
        self._logged_in.clear()
        self._connected.clear()

    def _on_message(self, client: Any, message: Any) -> None:
        """Route all incoming FIX messages."""
        try:
            logger.debug(f"FIX RAW: {message.getMessage()[:200]}")
            msg_type = message.getFieldValue(35)

            if msg_type == MSG_LOGON:
                logger.success("FIXCollector | Logon confirmed ✅")
                self._logged_in.set()

            elif msg_type == MSG_LOGOUT:
                reason = message.getFieldValue(58) or "unknown"
                logger.warning(f"FIXCollector | Logout: {reason}")
                self._logged_in.clear()

            elif msg_type == MSG_HEARTBEAT:
                pass  # normal keepalive

            elif msg_type == MSG_TEST_REQUEST:
                # Reply with heartbeat
                test_req_id = message.getFieldValue(112)
                self._send_heartbeat(test_req_id)

            elif msg_type == MSG_MARKET_DATA_SNAPSHOT:
                self._handle_snapshot(message)

            elif msg_type == MSG_MARKET_DATA_INCREMENTAL:
                self._handle_incremental(message)

            elif msg_type == MSG_MARKET_DATA_REJECT:
                reason = message.getFieldValue(58) or "unknown"
                symbol = message.getFieldValue(55) or "unknown"
                logger.warning(f"FIXCollector | MD Reject {symbol}: {reason}")

        except Exception as exc:
            logger.error(f"FIXCollector | Message handler error: {exc}")

    # ── Market data handlers ───────────────────────────────────────────────────

    def _handle_snapshot(self, message: Any) -> None:
        """
        Handle MarketDataSnapshotFullRefresh (W).
        Parse raw FIX string directly — library does not support indexed repeating groups.
        """
        ID_TO_SYMBOL = {
            "1": "EURUSD", "2": "GBPUSD",  "3": "EURJPY",
            "4": "USDJPY", "5": "AUDUSD",  "6": "USDCHF",
            "7": "GBPJPY", "8": "USDCAD",  "9": "EURGBP",
            "11": "AUDJPY","12": "NZDUSD", "21": "NZDJPY",
        }
        try:
            raw = message.getMessage()
            entries = self._parse_repeating_groups(raw)
            if not entries:
                return

            # Get symbol from top-level field 55 (numeric ID)
            sym_id = None
            for part in raw.replace("\x01", "|").split("|"):
                if part.startswith("55="):
                    sym_id = part.split("=", 1)[1]
                    break
            symbol = ID_TO_SYMBOL.get(sym_id or "", "")
            if not symbol:
                return

            now = datetime.now(timezone.utc)
            bids: List[Dict] = []
            asks: List[Dict] = []

            for entry in entries:
                price      = entry.get("270")
                entry_type = entry.get("269")
                size       = entry.get("271")
                if price is None or entry_type is None:
                    continue
                price = float(price)
                size  = float(size) if size else 0.0
                if entry_type == MD_BID:
                    bids.append({"price": price, "size": size})
                elif entry_type == MD_ASK:
                    asks.append({"price": price, "size": size})

            if not bids or not asks:
                return

            bids.sort(key=lambda x: x["price"], reverse=True)
            asks.sort(key=lambda x: x["price"])
            best_bid = bids[0]["price"]
            best_ask = asks[0]["price"]

            self._buffer_tick(symbol, best_bid, best_ask, now)
            self._cache_orderbook(symbol, bids[:10], asks[:10], now)
            self._store_orderbook_db(symbol, bids[:10], asks[:10], now)

        except Exception as exc:
            logger.error(f"FIXCollector | Snapshot error: {exc}")

    def _handle_incremental(self, message: Any) -> None:
        """
        Handle MarketDataIncrementalRefresh (X).
        Parse raw FIX string directly — library does not support indexed repeating groups.
        """
        ID_TO_SYMBOL = {
            "1": "EURUSD", "2": "GBPUSD",  "3": "EURJPY",
            "4": "USDJPY", "5": "AUDUSD",  "6": "USDCHF",
            "7": "GBPJPY", "8": "USDCAD",  "9": "EURGBP",
            "11": "AUDJPY","12": "NZDUSD", "21": "NZDJPY",
        }
        try:
            raw = message.getMessage()
            entries = self._parse_repeating_groups(raw)
            now = datetime.now(timezone.utc)
            ticks: Dict[str, Dict] = {}

            for entry in entries:
                sym_id     = entry.get("55")
                entry_type = entry.get("269")
                price      = entry.get("270")
                symbol     = ID_TO_SYMBOL.get(sym_id or "", "")
                if not symbol or price is None:
                    continue
                price = float(price)
                if symbol not in ticks:
                    ticks[symbol] = {}
                if entry_type == MD_BID:
                    ticks[symbol]["bid"] = price
                elif entry_type == MD_ASK:
                    ticks[symbol]["ask"] = price

            for symbol, data in ticks.items():
                if "bid" in data and "ask" in data:
                    self._buffer_tick(symbol, data["bid"], data["ask"], now)

        except Exception as exc:
            logger.error(f"FIXCollector | Incremental error: {exc}")

    def _parse_repeating_groups(self, raw: str) -> List[Dict[str, str]]:
        """
        Parse FIX repeating groups from raw message string.
        Returns list of dicts, one per group entry (split on field 269).
        Handles both | and \x01 delimiters.
        """
        raw = raw.replace("\x01", "|")
        fields: Dict[str, str] = {}
        groups: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        for part in raw.split("|"):
            if "=" not in part:
                continue
            tag, val = part.split("=", 1)
            tag = tag.strip()
            if tag == "269":
                if current:
                    groups.append(current)
                current = {"269": val}
            elif tag in ("270", "271", "55", "55"):
                current[tag] = val
            else:
                fields[tag] = val
        if current:
            groups.append(current)
        # Attach top-level symbol to each group if not present
        sym = fields.get("55")
        if sym:
            for g in groups:
                g.setdefault("55", sym)
        return groups

    def _buffer_tick(
        self,
        symbol: str,
        bid: float,
        ask: float,
        ts: datetime,
    ) -> None:
        """Buffer tick for batch DB insert + update Redis cache."""
        # Update Redis immediately (real-time)
        self._cache_tick(symbol, bid, ask, ts)

        # Buffer for batch DB insert
        with self._lock:
            self._tick_buffer.append({
                "symbol": symbol,
                "time":   ts,
                "bid":    bid,
                "ask":    ask,
            })

        # Flush if buffer is full
        if len(self._tick_buffer) >= TICK_BATCH_SIZE:
            self._flush_ticks()

    def _maybe_flush_ticks(self) -> None:
        """Flush if enough time has passed."""
        if time.time() - self._last_flush >= TICK_FLUSH_SECS:
            self._flush_ticks()

    def _flush_ticks(self) -> None:
        """Batch insert tick buffer to TimescaleDB."""
        with self._lock:
            if not self._tick_buffer:
                return
            batch = self._tick_buffer.copy()
            self._tick_buffer.clear()

        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text

            with db.connection() as conn:
                conn.execute(
                    text("""
                        INSERT INTO ticks (symbol, time, bid, ask)
                        VALUES (:symbol, :time, :bid, :ask)
                        ON CONFLICT (symbol, time) DO NOTHING
                    """),
                    batch
                )
                conn.commit()

            self._last_flush = time.time()
            logger.debug(f"FIXCollector | Flushed {len(batch)} ticks to DB")

        except Exception as exc:
            logger.error(f"FIXCollector | Tick flush failed: {exc}")

    def _store_orderbook_db(
        self,
        symbol: str,
        bids: List[Dict],
        asks: List[Dict],
        ts: datetime,
    ) -> None:
        """Store L2 order book snapshot to TimescaleDB (JSONB schema)."""
        try:
            import json
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text

            total_bid_vol = sum(b["size"] for b in bids)
            total_ask_vol = sum(a["size"] for a in asks)
            denom = total_bid_vol + total_ask_vol
            imbalance = (total_bid_vol - total_ask_vol) / denom if denom > 0 else 0.0

            with db.connection() as conn:
                conn.execute(
                    text("""
                        INSERT INTO order_book_snapshots
                            (symbol, time, bids, asks,
                             bid_ask_imbalance, total_bid_volume,
                             total_ask_volume, depth_levels)
                        VALUES
                            (:symbol, :time,
                             CAST(:bids AS jsonb), CAST(:asks AS jsonb),
                             :imbalance, :total_bid_vol,
                             :total_ask_vol, :depth_levels)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "symbol":        symbol,
                        "time":          ts,
                        "bids":          json.dumps(bids),
                        "asks":          json.dumps(asks),
                        "imbalance":     imbalance,
                        "total_bid_vol": total_bid_vol,
                        "total_ask_vol": total_ask_vol,
                        "depth_levels":  len(bids),
                    }
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"FIXCollector | Orderbook DB store failed: {exc}")

    def _cache_tick(
        self,
        symbol: str,
        bid: float,
        ask: float,
        ts: datetime,
    ) -> None:
        """Cache latest tick in Redis for real-time feature access."""
        try:
            r = self._get_redis()
            r.set(
                f"tick:{symbol}",
                json.dumps({
                    "bid":  bid,
                    "ask":  ask,
                    "mid":  (bid + ask) / 2,
                    "spread": round(ask - bid, 6),
                    "time": ts.isoformat(),
                }),
                ex=60  # expire after 60s if no updates
            )
        except Exception as exc:
            logger.debug(f"FIXCollector | Redis tick cache failed: {exc}")

    def _cache_orderbook(
        self,
        symbol: str,
        bids: List[Dict],
        asks: List[Dict],
        ts: datetime,
    ) -> None:
        """Cache L2 order book in Redis."""
        try:
            r = self._get_redis()
            r.set(
                f"orderbook:{symbol}",
                json.dumps({
                    "bids": bids,
                    "asks": asks,
                    "time": ts.isoformat(),
                }),
                ex=60
            )
        except Exception as exc:
            logger.debug(f"FIXCollector | Redis orderbook cache failed: {exc}")

    # ── FIX message senders ────────────────────────────────────────────────────

    def _send_logon(self) -> None:
        """Send FIX Logon (A) message."""
        try:
            from ctrader_fix import LogonRequest
            config  = self._build_config()
            logon   = LogonRequest(config)
            self._client.send(logon)
            logger.debug("FIXCollector | Logon sent")
        except Exception as exc:
            logger.error(f"FIXCollector | Logon send failed: {exc}")

    def _send_heartbeat(self, test_req_id: str = "") -> None:
        """Reply to TestRequest with Heartbeat."""
        try:
            from ctrader_fix import HeartbeatRequest
            config = self._build_config()
            hb = HeartbeatRequest(config)
            if test_req_id:
                hb.setField(112, test_req_id)
            self._client.send(hb)
        except Exception as exc:
            logger.debug(f"FIXCollector | Heartbeat send failed: {exc}")

    def _subscribe_all(self) -> None:
        """Subscribe to market data for all 12 G10 pairs."""
        for pair in G10_PAIRS:
            try:
                self._subscribe_pair(pair)
                time.sleep(0.1)  # small delay between subscriptions
            except Exception as exc:
                logger.error(f"FIXCollector | Subscribe {pair} failed: {exc}")

    def _subscribe_pair(self, symbol: str) -> None:
        """Send MarketDataRequest using correct repeating group format."""
        from ctrader_fix import MarketDataRequest

        SYMBOL_IDS = {
            "EURUSD": "1",  "GBPUSD": "2",  "EURJPY": "3",
            "USDJPY": "4",  "AUDUSD": "5",  "USDCHF": "6",
            "GBPJPY": "7",  "USDCAD": "8",  "EURGBP": "9",
            "AUDJPY": "11", "NZDUSD": "12", "NZDJPY": "21",
        }
        sym_id = SYMBOL_IDS.get(symbol, symbol)

        self._req_id += 1
        config = self._build_config()

        req = MarketDataRequest(config)
        req.MDReqID                 = f"MDR_{self._req_id}_{symbol}"
        req.SubscriptionRequestType = "1"
        req.MarketDepth             = "1"
        req.MDUpdateType            = "1"
        req.NoMDEntryTypes          = "1"   # send 1 entry only — bid
        req.MDEntryType             = "0"   # bid only first
        req.NoRelatedSym            = "1"
        req.Symbol                  = sym_id

        self._client.send(req)

        #   Second subscription for ask
        self._req_id += 1
        req2 = MarketDataRequest(config)
        req2.MDReqID                 = f"MDR_{self._req_id}_{symbol}_ASK"
        req2.SubscriptionRequestType = "1"
        req2.MarketDepth             = "1"
        req2.MDUpdateType            = "1"
        req2.NoMDEntryTypes          = "1"
        req2.MDEntryType             = "1"   # ask only
        req2.NoRelatedSym            = "1"
        req2.Symbol                  = sym_id

        self._client.send(req2)
        logger.debug(f"FIXCollector | Subscribed to {symbol} (id={sym_id})")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_config(self) -> Dict:
        """Build config dict for cTraderFixPy."""
        return {
            "Host":          self._host,
            "Port":          self._port,
            "SSL":           False,
            "Username":      self._username,
            "Password":      self._password,
            "BeginString":   "FIX.4.4",
            "SenderCompID":  self._sender,
            "SenderSubID":   "QUOTE",
            "TargetCompID":  self._target,
            "TargetSubID":   "QUOTE",
            "HeartBeat":     "30",
        }

    def _get_redis(self):
        """Lazy Redis connection."""
        if self._redis is None:
            import redis
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )
        return self._redis


# ── Module singleton ──────────────────────────────────────────────────────────
fix_collector = FIXTickCollector()
