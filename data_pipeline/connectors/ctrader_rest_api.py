"""
data_pipeline/connectors/ctrader_rest_api.py
=============================================
cTrader Open API Connector — Protobuf/TCP Protocol

Uses the official ctrader-open-api (Twisted + Protobuf over TCP).

Key fix vs previous version:
    Twisted reactor MUST be running before startService() is called.
    Solution: start reactor thread first, wait for it to be running,
    then schedule startService() inside the reactor via callFromThread.

Auth flow (per session):
    TCP connect
    → ProtoOAApplicationAuthReq  (client_id + client_secret)
    → ProtoOAAccountAuthReq      (access_token + account_id)
    → Ready

Usage:
    from data_pipeline.connectors.ctrader_rest_api import connector
    connector.connect()
    df = connector.get_historical_bars('EURUSD', 'M15', start, end)
    connector.disconnect()
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Load .env early so settings are available
load_dotenv()

# ── Lazy settings (avoids circular imports) ───────────────────────────────────
def _get(key: str, default: str = "") -> str:
    try:
        from config.settings import settings
        return str(getattr(settings, key, os.getenv(key, default)))
    except Exception:
        return os.getenv(key, default)


# ── Constants ─────────────────────────────────────────────────────────────────

TIMEFRAME_PERIOD: Dict[str, int] = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5,
    "M10": 6, "M15": 7, "M30": 8,
    "H1": 9, "H4": 10, "H12": 11,
    "D1": 12, "W1": 13, "MN1": 14,
}

MAX_BARS_PER_REQUEST    = 5000
HISTORICAL_RATE_LIMIT   = 0.21   # seconds between historical requests (max 5/s)
REQUEST_TIMEOUT         = 30.0   # seconds to wait for any response
REACTOR_START_TIMEOUT   = 10.0   # seconds to wait for reactor to start
JPY_PAIRS               = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}


# ── Connector ─────────────────────────────────────────────────────────────────

class CTraderOpenAPIConnector:
    """
    Thread-safe synchronous wrapper around ctrader-open-api (Twisted).

    The Twisted reactor runs in a dedicated background daemon thread.
    All public methods block the calling thread until a response arrives.

    Threading model:
        Main thread          │  Reactor thread (daemon)
        ─────────────────────┼──────────────────────────
        connector.connect()  │  reactor.run()
          └─ waits on Event  │    TCP connect
                             │    onConnected() → send app auth
                             │    onMessage()   → send account auth
                             │    onMessage()   → set _account_authed
          └─ Event fires ────┤
        connector.get_bars() │
          └─ waits on Event  │  onMessage() → set _response_event
          └─ returns df ─────┤
    """

    def __init__(self) -> None:
        self._client: Any = None

        # Auth state
        self._account_authed = threading.Event()

        # Single pending-response slot
        self._response_lock  = threading.Lock()
        self._response_event = threading.Event()
        self._last_response: Any = None

        # Symbol cache
        self._symbol_map: Dict[str, int] = {}
        self._tick_buffer: List[Dict] = []
        self._tick_buffer_lock = threading.Lock()
        self._last_tick_flush = time.time()

        # Rate limiting
        self._last_hist_request: float = 0.0

        # Demo / live
        env = os.getenv("CTRADER_ENV", "demo").lower()
        self._is_demo = (env != "live")

        logger.info(
            f"CTraderOpenAPIConnector | "
            f"mode={'DEMO' if self._is_demo else 'LIVE'}"
        )

    # =========================================================================
    # Connection lifecycle
    # =========================================================================

    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to cTrader and fully authenticate.

        The fix: reactor starts first in a background thread.
        Once reactor.running is True, startService() is scheduled
        inside the reactor via callFromThread — this is the correct
        Twisted pattern for threading.

        Returns True if authenticated and ready.
        """
        from ctrader_open_api import Client, EndPoints, TcpProtocol
        from twisted.internet import reactor

        host = (
            EndPoints.PROTOBUF_DEMO_HOST
            if self._is_demo
            else EndPoints.PROTOBUF_LIVE_HOST
        )
        port = EndPoints.PROTOBUF_PORT

        logger.info(f"Connecting | {host}:{port}")

        # Build client
        self._client = Client(host, port, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        # ── Step 1: start reactor in background thread ────────────────────
        reactor_ready = threading.Event()

        def _run_reactor():
            # Signal that reactor is about to start
            reactor.callWhenRunning(reactor_ready.set)
            reactor.run(installSignalHandlers=False)

        t = threading.Thread(target=_run_reactor, name="ctrader-reactor", daemon=True)
        t.start()

        # ── Step 2: wait for reactor to be running ────────────────────────
        if not reactor_ready.wait(timeout=REACTOR_START_TIMEOUT):
            logger.error("Twisted reactor failed to start")
            return False

        # ── Step 3: schedule startService() inside the reactor ────────────
        # This is the correct Twisted threading pattern
        reactor.callFromThread(self._client.startService)

        # ── Step 4: wait for account auth ─────────────────────────────────
        if not self._account_authed.wait(timeout=timeout):
            logger.error(
                f"Authentication timed out after {timeout}s.\n"
                "  → Check CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID"
            )
            return False

        logger.success("cTrader Open API | authenticated ✅")
        return True

    def disconnect(self) -> None:
        """Stop client service and Twisted reactor."""
        logger.info("Disconnecting...")
        if self._client:
            try:
                from twisted.internet import reactor
                reactor.callFromThread(self._client.stopService)
            except Exception as exc:
                logger.debug(f"stopService: {exc}")
        try:
            from twisted.internet import reactor
            if reactor.running:
                reactor.callFromThread(reactor.stop)
        except Exception as exc:
            logger.debug(f"reactor.stop: {exc}")

    # =========================================================================
    # Twisted callbacks (reactor thread)
    # =========================================================================

    def _on_connected(self, client: Any) -> None:
        logger.info("TCP connected — sending app auth...")
        self._send_app_auth()

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        logger.warning(f"Disconnected: {reason}")
        self._account_authed.clear()
        # Wake any blocked thread
        with self._response_lock:
            self._last_response = None
            self._response_event.set()

    def _on_message(self, client: Any, message: Any) -> None:
        """Route all incoming Protobuf messages."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthRes,
            ProtoOAAccountAuthRes,
            ProtoOAGetTrendbarsRes,
            ProtoOAGetTickDataRes,
            ProtoOATraderRes,
            ProtoOAReconcileRes,
            ProtoOADealListRes,
        )
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAErrorRes,
            ProtoOAOrderErrorEvent,
            ProtoOAExecutionEvent,
            ProtoOASpotEvent,
            ProtoOADepthEvent,
        )
        

        msg = Protobuf.extract(message)

        if isinstance(msg, ProtoOAApplicationAuthRes):
            logger.info("App auth OK — sending account auth...")
            self._send_account_auth()

        elif isinstance(msg, ProtoOAAccountAuthRes):
            logger.info(f"Account auth OK | id={_get('CTRADER_ACCOUNT_ID')}")
            self._account_authed.set()

        elif isinstance(msg, ProtoOAErrorRes):
            logger.error(f"cTrader error [{msg.errorCode}]: {msg.description}")
            # ALREADY_LOGGED_IN means app session is alive on server side —
            # treat it as app auth success and proceed to account auth.
            if msg.errorCode == "ALREADY_LOGGED_IN":
                logger.info("ALREADY_LOGGED_IN → app already authed, sending account auth...")
                self._send_account_auth()
            else:
                with self._response_lock:
                    self._last_response = msg
                    self._response_event.set()

        elif isinstance(msg, (
            ProtoOAGetTrendbarsRes,
            ProtoOAGetTickDataRes,
            ProtoOATraderRes,
            ProtoOAReconcileRes,
            ProtoOADealListRes,
        )):
            with self._response_lock:
                self._last_response = msg
                self._response_event.set()

        elif isinstance(msg, ProtoOAOrderErrorEvent):
            logger.error(
                f"OrderError | code={msg.errorCode} | "
                f"desc={getattr(msg, 'description', '?')} | "
                f"orderId={getattr(msg, 'clientOrderId', '?')}"
            )
            with self._response_lock:
                self._last_response = msg
                self._response_event.set()
        elif isinstance(msg, ProtoOAExecutionEvent):
            logger.success(
                f"Execution | type={msg.executionType} | "
                f"pos={getattr(msg.position, 'positionId', '?')} | "
                f"price={getattr(msg.position, 'price', '?')}"
            )
            with self._response_lock:
                self._last_response = msg
                self._response_event.set()
        elif isinstance(msg, ProtoOASpotEvent):
            self._handle_spot_event(msg)
        elif isinstance(msg, ProtoOADepthEvent):
            self._handle_depth_event(msg)
        else:
            logger.debug(f"Unhandled: {type(msg).__name__}")

    # =========================================================================
    # Live price + depth subscription (piggybacks on existing auth)
    # =========================================================================

    def subscribe_spots(self, pairs: List[str]) -> None:
        """
        Subscribe to real-time spot prices for given pairs.
        Reuses existing authenticated connection — no separate auth needed.
        Results stored in Redis: tick:{PAIR} and orderbook:{PAIR}
        """
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOASubscribeSpotsReq,
        )
        import redis as redis_lib
        import os

        if not self._account_authed.is_set():
            logger.warning("REST API | Cannot subscribe — not authenticated")
            return

        symbol_map = self.load_symbols()
        subscribed = []

        for pair in pairs:
            sid = symbol_map.get(pair)
            if not sid:
                logger.warning(f"REST API | Unknown symbol: {pair}")
                continue
            try:
                req = ProtoOASubscribeSpotsReq()
                req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
                req.symbolId.append(sid)
                req.subscribeToSpotTimestamp = True
                from twisted.internet import reactor
                reactor.callFromThread(self._client.send, req)
                subscribed.append(pair)
            except Exception as exc:
                logger.error(f"REST API | Subscribe {pair} failed: {exc}")

        if subscribed:
            logger.success(f"REST API | Subscribed to spots: {subscribed}")

        # Also subscribe to L2 depth quotes
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOASubscribeDepthQuotesReq,
            )
            for pair in pairs:
                sid = symbol_map.get(pair)
                if not sid:
                    continue
                req = ProtoOASubscribeDepthQuotesReq()
                req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
                req.symbolId.append(sid)
                from twisted.internet import reactor
                reactor.callFromThread(self._client.send, req)
            logger.success(f"REST API | Subscribed to L2 depth for {len(pairs)} pairs")
        except Exception as exc:
            logger.warning(f"REST API | Depth subscription failed: {exc}")

    def _handle_spot_event(self, msg: Any) -> None:
        """Handle incoming spot price event — cache to Redis."""
        try:
            import redis as redis_lib, os, json
            from datetime import datetime, timezone

            r = redis_lib.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )

            symbol_map = self.load_symbols()
            rev_map    = {v: k for k, v in symbol_map.items()}
            pair       = rev_map.get(msg.symbolId, f"SYM_{msg.symbolId}")

            # cTrader sends prices as integers (price * 100000)
            # cTrader: non-JPY = price × 100000, JPY = price × 1000
            JPY_PAIRS = {"USDJPY","EURJPY","GBPJPY","AUDJPY","NZDJPY"}
            # cTrader spot event scaling:
            # Non-JPY: raw / 100000 → correct price
            # JPY:     raw / 1000   → correct price
            # DEBUG — log raw integer before division
            # cTrader OpenAPI: all pairs use 100000 divisor (5 decimal places)
            divisor = 100000.0
            bid = msg.bid / divisor if msg.HasField("bid") else 0.0
            ask = msg.ask / divisor if msg.HasField("ask") else 0.0

            if bid <= 0 or ask <= 0:
                return

            now = datetime.now(timezone.utc)
            tick = {
                "bid":    round(bid, 6),
                "ask":    round(ask, 6),
                "mid":    round((bid + ask) / 2, 6),
                "spread": round(ask - bid, 6),
                "time":   now.isoformat(),
            }
            r.set(f"tick:{pair}", json.dumps(tick), ex=60)

            # Store to DB ticks table (batch via Redis list)
            # Batch tick to buffer — flush every 100 ticks or 10 seconds
            with self._tick_buffer_lock:
                self._tick_buffer.append({
                    "sym": pair,
                    "ts":  now,
                    "bid": round(bid, 6),
                    "ask": round(ask, 6),
                })
                should_flush = (
                    len(self._tick_buffer) >= 100 or
                    time.time() - self._last_tick_flush >= 10
                )

            if should_flush:
                self._flush_tick_buffer()

        except Exception as exc:
            logger.debug(f"REST API | Spot event failed: {exc}")
    
    def _flush_tick_buffer(self) -> None:
        """Flush tick buffer to DB in one batch INSERT."""
        with self._tick_buffer_lock:
            if not self._tick_buffer:
                return
            batch = self._tick_buffer.copy()
            self._tick_buffer.clear()
            self._last_tick_flush = time.time()

        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text as sa_text
            with db.connection() as conn:
                rows = [{"sym": t["sym"],
                         "ts":  t["ts"].isoformat() if hasattr(t["ts"], "isoformat") else str(t["ts"]),
                         "bid": t["bid"],
                         "ask": t["ask"]}
                        for t in batch]
                conn.execute(sa_text("""
                    INSERT INTO ticks (symbol, time, bid, ask)
                    VALUES (:sym, :ts, :bid, :ask)
                    ON CONFLICT (symbol, time) DO NOTHING
                """), rows)
                conn.commit()
            logger.debug(f"REST API | Flushed {len(batch)} ticks to DB")
        except Exception as exc:
            logger.debug(f"REST API | Tick flush failed: {exc}")

    def _handle_depth_event(self, msg: Any) -> None:
        """Handle incoming L2 depth event — cache to Redis."""
        try:
            import redis as redis_lib, os, json
            from datetime import datetime, timezone

            r = redis_lib.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )

            symbol_map = self.load_symbols()
            rev_map    = {v: k for k, v in symbol_map.items()}
            pair       = rev_map.get(msg.symbolId, f"SYM_{msg.symbolId}")

            divisor = 100000.0
            bids, asks = [], []

            for entry in msg.newQuotes:
                size = entry.size if entry.HasField("size") else 0.0
                bid  = entry.bid / divisor if entry.HasField("bid") else 0.0
                ask  = entry.ask / divisor if entry.HasField("ask") else 0.0
                if bid > 0:
                    bids.append({"price": round(bid, 6), "size": size})
                if ask > 0:
                    asks.append({"price": round(ask, 6), "size": size})

            bids.sort(key=lambda x: x["price"], reverse=True)
            asks.sort(key=lambda x: x["price"])

            book = {
                "bids": bids[:20],
                "asks": asks[:20],
                "time": datetime.now(timezone.utc).isoformat(),
            }
            r.set(f"orderbook:{pair}", json.dumps(book), ex=60)

            # Store snapshot to DB for MMM model training
            try:
                from data_pipeline.storage.timeseries_db import db
                from sqlalchemy import text as sa_text
                import json as _json

                total_bid_vol = sum(b["size"] for b in bids)
                total_ask_vol = sum(a["size"] for a in asks)
                imbalance = (
                    (total_bid_vol - total_ask_vol) /
                    (total_bid_vol + total_ask_vol)
                    if (total_bid_vol + total_ask_vol) > 0 else 0.0
                )

                with db.connection() as conn:
                    conn.execute(sa_text("""
                        INSERT INTO order_book_snapshots
                            (time, symbol, bids, asks,
                             bid_ask_imbalance, total_bid_volume,
                             total_ask_volume, depth_levels)
                        VALUES
                            (:ts, :sym, :bids, :asks,
                             :imbalance, :bid_vol,
                             :ask_vol, :levels)
                        ON CONFLICT DO NOTHING
                    """), {
                        "ts":        datetime.now(timezone.utc).isoformat(),
                        "sym":       pair,
                        "bids":      _json.dumps(bids[:20]),
                        "asks":      _json.dumps(asks[:20]),
                        "imbalance": round(imbalance, 6),
                        "bid_vol":   total_bid_vol,
                        "ask_vol":   total_ask_vol,
                        "levels":    min(len(bids), len(asks), 20),
                    })
                    conn.commit()
            except Exception as db_exc:
                logger.debug(f"REST API | Depth DB store failed: {db_exc}")

        except Exception as exc:
            logger.debug(f"REST API | Depth event failed: {exc}")

    # =========================================================================
    # Auth senders (called from reactor thread)
    # =========================================================================

    def _send_app_auth(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthReq,
        )
        req = ProtoOAApplicationAuthReq()
        req.clientId     = _get("CTRADER_CLIENT_ID")
        req.clientSecret = _get("CTRADER_CLIENT_SECRET")
        self._client.send(req).addErrback(
            lambda f: logger.error(f"App auth send: {f}")
        )

    def _send_account_auth(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAAccountAuthReq,
        )
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
        req.accessToken         = _get("CTRADER_ACCESS_TOKEN")
        self._client.send(req).addErrback(
            lambda f: logger.error(f"Account auth send: {f}")
        )

    # =========================================================================
    # Synchronous request helper
    # =========================================================================

    def _send_request(self, request: Any, timeout: float = REQUEST_TIMEOUT) -> Any:
        """
        Send a Protobuf request and block until response arrives.
        Thread-safe — uses threading.Event to bridge reactor ↔ caller.
        """
        if not self._account_authed.is_set():
            raise RuntimeError("Not authenticated. Call connect() first.")

        with self._response_lock:
            self._response_event.clear()
            self._last_response = None

        from twisted.internet import reactor
        reactor.callFromThread(
            lambda: self._client.send(request).addErrback(
                lambda f: logger.error(f"Send error: {f}")
            )
        )

        if not self._response_event.wait(timeout=timeout):
            logger.error(f"Timeout ({timeout}s) | {type(request).__name__}")
            return None

        with self._response_lock:
            return self._last_response

    # =========================================================================
    # Symbol management
    # =========================================================================

    def load_symbols(self) -> Dict[str, int]:
        """Load symbol name → ID map from ctrader_config.yaml."""
        """Load symbol name → ID map from ctrader_config.yaml."""
        if self._symbol_map:          # ← ADD THIS LINE
            return self._symbol_map   # ← AND THIS LINE
        import yaml
        from pathlib import Path

        config_path = Path(_get("PROJECT_ROOT", "/workspaces/Trading_bot")) / \
                      "config" / "ctrader_config.yaml"
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            raw = cfg.get("fix", {}).get("symbol_map", {})
            self._symbol_map = {k: int(v) for k, v in raw.items()}
            logger.info(f"Loaded {len(self._symbol_map)} symbols from config")
        except Exception as exc:
            logger.warning(f"Symbol config failed ({exc}) — using defaults")
            self._symbol_map = {
                "EURUSD": 1,  "GBPUSD": 2,  "USDJPY": 3,  "USDCHF": 4,
                "AUDUSD": 5,  "NZDUSD": 6,  "USDCAD": 7,  "EURGBP": 8,
                "EURJPY": 9,  "GBPJPY": 10, "AUDJPY": 11, "NZDJPY": 12,
            }
        return self._symbol_map

    def get_symbol_id(self, symbol: str) -> int:
        if not self._symbol_map:
            self.load_symbols()
        sid = self._symbol_map.get(symbol)
        if sid is None:
            raise ValueError(
                f"Unknown symbol '{symbol}'. "
                f"Available: {sorted(self._symbol_map)}"
            )
        return sid

    # =========================================================================
    # Historical OHLCV
    # =========================================================================

    def get_historical_bars(
        self,
        symbol:      str,
        timeframe:   str = "M15",
        start:       Optional[datetime] = None,
        end:         Optional[datetime] = None,
        store_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars (ProtoOAGetTrendbarsReq).
        Auto-paginates. Rate-limited to 5 req/s.

        Args:
            symbol:      e.g. "EURUSD"
            timeframe:   M1 M5 M15 M30 H1 H4 D1 W1 MN1
            start:       UTC start (default 365d ago)
            end:         UTC end   (default now)
            store_to_db: upsert into TimescaleDB

        Returns:
            DataFrame[open, high, low, close, volume] indexed by UTC time.
        """
        if start is None:
            start = datetime.now(tz=timezone.utc) - timedelta(days=365)
        if end is None:
            end = datetime.now(tz=timezone.utc)

        if timeframe not in TIMEFRAME_PERIOD:
            raise ValueError(f"Invalid timeframe '{timeframe}'. Valid: {list(TIMEFRAME_PERIOD)}")

        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq

        symbol_id = self.get_symbol_id(symbol)
        period    = TIMEFRAME_PERIOD[timeframe]
        chunk_end = end
        all_bars: List[Any] = []

        logger.info(
            f"Fetching {symbol} {timeframe} | "
            f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"
        )

        while chunk_end > start:
            # Rate limit
            wait = HISTORICAL_RATE_LIMIT - (time.monotonic() - self._last_hist_request)
            if wait > 0:
                time.sleep(wait)

            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
            req.symbolId            = symbol_id
            req.period              = period
            req.fromTimestamp       = int(start.timestamp() * 1000)
            req.toTimestamp         = int(chunk_end.timestamp() * 1000)
            req.count               = MAX_BARS_PER_REQUEST

            self._last_hist_request = time.monotonic()
            response = self._send_request(req)

            if response is None:
                logger.error(f"No response for {symbol} — stopping")
                break

            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAErrorRes
            if isinstance(response, ProtoOAErrorRes):
                logger.error(f"API error: [{response.errorCode}] {response.description}")
                break

            bars = list(response.trendbar)
            if not bars:
                logger.debug(f"  {symbol}: empty chunk — done")
                break

            all_bars.extend(bars)
            logger.debug(f"  +{len(bars)} bars (total {len(all_bars)})")

            chunk_end = datetime.fromtimestamp(
                bars[0].utcTimestampInMinutes * 60, tz=timezone.utc
            ) - timedelta(milliseconds=1)

            if len(bars) < MAX_BARS_PER_REQUEST:
                break

        if not all_bars:
            logger.warning(f"No bars for {symbol} {timeframe}")
            return pd.DataFrame()

        df = self._bars_to_dataframe(all_bars, symbol)
        df = df.loc[
            (df.index >= pd.Timestamp(start)) &
            (df.index <= pd.Timestamp(end))
        ]
        df = df[~df.index.duplicated(keep="last")].sort_index()

        logger.success(
            f"✅ {symbol} {timeframe}: {len(df):,} bars | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )

        if store_to_db and not df.empty:
            self._store_ohlcv(df, symbol, timeframe)

        return df

    def get_all_pairs_historical(
        self,
        timeframe:   str  = "M15",
        days_back:   int  = 365,
        store_to_db: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch history for all G10 pairs. Returns {symbol: DataFrame}."""
        end    = datetime.now(tz=timezone.utc)
        start  = end - timedelta(days=days_back)
        pairs  = os.getenv("G10_PAIRS", "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,NZDUSD,USDCAD,EURGBP,EURJPY,GBPJPY,AUDJPY,NZDJPY").split(",")

        try:
            from config.settings import settings
            pairs = settings.G10_PAIRS
        except Exception:
            pass

        results: Dict[str, pd.DataFrame] = {}
        failed:  List[str] = []

        logger.info(f"Bulk fetch | {len(pairs)} pairs | {timeframe} | {days_back}d")

        for i, symbol in enumerate(pairs, 1):
            logger.info(f"[{i}/{len(pairs)}] {symbol}")
            try:
                results[symbol] = self.get_historical_bars(
                    symbol=symbol, timeframe=timeframe,
                    start=start, end=end, store_to_db=store_to_db,
                )
            except Exception as exc:
                logger.error(f"  ❌ {symbol}: {exc}")
                failed.append(symbol)

        logger.info(
            f"Done | ✅ {len(results)} | ❌ {len(failed)}"
            + (f" {failed}" if failed else "")
        )
        return results

    # =========================================================================
    # DataFrame helpers
    # =========================================================================

    def _bars_to_dataframe(self, bars: List[Any], symbol: str) -> pd.DataFrame:
        """
        ProtoOATrendbar list → OHLCV DataFrame.

        cTrader delta encoding:
            high  = low + deltaHigh
            open  = low + deltaOpen
            close = low + deltaClose
        Prices are integers scaled by 100 000 (5-dp) or 1 000 (JPY).
        """
        divisor = 1_000 if symbol in JPY_PAIRS else 100_000
        records = []

        for bar in bars:
            try:
                low   = bar.low / divisor
                high  = (bar.low + bar.deltaHigh)  / divisor
                open_ = (bar.low + bar.deltaOpen)  / divisor
                close = (bar.low + bar.deltaClose) / divisor
                records.append({
                    "time":   pd.Timestamp(bar.utcTimestampInMinutes * 60, unit="s", tz="UTC"),
                    "open":   open_,
                    "high":   high,
                    "low":    low,
                    "close":  close,
                    "volume": bar.volume,
                })
            except Exception as exc:
                logger.warning(f"Skipping bar: {exc}")

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("time").sort_index()
        df = df[(df["high"] >= df["low"]) & (df["open"] > 0) & (df["close"] > 0)]
        return df

    def _store_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Upsert OHLCV rows into TimescaleDB."""
        try:
            from data_pipeline.storage.timeseries_db import db
            rows = [
                {
                    "time":      row.Index.to_pydatetime(),
                    "symbol":    symbol,
                    "timeframe": timeframe,
                    "open":      float(row.open),
                    "high":      float(row.high),
                    "low":       float(row.low),
                    "close":     float(row.close),
                    "volume":    float(row.volume),
                    "source":    "ctrader_openapi",
                }
                for row in df.itertuples()
            ]
            inserted = db.insert_ohlcv(rows)
            logger.debug(f"DB | {symbol} {timeframe} | {inserted} rows upserted")
        except Exception as exc:
            logger.error(f"DB upsert failed for {symbol}: {exc}")

    # =========================================================================
    # Account management
    # =========================================================================

    def get_account_balance(self) -> Dict[str, Any]:
        """Fetch balance via ProtoOATraderReq."""
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOATraderReq
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
        response = self._send_request(req)
        if response is None:
            return {}
        t = response.trader
        result = {
            "balance":    t.balance / 100,
            "currency":   getattr(t, "depositCurrency", "USD"),
            "account_id": t.ctidTraderAccountId,
        }
        logger.info(f"Balance: {result['balance']:.2f} {result['currency']}")
        return result

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch open positions via ProtoOAReconcileReq."""
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAReconcileReq
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
        response = self._send_request(req)
        if response is None:
            return []
        return [
            {
                "position_id": p.positionId,
                "symbol_id":   p.tradeData.symbolId,
                "direction":   "BUY" if p.tradeData.tradeSide == 1 else "SELL",
                "volume":      p.tradeData.volume,
                "entry_price": p.price,
                "stop_loss":   getattr(p, "stopLoss",   0.0),
                "take_profit": getattr(p, "takeProfit", 0.0),
                "commission":  p.commission / 100,
                "swap":        p.swap / 100,
            }
            for p in response.position
        ]

    def get_trade_history(
        self,
        start: datetime,
        end:   Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch closed deals via ProtoOADealListReq."""
        if end is None:
            end = datetime.now(tz=timezone.utc)
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOADealListReq
        req = ProtoOADealListReq()
        req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
        req.fromTimestamp       = int(start.timestamp() * 1000)
        req.toTimestamp         = int(end.timestamp()   * 1000)
        response = self._send_request(req)
        if response is None:
            return []
        deals = []
        for d in response.deal:
            pnl = 0.0
            try:
                if d.HasField("closePositionDetail"):
                    pnl = d.closePositionDetail.grossProfit / 100
            except Exception:
                pass
            deals.append({
                "deal_id":     d.dealId,
                "position_id": d.positionId,
                "symbol_id":   d.symbolId,
                "direction":   "BUY" if d.tradeSide == 1 else "SELL",
                "volume":      d.filledVolume,
                "price":       d.executionPrice,
                "close_ts":    d.executionTimestamp,
                "pnl":         pnl,
                "commission":  d.commission / 100,
                "swap":        d.swap / 100,
            })
        return deals

    # =========================================================================
    # Context manager
    # =========================================================================

    def __enter__(self) -> "CTraderOpenAPIConnector":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()


# ── Token refresh utility ─────────────────────────────────────────────────────

def refresh_access_token() -> Dict[str, str]:
    """
    Refresh cTrader access token via Auth.refreshToken().
    Refresh tokens are SINGLE-USE — update .env immediately after calling.
    """
    from ctrader_open_api import Auth
    auth  = Auth(_get("CTRADER_CLIENT_ID"), _get("CTRADER_CLIENT_SECRET"), "https://localhost")
    token = auth.refreshToken(_get("CTRADER_REFRESH_TOKEN"))
    logger.success("Token refreshed — update .env now!")
    logger.info(f"  access:  {token.accessToken[:20]}...")
    logger.info(f"  refresh: {token.refreshToken[:20]}...")
    return {
        "accessToken":  token.accessToken,
        "refreshToken": token.refreshToken,
        "expiresIn":    token.expiresIn,
    }


# ── Module singleton ──────────────────────────────────────────────────────────
connector = CTraderOpenAPIConnector()
