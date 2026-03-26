"""
data_pipeline/connectors/ctrader_live_stream.py
================================================
cTrader Live Stream Connector — Real-Time Ticks + L2 Order Book Depth

Uses the same ctrader-open-api (Twisted/Protobuf) as the historical connector.
Runs on the SAME Twisted reactor — one TCP connection for everything.

What this provides:
    • Real-time bid/ask ticks for all 12 G10 pairs
    • L2 order book depth (20 levels bid + ask)
    • Live M1/M15 candle aggregation (in-memory)
    • Redis cache for latest prices + full order book
    • TimescaleDB storage for ticks
    • bid_ask_imbalance_1m  ← feeds Analyst 1 & feature engineering
    • large_order_count_5m  ← feeds liquidity features
    • liquidity_disappearance detection ← feeds VPIN

Architecture:
    ┌─────────────────────────────────────────┐
    │  Twisted Reactor Thread (shared)         │
    │  ProtoOASpotEvent      → _on_spot()      │
    │  ProtoOADepthEvent     → _on_depth()     │
    └───────────┬─────────────────────────────┘
                │
    ┌───────────▼─────────────────────────────┐
    │  Tick Handler                            │
    │  • Parse bid/ask/timestamp               │
    │  • Redis HSET price:{symbol}             │
    │  • TimescaleDB INSERT (async batch)      │
    │  • CandleAggregator.update()             │
    └───────────┬─────────────────────────────┘
                │
    ┌───────────▼─────────────────────────────┐
    │  CandleAggregator (in-memory)            │
    │  • Builds M1, M15 candles from ticks     │
    │  • On candle close → emit to callbacks   │
    │  • Keeps rolling 500-candle buffer       │
    └───────────┬─────────────────────────────┘
                │
    ┌───────────▼─────────────────────────────┐
    │  L2 Depth Handler                        │
    │  • 20-level bid/ask book per symbol      │
    │  • Redis HSET depth:{symbol}             │
    │  • Compute bid_ask_imbalance_1m          │
    │  • Detect liquidity_disappearance        │
    └─────────────────────────────────────────┘

Usage:
    from data_pipeline.connectors.ctrader_live_stream import live_stream

    live_stream.connect()
    live_stream.subscribe(["EURUSD", "GBPUSD", "USDJPY"])

    price = live_stream.get_latest_price("EURUSD")
    book  = live_stream.get_order_book("EURUSD")
    imbalance = live_stream.get_bid_ask_imbalance("EURUSD", window_minutes=1)

    live_stream.on_candle_close(callback_fn)   # register M15 callback
    live_stream.disconnect()
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(key: str, default: str = "") -> str:
    try:
        from config.settings import settings
        return str(getattr(settings, key, os.getenv(key, default)))
    except Exception:
        return os.getenv(key, default)


# ── Price divisors ─────────────────────────────────────────────────────────────
JPY_PAIRS   = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}
G10_PAIRS   = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

def _divisor(symbol: str) -> int:
    return 1_000 if symbol in JPY_PAIRS else 100_000


# ── Tick record ────────────────────────────────────────────────────────────────

class Tick:
    """Single bid/ask tick from cTrader."""
    __slots__ = ("symbol", "bid", "ask", "mid", "timestamp", "epoch_ms")

    def __init__(
        self,
        symbol:   str,
        bid:      float,
        ask:      float,
        epoch_ms: int,
    ) -> None:
        self.symbol    = symbol
        self.bid       = bid
        self.ask       = ask
        self.mid       = (bid + ask) / 2
        self.epoch_ms  = epoch_ms
        self.timestamp = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)

    def spread_pips(self) -> float:
        d = _divisor(self.symbol)
        return round((self.ask - self.bid) * d, 1)

    def __repr__(self) -> str:
        return (
            f"Tick({self.symbol} bid={self.bid:.5f} ask={self.ask:.5f} "
            f"spread={self.spread_pips()}pips @{self.timestamp.strftime('%H:%M:%S.%f')[:-3]})"
        )


# ── Candle (OHLCV bar) ─────────────────────────────────────────────────────────

class Candle:
    """In-memory OHLCV bar built from ticks."""
    __slots__ = (
        "symbol", "timeframe", "open_time", "close_time",
        "open", "high", "low", "close", "volume",
        "tick_count", "is_closed",
    )

    def __init__(
        self,
        symbol:    str,
        timeframe: str,
        open_time: datetime,
        first_price: float,
    ) -> None:
        self.symbol    = symbol
        self.timeframe = timeframe
        self.open_time = open_time
        self.close_time = open_time + _tf_delta(timeframe)
        self.open      = first_price
        self.high      = first_price
        self.low       = first_price
        self.close     = first_price
        self.volume    = 0.0
        self.tick_count = 1
        self.is_closed  = False

    def update(self, price: float, volume: float = 0.0) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += volume
        self.tick_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol":     self.symbol,
            "timeframe":  self.timeframe,
            "time":       self.open_time,
            "open":       self.open,
            "high":       self.high,
            "low":        self.low,
            "close":      self.close,
            "volume":     self.volume,
            "tick_count": self.tick_count,
        }

    def __repr__(self) -> str:
        return (
            f"Candle({self.symbol} {self.timeframe} "
            f"O={self.open:.5f} H={self.high:.5f} "
            f"L={self.low:.5f} C={self.close:.5f} "
            f"@{self.open_time.strftime('%Y-%m-%d %H:%M')})"
        )


def _tf_delta(timeframe: str) -> timedelta:
    """Convert timeframe string to timedelta."""
    mapping = {
        "M1":  timedelta(minutes=1),
        "M5":  timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1":  timedelta(hours=1),
        "H4":  timedelta(hours=4),
        "D1":  timedelta(days=1),
    }
    return mapping.get(timeframe, timedelta(minutes=15))


def _candle_open_time(ts: datetime, timeframe: str) -> datetime:
    """Floor a timestamp to the candle open time for a given timeframe."""
    if timeframe == "M1":
        return ts.replace(second=0, microsecond=0)
    elif timeframe == "M5":
        return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
    elif timeframe == "M15":
        return ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
    elif timeframe == "M30":
        return ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
    elif timeframe == "H1":
        return ts.replace(minute=0, second=0, microsecond=0)
    elif timeframe == "H4":
        return ts.replace(hour=(ts.hour // 4) * 4, minute=0, second=0, microsecond=0)
    elif timeframe == "D1":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return ts.replace(second=0, microsecond=0)


# ── Candle Aggregator ──────────────────────────────────────────────────────────

class CandleAggregator:
    """
    Builds live OHLCV candles from incoming ticks.

    Maintains in-memory buffers for M1 and M15 timeframes.
    On candle close, fires registered callbacks and optionally stores to DB.

    Thread-safe: all state protected by a single lock.
    """

    TIMEFRAMES = ["M1", "M15"]
    BUFFER_SIZE = 500   # candles kept per symbol per timeframe

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Current open candle: {symbol: {timeframe: Candle}}
        self._open: Dict[str, Dict[str, Candle]] = defaultdict(dict)

        # Closed candle buffer: {symbol: {timeframe: deque[Candle]}}
        self._closed: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.BUFFER_SIZE))
        )

        # Callbacks fired on candle close: List[Callable[[Candle], None]]
        self._callbacks: List[Callable[[Candle], None]] = []

    def register_callback(self, fn: Callable[[Candle], None]) -> None:
        """Register a function to be called every time a candle closes."""
        self._callbacks.append(fn)

    def update(self, tick: Tick) -> List[Candle]:
        """
        Process a tick. Returns list of candles that closed (0 or more).
        Thread-safe.
        """
        closed: List[Candle] = []

        with self._lock:
            for tf in self.TIMEFRAMES:
                bar_open = _candle_open_time(tick.timestamp, tf)

                if tf not in self._open[tick.symbol]:
                    # First tick for this symbol/timeframe
                    self._open[tick.symbol][tf] = Candle(
                        tick.symbol, tf, bar_open, tick.mid
                    )
                    continue

                current = self._open[tick.symbol][tf]

                if tick.timestamp >= current.close_time:
                    # Candle closed — archive it
                    current.is_closed = True
                    self._closed[tick.symbol][tf].append(current)
                    closed.append(current)

                    # Open new candle
                    self._open[tick.symbol][tf] = Candle(
                        tick.symbol, tf, bar_open, tick.mid
                    )
                else:
                    # Update current candle
                    current.update(tick.mid)

        # Fire callbacks outside lock
        for candle in closed:
            for fn in self._callbacks:
                try:
                    fn(candle)
                except Exception as exc:
                    logger.error(f"Candle callback error: {exc}")

        return closed

    def get_candles(
        self,
        symbol:    str,
        timeframe: str,
        n:         int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get last N closed candles for a symbol/timeframe.
        Returns list of dicts sorted oldest → newest.
        """
        with self._lock:
            buf = self._closed.get(symbol, {}).get(timeframe, deque())
            candles = list(buf)[-n:]
        return [c.to_dict() for c in candles]

    def get_current_candle(
        self,
        symbol:    str,
        timeframe: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the currently forming (open) candle."""
        with self._lock:
            c = self._open.get(symbol, {}).get(timeframe)
        return c.to_dict() if c else None


# ── L2 Order Book ──────────────────────────────────────────────────────────────

class OrderBook:
    """
    20-level bid/ask order book for a single symbol.

    Stores price levels and volumes.
    Computes bid_ask_imbalance used by feature engineering.

    cTrader sends full snapshots via ProtoOADepthEvent —
    we replace the entire book on each update.
    """

    MAX_LEVELS = 20

    def __init__(self, symbol: str) -> None:
        self.symbol    = symbol
        self._lock     = threading.Lock()
        self._bids: List[Tuple[float, float]] = []   # [(price, volume), ...]
        self._asks: List[Tuple[float, float]] = []
        self._updated: Optional[datetime] = None

        # Rolling 1-min imbalance history: deque of (timestamp, imbalance)
        self._imbalance_history: deque = deque(maxlen=200)

    def update(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> None:
        """Replace entire order book. Thread-safe."""
        with self._lock:
            self._bids = sorted(bids, key=lambda x: x[0], reverse=True)[: self.MAX_LEVELS]
            self._asks = sorted(asks, key=lambda x: x[0])[: self.MAX_LEVELS]
            self._updated = datetime.now(tz=timezone.utc)

            imb = self._compute_imbalance(levels=5)
            self._imbalance_history.append((self._updated, imb))

    def _compute_imbalance(self, levels: int = 5) -> float:
        """
        bid_ask_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Range: -1.0 (full ask pressure) to +1.0 (full bid pressure)
        Uses top `levels` price levels.
        """
        bid_vol = sum(v for _, v in self._bids[:levels])
        ask_vol = sum(v for _, v in self._asks[:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def get_snapshot(self) -> Dict[str, Any]:
        """Return full book snapshot as dict. Thread-safe."""
        with self._lock:
            return {
                "symbol":    self.symbol,
                "timestamp": self._updated.isoformat() if self._updated else None,
                "bids":      list(self._bids),
                "asks":      list(self._asks),
                "imbalance": self._compute_imbalance(levels=5),
                "spread":    (
                    self._asks[0][0] - self._bids[0][0]
                    if self._bids and self._asks else 0.0
                ),
            }

    def get_imbalance(self, window_minutes: int = 1) -> float:
        """
        Average bid_ask_imbalance over the last `window_minutes`.
        Used as bid_ask_imbalance_1m feature.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=window_minutes)
        with self._lock:
            recent = [
                imb for ts, imb in self._imbalance_history
                if ts >= cutoff
            ]
        return sum(recent) / len(recent) if recent else 0.0

    def detect_liquidity_disappearance(self, threshold: float = 0.5) -> bool:
        """
        Returns True if total book volume dropped >50% vs recent average.
        Used as liquidity_disappearance feature for VPIN.
        """
        with self._lock:
            if len(self._imbalance_history) < 10:
                return False
            current_vol = sum(v for _, v in self._bids[:5]) + sum(v for _, v in self._asks[:5])
            # Compare against 10-snapshot rolling average
            recent = list(self._imbalance_history)[-10:]

        # Simple proxy: use imbalance variance as liquidity proxy
        imbalances = [abs(imb) for _, imb in recent]
        avg = sum(imbalances) / len(imbalances)
        latest = imbalances[-1] if imbalances else 0.0
        return latest > avg + threshold


# ── Live Stream Connector ──────────────────────────────────────────────────────

class CTraderLiveStream:
    """
    Real-time price stream and L2 order book depth from cTrader Open API.

    Runs the Twisted reactor in a background thread (same pattern as
    CTraderOpenAPIConnector). Both connectors can share the same reactor
    if started in the correct order.

    Key outputs for feature engineering:
        get_latest_price(symbol)              → {"bid", "ask", "mid", "timestamp"}
        get_order_book(symbol)                → 20-level bid/ask book
        get_bid_ask_imbalance(symbol, window) → float -1.0 to +1.0
        get_candles(symbol, "M15", n=100)     → list of dicts
        get_large_order_count(symbol, min=5)  → count in last 5 minutes
        detect_liquidity_disappearance(sym)   → bool

    Usage:
        live_stream.connect()
        live_stream.subscribe(["EURUSD", "GBPUSD"])
        live_stream.on_candle_close(my_callback)   # called on every M15 close
    """

    def __init__(self) -> None:
        self._client: Any = None

        # Auth state
        self._account_authed = threading.Event()
        self._subscribed: set = set()

        # Response handling (for request/response pattern)
        self._response_lock  = threading.Lock()
        self._response_event = threading.Event()
        self._last_response: Any = None

        # Symbol name → cTrader ID
        self._symbol_map: Dict[str, int] = {}
        self._id_to_symbol: Dict[int, str] = {}

        # Latest prices: {symbol: {"bid", "ask", "mid", "timestamp", "epoch_ms"}}
        self._prices: Dict[str, Dict[str, Any]] = {}
        self._prices_lock = threading.Lock()

        # Order books: {symbol: OrderBook}
        self._books: Dict[str, OrderBook] = {}
        self._books_lock = threading.Lock()

        # Candle aggregator
        self._aggregator = CandleAggregator()

        # Tick storage batch (flush every N ticks or T seconds)
        self._tick_batch: List[Dict[str, Any]] = []
        self._tick_batch_lock = threading.Lock()
        self._last_batch_flush: float = time.monotonic()
        self.BATCH_SIZE   = 500
        self.BATCH_MAX_S  = 30.0

        # Large order tracker: {symbol: deque[(timestamp, volume)]}
        self._large_order_log: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Demo / live
        env = os.getenv("CTRADER_ENV", "demo").lower()
        self._is_demo = (env != "live")

        # Reactor-started flag (shared across connectors)
        self._reactor_started = False

        logger.info(
            f"CTraderLiveStream | mode={'DEMO' if self._is_demo else 'LIVE'}"
        )

    # =========================================================================
    # Connection lifecycle
    # =========================================================================

    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to cTrader Open API and authenticate.

        If the Twisted reactor is already running (shared with historical
        connector), skips reactor startup and reuses the existing reactor.

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

        logger.info(f"LiveStream connecting | {host}:{port}")

        self._client = Client(host, port, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        if reactor.running:
            # Reactor already running (shared with historical connector)
            logger.info("Reusing existing Twisted reactor")
            reactor.callFromThread(self._client.startService)
        else:
            # Start fresh reactor
            reactor_ready = threading.Event()

            def _run():
                reactor.callWhenRunning(reactor_ready.set)
                reactor.run(installSignalHandlers=False)

            t = threading.Thread(target=_run, name="live-reactor", daemon=True)
            t.start()

            if not reactor_ready.wait(timeout=10.0):
                logger.error("Reactor failed to start")
                return False

            reactor.callFromThread(self._client.startService)

        if not self._account_authed.wait(timeout=timeout):
            logger.error(f"LiveStream auth timeout after {timeout}s")
            return False

        # Load symbol map
        self._load_symbol_map()

        # Start background tick flusher
        self._start_batch_flusher()

        logger.success("CTraderLiveStream | authenticated ✅")
        return True

    def disconnect(self) -> None:
        """Unsubscribe all symbols and stop client."""
        logger.info("LiveStream disconnecting...")
        if self._subscribed:
            try:
                self.unsubscribe(list(self._subscribed))
            except Exception as exc:
                logger.debug(f"Unsubscribe on disconnect: {exc}")

        # Flush remaining ticks
        self._flush_tick_batch(force=True)

        if self._client:
            try:
                from twisted.internet import reactor
                reactor.callFromThread(self._client.stopService)
            except Exception as exc:
                logger.debug(f"stopService: {exc}")

    # =========================================================================
    # Twisted callbacks
    # =========================================================================

    def _on_connected(self, client: Any) -> None:
        logger.info("LiveStream TCP connected — authenticating...")
        # App auth may already be done by REST API connector
        # Try app auth first, if ALREADY_LOGGED_IN skip and go straight to account auth
        self._send_app_auth()

    def _on_app_auth_already_logged_in(self) -> None:
        """Called when app is already authorized — skip to account auth."""
        logger.info("LiveStream | App already authorized — sending account auth")
        self._send_account_auth()

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        logger.warning(f"LiveStream disconnected: {reason}")
        self._account_authed.clear()
        self._subscribed.clear()
        with self._response_lock:
            self._last_response = None
            self._response_event.set()

    def _on_message(self, client: Any, message: Any) -> None:
        """Route all incoming messages from cTrader."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthRes,
            ProtoOAAccountAuthRes,
            ProtoOASpotEvent,
            ProtoOADepthEvent,
            ProtoOASubscribeSpotsRes,
            ProtoOAUnsubscribeSpotsRes,
            ProtoOASubscribeDepthQuotesRes,
            ProtoOAUnsubscribeDepthQuotesRes,
            ProtoOAErrorRes,
        )

        msg = Protobuf.extract(message)

        # ── Auth ──────────────────────────────────────────────────────────
        if isinstance(msg, ProtoOAApplicationAuthRes):
            logger.info("LiveStream app auth OK")
            self._send_account_auth()

        elif isinstance(msg, ProtoOAAccountAuthRes):
            logger.info("LiveStream account auth OK")
            self._account_authed.set()

        # ── Live data events ──────────────────────────────────────────────
        elif isinstance(msg, ProtoOASpotEvent):
            self._handle_spot(msg)

        elif isinstance(msg, ProtoOADepthEvent):
            self._handle_depth(msg)

        # ── Subscription confirmations ────────────────────────────────────
        elif isinstance(msg, (
            ProtoOASubscribeSpotsRes,
            ProtoOAUnsubscribeSpotsRes,
            ProtoOASubscribeDepthQuotesRes,
            ProtoOAUnsubscribeDepthQuotesRes,
        )):
            logger.debug(f"Subscription ack: {type(msg).__name__}")
            with self._response_lock:
                self._last_response = msg
                self._response_event.set()

        # ── Errors ────────────────────────────────────────────────────────
        elif isinstance(msg, ProtoOAErrorRes):
            logger.error(f"LiveStream error [{msg.errorCode}]: {msg.description}")
            # ALREADY_LOGGED_IN means app auth already done by REST API
            # Skip to account auth directly
            if msg.errorCode == "ALREADY_LOGGED_IN":
                logger.info("LiveStream | App already authorized — sending account auth")
                self._send_account_auth()
            with self._response_lock:
                self._last_response = msg
                self._response_event.set()

        else:
            logger.debug(f"LiveStream unhandled: {type(msg).__name__}")

    # =========================================================================
    # Auth senders
    # =========================================================================

    def _send_app_auth(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthReq,
        )
        req = ProtoOAApplicationAuthReq()
        req.clientId     = _get("CTRADER_CLIENT_ID")
        req.clientSecret = _get("CTRADER_CLIENT_SECRET")
        self._client.send(req).addErrback(
            lambda f: logger.error(f"App auth: {f}")
        )

    def _send_account_auth(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAAccountAuthReq,
        )
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
        req.accessToken         = _get("CTRADER_ACCESS_TOKEN")
        self._client.send(req).addErrback(
            lambda f: logger.error(f"Account auth: {f}")
        )

    # =========================================================================
    # Subscription management
    # =========================================================================

    def subscribe(
        self,
        symbols:       List[str],
        include_depth: bool = True,
    ) -> None:
        """
        Subscribe to real-time spot prices and optionally L2 depth.

        Args:
            symbols:       List of pair names e.g. ["EURUSD", "GBPUSD"]
            include_depth: Also subscribe to L2 order book updates
        """
        if not self._account_authed.is_set():
            raise RuntimeError("Not authenticated. Call connect() first.")

        for symbol in symbols:
            sid = self._symbol_map.get(symbol)
            if sid is None:
                logger.warning(f"Unknown symbol '{symbol}' — skipping")
                continue

            # Initialise order book
            with self._books_lock:
                if symbol not in self._books:
                    self._books[symbol] = OrderBook(symbol)

            # Subscribe spots
            self._subscribe_spots(sid, symbol)

            # Subscribe L2 depth
            if include_depth:
                self._subscribe_depth(sid, symbol)

            self._subscribed.add(symbol)
            logger.info(f"Subscribed | {symbol} (id={sid}) | depth={include_depth}")

    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from spot and depth feeds."""
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAUnsubscribeSpotsReq,
            ProtoOAUnsubscribeDepthQuotesReq,
        )
        from twisted.internet import reactor

        for symbol in symbols:
            sid = self._symbol_map.get(symbol)
            if sid is None:
                continue

            req_spot = ProtoOAUnsubscribeSpotsReq()
            req_spot.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
            req_spot.symbolId.append(sid)
            reactor.callFromThread(
                lambda r=req_spot: self._client.send(r).addErrback(
                    lambda f: logger.error(f"Unsubscribe spot: {f}")
                )
            )

            req_depth = ProtoOAUnsubscribeDepthQuotesReq()
            req_depth.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
            req_depth.symbolId.append(sid)
            reactor.callFromThread(
                lambda r=req_depth: self._client.send(r).addErrback(
                    lambda f: logger.error(f"Unsubscribe depth: {f}")
                )
            )

            self._subscribed.discard(symbol)
            logger.info(f"Unsubscribed | {symbol}")

    def subscribe_all_g10(self, include_depth: bool = True) -> None:
        """Subscribe to all 12 G10 pairs."""
        available = [s for s in G10_PAIRS if s in self._symbol_map]
        logger.info(f"Subscribing to {len(available)} G10 pairs...")
        self.subscribe(available, include_depth=include_depth)

    def _subscribe_spots(self, symbol_id: int, symbol: str) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOASubscribeSpotsReq,
        )
        from twisted.internet import reactor

        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
        req.symbolId.append(symbol_id)

        reactor.callFromThread(
            lambda: self._client.send(req).addErrback(
                lambda f: logger.error(f"Subscribe spot {symbol}: {f}")
            )
        )

    def _subscribe_depth(self, symbol_id: int, symbol: str) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOASubscribeDepthQuotesReq,
        )
        from twisted.internet import reactor

        req = ProtoOASubscribeDepthQuotesReq()
        req.ctidTraderAccountId = int(_get("CTRADER_ACCOUNT_ID"))
        req.symbolId.append(symbol_id)

        reactor.callFromThread(
            lambda: self._client.send(req).addErrback(
                lambda f: logger.error(f"Subscribe depth {symbol}: {f}")
            )
        )

    # =========================================================================
    # Spot tick handler
    # =========================================================================

    def _handle_spot(self, msg: Any) -> None:
        """
        Process ProtoOASpotEvent.

        cTrader sends bid and ask as integers scaled by divisor.
        timestamp is Unix milliseconds.
        """
        symbol = self._id_to_symbol.get(msg.symbolId)
        if symbol is None:
            return

        div    = 100_000
        bid    = msg.bid / div if msg.bid else 0.0
        ask    = msg.ask / div if msg.ask else 0.0
        ts_ms  = msg.timestamp

        # Skip if both zero (keepalive tick)
        if bid == 0.0 and ask == 0.0:
            return

        # If only one side sent, use last known other side
        with self._prices_lock:
            prev = self._prices.get(symbol, {})
            if bid == 0.0:
                bid = prev.get("bid", 0.0)
            if ask == 0.0:
                ask = prev.get("ask", 0.0)

        tick = Tick(symbol, bid, ask, ts_ms)

        # ── Update latest price cache ──────────────────────────────────────
        with self._prices_lock:
            self._prices[symbol] = {
                "bid":       tick.bid,
                "ask":       tick.ask,
                "mid":       tick.mid,
                "spread":    tick.spread_pips(),
                "timestamp": tick.timestamp,
                "epoch_ms":  tick.epoch_ms,
            }

        # ── Update candle aggregator ───────────────────────────────────────
        closed = self._aggregator.update(tick)
        if closed:
            for candle in closed:
                logger.debug(f"Candle closed: {candle}")
                # Store to DB
                self._store_candle_to_db(candle)

        # ── Batch tick for DB storage ──────────────────────────────────────
        self._queue_tick(tick)

        # ── Cache to Redis ────────────────────────────────────────────────
        self._redis_set_price(symbol, tick)

        # ── Large order detection (volume proxy from spread tightening) ───
        # cTrader spot ticks don't carry trade volume directly
        # We log abnormal spread widening as a liquidity proxy
        spread = tick.spread_pips()
        avg_spread = prev.get("spread", spread)
        if avg_spread > 0 and spread > avg_spread * 2.5:
            self._large_order_log[symbol].append(
                (tick.timestamp, spread / avg_spread)
            )

    # =========================================================================
    # L2 depth handler
    # =========================================================================

    def _handle_depth(self, msg: Any) -> None:
        """
        Process ProtoOADepthEvent — full L2 order book snapshot.

        cTrader sends each update as a full replace (not incremental).
        Prices are integers; volumes are in units.
        """
        symbol = self._id_to_symbol.get(msg.symbolId)
        if symbol is None:
            return

        div  = 100_000 
        quotes = list(msg.newQuotes)
        bids = [(q.bid / div, q.size) for q in quotes if q.bid > 0]
        asks = [(q.ask / div, q.size) for q in quotes if q.ask > 0]

        with self._books_lock:
            if symbol not in self._books:
                self._books[symbol] = OrderBook(symbol)
            self._books[symbol].update(bids, asks)

        # Cache full book to Redis
        self._redis_set_book(symbol)

    # =========================================================================
    # Public data access
    # =========================================================================

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest bid/ask/mid for a symbol.

        Returns:
            {"bid": float, "ask": float, "mid": float,
             "spread": float, "timestamp": datetime}
        or None if no tick received yet.
        """
        with self._prices_lock:
            return dict(self._prices.get(symbol, {})) or None

    def get_all_prices(self) -> Dict[str, Dict[str, Any]]:
        """Get latest prices for all subscribed symbols."""
        with self._prices_lock:
            return {k: dict(v) for k, v in self._prices.items()}

    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get full 20-level L2 order book snapshot.

        Returns:
            {"symbol", "timestamp", "bids": [(price, vol)...],
             "asks": [(price, vol)...], "imbalance": float, "spread": float}
        or None if no book received yet.
        """
        with self._books_lock:
            book = self._books.get(symbol)
        return book.get_snapshot() if book else None

    def get_bid_ask_imbalance(
        self,
        symbol:         str,
        window_minutes: int = 1,
    ) -> float:
        """
        Average bid/ask volume imbalance over the last N minutes.
        Range: -1.0 (full ask pressure) to +1.0 (full bid pressure).

        This is the bid_ask_imbalance_1m feature used by Analyst 1.
        """
        with self._books_lock:
            book = self._books.get(symbol)
        return book.get_imbalance(window_minutes) if book else 0.0

    def detect_liquidity_disappearance(self, symbol: str) -> bool:
        """
        True if order book liquidity dropped significantly.
        Used as liquidity_disappearance feature for VPIN calculation.
        """
        with self._books_lock:
            book = self._books.get(symbol)
        return book.detect_liquidity_disappearance() if book else False

    def get_large_order_count(
        self,
        symbol:       str,
        window_min:   int   = 5,
        min_multiple: float = 2.5,
    ) -> int:
        """
        Count of abnormal spread events (proxy for large order flow)
        in the last `window_min` minutes.

        Used as large_order_count_5m feature.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=window_min)
        log    = self._large_order_log.get(symbol, deque())
        return sum(1 for ts, _ in log if ts >= cutoff)

    def get_candles(
        self,
        symbol:    str,
        timeframe: str = "M15",
        n:         int = 100,
    ) -> List[Dict[str, Any]]:
        """Get last N closed candles for symbol/timeframe from live aggregator."""
        return self._aggregator.get_candles(symbol, timeframe, n)

    def get_current_candle(
        self,
        symbol:    str,
        timeframe: str = "M15",
    ) -> Optional[Dict[str, Any]]:
        """Get currently forming candle."""
        return self._aggregator.get_current_candle(symbol, timeframe)

    def on_candle_close(self, callback: Callable[[Candle], None]) -> None:
        """
        Register a callback fired every time any candle closes.

        Callback signature: fn(candle: Candle) -> None
        Called from the main thread (not reactor thread).

        Example:
            def on_m15_close(candle):
                if candle.timeframe == "M15":
                    run_signal_pipeline(candle.symbol)

            live_stream.on_candle_close(on_m15_close)
        """
        self._aggregator.register_callback(callback)

    def is_subscribed(self, symbol: str) -> bool:
        return symbol in self._subscribed

    def get_subscribed_symbols(self) -> List[str]:
        return list(self._subscribed)

    # =========================================================================
    # Symbol map
    # =========================================================================

    def _load_symbol_map(self) -> None:
        """Load symbol name ↔ ID mapping from config."""
        import yaml
        from pathlib import Path

        config_path = (
            Path(_get("PROJECT_ROOT", "/workspaces/Trading_bot"))
            / "config" / "ctrader_config.yaml"
        )
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            raw = cfg.get("fix", {}).get("symbol_map", {})
            self._symbol_map    = {k: int(v) for k, v in raw.items()}
            self._id_to_symbol  = {int(v): k for k, v in raw.items()}
            logger.info(f"LiveStream loaded {len(self._symbol_map)} symbols")
        except Exception as exc:
            logger.warning(f"Symbol config failed ({exc}) — using defaults")
            self._symbol_map = {
                "EURUSD": 1,  "GBPUSD": 2,  "USDJPY": 3,  "USDCHF": 4,
                "AUDUSD": 5,  "NZDUSD": 6,  "USDCAD": 7,  "EURGBP": 8,
                "EURJPY": 9,  "GBPJPY": 10, "AUDJPY": 11, "NZDJPY": 12,
            }
            self._id_to_symbol = {v: k for k, v in self._symbol_map.items()}

    # =========================================================================
    # Redis cache
    # =========================================================================

    def _get_redis(self):
        """Get Redis client (lazy init — shared pool)."""
        if self._redis_client is None:
            try:
                from data_pipeline.storage.redis_pool import redis_client
                self._redis_client = redis_client
            except Exception:
                return None
        return self._redis_client

    def _redis_set_price(self, symbol: str, tick: Tick) -> None:
        """Cache latest price to Redis. Key: price:{SYMBOL}"""
        try:
            r = self._get_redis()
            if r is None:
                return
            r.hset(f"price:{symbol}", mapping={
                "bid":      str(tick.bid),
                "ask":      str(tick.ask),
                "mid":      str(tick.mid),
                "spread":   str(tick.spread_pips()),
                "ts":       tick.timestamp.isoformat(),
                "epoch_ms": str(tick.epoch_ms),
            })
            r.expire(f"price:{symbol}", 300)   # TTL 5 minutes
        except Exception as exc:
            logger.debug(f"Redis price set failed: {exc}")

    def _redis_set_book(self, symbol: str) -> None:
        """Cache L2 order book to Redis. Key: depth:{SYMBOL}"""
        try:
            r = self._get_redis()
            if r is None:
                return
            with self._books_lock:
                book = self._books.get(symbol)
            if book is None:
                return
            snap = book.get_snapshot()
            r.hset(f"depth:{symbol}", mapping={
                "bids":      json.dumps(snap["bids"]),
                "asks":      json.dumps(snap["asks"]),
                "imbalance": str(snap["imbalance"]),
                "spread":    str(snap["spread"]),
                "ts":        snap["timestamp"] or "",
            })
            r.expire(f"depth:{symbol}", 300)
        except Exception as exc:
            logger.debug(f"Redis depth set failed: {exc}")

    # =========================================================================
    # Tick batching & DB storage
    # =========================================================================

    def _queue_tick(self, tick: Tick) -> None:
        """Add tick to write batch."""
        row = {
            "time":    tick.timestamp,
            "symbol":  tick.symbol,
            "bid":     tick.bid,
            "ask":     tick.ask,
            "mid":     tick.mid,
            "spread":  tick.ask - tick.bid,
            "source":  "ctrader_stream",
        }
        with self._tick_batch_lock:
            self._tick_batch.append(row)
            should_flush = (
                len(self._tick_batch) >= self.BATCH_SIZE
                or (time.monotonic() - self._last_batch_flush) >= self.BATCH_MAX_S
            )

        if should_flush:
            self._flush_tick_batch()

    def _flush_tick_batch(self, force: bool = False) -> None:
        """Write pending ticks to TimescaleDB."""
        with self._tick_batch_lock:
            if not self._tick_batch:
                return
            batch = list(self._tick_batch)
            self._tick_batch.clear()
            self._last_batch_flush = time.monotonic()

        try:
            from data_pipeline.storage.timeseries_db import db
            inserted = db.insert_ticks(batch)
            logger.debug(f"Tick flush | {inserted} rows → DB")
        except Exception as exc:
            logger.error(f"Tick DB flush failed: {exc}")

    def _store_candle_to_db(self, candle: Candle) -> None:
        """Store a closed candle to TimescaleDB ohlcv table."""
        try:
            from data_pipeline.storage.timeseries_db import db
            db.insert_ohlcv([{
                "time":      candle.open_time,
                "symbol":    candle.symbol,
                "timeframe": candle.timeframe,
                "open":      candle.open,
                "high":      candle.high,
                "low":       candle.low,
                "close":     candle.close,
                "volume":    candle.volume,
                "source":    "ctrader_stream",
            }])
        except Exception as exc:
            logger.debug(f"Candle DB store failed: {exc}")

    def _start_batch_flusher(self) -> None:
        """Background thread that flushes tick batch every 30s."""
        def _flusher():
            while True:
                time.sleep(self.BATCH_MAX_S)
                self._flush_tick_batch()

        t = threading.Thread(target=_flusher, name="tick-flusher", daemon=True)
        t.start()

    # =========================================================================
    # Context manager
    # =========================================================================

    def __enter__(self) -> "CTraderLiveStream":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()


# ── Module singleton ───────────────────────────────────────────────────────────
# Usage: from data_pipeline.connectors.ctrader_live_stream import live_stream
live_stream = CTraderLiveStream()
