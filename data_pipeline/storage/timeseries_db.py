"""
data_pipeline/storage/timeseries_db.py
=======================================
TimescaleDB interface for the G10 Dominance Architecture.

Provides:
- Connection pool management (SQLAlchemy)
- CRUD for all major tables
- Bulk insert helpers for high-throughput tick/candle data
- Data freshness tracking
- Thread-safe singleton access

Usage:
    from data_pipeline.storage.timeseries_db import db
    df = db.get_latest_ohlcv("EURUSD", "M15", limit=100)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from config.settings import settings


class TimescaleDB:
    """
    Thread-safe TimescaleDB client.

    Uses SQLAlchemy connection pooling with psycopg2.
    All public methods are safe to call from multiple threads concurrently.
    """

    def __init__(self) -> None:
        self._engine: Optional[Engine] = None
        self._lock = threading.Lock()

    # ── Connection management ─────────────────────────────────────────────────

    def _get_engine(self) -> Engine:
        """Lazy initialise the connection pool (thread-safe)."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:   # double-checked locking
                    self._engine = create_engine(
                        settings.DB_URL,
                        poolclass=QueuePool,
                        pool_size=settings.DB_POOL_SIZE,
                        max_overflow=settings.DB_MAX_OVERFLOW,
                        pool_pre_ping=True,     # validate connections on checkout
                        pool_recycle=3600,      # recycle connections after 1 hour
                        echo=False,
                    )
                    logger.info(
                        f"TimescaleDB pool created | "
                        f"host={settings.DB_HOST}:{settings.DB_PORT} "
                        f"db={settings.DB_NAME} "
                        f"pool_size={settings.DB_POOL_SIZE}"
                    )
        return self._engine

    @contextmanager
    def connection(self) -> Generator:
        """Context manager that yields a raw DB connection."""
        engine = self._get_engine()
        conn = engine.connect()
        try:
            yield conn
            conn.commit()
        except SQLAlchemyError as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def health_check(self) -> bool:
        """Return True if the database is reachable."""
        try:
            with self.connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"DB health check failed: {e}")
            return False

    # ── OHLCV ─────────────────────────────────────────────────────────────────

    def insert_ohlcv(self, rows: List[Dict[str, Any]]) -> int:
        """
        Bulk insert OHLCV candles.

        Args:
            rows: List of dicts with keys: time, symbol, timeframe, open, high, low, close, volume

        Returns:
            Number of rows inserted.
        """
        if not rows:
            return 0

        sql = text("""
            INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume, source)
            VALUES (:time, :symbol, :timeframe, :open, :high, :low, :close, :volume, :source)
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume
        """)

        with self.connection() as conn:
            conn.execute(sql, rows)

        self._update_freshness("ohlcv_ctrader", len(rows))
        logger.debug(f"Inserted {len(rows)} OHLCV rows for {rows[0]['symbol']}")
        return len(rows)

    def get_latest_ohlcv(
        self,
        symbol: str,
        timeframe: str = "M15",
        limit: int = 200,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch the most recent OHLCV candles for a symbol.

        Args:
            symbol: e.g. "EURUSD"
            timeframe: e.g. "M15", "H1"
            limit: number of candles to return
            end_time: if set, return candles ending at/before this time

        Returns:
            DataFrame with columns [time, open, high, low, close, volume], indexed by time.
        """
        where_time = "AND time <= :end_time" if end_time else ""
        sql = text(f"""
            SELECT time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe
            {where_time}
            ORDER BY time DESC
            LIMIT :limit
        """)

        params: Dict[str, Any] = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        if end_time:
            params["end_time"] = end_time

        with self.connection() as conn:
            result = conn.execute(sql, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if df.empty:
            return df

        df = df.sort_values("time").set_index("time")
        return df

    def get_ohlcv_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "M15",
    ) -> pd.DataFrame:
        """Fetch candles within a date range (inclusive)."""
        sql = text("""
            SELECT time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = :symbol
              AND timeframe = :timeframe
              AND time BETWEEN :start AND :end
            ORDER BY time ASC
        """)

        with self.connection() as conn:
            result = conn.execute(sql, {
                "symbol": symbol, "timeframe": timeframe,
                "start": start, "end": end
            })
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if not df.empty:
            df = df.set_index("time")
        return df

    # ── Ticks ─────────────────────────────────────────────────────────────────

    def insert_ticks(self, rows: List[Dict[str, Any]]) -> int:
        """Bulk insert tick data. Rows: {time, symbol, bid, ask}."""
        if not rows:
            return 0

        sql = text("""
            INSERT INTO ticks (time, symbol, bid, ask)
            VALUES (:time, :symbol, :bid, :ask)
        """)

        with self.connection() as conn:
            conn.execute(sql, rows)
        return len(rows)

    # ── Bond yields ───────────────────────────────────────────────────────────

    def insert_bond_yield(
        self,
        currency: str,
        tenor: str,
        yield_pct: float,
        source: str = "manual",
        time: Optional[datetime] = None,
    ) -> None:
        """Insert or update a bond yield observation."""
        if time is None:
            time = datetime.now(tz=timezone.utc)

        sql = text("""
            INSERT INTO bond_yields (time, currency, tenor, yield_pct, source)
            VALUES (:time, :currency, :tenor, :yield_pct, :source)
            ON CONFLICT (time, currency, tenor) DO UPDATE SET
                yield_pct = EXCLUDED.yield_pct,
                source    = EXCLUDED.source
        """)

        with self.connection() as conn:
            conn.execute(sql, {
                "time": time, "currency": currency, "tenor": tenor,
                "yield_pct": yield_pct, "source": source
            })

        self._update_freshness("bond_yields_manual" if source == "manual" else "bond_yields_scraped")
        logger.debug(f"Yield inserted: {currency} {tenor} = {yield_pct:.4f}% ({source})")

    def get_latest_yields(self) -> Dict[str, Dict[str, float]]:
        """
        Return latest yield for all G10 currencies and tenors.

        Returns:
            {currency: {tenor: yield_pct}}  e.g. {"USD": {"2Y": 4.25, "10Y": 4.50}, ...}
        """
        sql = text("""
            SELECT DISTINCT ON (currency, tenor)
                currency, tenor, yield_pct
            FROM bond_yields
            ORDER BY currency, tenor, time DESC
        """)

        with self.connection() as conn:
            rows = conn.execute(sql).fetchall()

        result: Dict[str, Dict[str, float]] = {}
        for row in rows:
            currency, tenor, yield_pct = row
            if currency not in result:
                result[currency] = {}
            result[currency][tenor] = yield_pct
        return result

    # ── Economic events ───────────────────────────────────────────────────────

    def get_upcoming_events(
        self,
        hours_ahead: float = 24.0,
        impact: str = "HIGH",
    ) -> pd.DataFrame:
        """
        Get upcoming economic events.

        Args:
            hours_ahead: how many hours into the future to look
            impact: filter by impact level ("HIGH", "MEDIUM", "LOW", or None for all)

        Returns:
            DataFrame with event details.
        """
        where_impact = "AND impact = :impact" if impact else ""
        sql = text(f"""
            SELECT id, event_time, country, currency, event_name, impact, forecast, previous
            FROM economic_events
            WHERE event_time BETWEEN NOW() AND NOW() + :hours * INTERVAL '1 hour'
            {where_impact}
            ORDER BY event_time ASC
        """)

        params: Dict[str, Any] = {"hours": hours_ahead}
        if impact:
            params["impact"] = impact

        with self.connection() as conn:
            result = conn.execute(sql, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        return df

    def insert_economic_events(self, events: List[Dict[str, Any]]) -> int:
        """Insert economic calendar events (upsert on event_time + country + event_name)."""
        if not events:
            return 0

        sql = text("""
            INSERT INTO economic_events
                (event_time, country, currency, event_name, impact, actual, forecast, previous, source)
            VALUES
                (:event_time, :country, :currency, :event_name, :impact, :actual, :forecast, :previous, :source)
            ON CONFLICT (event_time, country, event_name) DO NOTHING
        """)

        with self.connection() as conn:
            conn.execute(sql, events)

        self._update_freshness("economic_events", len(events))
        return len(events)

    # ── COT reports ───────────────────────────────────────────────────────────

    def insert_cot_report(self, row: Dict[str, Any]) -> None:
        """Insert or update a COT report row."""
        sql = text("""
            INSERT INTO cot_reports
                (report_date, currency, nc_long, nc_short, nc_net_prev, nc_net_change,
                 comm_long, comm_short, open_interest, nc_net_pct_oi)
            VALUES
                (:report_date, :currency, :nc_long, :nc_short, :nc_net_prev, :nc_net_change,
                 :comm_long, :comm_short, :open_interest, :nc_net_pct_oi)
            ON CONFLICT (report_date, currency) DO UPDATE SET
                nc_long        = EXCLUDED.nc_long,
                nc_short       = EXCLUDED.nc_short,
                nc_net_prev    = EXCLUDED.nc_net_prev,
                nc_net_change  = EXCLUDED.nc_net_change,
                comm_long      = EXCLUDED.comm_long,
                comm_short     = EXCLUDED.comm_short,
                open_interest  = EXCLUDED.open_interest,
                nc_net_pct_oi  = EXCLUDED.nc_net_pct_oi
        """)

        with self.connection() as conn:
            conn.execute(sql, row)
        self._update_freshness("cot_reports")

    def get_latest_cot(self) -> Dict[str, Dict[str, Any]]:
        """Return latest COT report for all currencies."""
        sql = text("""
            SELECT DISTINCT ON (currency)
                report_date, currency, nc_long, nc_short, nc_net,
                nc_net_change, open_interest, nc_net_pct_oi
            FROM cot_reports
            ORDER BY currency, report_date DESC
        """)

        with self.connection() as conn:
            rows = conn.execute(sql).fetchall()

        return {
            row.currency: dict(row._mapping)
            for row in rows
        }

    # ── Analyst signals ───────────────────────────────────────────────────────

    def insert_analyst_signal(self, signal: Dict[str, Any]) -> None:
        """Persist an analyst output signal."""
        sql = text("""
            INSERT INTO analyst_signals
                (time, analyst_id, analyst_name, pair, currency, signal_type,
                 signal_value, confidence, metadata, model_version)
            VALUES
                (:time, :analyst_id, :analyst_name, :pair, :currency, :signal_type,
                 :signal_value, :confidence, :metadata::jsonb, :model_version)
        """)

        import json
        signal = dict(signal)
        signal["metadata"] = json.dumps(signal.get("metadata", {}))

        with self.connection() as conn:
            conn.execute(sql, signal)

    def get_latest_signals(
        self,
        analyst_id: Optional[int] = None,
        pair: Optional[str] = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Retrieve recent analyst signals."""
        conditions = []
        params: Dict[str, Any] = {"limit": limit}

        if analyst_id is not None:
            conditions.append("analyst_id = :analyst_id")
            params["analyst_id"] = analyst_id
        if pair is not None:
            conditions.append("pair = :pair")
            params["pair"] = pair

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = text(f"""
            SELECT * FROM analyst_signals
            {where}
            ORDER BY time DESC
            LIMIT :limit
        """)

        with self.connection() as conn:
            result = conn.execute(sql, params)
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    # ── Trades ────────────────────────────────────────────────────────────────

    def insert_trade(self, trade: Dict[str, Any]) -> int:
        """Insert a new trade record. Returns the trade ID."""
        sql = text("""
            INSERT INTO trades (
                fix_order_id, ctrader_position_id, symbol, direction,
                volume_units, entry_price, stop_loss, take_profit, entry_time,
                status, regime, signal_1_regime, signal_1_confidence,
                signal_2_fund_score, signal_3_tft_prob, signal_4_tsmixer,
                signal_5_sentiment, signal_6_bias, requested_price,
                risk_pct, position_size_pct, trading_mode
            ) VALUES (
                :fix_order_id, :ctrader_position_id, :symbol, :direction,
                :volume_units, :entry_price, :stop_loss, :take_profit, :entry_time,
                :status, :regime, :signal_1_regime, :signal_1_confidence,
                :signal_2_fund_score, :signal_3_tft_prob, :signal_4_tsmixer,
                :signal_5_sentiment, :signal_6_bias, :requested_price,
                :risk_pct, :position_size_pct, :trading_mode
            )
            RETURNING id
        """)

        with self.connection() as conn:
            result = conn.execute(sql, trade)
            trade_id = result.scalar()

        logger.bind(TRADE=True).info(
            f"Trade inserted | id={trade_id} | {trade['symbol']} {trade['direction']} "
            f"@ {trade['entry_price']} | sl={trade['stop_loss']} tp={trade['take_profit']}"
        )
        return trade_id

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: datetime,
        status: str,
        pnl_pips: float,
        pnl_usd: float,
    ) -> None:
        """Mark a trade as closed with its outcome."""
        sql = text("""
            UPDATE trades
            SET exit_price = :exit_price,
                exit_time  = :exit_time,
                status     = :status,
                pnl_pips   = :pnl_pips,
                pnl_usd    = :pnl_usd,
                updated_at = NOW()
            WHERE id = :trade_id
        """)

        with self.connection() as conn:
            conn.execute(sql, {
                "trade_id": trade_id, "exit_price": exit_price,
                "exit_time": exit_time, "status": status,
                "pnl_pips": pnl_pips, "pnl_usd": pnl_usd
            })

        logger.bind(TRADE=True).info(
            f"Trade closed | id={trade_id} | status={status} | "
            f"pnl={pnl_pips:+.1f} pips ({pnl_usd:+.2f} USD)"
        )

    def get_open_trades(self) -> pd.DataFrame:
        """Return all currently open trades."""
        sql = text("""
            SELECT * FROM trades
            WHERE status = 'OPEN'
            ORDER BY entry_time ASC
        """)

        with self.connection() as conn:
            result = conn.execute(sql)
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    # ── Performance ───────────────────────────────────────────────────────────

    def upsert_performance_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Insert or update today's performance snapshot."""
        sql = text("""
            INSERT INTO performance_snapshots (
                snapshot_date, equity, balance, daily_pnl, daily_pnl_pct,
                trades_today, wins_today, losses_today, total_trades,
                win_rate, profit_factor, sharpe_ratio, max_drawdown,
                daily_loss_pct, trading_halted, halt_reason
            ) VALUES (
                :snapshot_date, :equity, :balance, :daily_pnl, :daily_pnl_pct,
                :trades_today, :wins_today, :losses_today, :total_trades,
                :win_rate, :profit_factor, :sharpe_ratio, :max_drawdown,
                :daily_loss_pct, :trading_halted, :halt_reason
            )
            ON CONFLICT (snapshot_date) DO UPDATE SET
                equity         = EXCLUDED.equity,
                balance        = EXCLUDED.balance,
                daily_pnl      = EXCLUDED.daily_pnl,
                daily_pnl_pct  = EXCLUDED.daily_pnl_pct,
                trades_today   = EXCLUDED.trades_today,
                wins_today     = EXCLUDED.wins_today,
                losses_today   = EXCLUDED.losses_today,
                total_trades   = EXCLUDED.total_trades,
                win_rate       = EXCLUDED.win_rate,
                profit_factor  = EXCLUDED.profit_factor,
                sharpe_ratio   = EXCLUDED.sharpe_ratio,
                max_drawdown   = EXCLUDED.max_drawdown,
                daily_loss_pct = EXCLUDED.daily_loss_pct,
                trading_halted = EXCLUDED.trading_halted,
                halt_reason    = EXCLUDED.halt_reason,
                created_at     = NOW()
        """)

        with self.connection() as conn:
            conn.execute(sql, snapshot)

    # ── Data freshness ────────────────────────────────────────────────────────

    def _update_freshness(self, source: str, record_count: Optional[int] = None) -> None:
        """Update data freshness tracker."""
        sql = text("""
            INSERT INTO data_freshness (source, last_update, status, record_count, updated_at)
            VALUES (:source, NOW(), 'OK', :record_count, NOW())
            ON CONFLICT (source) DO UPDATE SET
                last_update  = NOW(),
                status       = 'OK',
                record_count = COALESCE(:record_count, data_freshness.record_count),
                error_message = NULL,
                updated_at   = NOW()
        """)

        try:
            with self.connection() as conn:
                conn.execute(sql, {"source": source, "record_count": record_count})
        except Exception as e:
            logger.warning(f"Failed to update freshness for {source}: {e}")

    def get_data_freshness(self) -> pd.DataFrame:
        """Return freshness status for all data sources."""
        sql = text("SELECT * FROM data_freshness ORDER BY source")
        with self.connection() as conn:
            result = conn.execute(sql)
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def mark_freshness_error(self, source: str, error: str) -> None:
        """Mark a data source as errored."""
        sql = text("""
            UPDATE data_freshness
            SET status = 'ERROR', error_message = :error, updated_at = NOW()
            WHERE source = :source
        """)
        try:
            with self.connection() as conn:
                conn.execute(sql, {"source": source, "error": error})
        except Exception:
            pass  # don't let freshness tracking crash the main flow

    # ── Feature vectors ───────────────────────────────────────────────────────

    def insert_feature_vector(self, fv: Dict[str, Any]) -> None:
        """Upsert a pre-computed feature vector for (time, pair)."""
        # Build dynamic INSERT from dict keys
        cols = ", ".join(fv.keys())
        placeholders = ", ".join(f":{k}" for k in fv.keys())

        update_cols = ", ".join(
            f"{k} = EXCLUDED.{k}" for k in fv.keys()
            if k not in ("time", "pair")
        )

        sql = text(f"""
            INSERT INTO feature_vectors ({cols})
            VALUES ({placeholders})
            ON CONFLICT (time, pair) DO UPDATE SET {update_cols}
        """)

        with self.connection() as conn:
            conn.execute(sql, fv)

        self._update_freshness("feature_vectors")

    def get_feature_vectors(
        self,
        pair: str,
        limit: int = 200,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve feature vectors for ML model inference."""
        where_time = "AND time <= :end_time" if end_time else ""
        sql = text(f"""
            SELECT * FROM feature_vectors
            WHERE pair = :pair {where_time}
            ORDER BY time DESC
            LIMIT :limit
        """)

        params: Dict[str, Any] = {"pair": pair, "limit": limit}
        if end_time:
            params["end_time"] = end_time

        with self.connection() as conn:
            result = conn.execute(sql, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if not df.empty:
            df = df.sort_values("time").set_index("time")
        return df


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this everywhere: `from data_pipeline.storage.timeseries_db import db`
db = TimescaleDB()
