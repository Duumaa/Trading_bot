"""
config/settings.py
==================
Central configuration for the G10 Dominance Architecture.

All settings are loaded from environment variables (via .env file).
This module is the SINGLE source of truth for all configuration.
Import this instead of using os.environ directly anywhere in the codebase.

Usage:
    from config.settings import settings
    print(settings.DB_URL)
    print(settings.G10_PAIRS)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from loguru import logger

# ── Load .env from project root ───────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=False)


# ── Enums ─────────────────────────────────────────────────────────────────────

class TradingMode(str, Enum):
    """System operating mode — controls order routing."""
    PAPER = "PAPER"     # No real orders; simulated execution
    LIVE  = "LIVE"      # Real money; real FIX orders


class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


# ── Helper ────────────────────────────────────────────────────────────────────

def _require(key: str) -> str:
    """Return env var or raise a descriptive error at startup."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file at {_ROOT / '.env'}"
        )
    return value


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _get_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _get_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ── Settings dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """
    Immutable settings object. Loaded once at import time.

    All values are validated at construction — the system fails fast
    at startup rather than mid-run on a missing credential.
    """

    # ── System ────────────────────────────────────────────────────────────────
    TRADING_MODE: TradingMode = field(
        default_factory=lambda: TradingMode(_get("TRADING_MODE", "PAPER"))
    )
    LOG_LEVEL: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))
    PROJECT_ROOT: Path = field(default_factory=lambda: _ROOT)

    # ── Trading universe ──────────────────────────────────────────────────────
    G10_PAIRS: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
        "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
    ])
    G10_CURRENCIES: List[str] = field(default_factory=lambda: [
        "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"
    ])
    PRIMARY_TIMEFRAME: str = "M15"
    CANDLE_SECONDS: int = 900   # 15 * 60

    # ── cTrader FIX API ───────────────────────────────────────────────────────
    FIX_HOST: str = field(default_factory=lambda: _get("CTRADER_FIX_HOST", "h51.p.ctrader.com"))
    FIX_PORT: int = field(default_factory=lambda: _get_int("CTRADER_FIX_PORT", 5201))
    FIX_SENDER_COMP_ID: str = field(default_factory=lambda: _require("CTRADER_SENDER_COMP_ID"))
    FIX_TARGET_COMP_ID: str = field(default_factory=lambda: _get("CTRADER_TARGET_COMP_ID", "CSERVER"))
    FIX_USERNAME: str = field(default_factory=lambda: _require("CTRADER_FIX_USERNAME"))
    FIX_PASSWORD: str = field(default_factory=lambda: _require("CTRADER_FIX_PASSWORD"))
    FIX_HEARTBEAT_SECONDS: int = 30
    FIX_RECONNECT_MAX_RETRIES: int = 5
    FIX_ORDER_BOOK_DEPTH: int = 20

    # ── cTrader REST / Open API ───────────────────────────────────────────────
    CTRADER_CLIENT_ID: str = field(default_factory=lambda: _require("CTRADER_CLIENT_ID"))
    CTRADER_CLIENT_SECRET: str = field(default_factory=lambda: _require("CTRADER_CLIENT_SECRET"))
    CTRADER_ACCOUNT_ID: str = field(default_factory=lambda: _require("CTRADER_ACCOUNT_ID"))
    CTRADER_ACCESS_TOKEN: str = field(default_factory=lambda: _get("CTRADER_ACCESS_TOKEN", ""))
    CTRADER_REFRESH_TOKEN: str = field(default_factory=lambda: _get("CTRADER_REFRESH_TOKEN", ""))
    CTRADER_REST_BASE_URL: str = "https://api.ctrader.com"
    CTRADER_REST_MAX_RPS: int = 50   # max requests per second

    # ── TimescaleDB ───────────────────────────────────────────────────────────
    DB_HOST: str = field(default_factory=lambda: _get("DB_HOST", "localhost"))
    DB_PORT: int = field(default_factory=lambda: _get_int("DB_PORT", 5432))
    DB_NAME: str = field(default_factory=lambda: _get("DB_NAME", "g10bot"))
    DB_USER: str = field(default_factory=lambda: _get("DB_USER", "g10bot"))
    DB_PASSWORD: str = field(default_factory=lambda: _require("DB_PASSWORD"))
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    @property
    def DB_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def DB_URL_ASYNC(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_HOST: str = field(default_factory=lambda: _get("REDIS_HOST", "localhost"))
    REDIS_PORT: int = field(default_factory=lambda: _get_int("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = field(default_factory=lambda: _get("REDIS_PASSWORD", ""))
    REDIS_DB: int = 0
    REDIS_CACHE_TTL_SECONDS: int = 300   # 5 min default cache

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── InfluxDB ──────────────────────────────────────────────────────────────
    INFLUX_URL: str = field(default_factory=lambda: _get("INFLUX_URL", "http://localhost:8086"))
    INFLUX_TOKEN: str = field(default_factory=lambda: _require("INFLUX_ADMIN_TOKEN"))
    INFLUX_ORG: str = field(default_factory=lambda: _get("INFLUX_ORG", "g10_trading"))
    INFLUX_BUCKET: str = field(default_factory=lambda: _get("INFLUX_BUCKET", "system_metrics"))

    # ── MLflow ────────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = field(
        default_factory=lambda: _get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    MLFLOW_EXPERIMENT_NAME: str = "g10_dominance"

    # ── LLM APIs ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = field(default_factory=lambda: _require("GROQ_API_KEY"))
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_RPD: int = 14400          # free tier daily limit

    ANTHROPIC_API_KEY: str = field(default_factory=lambda: _require("ANTHROPIC_API_KEY"))
    ANTHROPIC_MODEL: str = "claude-opus-4-5"   # weekly deep analysis

    # ── Flask Web Portal ──────────────────────────────────────────────────────
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 8080
    FLASK_SECRET_KEY: str = field(default_factory=lambda: _require("FLASK_SECRET_KEY"))
    FLASK_DEBUG: bool = field(default_factory=lambda: _get_bool("FLASK_DEBUG", False))
    ADMIN_USERNAME: str = field(default_factory=lambda: _get("ADMIN_USERNAME", "admin"))
    ADMIN_PASSWORD: str = field(default_factory=lambda: _require("ADMIN_PASSWORD"))

    # ── Telegram alerts ───────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = field(default_factory=lambda: _get("TELEGRAM_BOT_TOKEN", ""))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: _get("TELEGRAM_CHAT_ID", ""))

    @property
    def TELEGRAM_ENABLED(self) -> bool:
        return bool(self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID)

    # ── Risk management ───────────────────────────────────────────────────────
    MAX_RISK_PCT_PER_TRADE: float = 0.02       # 2% of equity
    MAX_CONCURRENT_POSITIONS: int = 3
    DAILY_MAX_LOSS_PCT: float = 0.05           # -5% kill switch
    PRE_EVENT_FLAT_HOURS: float = 2.0          # flatten before FOMC/NFP
    MIN_RISK_REWARD: float = 1.5
    MAX_LEVERAGE: float = 10.0
    NORMAL_REGIME_SIZE_FACTOR: float = 1.0
    HIGH_VOL_REGIME_SIZE_FACTOR: float = 0.5
    CRISIS_REGIME_SIZE_FACTOR: float = 0.0    # flat

    # ── Signal gate thresholds ────────────────────────────────────────────────
    FUND_MIN_CONFIDENCE: float = 0.60          # Analyst 1 min confidence
    FUND_MIN_SCORE_ABS: float = 0.30           # |Analyst 2 score| threshold
    TECH_MIN_PROB: float = 0.55                # TFT min directional probability
    TECH_MIN_AUC: float = 0.55                 # TSMixer min AUC
    SENT_MIN_SCORE_ABS: float = 0.20           # |Sentiment score| threshold
    SENT_MIN_CONFIDENCE: float = 0.60

    # ── Data sources ──────────────────────────────────────────────────────────
    RSS_FEEDS: List[str] = field(default_factory=lambda: [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.ecb.europa.eu/rss/press.html",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    ])
    RSS_SCRAPE_INTERVAL_MINUTES: int = 15
    COT_DOWNLOAD_URL: str = "https://www.cftc.gov/dea/newcot/f_disagg.zip"
    FOREXFACTORY_URL: str = "https://www.forexfactory.com/calendar"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_DIR: Path = field(default_factory=lambda: _ROOT / "logs")
    LOG_ROTATION: str = "100 MB"
    LOG_RETENTION: str = "10 files"

    def __post_init__(self) -> None:
        """Validate settings after init and create required directories."""
        # Create log directory
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Validate trading mode
        if self.TRADING_MODE == TradingMode.LIVE:
            logger.warning(
                "⚠️  LIVE TRADING MODE ACTIVE — real orders will be sent to cTrader!"
            )
        else:
            logger.info("📄 PAPER trading mode — no real orders will be executed.")

        # Warn if Telegram not configured
        if not self.TELEGRAM_ENABLED:
            logger.warning("Telegram alerts disabled — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")

    def is_live(self) -> bool:
        """Convenience: are we in live trading mode?"""
        return self.TRADING_MODE == TradingMode.LIVE

    def is_paper(self) -> bool:
        """Convenience: are we in paper trading mode?"""
        return self.TRADING_MODE == TradingMode.PAPER


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this everywhere: `from config.settings import settings`
try:
    settings = Settings()
except EnvironmentError as e:
    logger.error(f"Configuration error: {e}")
    raise
