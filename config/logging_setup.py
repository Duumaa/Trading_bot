"""
config/logging_setup.py
=======================
Configures loguru for the entire G10 Dominance system.

Call setup_logging() once at startup (in main.py).
All modules then use: from loguru import logger
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings


def setup_logging() -> None:
    """
    Configure loguru with:
    - Console output (colourised, human-readable)
    - Rotating file output (JSON-structured for machine parsing)
    - Trade-specific log file (all trade events)
    - Error-only log file

    Call once at process startup.
    """
    # Remove default handler
    logger.remove()

    log_dir: Path = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Console: human-readable ───────────────────────────────────────────────
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── File: all logs (JSON, rotating 100MB, keep 10) ────────────────────────
    logger.add(
        log_dir / "g10_{time:YYYY-MM-DD}.log",
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="gz",
        enqueue=True,           # thread-safe async writes
        backtrace=True,
        diagnose=False,         # no variable inspection in file logs (security)
    )

    # ── File: trade events only ───────────────────────────────────────────────
    logger.add(
        log_dir / "trades_{time:YYYY-MM-DD}.log",
        level="INFO",
        filter=lambda record: "TRADE" in record["extra"],
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        rotation="1 day",
        retention="30 files",
        compression="gz",
        enqueue=True,
    )

    # ── File: errors only ────────────────────────────────────────────────────
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}\n{exception}",
        rotation=settings.LOG_ROTATION,
        retention="30 files",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info(
        f"Logging initialised | level={settings.LOG_LEVEL} | "
        f"log_dir={log_dir} | mode={settings.TRADING_MODE.value}"
    )