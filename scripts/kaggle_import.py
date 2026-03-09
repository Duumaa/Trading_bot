"""
scripts/kaggle_import.py
========================
Kaggle-аас татаж авсан түүхэн өгөгдлийг TimescaleDB-д импортлоно.

Импортлох өгөгдлүүд:
  1. M15 OHLCV  (2011–2026) — /data/kaggle/m15/*.csv
  2. D1  OHLCV  (2007–2026) — /data/kaggle/d1/*.csv
  3. Bond yields (2Y/10Y)   — /data/kaggle/bond/*.csv
  4. ForexFactory events    — /data/kaggle/g10_forexfactory_utc.csv

DB Schema (баталгаажсан 2026-03-09):
  ohlcv          → time, symbol, timeframe, open, high, low, close, volume, source
  yield_data     → date, currency, tenor, yield_pct
  economic_events→ event_time, country, currency, event_name, impact,
                   actual, forecast, previous, source

Usage:
  python3 << 'PYEOF'
  from dotenv import load_dotenv
  load_dotenv('/workspaces/Trading_bot/.env')
  import sys
  sys.path.insert(0, '/workspaces/Trading_bot')
  from scripts.kaggle_import import importer
  importer.run_all()
  PYEOF
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy import text

# ── project root ──────────────────────────────────────────────
ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

from data_pipeline.storage.timeseries_db import db  # noqa: E402

# ── paths ─────────────────────────────────────────────────────
KAGGLE_DIR   = ROOT / "data" / "kaggle"
M15_DIR      = KAGGLE_DIR / "m15"
D1_DIR       = KAGGLE_DIR / "d1"
BOND_DIR     = KAGGLE_DIR / "bond"
FF_FILE      = KAGGLE_DIR / "g10_forexfactory_utc.csv"

# ── constants ─────────────────────────────────────────────────
BATCH_SIZE   = 1000   # DB insert batch size

# Манай 12 хос — зөвхөн эдгээрийг импортлоно
TARGET_PAIRS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
}

# JPY pair-уудыг ÷100 хийнэ (cTrader points → бодит үнэ)
JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"}

# Bond файл нэр → (currency, tenor) map
# Файл нэр: AUD10.csv, US2.csv, EUR10.csv гэх мэт
BOND_FILE_MAP: Dict[str, Tuple[str, str]] = {
    "AUD10": ("AUD", "10Y"), "AUD2":  ("AUD", "2Y"),
    "CAD10": ("CAD", "10Y"), "CAD2":  ("CAD", "2Y"),
    "CHF10": ("CHF", "10Y"), "CHF2":  ("CHF", "2Y"),
    "EUR10": ("EUR", "10Y"), "EUR2":  ("EUR", "2Y"),
    "GBP10": ("GBP", "10Y"), "GBP2":  ("GBP", "2Y"),
    "JPY10": ("JPY", "10Y"), "JPY2":  ("JPY", "2Y"),
    "NOK10": ("NOK", "10Y"), "NOK3":  ("NOK", "3M"),
    "NZD10": ("NZD", "10Y"), "NZD2":  ("NZD", "2Y"),
    "SEK10": ("SEK", "10Y"), "SEK2":  ("SEK", "2Y"),
    "US10":  ("USD", "10Y"), "US2":   ("USD", "2Y"),
}

# ForexFactory impact → uppercase стандарт
IMPACT_MAP = {
    "high":   "HIGH",
    "medium": "MEDIUM",
    "med":    "MEDIUM",
    "low":    "LOW",
    "High":   "HIGH",
    "Medium": "MEDIUM",
    "Low":    "LOW",
    "HIGH":   "HIGH",
    "MEDIUM": "MEDIUM",
    "LOW":    "LOW",
}

# G10 + commodity currencies
VALID_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CHF",
    "AUD", "NZD", "CAD", "NOK", "SEK",
}


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def extract_pair_from_filename(filepath: Path) -> Optional[str]:
    """
    Файл нэрнээс хос нэрийг гаргана.
    EURUSD_M15.csv      → EURUSD
    GBPJPY_D1_19Y.csv   → GBPJPY
    """
    stem = filepath.stem                        # extension-гүй нэр
    match = re.match(r"^([A-Z]{6})", stem)
    if match:
        return match.group(1)
    return None


def fix_jpy_prices(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    JPY pair-уудыг ÷100 хийж бодит үнэ болгоно.
    GBPJPY: 13140.9 → 131.409
    """
    if symbol in JPY_PAIRS:
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                # Аль хэдийн зөв масштабтай эсэхийг шалгана
                # Хэрэв дундаж > 1000 бол ÷100 хэрэгтэй
                if df[col].mean() > 1000:
                    df[col] = df[col] / 100.0
    return df


def batch_insert(conn, sql: str, rows: List[dict]) -> int:
    """
    Өгөгдлийг BATCH_SIZE хэмжээгээр хуваан DB-д хийнэ.
    Returns: нийт оруулсан мөрийн тоо
    """
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        conn.execute(text(sql), batch)
        inserted += len(batch)
    conn.commit()
    return inserted


# ══════════════════════════════════════════════════════════════
# MAIN IMPORTER CLASS
# ══════════════════════════════════════════════════════════════

class KaggleImporter:
    """
    Kaggle CSV файлуудыг TimescaleDB-д импортлоно.

    Дараалал:
      1. import_ohlcv(timeframe="M15")  — M15 дата
      2. import_ohlcv(timeframe="D1")   — D1 дата
      3. import_yields()                — Bond yield дата
      4. import_events()                — ForexFactory events
    """

    # ── OHLCV ─────────────────────────────────────────────────

    def import_ohlcv(self, timeframe: str = "M15") -> Dict[str, int]:
        """
        M15 эсвэл D1 OHLCV CSV файлуудыг `ohlcv` хүснэгтэд импортлоно.

        Args:
            timeframe: "M15" эсвэл "D1"

        Returns:
            {symbol: inserted_count} dict
        """
        source_dir = M15_DIR if timeframe == "M15" else D1_DIR
        results: Dict[str, int] = {}

        csv_files = sorted(source_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"KaggleImporter | {source_dir} дотор CSV файл байхгүй!")
            return results

        logger.info(f"KaggleImporter | {timeframe} импорт эхэллээ — {len(csv_files)} файл олдлоо")

        sql = """
            INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume, source)
            VALUES (:time, :symbol, :timeframe, :open, :high, :low, :close, :volume, :source)
            ON CONFLICT (time, symbol, timeframe) DO NOTHING
        """

        for filepath in csv_files:
            symbol = extract_pair_from_filename(filepath)
            if not symbol:
                logger.warning(f"KaggleImporter | Хос нэр таних боломжгүй: {filepath.name}")
                continue

            if symbol not in TARGET_PAIRS:
                logger.debug(f"KaggleImporter | {symbol} манай 12 хос-д байхгүй — алгасав")
                continue

            try:
                logger.info(f"KaggleImporter | {symbol} {timeframe} уншиж байна: {filepath.name}")

                df = pd.read_csv(filepath, parse_dates=["timestamp"])
                df.columns = df.columns.str.lower()

                # timestamp → UTC timezone-тай болгоно
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
                else:
                    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

                # JPY ÷100 засвар
                df = fix_jpy_prices(df, symbol)

                # NaN шүүнэ
                df = df.dropna(subset=["open", "high", "low", "close"])

                # Volume байхгүй бол 0 болгоно
                if "volume" not in df.columns:
                    df["volume"] = 0.0

                rows = [
                    {
                        "time":      row["timestamp"].to_pydatetime(),
                        "symbol":    symbol,
                        "timeframe": timeframe,
                        "open":      float(row["open"]),
                        "high":      float(row["high"]),
                        "low":       float(row["low"]),
                        "close":     float(row["close"]),
                        "volume":    float(row.get("volume", 0)),
                        "source":    "kaggle",
                    }
                    for _, row in df.iterrows()
                ]

                with db.connection() as conn:
                    inserted = batch_insert(conn, sql, rows)

                results[symbol] = inserted
                logger.success(
                    f"KaggleImporter | {symbol} {timeframe}: "
                    f"{inserted:,} мөр оруулав "
                    f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})"
                )

            except Exception as exc:
                logger.error(f"KaggleImporter | {symbol} {timeframe} алдаа: {exc}")
                results[symbol] = 0

        total = sum(results.values())
        logger.success(f"KaggleImporter | {timeframe} дууслаа — нийт {total:,} мөр, {len(results)} хос")
        return results

    # ── BOND YIELDS ───────────────────────────────────────────

    def import_yields(self) -> int:
        """
        Bond yield CSV файлуудыг `yield_data` хүснэгтэд импортлоно.

        Файл нэрийн загвар: AUD10.csv → currency=AUD, tenor=10Y
        Column: timestamp, Price (бусад column-уудыг алгасна)

        Returns:
            Нийт оруулсан мөрийн тоо
        """
        csv_files = sorted(BOND_DIR.glob("*.csv"))
        if not csv_files:
            logger.error(f"KaggleImporter | {BOND_DIR} дотор CSV файл байхгүй!")
            return 0

        logger.info(f"KaggleImporter | Bond yield импорт эхэллээ — {len(csv_files)} файл")

        sql = """
            INSERT INTO yield_data (date, currency, tenor, yield_pct)
            VALUES (:date, :currency, :tenor, :yield_pct)
            ON CONFLICT (date, currency, tenor) DO NOTHING
        """

        total_inserted = 0

        for filepath in csv_files:
            stem = filepath.stem   # e.g. "AUD10", "US2", "NOK3"

            if stem not in BOND_FILE_MAP:
                logger.warning(f"KaggleImporter | Bond файл map-д байхгүй: {filepath.name} — алгасав")
                continue

            currency, tenor = BOND_FILE_MAP[stem]

            try:
                df = pd.read_csv(filepath, parse_dates=["timestamp"])
                df.columns = df.columns.str.strip()

                # timestamp → date болгоно
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
                else:
                    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

                df["date"] = df["timestamp"].dt.date

                # Price column шалгана
                if "Price" not in df.columns and "price" not in df.columns:
                    logger.error(f"KaggleImporter | {filepath.name}: 'Price' column байхгүй!")
                    continue

                price_col = "Price" if "Price" in df.columns else "price"

                # Тоо болгоно, NaN хасна
                df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
                df = df.dropna(subset=[price_col])

                # Нэг өдөрт олон мөр байвал сүүлийн утгыг авна
                df = df.sort_values("timestamp").drop_duplicates(subset=["date"], keep="last")

                rows = [
                    {
                        "date":      row["date"],
                        "currency":  currency,
                        "tenor":     tenor,
                        "yield_pct": float(row[price_col]),
                    }
                    for _, row in df.iterrows()
                ]

                with db.connection() as conn:
                    inserted = batch_insert(conn, sql, rows)

                total_inserted += inserted
                logger.success(
                    f"KaggleImporter | {stem} ({currency} {tenor}): "
                    f"{inserted:,} мөр "
                    f"({df['date'].min()} → {df['date'].max()})"
                )

            except Exception as exc:
                logger.error(f"KaggleImporter | {filepath.name} алдаа: {exc}")

        logger.success(f"KaggleImporter | Yield дууслаа — нийт {total_inserted:,} мөр")
        return total_inserted

    # ── FOREXFACTORY EVENTS ───────────────────────────────────

    def import_events(self) -> int:
        """
        ForexFactory CSV файлыг `economic_events` хүснэгтэд импортлоно.

        CSV columns: timestamp, currency, impact, event, actual, forecast, previous
        DB columns:  event_time, country, currency, event_name, impact,
                     actual, forecast, previous, source

        Returns:
            Нийт оруулсан мөрийн тоо
        """
        if not FF_FILE.exists():
            logger.error(f"KaggleImporter | ForexFactory файл олдсонгүй: {FF_FILE}")
            return 0

        logger.info(f"KaggleImporter | ForexFactory импорт эхэллээ: {FF_FILE.name}")

        try:
            df = pd.read_csv(FF_FILE, parse_dates=["timestamp"])
            df.columns = df.columns.str.strip().str.lower()

            logger.info(f"KaggleImporter | {len(df):,} мөр уншигдлаа")

            # timestamp UTC болгоно
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

            # Impact uppercase болгоно
            df["impact"] = df["impact"].astype(str).str.strip().map(
                lambda x: IMPACT_MAP.get(x, "LOW")
            )

            # Impact check constraint: HIGH, MEDIUM, LOW — бусдыг LOW болгоно
            valid_impacts = {"HIGH", "MEDIUM", "LOW"}
            invalid_mask = ~df["impact"].isin(valid_impacts)
            if invalid_mask.any():
                logger.warning(
                    f"KaggleImporter | {invalid_mask.sum()} мөрийн impact утга хүчинтэй биш → LOW болгов"
                )
                df.loc[invalid_mask, "impact"] = "LOW"

            # Currency шүүнэ — зөвхөн G10
            df["currency"] = df["currency"].astype(str).str.strip().str.upper()
            df = df[df["currency"].isin(VALID_CURRENCIES)]
            logger.info(f"KaggleImporter | G10 шүүсний дараа: {len(df):,} мөр")

            # NaN утгуудыг None болгоно (DB NULL)
            for col in ["actual", "forecast", "previous"]:
                if col in df.columns:
                    df[col] = df[col].where(df[col].notna(), None)
                    # VARCHAR(20) хязгаар
                    df[col] = df[col].apply(
                        lambda x: str(x)[:20] if x is not None else None
                    )
                else:
                    df[col] = None

            # event_name VARCHAR(200) хязгаар
            event_col = "event" if "event" in df.columns else "event_name"
            df["event_name"] = df[event_col].astype(str).str[:200]

            sql = """
                INSERT INTO economic_events
                    (event_time, country, currency, event_name, impact,
                     actual, forecast, previous, source)
                VALUES
                    (:event_time, :country, :currency, :event_name, :impact,
                     :actual, :forecast, :previous, :source)
                ON CONFLICT DO NOTHING
            """

            rows = [
                {
                    "event_time": row["timestamp"].to_pydatetime(),
                    "country":    row["currency"],   # country = currency code
                    "currency":   row["currency"],
                    "event_name": row["event_name"],
                    "impact":     row["impact"],
                    "actual":     row.get("actual"),
                    "forecast":   row.get("forecast"),
                    "previous":   row.get("previous"),
                    "source":     "forexfactory_kaggle",
                }
                for _, row in df.iterrows()
            ]

            with db.connection() as conn:
                inserted = batch_insert(conn, sql, rows)

            logger.success(
                f"KaggleImporter | ForexFactory дууслаа — "
                f"{inserted:,} мөр "
                f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})"
            )
            return inserted

        except Exception as exc:
            logger.error(f"KaggleImporter | ForexFactory алдаа: {exc}")
            return 0

    # ── VERIFY ────────────────────────────────────────────────

    def verify(self) -> None:
        """
        Импортын дараа DB дахь мөрийн тоог шалгаж хэвлэнэ.
        """
        logger.info("KaggleImporter | Баталгаажуулалт эхэллээ...")

        queries = {
            "ohlcv M15 (нийт мөр)": """
                SELECT COUNT(*) FROM ohlcv WHERE timeframe='M15' AND source='kaggle'
            """,
            "ohlcv D1  (нийт мөр)": """
                SELECT COUNT(*) FROM ohlcv WHERE timeframe='D1' AND source='kaggle'
            """,
            "ohlcv хос бүрийн тоо": """
                SELECT symbol, timeframe, COUNT(*) as cnt,
                       MIN(time)::date as from_date,
                       MAX(time)::date as to_date
                FROM ohlcv WHERE source='kaggle'
                GROUP BY symbol, timeframe
                ORDER BY timeframe, symbol
            """,
            "yield_data (нийт мөр)": """
                SELECT COUNT(*) FROM yield_data
            """,
            "yield_data хос бүрийн тоо": """
                SELECT currency, tenor, COUNT(*) as cnt,
                       MIN(date) as from_date, MAX(date) as to_date
                FROM yield_data
                GROUP BY currency, tenor
                ORDER BY currency, tenor
            """,
            "economic_events (нийт мөр)": """
                SELECT COUNT(*) FROM economic_events WHERE source='forexfactory_kaggle'
            """,
        }

        with db.connection() as conn:
            for label, sql in queries.items():
                try:
                    result = conn.execute(text(sql))
                    rows = result.fetchall()
                    if len(rows) == 1 and len(rows[0]) == 1:
                        logger.info(f"  {label}: {rows[0][0]:,}")
                    else:
                        logger.info(f"\n  {label}:")
                        for row in rows:
                            logger.info(f"    {row}")
                except Exception as exc:
                    logger.error(f"  {label} алдаа: {exc}")

    # ── RUN ALL ───────────────────────────────────────────────

    def run_all(self) -> None:
        """
        Бүх импортыг дараалан ажиллуулна.
        Дараалал: M15 → D1 → Yields → Events → Verify
        """
        logger.info("=" * 60)
        logger.info("KaggleImporter | БҮРЭН ИМПОРТ ЭХЭЛЛЭЭ")
        logger.info("=" * 60)

        # 1. M15 OHLCV
        logger.info("\n[1/4] M15 OHLCV импорт...")
        m15_results = self.import_ohlcv("M15")

        # 2. D1 OHLCV
        logger.info("\n[2/4] D1 OHLCV импорт...")
        d1_results = self.import_ohlcv("D1")

        # 3. Bond Yields
        logger.info("\n[3/4] Bond yield импорт...")
        yield_count = self.import_yields()

        # 4. ForexFactory Events
        logger.info("\n[4/4] ForexFactory events импорт...")
        event_count = self.import_events()

        # 5. Verify
        logger.info("\n[✓] Баталгаажуулалт...")
        self.verify()

        logger.info("=" * 60)
        logger.success("KaggleImporter | БҮРЭН ИМПОРТ ДУУСЛАА")
        logger.info(f"  M15 хос: {len(m15_results)}, "
                    f"D1 хос: {len(d1_results)}, "
                    f"Yield: {yield_count:,}, "
                    f"Events: {event_count:,}")
        logger.info("=" * 60)


# ── Module singleton ──────────────────────────────────────────
importer = KaggleImporter()


# ── Direct run ───────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")
    importer.run_all()