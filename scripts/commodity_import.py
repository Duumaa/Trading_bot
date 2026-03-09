"""
scripts/commodity_import.py
============================
Imports WTI Crude Oil and Gold (XAU/USD) historical prices
into the macro_indicators table.

Commodity → currency correlation used by Analyst 2:
  WTI Oil  → CAD (Canada exports oil  → oil up = CAD bullish)
  WTI Oil  → NOK (Norway exports oil  → oil up = NOK bullish)
  Gold     → AUD (Australia exports gold → gold up = AUD bullish)
  Gold     → CHF (safe haven → gold up = CHF bullish)
  Gold     → JPY (safe haven → gold up = JPY bullish)

DB table: macro_indicators
  country   = "OIL" or "XAU"
  indicator = "WTI_CLOSE" or "GOLD_CLOSE"
  actual    = daily closing price
  source    = "kaggle"

Usage:
  python3 << 'PYEOF'
  from dotenv import load_dotenv
  load_dotenv('/workspaces/Trading_bot/.env')
  import sys; sys.path.insert(0, '/workspaces/Trading_bot')
  from scripts.commodity_import import commodity_importer
  commodity_importer.run()
  PYEOF
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import text

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

from data_pipeline.storage.timeseries_db import db  # noqa: E402

# ── file paths ────────────────────────────────────────────────
COMM_DIR = ROOT / "data" / "kaggle" / "commoduty"
OIL_FILE  = COMM_DIR / "Crude Oil WTI Futures Historical Data.csv"
GOLD_FILE = COMM_DIR / "XAU_USD Historical Data.csv"

BATCH_SIZE = 500


# ── helpers ───────────────────────────────────────────────────

def _clean_price(val: str) -> Optional[float]:
    """
    '5,043.11' → 5043.11   |   '62.89' → 62.89   |   '' → None
    Strips commas, percentage signs, whitespace.
    """
    if pd.isna(val):
        return None
    cleaned = str(val).replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _load_commodity(filepath: Path, country: str, indicator: str) -> pd.DataFrame:
    """
    Loads a commodity CSV and returns a clean DataFrame ready for DB insert.

    CSV format:
      Date       : MM/DD/YYYY  (e.g. "02/13/2026")
      Price      : closing price (may have commas e.g. "5,043.11")
      Open/High/Low: ignored for now
      Vol.       : ignored
      Change %   : ignored

    Returns DataFrame with columns:
      time, country, indicator, actual
    """
    if not filepath.exists():
        logger.error(f"CommodityImporter | File not found: {filepath}")
        return pd.DataFrame()

    logger.info(f"CommodityImporter | Reading {filepath.name}...")

    df = pd.read_csv(filepath)

    # Normalise column names
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Parse date — format MM/DD/YYYY
    df["time"] = pd.to_datetime(df["Date"].str.strip().str.replace('"', ''),
                                format="%m/%d/%Y", utc=True)

    # Clean price
    df["actual"] = df["Price"].apply(_clean_price)

    # Drop rows with no price
    df = df.dropna(subset=["actual"])

    # Sort ascending
    df = df.sort_values("time").drop_duplicates(subset=["time"])

    df["country"]   = country
    df["indicator"] = indicator
    df["source"]    = "kaggle"

    logger.info(
        f"CommodityImporter | {indicator}: {len(df):,} rows "
        f"({df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()})"
    )

    return df[["time", "country", "indicator", "actual", "source"]]


def _insert(df: pd.DataFrame) -> int:
    """
    Batch-inserts commodity rows into macro_indicators.
    ON CONFLICT (time, country, indicator) → DO NOTHING.

    Note: macro_indicators has no explicit UNIQUE constraint on
    (time, country, indicator) in schema, so we use DO NOTHING
    via the existing index idx_macro_country_indicator.
    We rely on the index to handle duplicates gracefully.
    """
    if df.empty:
        return 0

    sql = """
        INSERT INTO macro_indicators
            (time, country, indicator, actual, source)
        VALUES
            (:time, :country, :indicator, :actual, :source)
        ON CONFLICT DO NOTHING
    """

    rows = df.to_dict("records")
    inserted = 0

    with db.connection() as conn:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            conn.execute(text(sql), batch)
            inserted += len(batch)
        conn.commit()

    return inserted


# ── main class ────────────────────────────────────────────────

class CommodityImporter:
    """
    Imports WTI Oil and Gold prices into macro_indicators table.

    These prices are used by Analyst 2 (score_calculator.py) to
    compute commodity correlation scores for commodity-linked
    currency pairs: CAD, NOK, AUD, NZD, CHF, JPY.
    """

    def run(self) -> None:
        """Import Oil and Gold, then verify."""
        logger.info("=" * 55)
        logger.info("CommodityImporter | COMMODITY IMPORT START")
        logger.info("=" * 55)

        # 1. WTI Crude Oil
        logger.info("\n[1/2] WTI Crude Oil...")
        oil_df = _load_commodity(OIL_FILE, country="OIL", indicator="WTI_CLOSE")
        oil_inserted = _insert(oil_df)
        logger.success(f"CommodityImporter | WTI Oil: {oil_inserted:,} rows inserted")

        # 2. Gold (XAU/USD)
        logger.info("\n[2/2] Gold (XAU/USD)...")
        gold_df = _load_commodity(GOLD_FILE, country="XAU", indicator="GOLD_CLOSE")
        gold_inserted = _insert(gold_df)
        logger.success(f"CommodityImporter | Gold: {gold_inserted:,} rows inserted")

        # 3. Verify
        self._verify()

        logger.info("=" * 55)
        logger.success(
            f"CommodityImporter | DONE — "
            f"Oil: {oil_inserted:,} | Gold: {gold_inserted:,}"
        )
        logger.info("=" * 55)

    def _verify(self) -> None:
        """Print row counts and date ranges from DB."""
        sql = """
            SELECT country, indicator, COUNT(*) as cnt,
                   MIN(time)::date as from_date,
                   MAX(time)::date as to_date
            FROM macro_indicators
            WHERE country IN ('OIL', 'XAU')
            GROUP BY country, indicator
            ORDER BY country
        """
        with db.connection() as conn:
            rows = conn.execute(text(sql)).fetchall()

        logger.info("\n  Verification:")
        for row in rows:
            logger.info(
                f"  {row[0]} | {row[1]} | "
                f"{row[2]:,} rows | {row[3]} → {row[4]}"
            )


# ── singleton ─────────────────────────────────────────────────
commodity_importer = CommodityImporter()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")
    commodity_importer.run()