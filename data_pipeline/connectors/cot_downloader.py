"""
data_pipeline/connectors/cot_downloader.py
==========================================
CFTC Commitment of Traders (COT) Report Downloader

Follows the exact methodology from the G10 course:
    • Focus ONLY on non-commercial (hedge fund) positioning
    • Three CFTC data periods with different column names:
        1986–2006: "Non-Commercial Positions Long/Short All"
        2007–2014: "Non-Commercial Positions Long/Short All"  (same columns)
        2015–now:  "Leveraged Money Positions Long/Short All" (same definition, new name)
    • Key metric: THE FLIP = %NC_Long - %NC_Short (net hedge fund positioning)
    • Positive flip = hedge funds net long = bullish conviction
    • Negative flip = hedge funds net short = bearish conviction

G10 currency futures on CME (not all pairs available):
    EURUSD → Euro FX
    GBPUSD → British Pound
    USDJPY → Japanese Yen
    USDCHF → Swiss Franc
    AUDUSD → Australian Dollar
    NZDUSD → New Zealand Dollar
    USDCAD → Canadian Dollar
    (EURGBP, EURJPY, GBPJPY, AUDJPY, NZDJPY — crosses, not directly available)
    → For crosses: derive from individual currency net positions

CFTC Data URLs:
    Historical zip: https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip
    Current year:   https://www.cftc.gov/dea/newcot/f_disagg.txt

Usage:
    from data_pipeline.connectors.cot_downloader import cot_downloader

    # Weekly update (run every Friday ~16:00 EST)
    cot_downloader.fetch_latest()

    # Full historical backfill
    cot_downloader.backfill(start_year=1986)

    # Get flip score for a currency
    flip = cot_downloader.get_flip_score("AUD")
    # Returns: {"flip": -0.23, "direction": "SHORT", "change": -0.05}
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

# ── Constants ──────────────────────────────────────────────────────────────────

# CME futures contract names → currency code
# Source: CFTC disaggregated futures reports
CONTRACT_MAP: Dict[str, str] = {
    "EURO FX":                    "EUR",
    "BRITISH POUND":              "GBP",
    "JAPANESE YEN":               "JPY",
    "SWISS FRANC":                "CHF",
    "AUSTRALIAN DOLLAR":         "AUD",
    "NEW ZEALAND DOLLAR":        "NZD",
    "CANADIAN DOLLAR":           "CAD",
    # Alternative spellings in older reports
    "EURO FX - CHICAGO MERCANTILE EXCHANGE":           "EUR",
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE":      "JPY",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE":       "CHF",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUD",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE":"NZD",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE":   "CAD",
}

# CFTC URLs
CFTC_CURRENT_URL    = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
CFTC_LEGACY_URL     = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"

UTC = timezone.utc

# ── Column mappings per period ─────────────────────────────────────────────────
#
# Course specification:
#   1986–2014: Non-Commercial Longs/Shorts + % OI Non-Commercial Long/Short
#   2015–now:  Leveraged Money Longs/Shorts + % OI Leveraged Money Long/Short
#   These are the SAME definition, just renamed by CFTC in 2015.
#
# Disaggregated report columns (f_disagg.txt / fut_disagg_txt_YYYY.zip):

# TFF (Traders in Financial Futures) column names — consistent across all years
# Leveraged Money = hedge funds (course definition: non-commercial equivalent)
# These columns exist in ALL TFF files (2009–present)
COLS_TFF = {
    "report_date":   "Report_Date_as_YYYY-MM-DD",
    "contract":      "Market_and_Exchange_Names",
    "lm_long":       "Lev_Money_Positions_Long_All",
    "lm_short":      "Lev_Money_Positions_Short_All",
    "lm_long_chg":   "Change_in_Lev_Money_Long_All",
    "lm_short_chg":  "Change_in_Lev_Money_Short_All",
    "lm_pct_long":   "Pct_of_OI_Lev_Money_Long_All",
    "lm_pct_short":  "Pct_of_OI_Lev_Money_Short_All",
    "open_interest": "Open_Interest_All",
}
# Alias for compatibility
COLS_2015    = COLS_TFF
COLS_PRE2015 = COLS_TFF


# ── Parser ─────────────────────────────────────────────────────────────────────

def _parse_cot_df(df: pd.DataFrame, is_2015_format: bool) -> pd.DataFrame:
    """
    Parse raw CFTC DataFrame → standardised COT records for G10 currencies.

    Applies the course methodology:
        flip = pct_long - pct_short  (net hedge fund positioning)
        flip > 0 = net long (bullish)
        flip < 0 = net short (bearish)

    Returns DataFrame with columns:
        report_date, currency, nc_long, nc_short, nc_long_chg, nc_short_chg,
        nc_pct_long, nc_pct_short, flip, open_interest
    """
    cols = COLS_2015 if is_2015_format else COLS_PRE2015

    # Check required columns exist
    contract_col = cols["contract"]
    date_col     = cols["report_date"]

    if contract_col not in df.columns:
        logger.warning(f"Contract column '{contract_col}' not found")
        logger.debug(f"Available: {list(df.columns[:10])}")
        return pd.DataFrame()

    # Filter to G10 currency futures only
    df["_currency"] = (
        df[contract_col]
        .str.upper()
        .str.strip()
        .map(lambda x: next(
            (v for k, v in CONTRACT_MAP.items() if k in x), None
        ))
    )
    df = df[df["_currency"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    # Map column names based on period
    if is_2015_format:
        long_col     = cols["lm_long"]
        short_col    = cols["lm_short"]
        long_chg     = cols["lm_long_chg"]
        short_chg    = cols["lm_short_chg"]
        pct_long     = cols["lm_pct_long"]
        pct_short    = cols["lm_pct_short"]
    else:
        long_col     = cols["nc_long"]
        short_col    = cols["nc_short"]
        long_chg     = cols["nc_long_chg"]
        short_chg    = cols["nc_short_chg"]
        pct_long     = cols["nc_pct_long"]
        pct_short    = cols["nc_pct_short"]

    oi_col = cols["open_interest"]

    def _safe_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")

    # Parse date
    try:
        report_dates = pd.to_datetime(df[date_col], errors="coerce")
        # Ensure UTC
        if report_dates.dt.tz is None:
            report_dates = report_dates.dt.tz_localize("UTC")
        else:
            report_dates = report_dates.dt.tz_convert("UTC")
    except Exception as exc:
        logger.warning(f"Date parse failed: {exc}")
        return pd.DataFrame()

    nc_long_s      = _safe_float(df[long_col])   if long_col  in df.columns else pd.Series(0, index=df.index)
    nc_short_s     = _safe_float(df[short_col])  if short_col in df.columns else pd.Series(0, index=df.index)
    nc_long_chg_s  = _safe_float(df[long_chg])   if long_chg  in df.columns else pd.Series(0, index=df.index)
    nc_short_chg_s = _safe_float(df[short_chg])  if short_chg in df.columns else pd.Series(0, index=df.index)
    pct_long_s     = _safe_float(df[pct_long])   if pct_long  in df.columns else pd.Series(0, index=df.index)
    pct_short_s    = _safe_float(df[pct_short])  if pct_short in df.columns else pd.Series(0, index=df.index)
    oi_s           = _safe_float(df[oi_col])     if oi_col    in df.columns else pd.Series(0, index=df.index)

    # THE FLIP: net hedge fund positioning (course formula)
    flip_s = pct_long_s - pct_short_s

    records = []
    for i in range(len(df)):
        try:
            nc_long_val  = nc_long_s.iloc[i]
            nc_short_val = nc_short_s.iloc[i]
            pct_l        = pct_long_s.iloc[i]
            pct_s        = pct_short_s.iloc[i]
            flip_val     = flip_s.iloc[i]
            oi_val       = oi_s.iloc[i]

            # nc_net_pct_oi: net as % of open interest (for normalisation)
            nc_net       = nc_long_val - nc_short_val
            nc_net_pct   = (nc_net / oi_val * 100) if oi_val and oi_val > 0 else 0.0

            records.append({
                "report_date":    report_dates.iloc[i],
                "currency":       df["_currency"].iloc[i],
                "nc_long":        int(nc_long_val)        if pd.notna(nc_long_val)  else 0,
                "nc_short":       int(nc_short_val)       if pd.notna(nc_short_val) else 0,
                "nc_net_prev":    0,   # filled during DB upsert via lag
                "nc_net_change":  int(nc_long_chg_s.iloc[i] - nc_short_chg_s.iloc[i])
                                  if pd.notna(nc_long_chg_s.iloc[i]) else 0,
                "nc_pct_long":    float(pct_l)   if pd.notna(pct_l)    else 0.0,
                "nc_pct_short":   float(pct_s)   if pd.notna(pct_s)    else 0.0,
                "flip":           float(flip_val) if pd.notna(flip_val) else 0.0,
                "open_interest":  int(oi_val)     if pd.notna(oi_val)   else 0,
                "nc_net_pct_oi":  round(nc_net_pct, 4),
            })
        except Exception as exc:
            logger.debug(f"Row parse error: {exc}")
            continue

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df
    # Keep only one row per (report_date, currency) — highest open interest
    # This deduplicates EUR which appears in multiple contracts (EURUSD, EURJPY etc.)
    result_df = (
        result_df
        .sort_values("open_interest", ascending=False)
        .drop_duplicates(subset=["report_date", "currency"])
        .sort_values(["currency", "report_date"])
        .reset_index(drop=True)
    )
    return result_df


def _detect_format(df: pd.DataFrame) -> bool:
    """TFF files always use Leveraged Money columns — always return True."""
    return True


# ── Downloader class ───────────────────────────────────────────────────────────

class COTDownloader:
    """
    CFTC Commitment of Traders report downloader for G10 currencies.

    Implements the course methodology:
        1. Download disaggregated futures report (f_disagg)
        2. Extract non-commercial / leveraged money positioning
        3. Calculate THE FLIP (net % positioning)
        4. Store to TimescaleDB cot_reports table
        5. Provide flip scores for Analyst 2 (Endo/Exo)

    Weekly schedule: CFTC releases every Friday ~15:30 EST (20:30 UTC)
    Data covers positions as of previous Tuesday.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        })

    # =========================================================================
    # Download methods
    # =========================================================================

    def fetch_latest(self) -> pd.DataFrame:
        """
        Download and store the most recent COT report (current year YTD).
        Run every Friday after 20:30 UTC.
        Uses TFF (Traders in Financial Futures) zip file.
        """
        year = datetime.now(UTC).year
        return self.fetch_year(year)

    def fetch_year(self, year: int) -> pd.DataFrame:
        """Download and parse one year of COT TFF data."""
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
        logger.info(f"COT | Downloading TFF {year} from {url}...")

        try:
            resp = self._session.get(url, timeout=120)
            resp.raise_for_status()
        except Exception as exc:
            logger.error(f"COT {year} download failed: {exc}")
            return pd.DataFrame()

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                txt_files = [f for f in z.namelist() if f.endswith(".txt")]
                if not txt_files:
                    logger.error(f"No .txt in zip for {year}")
                    return pd.DataFrame()
                with z.open(txt_files[0]) as f:
                    df_raw = pd.read_csv(f, low_memory=False)
        except Exception as exc:
            logger.error(f"COT {year} unzip failed: {exc}")
            return pd.DataFrame()

        logger.info(f"COT | {year} raw rows: {len(df_raw)}")
        df = _parse_cot_df(df_raw, is_2015_format=True)

        if not df.empty:
            self._store(df)
            logger.info(f"COT | {year}: {len(df)} records stored")
        else:
            logger.warning(f"COT | {year}: No G10 records found")

        return df

    def backfill(
        self,
        start_year: int = 2007,
        end_year:   Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Full historical backfill from start_year to present.

        CFTC data:
            2007–2014: fut_disagg_txt_2007.zip ... fut_disagg_txt_2014.zip
            2015–now:  fut_disagg_txt_2015.zip ... current year

        Args:
            start_year: First year (default 2007 — course recommendation)
            end_year:   Last year (default: current year)
        """
        if end_year is None:
            end_year = datetime.now(UTC).year

        logger.info(f"COT | Backfill {start_year}–{end_year}")
        all_dfs: List[pd.DataFrame] = []

        for year in range(start_year, end_year + 1):
            df = self.fetch_year(year)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        full = pd.concat(all_dfs, ignore_index=True).drop_duplicates(
            subset=["report_date", "currency"]
        )
        logger.success(f"COT | Backfill complete | {len(full)} total records")
        return full

    # =========================================================================
    # Flip score — course formula
    # =========================================================================

    def get_flip_score(
        self,
        currency: str,
        lookback_weeks: int = 52,
    ) -> Dict[str, Any]:
        """
        Get the current FLIP score for a currency.

        THE FLIP (course formula):
            flip = pct_nc_long - pct_nc_short

        Interpretation:
            flip > 0  = hedge funds net LONG = bullish conviction
            flip < 0  = hedge funds net SHORT = bearish conviction
            flip crossing zero = potential trend change signal

        Args:
            currency:       e.g. "AUD", "EUR", "JPY"
            lookback_weeks: history for trend analysis

        Returns:
            {
                "flip":           float,   # current net positioning
                "flip_prev":      float,   # previous week
                "flip_change":    float,   # week-on-week change
                "direction":      "LONG" / "SHORT" / "NEUTRAL",
                "crossover":      bool,    # flipped sign this week
                "extreme_long":   bool,    # >2 std above mean (contrarian warning)
                "extreme_short":  bool,    # >2 std below mean
                "percentile":     float,   # current flip percentile vs history
                "weeks_in_trend": int,     # consecutive weeks same direction
            }
        """
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text
            with db.connection() as conn:
                result = conn.execute(text("""
                    SELECT report_date, nc_long, nc_short, nc_net_pct_oi,
                           open_interest,
                           (nc_long::float / NULLIF(open_interest, 0) * 100) as pct_long,
                           (nc_short::float / NULLIF(open_interest, 0) * 100) as pct_short
                    FROM cot_reports
                    WHERE currency = :currency
                    ORDER BY report_date DESC
                    LIMIT :limit
                """), {"currency": currency, "limit": lookback_weeks})
                rows = result.fetchall()

            if not rows:
                return self._empty_flip()

            df = pd.DataFrame(rows, columns=result.keys())

        except Exception as exc:
            logger.error(f"get_flip_score DB error: {exc}")
            return self._empty_flip()

        if df.empty or len(df) < 2:
            return self._empty_flip()

        # Calculate flip from DB data
        df["flip"] = df["pct_long"] - df["pct_short"]
        df = df.sort_values("report_date")

        current  = df["flip"].iloc[-1]
        previous = df["flip"].iloc[-2]
        change   = current - previous

        # Percentile rank
        percentile = (df["flip"] < current).mean() * 100

        # Extremes (contrarian warning per course)
        mean = df["flip"].mean()
        std  = df["flip"].std()
        extreme_long  = current > mean + 2 * std
        extreme_short = current < mean - 2 * std

        # Crossover detection
        crossover = (current >= 0) != (previous >= 0)

        # Consecutive weeks in same direction
        direction_series = (df["flip"] >= 0)
        current_dir = direction_series.iloc[-1]
        weeks = 0
        for val in reversed(direction_series.values):
            if val == current_dir:
                weeks += 1
            else:
                break

        direction = "NEUTRAL"
        if current > 0.5:
            direction = "LONG"
        elif current < -0.5:
            direction = "SHORT"

        return {
            "currency":       currency,
            "flip":           round(float(current),  4),
            "flip_prev":      round(float(previous), 4),
            "flip_change":    round(float(change),   4),
            "direction":      direction,
            "crossover":      bool(crossover),
            "extreme_long":   bool(extreme_long),
            "extreme_short":  bool(extreme_short),
            "percentile":     round(float(percentile), 1),
            "weeks_in_trend": int(weeks),
        }

    def get_all_flips(self) -> Dict[str, Dict[str, Any]]:
        """Get flip scores for all 7 available COT currencies."""
        currencies = ["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
        return {c: self.get_flip_score(c) for c in currencies}

    def get_pair_cot_bias(self, symbol: str) -> Dict[str, Any]:
        """
        Derive COT bias for a currency pair from individual currency flips.

        For direct pairs (EURUSD): EUR flip vs USD flip
        For crosses (EURJPY): EUR flip vs JPY flip

        Returns:
            {
                "symbol":     "EURUSD",
                "bias":       "BULLISH" / "BEARISH" / "NEUTRAL",
                "score":      float,   # -1.0 to +1.0
                "base_flip":  float,
                "quote_flip": float,
            }
        """
        if len(symbol) != 6:
            return {"symbol": symbol, "bias": "NEUTRAL", "score": 0.0}

        base  = symbol[:3]
        quote = symbol[3:]

        base_flip  = self.get_flip_score(base).get("flip", 0.0)
        quote_flip = self.get_flip_score(quote).get("flip", 0.0)

        # For EURUSD: positive = EUR bullish vs USD
        # Base currency long = pair goes up
        # Quote currency long = pair goes down
        score = (base_flip - quote_flip) / 20.0  # normalise roughly to -1..+1
        score = max(-1.0, min(1.0, score))

        bias = "NEUTRAL"
        if score > 0.15:
            bias = "BULLISH"
        elif score < -0.15:
            bias = "BEARISH"

        return {
            "symbol":     symbol,
            "bias":       bias,
            "score":      round(score, 4),
            "base_flip":  base_flip,
            "quote_flip": quote_flip,
        }

    def _empty_flip(self) -> Dict[str, Any]:
        return {
            "flip": 0.0, "flip_prev": 0.0, "flip_change": 0.0,
            "direction": "NEUTRAL", "crossover": False,
            "extreme_long": False, "extreme_short": False,
            "percentile": 50.0, "weeks_in_trend": 0,
        }

    # =========================================================================
    # Storage
    # =========================================================================

    def _store(self, df: pd.DataFrame) -> int:
        """Store COT records to TimescaleDB cot_reports table."""
        if df.empty:
            return 0

        stored = 0
        try:
            from data_pipeline.storage.timeseries_db import db
            for _, row in df.iterrows():
                try:
                    db.insert_cot_report({
                        "report_date":   row["report_date"].to_pydatetime()
                                         if hasattr(row["report_date"], "to_pydatetime")
                                         else row["report_date"],
                        "currency":      row["currency"],
                        "nc_long":       int(row["nc_long"]),
                        "nc_short":      int(row["nc_short"]),
                        "nc_net_prev":   int(row.get("nc_net_prev", 0)),
                        "nc_net_change": int(row.get("nc_net_change", 0)),
                        "comm_long":     0,   # not extracted (course: not useful)
                        "comm_short":    0,
                        "open_interest": int(row["open_interest"]),
                        "nc_net_pct_oi": float(row["nc_net_pct_oi"]),
                    })
                    stored += 1
                except Exception as exc:
                    logger.debug(f"COT row store failed: {exc}")
        except Exception as exc:
            logger.error(f"COT DB store failed: {exc}")

        logger.debug(f"COT | Stored {stored}/{len(df)} records")
        return stored

    # =========================================================================
    # Weekly scheduler
    # =========================================================================

    def start_weekly_scheduler(self) -> None:
        """
        Background thread that auto-downloads COT every Friday at 21:00 UTC
        (CFTC releases ~20:30 UTC = 15:30 EST).
        """
        import threading

        def _scheduler():
            while True:
                now = datetime.now(UTC)
                # Next Friday 21:00 UTC
                days_until_friday = (4 - now.weekday()) % 7
                if days_until_friday == 0 and now.hour >= 21:
                    days_until_friday = 7
                next_run = now.replace(
                    hour=21, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_until_friday)
                wait_s = (next_run - now).total_seconds()
                logger.info(
                    f"COT scheduler | next run in {wait_s/3600:.1f}h "
                    f"({next_run.strftime('%Y-%m-%d %H:%M UTC')})"
                )
                time_module.sleep(wait_s)
                try:
                    logger.info("COT scheduler | fetching latest...")
                    self.fetch_latest()
                except Exception as exc:
                    logger.error(f"COT scheduler error: {exc}")

        import time as time_module
        t = threading.Thread(
            target=_scheduler, name="cot-scheduler", daemon=True
        )
        t.start()
        logger.info("COT weekly scheduler started ✅")


# ── Module singleton ───────────────────────────────────────────────────────────
cot_downloader = COTDownloader()