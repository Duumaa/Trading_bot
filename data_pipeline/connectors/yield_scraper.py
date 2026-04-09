"""
data_pipeline/connectors/yield_scraper.py
=========================================
Bond Yield Scraper — G10 Interest Rate Differentials
Sources: FRED (USD + secondary) + ECB (EUR) + BoE (GBP) + BoJ (JPY)

Why this matters (from course):
    "When the interest rate differential between Australia and the US widens,
     we get hot money flows into Australia — inflationary for AUD/USD."
    "It's not just about the absolute level — it's also the DIRECTION
     the interest rate differential is moving in."
    Yield spreads are THE primary driver of FX in Analyst 3 (TFT model).

Source strategy:
    USD  -> FRED DGS2/DGS10         (daily, same-day, gold standard)
    EUR  -> ECB Data Portal API      (daily, same-day, official, no key needed)
    GBP  -> Bank of England API      (daily, same-day, official, no key needed)
    JPY  -> Ministry of Finance CSV  (daily, same-day, official, no key needed)
    CHF/CAD/AUD/NZD/SEK/NOK
         -> FRED OECD series         (monthly ~4w lag, direction > precision)

ECB API:
    Base: https://data-api.ecb.europa.eu/service/data/YC/
    2Y:   B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y
    10Y:  B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y

BoE API:
    Base: https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp
    2Y:   IUDBEDR   10Y: IUDMNZC

BoJ (MoF):
    https://www.mof.go.jp/english/jgbs/reference/interest_rate/jgbcme.csv

Usage:
    from data_pipeline.connectors.yield_scraper import yield_scraper

    yields = yield_scraper.fetch_latest()
    diff   = yield_scraper.get_yield_differential("EURUSD", tenor="10Y")
    slope  = yield_scraper.get_curve_slope("EUR")   # now works with ECB 2Y data
    ranks  = yield_scraper.get_carry_ranking()
    yield_scraper.start_scheduler()
"""

from __future__ import annotations

import io
import os
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

UTC = timezone.utc

# ── FRED config ────────────────────────────────────────────────────────────────

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_KEY  = os.getenv("FRED_API_KEY", "")

FRED_SERIES: List[Tuple[str, str, str]] = [
    # USD daily (gold standard)
    ("USD", "2Y",  "DGS2"),
    ("USD", "10Y", "DGS10"),
    ("USD", "3M",  "DGS3MO"),
    # USD TIPS — real yields directly (no CPI adjustment needed)
    # FRED has no 2Y TIPS; DFII5 (5Y) is shortest available
    ("USD", "REAL_5Y",  "DFII5"),
    ("USD", "REAL_10Y", "DFII10"),
    # Secondary G10 monthly OECD (EUR/GBP/JPY overridden by CB sources below)
    ("EUR", "10Y", "IRLTLT01DEM156N"),
    ("GBP", "10Y", "IRLTLT01GBM156N"),
    ("JPY", "10Y", "IRLTLT01JPM156N"),
    ("CHF", "10Y", "IRLTLT01CHM156N"),
    ("CAD", "10Y", "IRLTLT01CAM156N"),
    ("AUD", "10Y", "IRLTLT01AUM156N"),
    ("NZD", "10Y", "IRLTLT01NZM156N"),
    ("SEK", "10Y", "IRLTLT01SEM156N"),
    ("NOK", "10Y", "IRLTLT01NOM156N"),
]

# ── CPI / inflation series for real rate computation ───────────────────────────
# Monthly index levels (2015=100) — YoY% computed on fetch.
# USD uses TIPS directly so no CPI series needed.
FRED_CPI_SERIES: List[Tuple[str, str]] = [
    ("EUR", "CP0000EZ19M086NEST"),   # EA HICP monthly index
    ("GBP", "GBRCPIALLMINMEI"),      # UK CPI monthly
    ("JPY", "JPNCPIALLMINMEI"),      # Japan CPI monthly
    ("CHF", "CHECPIALLMINMEI"),      # Switzerland CPI monthly
    ("CAD", "CANCPIALLMINMEI"),      # Canada CPI monthly
    ("SEK", "SWECPIALLMINMEI"),      # Sweden CPI monthly
    ("NOK", "NORCPIALLMINMEI"),      # Norway CPI monthly
    # AUD/NZD quarterly series not available on FRED — skipped
]

# ── ECB config ─────────────────────────────────────────────────────────────────

ECB_BASE   = "https://data-api.ecb.europa.eu/service/data/YC"
ECB_SERIES = {
    "EUR_2Y":  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
    "EUR_10Y": "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
}

# ── BoE config ─────────────────────────────────────────────────────────────────

BOE_BASE   = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
BOE_SERIES = {
    "GBP_2Y":  "IUDBEDR",
    "GBP_10Y": "IUDMNZC",
}

# ── BoJ / MoF config ──────────────────────────────────────────────────────────

BOJ_URL  = "https://www.mof.go.jp/english/jgbs/reference/interest_rate/jgbcme.csv"
BOJ_COLS = {
    "2y":  "JPY_2Y",
    "10y": "JPY_10Y",
}

# ── G10 pairs ──────────────────────────────────────────────────────────────────

G10_PAIRS: Dict[str, Tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"),
    "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"),
    "GBPJPY": ("GBP", "JPY"),
    "AUDJPY": ("AUD", "JPY"),
    "NZDJPY": ("NZD", "JPY"),
}

DIFF_THRESHOLDS = {
    "strong":   2.0,
    "moderate": 1.0,
    "weak":     0.3,
    "extreme":  4.0,
}


# ── Main scraper ───────────────────────────────────────────────────────────────

class YieldScraper:
    """
    Multi-source bond yield scraper for G10 interest rate differentials.

    Priority per currency:
        USD -> FRED daily
        EUR -> ECB API daily   (falls back to FRED monthly if ECB fails)
        GBP -> BoE API daily   (falls back to FRED monthly if BoE fails)
        JPY -> MoF CSV daily   (falls back to FRED monthly if MoF fails)
        Others -> FRED OECD monthly
    """

    FETCH_INTERVAL_S = 86400

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "G10Bot/1.0"})
        self._cache: Dict[str, pd.Series] = {}
        self._last_fetch: Optional[datetime] = None
        if not FRED_KEY:
            logger.warning("FRED_API_KEY not set")

    # =========================================================================
    # FRED
    # =========================================================================

    def _fetch_fred(self, series_id: str, start: str, end: str) -> pd.Series:
        params = {
            "series_id": series_id, "api_key": FRED_KEY,
            "file_type": "json",
            "observation_start": start, "observation_end": end,
        }
        try:
            resp = self._session.get(FRED_BASE, params=params, timeout=30)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
        except Exception as exc:
            logger.error(f"FRED [{series_id}]: {exc}")
            return pd.Series(dtype=float)

        records = {}
        for o in obs:
            v = o.get("value", ".")
            if v != ".":
                try:
                    records[o["date"]] = float(v)
                except (ValueError, KeyError):
                    pass

        if not records:
            return pd.Series(dtype=float)

        s = pd.Series(records)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="D")).ffill()
        logger.debug(f"FRED [{series_id}]: {len(s)} daily obs")
        return s

    # =========================================================================
    # ECB
    # =========================================================================

    def _fetch_ecb(self, key: str, series_key: str, start: str) -> pd.Series:
        """ECB yield curve data — no API key needed."""
        url = f"{ECB_BASE}/{series_key}"
        try:
            resp = self._session.get(
                url, params={"startPeriod": start, "format": "csvdata"}, timeout=30
            )
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
            df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
            s = df.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float).sort_index()
            s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="D")).ffill()
            logger.debug(f"ECB [{key}]: {len(s)} daily obs")
            return s
        except Exception as exc:
            logger.error(f"ECB [{key}]: {exc}")
            return pd.Series(dtype=float)

    # =========================================================================
    # BoE
    # =========================================================================

    def _fetch_boe(self, key: str, series_id: str, start: str) -> pd.Series:
        """Bank of England gilt yields — no API key needed."""
        try:
            start_dt  = datetime.strptime(start, "%Y-%m-%d")
            start_boe = start_dt.strftime("%d/%b/%Y")
            end_boe   = datetime.now(UTC).strftime("%d/%b/%Y")
        except Exception:
            start_boe = "01/Jan/2020"
            end_boe   = datetime.now(UTC).strftime("%d/%b/%Y")

        # BoE changed API in 2025 — new CSV endpoint requires csv.x=yes param
        # Old endpoint returns HTML page, new one returns application/csv directly
        BOE_CSV_URL = (
            "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp"
            f"?csv.x=yes&Datefrom={start_boe}&Dateto={end_boe}"
            f"&SeriesCodes={series_id}&CSVF=TN&UsingCodes=Y"
        )
        try:
            resp = self._session.get(
                BOE_CSV_URL,
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/csv,text/plain,*/*",
                    "Referer": "https://www.bankofengland.co.uk/",
                },
            )
            resp.raise_for_status()

            # Response format: "08 Jan 2024,5.25UDBEDR" per line
            # Strip the series code suffix that appears on first line
            lines = resp.text.strip().splitlines()
            clean_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Strip series code suffix if present (e.g. "5.25IUDBEDR")
                parts = line.split(",")
                if len(parts) >= 2:
                    val = parts[1]
                    # Remove any trailing non-numeric chars (series code)
                    import re
                    val = re.sub(r"[A-Za-z]+$", "", val).strip()
                    clean_lines.append(f"{parts[0]},{val}")

            clean_csv = "\n".join(clean_lines)
            df = pd.read_csv(
                io.StringIO(clean_csv),
                header=None,
                names=["date", "value"],
            )
            df["date"]  = pd.to_datetime(df["date"], format="%d %b %Y",
                                          errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().set_index("date").sort_index()

            s = df["value"]
            s = s.reindex(
                pd.date_range(s.index[0], s.index[-1], freq="D")
            ).ffill()

            logger.debug(f"BoE [{key}]: {len(s)} daily obs")
            return s

        except Exception as exc:
            logger.error(f"BoE [{key}]: {exc}")
            return pd.Series(dtype=float)

    # =========================================================================
    # BoJ / MoF
    # =========================================================================

    def _fetch_boj(self, start: str) -> Dict[str, pd.Series]:
        """Japanese Government Bond yields from Ministry of Finance CSV."""
        try:
            resp = self._session.get(BOJ_URL, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), skiprows=1, header=0)
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col)
            df = df[df.index >= pd.to_datetime(start)]

            cols_lower = {c.lower().replace(" ", ""): c for c in df.columns}
            results: Dict[str, pd.Series] = {}

            for search_key, out_key in BOJ_COLS.items():
                matched = next(
                    (orig for lwr, orig in cols_lower.items() if search_key in lwr), None
                )
                if matched:
                    s = pd.to_numeric(df[matched], errors="coerce").dropna().sort_index()
                    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="D")).ffill()
                    results[out_key] = s
                    logger.debug(f"BoJ [{out_key}]: {len(s)} daily obs")
                else:
                    logger.warning(
                        f"BoJ: col '{search_key}' not found. "
                        f"Available: {list(cols_lower.keys())[:8]}"
                    )
            return results

        except Exception as exc:
            logger.error(f"BoJ fetch failed: {exc}")
            return {}

    # =========================================================================
    # CPI / inflation series
    # =========================================================================

    def _fetch_cpi_series(self, lookback_years: int = 5) -> Dict[str, pd.Series]:
        """
        Fetch G10 CPI index levels from FRED, compute YoY% inflation rate.
        Returns dict keyed {CCY}_CPI with annualised YoY inflation %.
        USD excluded — real yields fetched directly via TIPS (DFII2/DFII10).
        """
        # Always use 6-year lookback — monthly CPI needs 13+ months after shift(12)
        cpi_start = (datetime.now(UTC) - timedelta(days=6 * 365)).strftime("%Y-%m-%d")
        end       = datetime.now(UTC).strftime("%Y-%m-%d")
        result: Dict[str, pd.Series] = {}

        for ccy, sid in FRED_CPI_SERIES:
            raw = self._fetch_fred(sid, cpi_start, end)
            if raw.empty:
                logger.warning(f"CPI | {ccy} ({sid}): no data")
                continue
            try:
                # YoY% = (index_now / index_12m_ago - 1) * 100
                # Monthly series: resample to month-end then shift 12 months
                monthly = raw.resample("ME").last().dropna()
                if len(monthly) < 14:
                    logger.warning(f"CPI | {ccy}: insufficient history ({len(monthly)} months)")
                    continue
                yoy = ((monthly / monthly.shift(12)) - 1.0) * 100.0
                yoy = yoy.dropna()
                # Forward-fill to daily granularity to align with yield_data
                daily = yoy.reindex(
                    pd.date_range(yoy.index[0], end, freq="D")
                ).ffill()
                result[f"{ccy}_CPI"] = daily
                logger.debug(
                    f"CPI | {ccy}: {len(daily)} daily obs, "
                    f"latest={round(float(daily.iloc[-1]), 2)}%"
                )
            except Exception as exc:
                logger.warning(f"CPI | {ccy} compute failed: {exc}")
            time.sleep(0.2)

        return result

    # =========================================================================
    # Orchestrated fetch
    # =========================================================================

    def _fetch_all_series(self, lookback_years: int = 5) -> Dict[str, pd.Series]:
        """
        Fetch all G10 yields using best source per currency.
        Central bank APIs override FRED monthly series where available.
        Also fetches CPI for real rate computation.
        """
        start = (datetime.now(UTC) - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
        end   = datetime.now(UTC).strftime("%Y-%m-%d")
        results: Dict[str, pd.Series] = {}

        # 1. FRED — USD daily + secondary monthly + USD TIPS real yields
        logger.info("Yields | [1/5] FRED (nominal + TIPS)...")
        for ccy, tenor, sid in FRED_SERIES:
            s = self._fetch_fred(sid, start, end)
            if not s.empty:
                results[f"{ccy}_{tenor}"] = s
            time.sleep(0.2)

        # 2. ECB — EUR daily (overrides FRED monthly)
        logger.info("Yields | [2/5] ECB (EUR same-day)...")
        for key, series_key in ECB_SERIES.items():
            s = self._fetch_ecb(key, series_key, start)
            if not s.empty:
                results[key] = s
                logger.info(f"  ECB {key}: {len(s)} obs ✅")
            else:
                logger.warning(f"  ECB {key} failed — using FRED fallback")
            time.sleep(0.3)

        # 3. BoE — GBP daily (overrides FRED monthly)
        logger.info("Yields | [3/5] BoE (GBP same-day)...")
        for key, sid in BOE_SERIES.items():
            s = self._fetch_boe(key, sid, start)
            if not s.empty:
                results[key] = s
                logger.info(f"  BoE {key}: {len(s)} obs ✅")
            else:
                logger.warning(f"  BoE {key} failed — using FRED fallback")
            time.sleep(0.3)

        # 4. BoJ — JPY daily (overrides FRED monthly)
        logger.info("Yields | [4/5] BoJ/MoF (JPY same-day)...")
        for key, s in self._fetch_boj(start).items():
            if not s.empty:
                results[key] = s
                logger.info(f"  BoJ {key}: {len(s)} obs ✅")

        # 5. CPI — G10 inflation for real rate computation
        logger.info("Yields | [5/5] FRED CPI (real rate base)...")
        cpi_series = self._fetch_cpi_series(lookback_years)
        results.update(cpi_series)
        logger.info(f"  CPI: {len(cpi_series)} currencies ✅")

        return results

    # =========================================================================
    # Fetch + store
    # =========================================================================

    def fetch_latest(self, lookback_years: int = 5) -> Dict[str, Any]:
        """Fetch all G10 yields, store to DB. Returns summary of latest values."""
        logger.info("Yields | Starting multi-source fetch...")
        series_map = self._fetch_all_series(lookback_years)

        if not series_map:
            logger.error("Yields | No data fetched")
            return {}

        rows   = self._build_rows(series_map)
        stored = self._store(rows)

        self._cache      = series_map
        self._last_fetch = datetime.now(UTC)

        summary = {
            k: round(float(v.iloc[-1]), 4)
            for k, v in series_map.items()
            if not v.empty
        }

        logger.success(
            f"Yields | {stored} rows | {len(series_map)} series | "
            f"USD10Y={summary.get('USD_10Y')}% "
            f"EUR10Y={summary.get('EUR_10Y')}% "
            f"EUR2Y={summary.get('EUR_2Y')}% "
            f"GBP10Y={summary.get('GBP_10Y')}% "
            f"JPY10Y={summary.get('JPY_10Y')}%"
        )
        return summary

    def _build_rows(self, series_map: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """
        Build flat DB rows + compute derived series:
          SLOPE     = 10Y - 2Y  (yield curve)
          REAL_10Y  = 10Y - CPI (non-USD only; USD uses TIPS directly)
          REAL_2Y   = 2Y  - CPI (non-USD, where 2Y data exists)
        """
        rows: List[Dict[str, Any]] = []

        for key, series in series_map.items():
            ccy, tenor = key.split("_", 1)
            for date, value in series.items():
                if pd.isna(value):
                    continue
                rows.append({
                    "date":      date.strftime("%Y-%m-%d"),
                    "currency":  ccy,
                    "tenor":     tenor,
                    "yield_pct": float(value),
                })

        all_ccys = set(k.split("_")[0] for k in series_map)

        for ccy in all_ccys:
            s10 = series_map.get(f"{ccy}_10Y")
            s2  = series_map.get(f"{ccy}_2Y")

            # SLOPE: 10Y - 2Y
            if s10 is not None and s2 is not None:
                common = s10.index.intersection(s2.index)
                slope  = s10.loc[common] - s2.loc[common]
                for date, value in slope.items():
                    if pd.isna(value):
                        continue
                    rows.append({
                        "date":      date.strftime("%Y-%m-%d"),
                        "currency":  ccy,
                        "tenor":     "SLOPE",
                        "yield_pct": float(value),
                    })
                logger.debug(f"Slope {ccy}: {len(slope)} rows")

            # REAL_10Y / REAL_2Y for non-USD: nominal - CPI inflation
            # USD already has TIPS real yields stored directly (DFII5/DFII10)
            if ccy == "USD":
                continue

            cpi = series_map.get(f"{ccy}_CPI")
            if cpi is None:
                continue

            if s10 is not None:
                common10 = s10.index.intersection(cpi.index)
                real10   = s10.loc[common10] - cpi.loc[common10]
                for date, value in real10.items():
                    if pd.isna(value):
                        continue
                    rows.append({
                        "date":      date.strftime("%Y-%m-%d"),
                        "currency":  ccy,
                        "tenor":     "REAL_10Y",
                        "yield_pct": float(value),
                    })
                logger.debug(f"Real10Y {ccy}: {len(real10)} rows")

            if s2 is not None:
                common2 = s2.index.intersection(cpi.index)
                real2   = s2.loc[common2] - cpi.loc[common2]
                for date, value in real2.items():
                    if pd.isna(value):
                        continue
                    rows.append({
                        "date":      date.strftime("%Y-%m-%d"),
                        "currency":  ccy,
                        "tenor":     "REAL_2Y",
                        "yield_pct": float(value),
                    })
                logger.debug(f"Real2Y {ccy}: {len(real2)} rows")

        return rows

    def _store(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert to yield_data table."""
        if not rows:
            return 0
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text
            stored = 0
            with db.connection() as conn:
                for i in range(0, len(rows), 1000):
                    chunk = rows[i: i + 1000]
                    conn.execute(
                        text("""
                            INSERT INTO yield_data (date, currency, tenor, yield_pct)
                            VALUES (:date, :currency, :tenor, :yield_pct)
                            ON CONFLICT (date, currency, tenor)
                            DO UPDATE SET yield_pct = EXCLUDED.yield_pct
                        """),
                        chunk,
                    )
                    stored += len(chunk)
            logger.debug(f"Yields DB | {stored} rows upserted")
            return stored
        except Exception as exc:
            logger.error(f"Yields DB store failed: {exc}")
            return 0

    # =========================================================================
    # Analysis
    # =========================================================================

    def get_latest_yields(self) -> Dict[str, float]:
        """Most recent yield per (currency, tenor) from DB."""
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text
            with db.connection() as conn:
                result = conn.execute(text("""
                    SELECT DISTINCT ON (currency, tenor)
                        currency, tenor, yield_pct
                    FROM yield_data
                    ORDER BY currency, tenor, date DESC
                """))
                return {f"{r[0]}_{r[1]}": round(r[2], 4) for r in result.fetchall()}
        except Exception as exc:
            logger.error(f"get_latest_yields: {exc}")
            return {}

    def get_yield_differential(self, pair: str, tenor: str = "10Y") -> Dict[str, Any]:
        """
        Yield differential for a G10 pair.
        differential = base_yield - quote_yield
        positive -> base favored, pair should appreciate.
        Score: -1.0 to +1.0. Extreme (>4%) -> lower conviction (already priced in).
        """
        pair_upper = pair.upper()
        if pair_upper not in G10_PAIRS:
            return self._empty_differential()

        base_ccy, quote_ccy = G10_PAIRS[pair_upper]
        yields  = self.get_latest_yields()
        base_y  = yields.get(f"{base_ccy}_{tenor}")
        quote_y = yields.get(f"{quote_ccy}_{tenor}")

        if base_y is None or quote_y is None:
            logger.warning(f"Missing: {base_ccy}_{tenor}={base_y} {quote_ccy}_{tenor}={quote_y}")
            return self._empty_differential()

        diff     = round(base_y - quote_y, 4)
        abs_diff = abs(diff)
        sign     = 1 if diff > 0 else -1

        if abs_diff < 0.05:
            direction, score, conviction = "NEUTRAL", 0.0, "NEUTRAL"
        elif abs_diff >= DIFF_THRESHOLDS["extreme"]:
            direction  = "BASE_FAVORED" if diff > 0 else "QUOTE_FAVORED"
            score, conviction = sign * 0.6, "EXTREME"
        elif abs_diff >= DIFF_THRESHOLDS["strong"]:
            direction  = "BASE_FAVORED" if diff > 0 else "QUOTE_FAVORED"
            score, conviction = sign * 0.9, "STRONG"
        elif abs_diff >= DIFF_THRESHOLDS["moderate"]:
            direction  = "BASE_FAVORED" if diff > 0 else "QUOTE_FAVORED"
            score, conviction = sign * 0.6, "MODERATE"
        else:
            direction  = "BASE_FAVORED" if diff > 0 else "QUOTE_FAVORED"
            score, conviction = sign * 0.3, "WEAK"

        chg_4w, chg_13w, trend = self._get_differential_trend(base_ccy, quote_ccy, tenor)

        return {
            "pair": pair_upper, "base_ccy": base_ccy, "quote_ccy": quote_ccy,
            "tenor": tenor, "base_yield": base_y, "quote_yield": quote_y,
            "differential": diff, "direction": direction,
            "score": round(score, 3), "conviction": conviction,
            "4w_change": chg_4w, "13w_change": chg_13w, "trend": trend,
        }

    def get_curve_slope(self, currency: str) -> Dict[str, Any]:
        """
        Yield curve slope (10Y - 2Y).
        Now works for EUR (ECB 2Y), GBP (BoE 2Y), USD (FRED 2Y).
        Inversion = recession / risk-off signal.
        """
        yields  = self.get_latest_yields()
        ten_y   = yields.get(f"{currency}_10Y")
        two_y   = yields.get(f"{currency}_2Y")
        slope_v = yields.get(f"{currency}_SLOPE")

        if slope_v is not None:
            slope = round(slope_v, 4)
        elif ten_y is not None and two_y is not None:
            slope = round(ten_y - two_y, 4)
        else:
            return {
                "currency": currency, "slope": None,
                "inverted": None, "signal": "UNKNOWN",
                "note": f"Missing 10Y={ten_y} 2Y={two_y}",
            }

        inverted = slope < 0
        signal   = "RISK_OFF" if inverted else ("NEUTRAL" if slope < 0.5 else "RISK_ON")

        return {
            "currency": currency, "slope": slope,
            "10Y": ten_y, "2Y": two_y,
            "inverted": inverted, "signal": signal,
        }

    def get_all_differentials(self, tenor: str = "10Y") -> Dict[str, Dict]:
        return {pair: self.get_yield_differential(pair, tenor) for pair in G10_PAIRS}

    def get_carry_ranking(self) -> List[Dict[str, Any]]:
        """Rank G10 by 10Y yield. High yield = carry attractiveness."""
        yields  = self.get_latest_yields()
        ranking = []
        for ccy in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]:
            y10 = yields.get(f"{ccy}_10Y")
            y2  = yields.get(f"{ccy}_2Y")
            if y10 is not None:
                ranking.append({
                    "currency": ccy, "10Y": y10, "2Y": y2,
                    "slope": round(y10 - y2, 4) if y2 else None,
                })
        ranking.sort(key=lambda x: x["10Y"] or 0, reverse=True)
        for i, item in enumerate(ranking):
            item["rank"] = i + 1
        return ranking

    def _get_differential_trend(
        self, base_ccy: str, quote_ccy: str, tenor: str
    ) -> Tuple[float, float, str]:
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text
            with db.connection() as conn:
                result = conn.execute(text("""
                    WITH base AS (
                        SELECT date, yield_pct FROM yield_data
                        WHERE currency=:base AND tenor=:tenor ORDER BY date DESC LIMIT 100
                    ), quote AS (
                        SELECT date, yield_pct FROM yield_data
                        WHERE currency=:quote AND tenor=:tenor ORDER BY date DESC LIMIT 100
                    ), diff AS (
                        SELECT b.date, b.yield_pct - q.yield_pct AS d
                        FROM base b JOIN quote q ON b.date=q.date ORDER BY b.date DESC
                    )
                    SELECT
                        MAX(CASE WHEN rn=1  THEN d END),
                        MAX(CASE WHEN rn=28 THEN d END),
                        MAX(CASE WHEN rn=91 THEN d END)
                    FROM (SELECT d, ROW_NUMBER() OVER (ORDER BY date DESC) rn FROM diff) r
                """), {"base": base_ccy, "quote": quote_ccy, "tenor": tenor})
                row = result.fetchone()
                if not row or row[0] is None:
                    return (0.0, 0.0, "STABLE")
                current  = float(row[0])
                diff_4w  = float(row[1]) if row[1] else current
                diff_13w = float(row[2]) if row[2] else current
                chg_4w   = round(current - diff_4w,  4)
                chg_13w  = round(current - diff_13w, 4)
                trend    = "WIDENING" if chg_4w > 0.1 else ("NARROWING" if chg_4w < -0.1 else "STABLE")
                return (chg_4w, chg_13w, trend)
        except Exception as exc:
            logger.debug(f"Trend calc: {exc}")
            return (0.0, 0.0, "STABLE")

    def _empty_differential(self) -> Dict[str, Any]:
        return {
            "pair": None, "base_ccy": None, "quote_ccy": None,
            "tenor": None, "base_yield": None, "quote_yield": None,
            "differential": 0.0, "direction": "NEUTRAL",
            "score": 0.0, "conviction": "NEUTRAL",
            "4w_change": 0.0, "13w_change": 0.0, "trend": "STABLE",
        }

    # =========================================================================
    # Scheduler
    # =========================================================================

    def start_scheduler(self) -> None:
        """Fetch daily at 18:00 UTC — after US bond market close + FRED update."""
        def _run():
            logger.info("Yield scheduler | daily at 18:00 UTC")
            while True:
                now    = datetime.now(UTC)
                target = now.replace(hour=18, minute=0, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                time.sleep((target - now).total_seconds())
                try:
                    self.fetch_latest()
                except Exception as exc:
                    logger.error(f"Yield scheduler error: {exc}")

        threading.Thread(target=_run, name="yield-scheduler", daemon=True).start()
        logger.info("Yield scheduler started")


# ── Module singleton ───────────────────────────────────────────────────────────
yield_scraper = YieldScraper()
