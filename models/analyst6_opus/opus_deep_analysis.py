"""
models/analyst6_opus/opus_deep_analysis.py
==========================================
Analyst 6: Claude Opus Weekly Deep Analysis

Uses Claude claude-sonnet-4-20250514 (via Anthropic API) to produce a structured
weekly macro analysis across three perspectives:

  Hawk    — Central bank policy divergence & rate trajectory
  Dove    — Growth risks, recession signals, easing pressure
  Strategist — Net FX bias per currency after weighing both sides

This is the highest-level fundamental signal in the system.
Runs ONCE per week (Sunday evening UTC) — computationally expensive
but provides the macro framework that guides all 12 pairs for the week.

Architecture position:
  SENTIMENT GROUP — Analyst 6 of 6.
  Weekly context sets the macro REGIME BIAS for Analyst 2 weighting.
  Does NOT override Analyst 1 regime or Analyst 2 score.
  Acts as a prior: strong Opus conviction → raise Analyst 2 weight.
  Weak/conflicted Opus → reduce Analyst 2 confidence threshold.

Output stored in Redis:
  features:opus:weekly          Full analysis JSON     TTL=7 days
  features:opus:bias:{CCY}      Per-currency bias      TTL=7 days
  features:opus:pair:{PAIR}     Per-pair conviction    TTL=7 days

Output structure:
  {
    "week_of":       "2026-03-09",
    "hawk_analysis": "Central banks remain cautious...",
    "dove_analysis": "Growth risks in EUR area...",
    "strategist":    "Net bias: short EUR, long USD...",
    "currency_biases": {
      "USD": {"bias": "BULLISH", "score": 0.6, "rationale": "..."},
      "EUR": {"bias": "BEARISH", "score": -0.4, "rationale": "..."},
      ...
    },
    "pair_convictions": {
      "EURUSD": {"bias": "SHORT_BASE", "conviction": 0.65, "rationale": "..."},
      ...
    },
    "top_trades": ["SHORT EURUSD", "LONG AUDJPY", "LONG GBPJPY"],
    "key_risks":  ["Strait of Hormuz escalation", "Fed pivot surprise"],
    "generated_at": "2026-03-09T20:00:00Z"
  }

Usage:
  from models.analyst6_opus.opus_deep_analysis import opus_analyst
  analysis = opus_analyst.run_weekly()          # full analysis
  bias = opus_analyst.get_currency_bias("USD")  # {"bias": "BULLISH", "score": 0.6}
  conv = opus_analyst.get_pair_conviction("EURUSD")
  opus_analyst.start_scheduler()                # Sunday 20:00 UTC weekly
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger
from sqlalchemy import text

ROOT = Path("/workspaces/Trading_bot")
sys.path.insert(0, str(ROOT))

from data_pipeline.storage.timeseries_db import db  # noqa: E402

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

PAIR_CURRENCIES: Dict[str, Tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"), "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
    "AUDJPY": ("AUD", "JPY"), "NZDJPY": ("NZD", "JPY"),
}

ALL_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPUS_MODEL        = "claude-sonnet-4-20250514"
OPUS_MAX_TOKENS   = 4000
OPUS_TIMEOUT      = 120    # seconds — Opus is slow but thorough

CACHE_TTL_WEEKLY  = 604800  # 7 days
SCHEDULER_DAY     = 6       # Sunday (0=Mon, 6=Sun)
SCHEDULER_HOUR    = 20      # 20:00 UTC

# ══════════════════════════════════════════════════════════════
# REDIS
# ══════════════════════════════════════════════════════════════

def _get_redis():
    import redis as redis_lib
    from dotenv import load_dotenv
    load_dotenv(str(ROOT / ".env"))
    return redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )


# ══════════════════════════════════════════════════════════════
# DATA COLLECTORS
# ══════════════════════════════════════════════════════════════

def _collect_yield_data() -> str:
    """Collect latest bond yields for all G10 currencies."""
    try:
        with db.connection() as conn:
            rows = conn.execute(text("""
                SELECT y.currency, y.tenor, y.yield_pct, y.date
                FROM yield_data y
                INNER JOIN (
                    SELECT currency, tenor, MAX(date) as max_date
                    FROM yield_data
                    WHERE tenor IN ('2Y', '10Y')
                    GROUP BY currency, tenor
                ) latest ON y.currency = latest.currency
                    AND y.tenor = latest.tenor
                    AND y.date = latest.max_date
                ORDER BY y.currency, y.tenor
            """)).fetchall()

        if not rows:
            return "Bond yield data unavailable."

        lines = ["BOND YIELDS (latest):"]
        by_ccy: Dict[str, Dict[str, float]] = {}
        for row in rows:
            ccy, tenor, yld, date = row
            if ccy not in by_ccy:
                by_ccy[ccy] = {}
            by_ccy[ccy][tenor] = yld

        # Carry ranking
        carry = sorted(
            [(c, v.get("10Y", 0)) for c, v in by_ccy.items()],
            key=lambda x: -x[1]
        )

        for ccy, tenors in sorted(by_ccy.items()):
            y2  = tenors.get("2Y",  0)
            y10 = tenors.get("10Y", 0)
            slope = y10 - y2
            lines.append(
                f"  {ccy}: 2Y={y2:.3f}% 10Y={y10:.3f}% slope={slope:+.3f}%"
            )

        lines.append("\nCarry ranking (10Y): " +
                     " > ".join(f"{c}({y:.2f}%)" for c, y in carry))

        # Key spreads
        usd10 = by_ccy.get("USD", {}).get("10Y", 0)
        lines.append("\nKey spreads vs USD 10Y:")
        for ccy, tenors in sorted(by_ccy.items()):
            if ccy != "USD":
                spread = usd10 - tenors.get("10Y", 0)
                lines.append(f"  USD-{ccy}: {spread:+.3f}%")

        return "\n".join(lines)
    except Exception as exc:
        return f"Yield data error: {exc}"


def _collect_cot_data() -> str:
    """Collect latest COT positioning for G10 currencies."""
    try:
        with db.connection() as conn:
            rows = conn.execute(text("""
                SELECT c.currency, c.report_date, c.nc_net_pct_oi,
                       c.nc_net_change, c.nc_net, c.open_interest
                FROM cot_reports c
                INNER JOIN (
                    SELECT currency, MAX(report_date) as max_date
                    FROM cot_reports
                    GROUP BY currency
                ) latest ON c.currency = latest.currency
                    AND c.report_date = latest.max_date
                ORDER BY c.currency
            """)).fetchall()

            # Also get prior week for trend
            prior = conn.execute(text("""
                SELECT c.currency, c.nc_net_pct_oi
                FROM cot_reports c
                INNER JOIN (
                    SELECT currency, MAX(report_date) as max_date
                    FROM cot_reports
                    WHERE report_date < (
                        SELECT MAX(report_date) FROM cot_reports
                    )
                    GROUP BY currency
                ) latest ON c.currency = latest.currency
                    AND c.report_date = latest.max_date
            """)).fetchall()

        if not rows:
            return "COT data unavailable."

        prior_dict = {r[0]: r[1] for r in prior}

        lines = ["COT POSITIONING (Leveraged Money, latest week):"]
        for row in rows:
            ccy, date, pct_oi, chg, net, oi = row
            prior_pct = prior_dict.get(ccy, pct_oi)
            direction = "▲" if pct_oi > prior_pct else "▼"
            extreme   = " ⚠EXTREME" if abs(pct_oi) > 30 else ""
            lines.append(
                f"  {ccy}: {pct_oi:+.1f}% OI {direction} "
                f"(chg: {chg:+,} contracts){extreme}"
            )

        return "\n".join(lines)
    except Exception as exc:
        return f"COT data error: {exc}"


def _collect_news_data() -> str:
    """Collect recent HIGH-impact news headlines for context."""
    try:
        with db.connection() as conn:
            rows = conn.execute(text("""
                SELECT title, published_at, currencies, source
                FROM news_articles
                WHERE impact_level = 'HIGH'
                  AND published_at >= NOW() - INTERVAL '7 days'
                ORDER BY published_at DESC
                LIMIT 20
            """)).fetchall()

        if not rows:
            return "No recent HIGH-impact news."

        lines = ["RECENT HIGH-IMPACT NEWS (last 7 days):"]
        for row in rows:
            title, pub_at, currencies, source = row
            date_str = pub_at.strftime("%b %d")
            ccy_str  = str(currencies).replace("{", "").replace("}", "")
            lines.append(f"  [{date_str}] [{ccy_str}] {title[:120]}")

        return "\n".join(lines)
    except Exception as exc:
        return f"News data error: {exc}"


def _collect_analyst2_scores() -> str:
    """Collect Analyst 2 exogenous scores from Redis."""
    try:
        r = _get_redis()
        lines = ["ANALYST 2 EXOGENOUS SCORES (current):"]
        for pair in PAIRS:
            val = r.get(f"features:analyst2:{pair}")
            if val:
                d = json.loads(val)
                lines.append(
                    f"  {pair}: {d.get('score', 0):+.3f} "
                    f"({d.get('bias', 'NEUTRAL')}) "
                    f"conf={d.get('confidence', 0):.2f}"
                )
        return "\n".join(lines) if len(lines) > 1 else "Analyst 2 scores unavailable."
    except Exception as exc:
        return f"Analyst 2 data error: {exc}"


def _collect_regime_data() -> str:
    """Collect Analyst 1 regime classifications from Redis."""
    try:
        r = _get_redis()
        lines = ["ANALYST 1 REGIME CLASSIFICATIONS (current):"]
        for pair in PAIRS:
            val = r.get(f"features:regime:{pair}")
            if val:
                d = json.loads(val)
                lines.append(
                    f"  {pair}: {d.get('regime', '?')} "
                    f"conf={d.get('confidence', 0):.2f}"
                )
        return "\n".join(lines) if len(lines) > 1 else "Regime data unavailable."
    except Exception as exc:
        return f"Regime data error: {exc}"


def _collect_sentiment_data() -> str:
    """Collect Analyst 5 Groq sentiment scores from Redis."""
    try:
        r = _get_redis()
        val = r.get("features:sentiment:currencies")
        if val:
            scores = json.loads(val)
            lines = ["ANALYST 5 GROQ SENTIMENT (currency):"]
            for ccy, score in scores.items():
                arrow = "▲" if score > 0.1 else "▼" if score < -0.1 else "─"
                lines.append(f"  {ccy}: {score:+.3f} {arrow}")
            return "\n".join(lines)
        return "Sentiment data unavailable."
    except Exception as exc:
        return f"Sentiment data error: {exc}"


def _collect_commodity_data() -> str:
    """Collect latest Oil and Gold prices."""
    try:
        with db.connection() as conn:
            rows = conn.execute(text("""
                SELECT country, indicator, actual, time
                FROM macro_indicators
                WHERE country IN ('OIL', 'XAU')
                ORDER BY country, time DESC
            """)).fetchall()

        if not rows:
            return "Commodity data unavailable."

        seen = {}
        lines = ["COMMODITY PRICES (latest):"]
        for row in rows:
            country, indicator, price, ts = row
            if country not in seen:
                seen[country] = (price, ts)

        oil  = seen.get("OIL")
        gold = seen.get("XAU")
        if oil:
            lines.append(f"  WTI Oil:  ${oil[0]:.2f} @ {oil[1].date()}")
        if gold:
            lines.append(f"  Gold XAU: ${gold[0]:.2f} @ {gold[1].date()}")

        # Trend
        try:
            with db.connection() as conn:
                for country, label in [("OIL", "WTI"), ("XAU", "Gold")]:
                    hist = conn.execute(text("""
                        SELECT actual FROM macro_indicators
                        WHERE country = :c
                        ORDER BY time DESC LIMIT 20
                    """), {"c": country}).fetchall()
                    if len(hist) >= 20:
                        now_p  = hist[0][0]
                        wk_p   = hist[4][0]
                        mo_p   = hist[19][0]
                        lines.append(
                            f"  {label} trend: "
                            f"1w={((now_p-wk_p)/wk_p*100):+.1f}% "
                            f"1m={((now_p-mo_p)/mo_p*100):+.1f}%"
                        )
        except Exception:
            pass

        return "\n".join(lines)
    except Exception as exc:
        return f"Commodity data error: {exc}"


# ══════════════════════════════════════════════════════════════
# OPUS PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior FX macro strategist at a tier-1 institutional hedge fund.
You combine central bank policy analysis, global capital flow dynamics, and geopolitical
risk assessment to produce actionable weekly FX trade bias for G10 currency pairs.

You think in three roles simultaneously:
  HAWK: Focus on inflation, rate differentials, central bank hawkishness/dovishness.
        Which central banks are tightening, holding, or about to cut?
  DOVE: Focus on growth risks, recession probability, safe-haven flows, risk-off signals.
        Where is growth slowing? Where is capital fleeing to?
  STRATEGIST: Synthesize Hawk + Dove views into net FX bias per currency and pair.
              Account for: what's already priced in vs what's a surprise.
              Extreme positioning = lower conviction (already priced in).

Output ONLY valid JSON matching this exact structure:
{
  "hawk_analysis": "2-3 sentences on rate differentials and CB policy divergence",
  "dove_analysis": "2-3 sentences on growth risks and safe-haven dynamics",
  "strategist_summary": "2-3 sentences on net FX bias and key themes for the week",
  "currency_biases": {
    "USD": {"bias": "BULLISH", "score": 0.6, "rationale": "one sentence"},
    "EUR": {"bias": "BEARISH", "score": -0.4, "rationale": "one sentence"},
    "GBP": {"bias": "NEUTRAL", "score": 0.1, "rationale": "one sentence"},
    "JPY": {"bias": "BULLISH", "score": 0.5, "rationale": "one sentence"},
    "CHF": {"bias": "BULLISH", "score": 0.4, "rationale": "one sentence"},
    "AUD": {"bias": "BEARISH", "score": -0.3, "rationale": "one sentence"},
    "NZD": {"bias": "BEARISH", "score": -0.2, "rationale": "one sentence"},
    "CAD": {"bias": "NEUTRAL", "score": 0.0, "rationale": "one sentence"}
  },
  "pair_convictions": {
    "EURUSD": {"bias": "SHORT_BASE", "conviction": 0.65, "rationale": "one sentence"},
    "GBPUSD": {"bias": "NEUTRAL", "conviction": 0.30, "rationale": "one sentence"},
    "USDJPY": {"bias": "SHORT_BASE", "conviction": 0.55, "rationale": "one sentence"},
    "USDCHF": {"bias": "SHORT_BASE", "conviction": 0.50, "rationale": "one sentence"},
    "AUDUSD": {"bias": "SHORT_BASE", "conviction": 0.45, "rationale": "one sentence"},
    "NZDUSD": {"bias": "SHORT_BASE", "conviction": 0.35, "rationale": "one sentence"},
    "USDCAD": {"bias": "NEUTRAL", "conviction": 0.25, "rationale": "one sentence"},
    "EURGBP": {"bias": "SHORT_BASE", "conviction": 0.40, "rationale": "one sentence"},
    "EURJPY": {"bias": "SHORT_BASE", "conviction": 0.50, "rationale": "one sentence"},
    "GBPJPY": {"bias": "LONG_BASE", "conviction": 0.60, "rationale": "one sentence"},
    "AUDJPY": {"bias": "LONG_BASE", "conviction": 0.55, "rationale": "one sentence"},
    "NZDJPY": {"bias": "LONG_BASE", "conviction": 0.45, "rationale": "one sentence"}
  },
  "top_trades": ["trade1", "trade2", "trade3"],
  "key_risks": ["risk1", "risk2", "risk3"],
  "market_regime": "RISK_OFF"
}

Rules:
- bias values: "BULLISH" | "BEARISH" | "NEUTRAL" for currencies
- bias values: "LONG_BASE" | "SHORT_BASE" | "NEUTRAL" for pairs
- score: -1.0 to +1.0 (currency) | conviction: 0.0 to 1.0 (pairs)
- If COT shows EXTREME positioning (>30% OI): reduce conviction by 0.2 (already priced in)
- market_regime: "RISK_ON" | "RISK_OFF" | "MIXED"
- Return ONLY the JSON object. No markdown, no explanation outside JSON."""


def _build_analysis_prompt(
    yields: str, cot: str, news: str,
    analyst2: str, regime: str,
    sentiment: str, commodities: str
) -> str:
    """Build the full data context prompt for Opus."""
    week_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""Produce your weekly G10 FX macro analysis for the week of {week_of}.

Here is all available data from our trading system:

{yields}

{cot}

{commodities}

{news}

{analyst2}

{regime}

{sentiment}

Based on all of the above data, produce your Hawk / Dove / Strategist analysis
and complete G10 FX bias for this week. Remember: extreme COT positioning means
lower conviction. Consider what is already priced in vs what is a fresh catalyst."""


# ══════════════════════════════════════════════════════════════
# RESPONSE PARSER
# ══════════════════════════════════════════════════════════════

def _parse_opus_response(response: str) -> Optional[Dict]:
    """
    Parse Opus JSON response with robust error handling.
    Handles markdown fences and partial JSON.
    """
    if not response:
        return None

    text = response.strip()

    # Strip markdown fences
    if "```json" in text:
        start = text.find("```json") + 7
        end   = text.find("```", start)
        text  = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end   = text.find("```", start)
        text  = text[start:end].strip()
    else:
        # Find outermost JSON object
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"OpusAnalyst | JSON parse failed: {exc}")
        logger.debug(f"OpusAnalyst | Raw response: {response[:500]}")
        return None

    # Validate required keys
    required = ["hawk_analysis", "dove_analysis", "strategist_summary",
                "currency_biases", "pair_convictions", "top_trades", "key_risks"]
    missing = [k for k in required if k not in data]
    if missing:
        logger.warning(f"OpusAnalyst | Missing keys in response: {missing}")

    return data


# ══════════════════════════════════════════════════════════════
# OPUS ANALYST
# ══════════════════════════════════════════════════════════════

class OpusDeepAnalyst:
    """
    Analyst 6: Claude Opus Weekly Deep Analysis.

    Aggregates all system data → calls Anthropic API with Opus model →
    produces structured weekly macro analysis → stores in Redis.

    The weekly analysis provides:
      1. Hawk/Dove/Strategist narrative (for human review)
      2. Currency biases (USD/EUR/GBP/JPY/CHF/AUD/NZD/CAD)
      3. Per-pair conviction scores (used by signal_aggregator)
      4. Top 3 trade ideas for the week
      5. Key macro risks to watch

    Architecture rule:
      Weekly analysis sets macro PRIOR for signal_aggregator.
      High Opus conviction on a pair → lowers Analyst 2 threshold to trade.
      Conflicted Opus → raises threshold (more conservative).
    """

    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv(str(ROOT / ".env"))
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._redis   = None
        if not self._api_key:
            logger.error("OpusAnalyst | ANTHROPIC_API_KEY not set")

    @property
    def r(self):
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    # ── Public API ────────────────────────────────────────────

    def get_weekly_analysis(self) -> Optional[Dict]:
        """Get cached weekly analysis from Redis."""
        try:
            val = self.r.get("features:opus:weekly")
            if val:
                return json.loads(val)
        except Exception as exc:
            logger.warning(f"OpusAnalyst | Redis read failed: {exc}")
        return None

    def get_currency_bias(self, currency: str) -> Dict:
        """
        Get Opus bias for a specific currency.
        Returns {"bias": "BULLISH"|"BEARISH"|"NEUTRAL", "score": float}
        """
        try:
            val = self.r.get(f"features:opus:bias:{currency}")
            if val:
                return json.loads(val)
        except Exception:
            pass
        return {"bias": "NEUTRAL", "score": 0.0, "rationale": "No analysis available"}

    def get_pair_conviction(self, pair: str) -> Dict:
        """
        Get Opus conviction for a specific pair.
        Returns {"bias": "LONG_BASE"|"SHORT_BASE"|"NEUTRAL", "conviction": float}
        """
        try:
            val = self.r.get(f"features:opus:pair:{pair}")
            if val:
                return json.loads(val)
        except Exception:
            pass
        return {"bias": "NEUTRAL", "conviction": 0.0, "rationale": "No analysis available"}

    def run_weekly(self) -> Optional[Dict]:
        """
        Run full weekly deep analysis.

        1. Collect all data from DB + Redis
        2. Call Anthropic API with Opus model
        3. Parse response → store in Redis
        4. Return full analysis dict

        Takes ~30-60 seconds (Opus is thorough).
        """
        logger.info("=" * 60)
        logger.info("OpusAnalyst | WEEKLY DEEP ANALYSIS — START")
        logger.info(f"OpusAnalyst | Model: {OPUS_MODEL}")
        logger.info("=" * 60)

        # ── Collect data ──────────────────────────────────────
        logger.info("OpusAnalyst | Collecting data from DB + Redis...")
        yields      = _collect_yield_data()
        cot         = _collect_cot_data()
        news        = _collect_news_data()
        analyst2    = _collect_analyst2_scores()
        regime      = _collect_regime_data()
        sentiment   = _collect_sentiment_data()
        commodities = _collect_commodity_data()

        logger.info("OpusAnalyst | Data collected — calling Anthropic API...")

        # ── Build prompt ──────────────────────────────────────
        user_prompt = _build_analysis_prompt(
            yields, cot, news, analyst2, regime, sentiment, commodities
        )

        # ── Call Anthropic API ────────────────────────────────
        response_text = self._call_anthropic(user_prompt)
        if not response_text:
            logger.error("OpusAnalyst | No response from Anthropic API")
            return None

        # ── Parse response ────────────────────────────────────
        analysis = _parse_opus_response(response_text)
        if not analysis:
            logger.error("OpusAnalyst | Failed to parse Opus response")
            return None

        # ── Add metadata ──────────────────────────────────────
        analysis["week_of"]      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        analysis["generated_at"] = datetime.now(timezone.utc).isoformat()
        analysis["model"]        = OPUS_MODEL

        # ── Store in Redis ────────────────────────────────────
        self._cache_analysis(analysis)

        # ── Log summary ───────────────────────────────────────
        self._log_summary(analysis)

        return analysis

    def start_scheduler(self) -> None:
        """
        Start background scheduler that runs weekly analysis
        every Sunday at 20:00 UTC.
        Also runs immediately on first start if no cached analysis.
        """
        def _loop() -> None:
            logger.info("OpusAnalyst | Scheduler started (Sunday 20:00 UTC)")

            # Run immediately if no cached analysis
            if not self.get_weekly_analysis():
                logger.info("OpusAnalyst | No cached analysis — running now...")
                self.run_weekly()

            while True:
                now = datetime.now(timezone.utc)
                # Calculate seconds until next Sunday 20:00 UTC
                days_until_sunday = (SCHEDULER_DAY - now.weekday()) % 7
                if days_until_sunday == 0 and now.hour >= SCHEDULER_HOUR:
                    days_until_sunday = 7
                next_run = now.replace(
                    hour=SCHEDULER_HOUR, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_until_sunday)
                wait_sec = (next_run - now).total_seconds()
                logger.info(
                    f"OpusAnalyst | Next run: {next_run.strftime('%Y-%m-%d %H:%M UTC')} "
                    f"(in {wait_sec/3600:.1f}h)"
                )
                time.sleep(wait_sec)
                try:
                    self.run_weekly()
                except Exception as exc:
                    logger.error(f"OpusAnalyst | Weekly run failed: {exc}")

        t = threading.Thread(target=_loop, daemon=True, name="opus-scheduler")
        t.start()
        logger.info("OpusAnalyst | Scheduler thread launched")

    def print_summary(self) -> None:
        """Print formatted summary of latest weekly analysis."""
        analysis = self.get_weekly_analysis()

        print("\n" + "=" * 70)
        print("ANALYST 6 — OPUS WEEKLY DEEP ANALYSIS")
        print("=" * 70)

        if not analysis:
            print("  No analysis available. Run opus_analyst.run_weekly() first.")
            print("=" * 70 + "\n")
            return

        print(f"  Week of: {analysis.get('week_of', 'unknown')}")
        print(f"  Model:   {analysis.get('model', OPUS_MODEL)}")
        print(f"  Regime:  {analysis.get('market_regime', 'UNKNOWN')}")
        print()

        print("HAWK:  " + analysis.get("hawk_analysis", "")[:120])
        print("DOVE:  " + analysis.get("dove_analysis", "")[:120])
        print("STRAT: " + analysis.get("strategist_summary", "")[:120])
        print()

        print(f"{'CURRENCY':<10} {'BIAS':<10} {'SCORE':>7}  RATIONALE")
        print("-" * 70)
        for ccy, data in analysis.get("currency_biases", {}).items():
            print(
                f"{ccy:<10} {data.get('bias','?'):<10} "
                f"{data.get('score', 0):>+7.2f}  "
                f"{data.get('rationale', '')[:45]}"
            )

        print()
        print(f"{'PAIR':<10} {'BIAS':<12} {'CONV':>6}  RATIONALE")
        print("-" * 70)
        for pair, data in analysis.get("pair_convictions", {}).items():
            print(
                f"{pair:<10} {data.get('bias','?'):<12} "
                f"{data.get('conviction', 0):>6.2f}  "
                f"{data.get('rationale', '')[:40]}"
            )

        print()
        top = analysis.get("top_trades", [])
        risks = analysis.get("key_risks", [])
        print(f"TOP TRADES: {' | '.join(top)}")
        print(f"KEY RISKS:  {' | '.join(risks)}")
        print("=" * 70 + "\n")

    # ── Internal ──────────────────────────────────────────────

    def _call_anthropic(self, user_prompt: str) -> Optional[str]:
        """Call Anthropic Messages API with Opus model."""
        headers = {
            "x-api-key":         self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        payload = {
            "model":      OPUS_MODEL,
            "max_tokens": OPUS_MAX_TOKENS,
            "system":     SYSTEM_PROMPT,
            "messages":   [{"role": "user", "content": user_prompt}],
        }

        for attempt in range(1, 4):
            try:
                logger.info(
                    f"OpusAnalyst | API call attempt {attempt} "
                    f"(model={OPUS_MODEL}, max_tokens={OPUS_MAX_TOKENS})..."
                )
                resp = httpx.post(
                    ANTHROPIC_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=OPUS_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()

                # Extract text from content blocks
                content = data.get("content", [])
                text_blocks = [b["text"] for b in content if b.get("type") == "text"]
                result = "\n".join(text_blocks)

                usage = data.get("usage", {})
                logger.success(
                    f"OpusAnalyst | Response received | "
                    f"input={usage.get('input_tokens',0)} "
                    f"output={usage.get('output_tokens',0)} tokens"
                )
                return result

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 529:
                    wait = 30 * attempt
                    logger.warning(f"OpusAnalyst | API overloaded — waiting {wait}s")
                    time.sleep(wait)
                elif exc.response.status_code == 429:
                    wait = 60
                    logger.warning(f"OpusAnalyst | Rate limited — waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(
                        f"OpusAnalyst | HTTP {exc.response.status_code}: "
                        f"{exc.response.text[:200]}"
                    )
                    return None

            except Exception as exc:
                logger.error(f"OpusAnalyst | Attempt {attempt} failed: {exc}")
                if attempt < 3:
                    time.sleep(10)

        return None

    def _cache_analysis(self, analysis: Dict) -> None:
        """Store full analysis + extracted per-currency/pair data in Redis."""
        try:
            # Full analysis
            self.r.setex(
                "features:opus:weekly",
                CACHE_TTL_WEEKLY,
                json.dumps(analysis)
            )

            # Per-currency biases
            for ccy, bias_data in analysis.get("currency_biases", {}).items():
                self.r.setex(
                    f"features:opus:bias:{ccy}",
                    CACHE_TTL_WEEKLY,
                    json.dumps(bias_data)
                )

            # Per-pair convictions
            for pair, conv_data in analysis.get("pair_convictions", {}).items():
                self.r.setex(
                    f"features:opus:pair:{pair}",
                    CACHE_TTL_WEEKLY,
                    json.dumps(conv_data)
                )

            logger.success(
                f"OpusAnalyst | Cached: "
                f"{len(analysis.get('currency_biases', {}))} currencies, "
                f"{len(analysis.get('pair_convictions', {}))} pairs (TTL=7d)"
            )

        except Exception as exc:
            logger.error(f"OpusAnalyst | Cache failed: {exc}")

    def _log_summary(self, analysis: Dict) -> None:
        """Log concise summary of weekly analysis."""
        regime = analysis.get("market_regime", "UNKNOWN")
        top    = analysis.get("top_trades", [])
        risks  = analysis.get("key_risks", [])

        logger.success(f"OpusAnalyst | Market regime: {regime}")
        logger.success(f"OpusAnalyst | Top trades: {top}")
        logger.info(f"OpusAnalyst | Key risks: {risks}")

        # Currency bias summary
        biases = analysis.get("currency_biases", {})
        bull = [c for c, d in biases.items() if d.get("bias") == "BULLISH"]
        bear = [c for c, d in biases.items() if d.get("bias") == "BEARISH"]
        logger.info(f"OpusAnalyst | BULLISH: {bull} | BEARISH: {bear}")


# ── Module singleton ──────────────────────────────────────────
opus_analyst = OpusDeepAnalyst()


# ── Direct run ────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")

    logger.info("Analyst 6 | Running Opus weekly deep analysis...")
    analysis = opus_analyst.run_weekly()
    if analysis:
        opus_analyst.print_summary()
    else:
        logger.error("Analyst 6 | Analysis failed — check API key and logs")