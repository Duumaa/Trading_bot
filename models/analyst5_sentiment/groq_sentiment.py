"""
models/analyst5_sentiment/groq_sentiment.py
============================================
Analyst 5: Groq LLaMA Daily Sentiment Engine

Processes unread news articles from the Redis queue using Groq's
LLaMA model to produce per-currency and per-pair sentiment scores.

Architecture position:
  SENTIMENT GROUP — runs AFTER Fundamental (Analysts 1+2) gate passes.
  Sentiment validates direction — it does NOT override fundamental bias.
  A trade requires: Regime ≠ PNS AND Analyst2 bias matches sentiment.

Flow:
  1. RSS scraper (rss_news_scraper.py) → DB + Redis queue (news:queue)
  2. This module → dequeues article IDs → fetches from Redis hash
  3. Calls Groq LLaMA (llama-3.3-70b-versatile) with structured prompt
  4. Parses JSON response → per-currency sentiment scores
  5. Aggregates to per-pair scores → stores in Redis
  6. Marks articles as processed in DB

Sentiment output per pair:
  {
    "pair":          "AUDUSD",
    "sentiment":     0.42,        # -1.0 (bearish) to +1.0 (bullish base)
    "bias":          "BULLISH",   # BULLISH | BEARISH | NEUTRAL
    "confidence":    0.65,        # 0.0 to 1.0
    "article_count": 8,           # articles processed in last 24h
    "currencies": {
        "AUD": 0.55,
        "USD": -0.10,
    },
    "updated_at":    "2026-03-09T07:00:00Z"
  }

Redis keys:
  news:queue                    → Redis LIST of unprocessed article IDs
  news:{article_id}             → Redis HASH with article data
  features:sentiment:{PAIR}     → per-pair sentiment score  TTL=8h
  features:sentiment:currencies → per-currency dict         TTL=8h

Schedule: Runs every 4 hours (or on-demand after news scrape)

Groq model: llama-3.3-70b-versatile
  - Fast inference (~2s per batch of 5 articles)
  - Strong financial text comprehension
  - Returns structured JSON sentiment per currency

Usage:
  from models.analyst5_sentiment.groq_sentiment import groq_sentiment
  scores = groq_sentiment.run_daily()          # process queue + update Redis
  score  = groq_sentiment.get_sentiment("AUDUSD")
  groq_sentiment.start_scheduler()             # background 4h loop
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

GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.3-70b-versatile"
GROQ_TIMEOUT   = 30       # seconds per API call
GROQ_MAX_RETRIES = 3
BATCH_SIZE     = 5        # articles per Groq call (stay within context limit)

CACHE_TTL_SENTIMENT  = 28800   # 8 hours for sentiment (slower-moving)
CACHE_TTL_CURRENCY   = 28800
SCHEDULER_INTERVAL   = 14400   # 4 hours
LOOKBACK_HOURS       = 24      # only use articles from last 24h for scoring
NEUTRAL_THRESHOLD    = 0.15    # |score| below this = NEUTRAL bias

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
# GROQ API CLIENT
# ══════════════════════════════════════════════════════════════

class GroqClient:
    """
    Thin wrapper around Groq's OpenAI-compatible chat API.
    Uses httpx for sync HTTP (no openai SDK dependency needed).
    """

    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv(str(ROOT / ".env"))
        self._api_key = os.getenv("GROQ_API_KEY", "")
        if not self._api_key:
            logger.error("GroqClient | GROQ_API_KEY not set in .env")

    def chat(self, system: str, user: str,
             temperature: float = 0.1,
             max_tokens: int = 1000) -> Optional[str]:
        """
        Call Groq LLaMA and return the response text.

        Args:
            system:      System prompt
            user:        User message
            temperature: Low = more deterministic (0.1 for structured output)
            max_tokens:  Max response length

        Returns:
            Response text or None on failure
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       GROQ_MODEL,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }

        for attempt in range(1, GROQ_MAX_RETRIES + 1):
            try:
                resp = httpx.post(
                    GROQ_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=GROQ_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"GroqClient | Rate limited — waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"GroqClient | HTTP {exc.response.status_code}: {exc}")
                    return None

            except Exception as exc:
                logger.error(f"GroqClient | Attempt {attempt} failed: {exc}")
                if attempt < GROQ_MAX_RETRIES:
                    time.sleep(2)

        return None


# ══════════════════════════════════════════════════════════════
# ARTICLE LOADER
# ══════════════════════════════════════════════════════════════

def _load_article_from_redis(r, article_id: str) -> Optional[Dict]:
    """
    Load article data from Redis hash.
    Key pattern: news:{article_id}
    """
    try:
        key = f"news:{article_id}"
        # Try as JSON string first
        val = r.get(key)
        if val:
            return json.loads(val)
        # Try as hash
        data = r.hgetall(key)
        if data:
            return data
    except Exception as exc:
        logger.debug(f"Sentiment | Redis article load failed {article_id}: {exc}")
    return None


def _load_unprocessed_from_db(hours: int = LOOKBACK_HOURS) -> List[Dict]:
    """
    Load unprocessed HIGH-impact articles from DB for last N hours.
    Falls back to ALL unprocessed if no HIGH-impact found.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    sql = """
        SELECT id, published_at, source, title, summary,
               currencies, impact_level
        FROM news_articles
        WHERE processed = false
          AND published_at >= :cutoff
        ORDER BY
            CASE WHEN impact_level = 'HIGH' THEN 0 ELSE 1 END,
            published_at DESC
        LIMIT 50
    """
    with db.connection() as conn:
        rows = conn.execute(text(sql), {"cutoff": cutoff}).fetchall()

    articles = []
    for row in rows:
        currencies = row[5]
        # Handle Postgres array format
        if isinstance(currencies, str):
            currencies = currencies.strip("{}").split(",")
        elif currencies is None:
            currencies = []

        articles.append({
            "db_id":       row[0],
            "published_at": row[1],
            "source":      row[2],
            "title":       row[3],
            "summary":     row[4] or "",
            "currencies":  [c.strip().strip('"') for c in currencies],
            "impact":      row[6],
        })

    logger.info(f"Sentiment | Loaded {len(articles)} unprocessed articles from DB")
    return articles


def _mark_processed(db_ids: List[int]) -> None:
    """Mark articles as processed in DB."""
    if not db_ids:
        return
    with db.connection() as conn:
        conn.execute(
            text("UPDATE news_articles SET processed = true WHERE id = ANY(:ids)"),
            {"ids": db_ids}
        )
        conn.commit()
    logger.debug(f"Sentiment | Marked {len(db_ids)} articles as processed")


# ══════════════════════════════════════════════════════════════
# SENTIMENT PROMPT & PARSER
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert FX macro analyst specializing in G10 currencies.
Your task is to analyze news articles and determine their short-term sentiment impact
on G10 currencies: USD, EUR, GBP, JPY, CHF, AUD, NZD, CAD.

Rules:
- Score each relevant currency from -1.0 (strongly bearish) to +1.0 (strongly bullish)
- 0.0 means neutral or not mentioned
- Consider: central bank policy signals, economic data surprises, risk sentiment,
  capital flows, trade/geopolitical events
- High-impact events (NFP, CPI, rate decisions) warrant scores of ±0.5 to ±1.0
- General market commentary warrants smaller scores of ±0.1 to ±0.3
- Risk-off events: USD bullish, JPY bullish, CHF bullish, AUD/NZD bearish
- Risk-on events: AUD/NZD bullish, JPY/CHF bearish

Return ONLY valid JSON in this exact format, no other text:
{
  "USD": 0.0,
  "EUR": 0.0,
  "GBP": 0.0,
  "JPY": 0.0,
  "CHF": 0.0,
  "AUD": 0.0,
  "NZD": 0.0,
  "CAD": 0.0,
  "reasoning": "one sentence explanation"
}"""


def _build_user_prompt(articles: List[Dict]) -> str:
    """Build the user prompt from a batch of articles."""
    lines = [f"Analyze these {len(articles)} FX news articles:\n"]
    for i, art in enumerate(articles, 1):
        title   = art.get("title", "")[:200]
        summary = art.get("summary", "")[:300]
        impact  = art.get("impact", "LOW")
        lines.append(f"[{i}] [{impact}] {title}")
        if summary and summary != title:
            lines.append(f"    {summary}")
    return "\n".join(lines)


def _parse_sentiment_response(response: str) -> Optional[Dict[str, float]]:
    """
    Parse Groq JSON response into currency sentiment dict.
    Handles common LLaMA formatting quirks (markdown fences, trailing text).
    """
    if not response:
        return None

    # Strip markdown code fences if present
    text = response.strip()
    if "```" in text:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
    elif text.startswith("{"):
        pass
    else:
        # Find JSON object
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(f"Sentiment | JSON parse failed: {exc} | text={text[:100]}")
        return None

    # Extract currency scores
    scores = {}
    for ccy in ALL_CURRENCIES:
        val = data.get(ccy, 0.0)
        try:
            scores[ccy] = float(max(-1.0, min(1.0, val)))
        except (TypeError, ValueError):
            scores[ccy] = 0.0

    reasoning = data.get("reasoning", "")
    if reasoning:
        logger.debug(f"Sentiment | LLaMA: {reasoning}")

    return scores


# ══════════════════════════════════════════════════════════════
# AGGREGATOR
# ══════════════════════════════════════════════════════════════

def _aggregate_currency_scores(
    batch_results: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate multiple batch sentiment results into a single
    per-currency score using exponential recency weighting.

    Most recent batch gets weight 1.0, older batches decay.
    """
    if not batch_results:
        return {ccy: 0.0 for ccy in ALL_CURRENCIES}

    n = len(batch_results)
    agg = {ccy: 0.0 for ccy in ALL_CURRENCIES}
    total_weight = 0.0

    for i, scores in enumerate(batch_results):
        # More recent = higher weight
        weight = 1.0 / (n - i)
        total_weight += weight
        for ccy in ALL_CURRENCIES:
            agg[ccy] += scores.get(ccy, 0.0) * weight

    if total_weight > 0:
        agg = {ccy: round(v / total_weight, 4) for ccy, v in agg.items()}

    return agg


def _currency_to_pair_scores(
    currency_scores: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Convert per-currency sentiment scores to per-pair sentiment dicts.

    Pair score = base_score - quote_score
    Positive = bullish base currency.
    """
    results = {}
    now = datetime.now(timezone.utc).isoformat()

    for pair in PAIRS:
        base, quote = PAIR_CURRENCIES[pair]
        base_score  = currency_scores.get(base, 0.0)
        quote_score = currency_scores.get(quote, 0.0)

        # Pair sentiment: positive = bullish base
        raw = base_score - quote_score
        sentiment = float(max(-1.0, min(1.0, raw / 2.0)))

        # Confidence: how strongly the currencies are scored
        abs_scores = [abs(base_score), abs(quote_score)]
        confidence = float(min(sum(abs_scores) / 2.0 * 1.5, 1.0))

        if sentiment > NEUTRAL_THRESHOLD:
            bias = "BULLISH"
        elif sentiment < -NEUTRAL_THRESHOLD:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        results[pair] = {
            "pair":          pair,
            "sentiment":     round(sentiment, 4),
            "bias":          bias,
            "confidence":    round(confidence, 4),
            "currencies": {
                base:  round(base_score, 4),
                quote: round(quote_score, 4),
            },
            "updated_at":    now,
        }

    return results


# ══════════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════

class GroqSentimentEngine:
    """
    Analyst 5: Daily Groq LLaMA Sentiment Engine.

    Processes news articles → per-pair sentiment scores cached in Redis.

    Architecture rule:
      This is Analyst 5 of the SENTIMENT GROUP.
      Output feeds signal_aggregator.py ONLY as a validation layer.
      NEVER ensemble with Analysts 1-4 (fundamental/technical).
      Sentiment gate: |sentiment| > 0.15 AND matches Analyst 2 bias.
    """

    def __init__(self) -> None:
        self._groq   = GroqClient()
        self._redis  = None

    @property
    def r(self):
        if self._redis is None:
            self._redis = _get_redis()
        return self._redis

    # ── Public API ────────────────────────────────────────────

    def get_sentiment(self, pair: str) -> Dict:
        """
        Get current sentiment for a pair from Redis cache.
        Returns neutral result if no data available.
        """
        try:
            val = self.r.get(f"features:sentiment:{pair}")
            if val:
                return json.loads(val)
        except Exception as exc:
            logger.warning(f"Sentiment | Redis read failed: {exc}")

        return self._neutral_result(pair)

    def get_all_sentiments(self) -> Dict[str, Dict]:
        """Get sentiment for all 12 pairs from Redis."""
        return {pair: self.get_sentiment(pair) for pair in PAIRS}

    def run_daily(self) -> Dict[str, Dict]:
        """
        Main entry point: process news queue → update Redis sentiment.

        Flow:
          1. Load unprocessed articles from DB (last 24h)
          2. Also drain Redis news:queue for any IDs not yet in DB
          3. Batch articles → call Groq LLaMA per batch
          4. Aggregate currency scores → compute pair scores
          5. Store in Redis + mark DB articles as processed

        Returns: {pair: sentiment_dict} for all 12 pairs
        """
        logger.info("=" * 55)
        logger.info("Sentiment | GROQ DAILY SENTIMENT RUN")
        logger.info("=" * 55)

        # ── Step 1: Load from DB ──────────────────────────────
        articles = _load_unprocessed_from_db(LOOKBACK_HOURS)

        # ── Step 2: Drain Redis queue for extra IDs ───────────
        redis_articles = self._drain_redis_queue()
        # Merge — avoid duplicates by title
        existing_titles = {a["title"] for a in articles}
        for art in redis_articles:
            if art.get("title", "") not in existing_titles:
                articles.append(art)
                existing_titles.add(art["title"])

        if not articles:
            logger.warning("Sentiment | No new articles to process")
            return self.get_all_sentiments()

        logger.info(f"Sentiment | Processing {len(articles)} articles in batches of {BATCH_SIZE}")

        # ── Step 3: Batch → Groq ──────────────────────────────
        batch_results: List[Dict[str, float]] = []
        db_ids_processed: List[int] = []

        batches = [articles[i:i+BATCH_SIZE]
                   for i in range(0, len(articles), BATCH_SIZE)]

        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Sentiment | Batch {batch_num}/{len(batches)} "
                        f"({len(batch)} articles)...")

            user_prompt = _build_user_prompt(batch)
            response    = self._groq.chat(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                temperature=0.1,
                max_tokens=500,
            )

            if response:
                scores = _parse_sentiment_response(response)
                if scores:
                    batch_results.append(scores)
                    logger.debug(
                        f"Sentiment | Batch {batch_num}: "
                        f"USD={scores.get('USD',0):+.2f} "
                        f"EUR={scores.get('EUR',0):+.2f} "
                        f"AUD={scores.get('AUD',0):+.2f} "
                        f"JPY={scores.get('JPY',0):+.2f}"
                    )

            # Collect DB IDs
            for art in batch:
                if "db_id" in art:
                    db_ids_processed.append(art["db_id"])

            # Rate limit courtesy pause
            if batch_num < len(batches):
                time.sleep(0.5)

        if not batch_results:
            logger.error("Sentiment | No valid responses from Groq")
            return self.get_all_sentiments()

        # ── Step 4: Aggregate + compute pairs ─────────────────
        currency_scores = _aggregate_currency_scores(batch_results)
        pair_scores     = _currency_to_pair_scores(currency_scores)

        # Add article count to each pair result
        for pair in pair_scores:
            pair_scores[pair]["article_count"] = len(articles)

        logger.info(f"Sentiment | Currency scores: " +
                    " ".join(f"{c}={v:+.2f}"
                             for c, v in currency_scores.items()))

        # ── Step 5: Store in Redis ────────────────────────────
        try:
            self.r.setex(
                "features:sentiment:currencies",
                CACHE_TTL_CURRENCY,
                json.dumps(currency_scores)
            )
            for pair, result in pair_scores.items():
                self.r.setex(
                    f"features:sentiment:{pair}",
                    CACHE_TTL_SENTIMENT,
                    json.dumps(result)
                )
            logger.success(
                f"Sentiment | Cached scores for "
                f"{len(pair_scores)} pairs (TTL=8h)"
            )
        except Exception as exc:
            logger.error(f"Sentiment | Redis write failed: {exc}")

        # ── Step 6: Mark DB articles processed ───────────────
        _mark_processed(db_ids_processed)

        # ── Summary ───────────────────────────────────────────
        self._log_summary(pair_scores, currency_scores)
        return pair_scores

    def start_scheduler(self) -> None:
        """Start background daemon that runs sentiment every 4 hours."""
        def _loop() -> None:
            logger.info("Sentiment | Scheduler started (4h interval)")
            while True:
                try:
                    self.run_daily()
                except Exception as exc:
                    logger.error(f"Sentiment | Scheduler error: {exc}")
                time.sleep(SCHEDULER_INTERVAL)

        t = threading.Thread(
            target=_loop, daemon=True, name="sentiment-scheduler"
        )
        t.start()
        logger.info("Sentiment | Scheduler thread launched")

    def print_summary(self) -> None:
        """Print formatted sentiment table for all pairs."""
        print("\n" + "=" * 65)
        print("ANALYST 5 — GROQ SENTIMENT SUMMARY")
        print("=" * 65)
        print(f"{'PAIR':<10} {'SENT':>6} {'BIAS':<10} {'CONF':>6} "
              f"{'BASE':>7} {'QUOTE':>7} {'N':>4}")
        print("-" * 65)

        for pair in PAIRS:
            r   = self.get_sentiment(pair)
            ccy = r.get("currencies", {})
            base, quote = PAIR_CURRENCIES[pair]
            print(
                f"{pair:<10} "
                f"{r['sentiment']:>+6.3f} "
                f"{r['bias']:<10} "
                f"{r['confidence']:>6.2f} "
                f"{ccy.get(base, 0):>+7.3f} "
                f"{ccy.get(quote, 0):>+7.3f} "
                f"{r.get('article_count', 0):>4}"
            )
        print("=" * 65)
        print("Sentiment gate: |sentiment| > 0.15 AND matches Analyst 2 bias")
        print("=" * 65 + "\n")

    # ── Internal helpers ──────────────────────────────────────

    def _drain_redis_queue(self) -> List[Dict]:
        """
        Drain article IDs from Redis news:queue.
        Fetches article data from news:{id} Redis hashes.
        Returns list of article dicts (those found in Redis).
        """
        articles = []
        try:
            queue_len = self.r.llen("news:queue")
            if not queue_len:
                return []

            logger.info(f"Sentiment | Draining Redis queue ({queue_len} items)...")

            # Pop up to 50 IDs
            ids = []
            for _ in range(min(queue_len, 50)):
                item = self.r.lpop("news:queue")
                if item:
                    ids.append(item.strip())

            for article_id in ids:
                art = _load_article_from_redis(self.r, article_id)
                if art:
                    # Normalise field names
                    article = {
                        "title":     art.get("title", ""),
                        "summary":   art.get("summary", ""),
                        "source":    art.get("source", ""),
                        "impact":    "HIGH" if art.get("is_high_impact") in
                                     (True, "True", "true", 1, "1") else "LOW",
                        "currencies": self._parse_currencies(
                            art.get("currencies", [])
                        ),
                    }
                    if article["title"]:
                        articles.append(article)

        except Exception as exc:
            logger.warning(f"Sentiment | Redis queue drain failed: {exc}")

        return articles

    def _parse_currencies(self, raw) -> List[str]:
        """Normalise currencies field from various formats."""
        if isinstance(raw, list):
            return [str(c).strip().strip("'\"") for c in raw]
        if isinstance(raw, str):
            cleaned = raw.strip("[]'\"")
            return [c.strip().strip("'\"")
                    for c in cleaned.split(",") if c.strip()]
        return []

    @staticmethod
    def _neutral_result(pair: str) -> Dict:
        """Return safe neutral sentiment when no data available."""
        base, quote = PAIR_CURRENCIES.get(pair, ("", ""))
        return {
            "pair":          pair,
            "sentiment":     0.0,
            "bias":          "NEUTRAL",
            "confidence":    0.0,
            "article_count": 0,
            "currencies":    {base: 0.0, quote: 0.0},
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }

    def _log_summary(self, pair_scores: Dict[str, Dict],
                     currency_scores: Dict[str, float]) -> None:
        """Log concise sentiment summary."""
        bullish = [p for p, r in pair_scores.items() if r["bias"] == "BULLISH"]
        bearish = [p for p, r in pair_scores.items() if r["bias"] == "BEARISH"]
        logger.success(
            f"Sentiment | BULLISH: {bullish} | BEARISH: {bearish}"
        )
        logger.info(
            "Sentiment | Currency bias: " +
            " ".join(
                f"{c}={'▲' if v > 0.1 else '▼' if v < -0.1 else '─'}"
                for c, v in currency_scores.items()
            )
        )


# ── Module singleton ──────────────────────────────────────────
groq_sentiment = GroqSentimentEngine()


# ── Direct run ────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/workspaces/Trading_bot/.env")

    logger.info("Analyst 5 | Running Groq sentiment...")
    groq_sentiment.run_daily()
    groq_sentiment.print_summary()