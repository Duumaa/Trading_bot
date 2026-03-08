"""
data_pipeline/connectors/rss_news_scraper.py
============================================
RSS News Scraper — G10 FX Headlines for Sentiment Analysis

Scrapes multiple RSS feeds, extracts G10-relevant headlines,
stores to TimescaleDB and Redis queue for Groq LLaMA processing.

Sources:
    • Reuters Business News
    • ForexLive
    • FXStreet
    • Investing.com FX News
    • DailyFX

Pipeline:
    RSS feeds → parse → filter G10 → deduplicate → TimescaleDB
                                                  → Redis queue (for Groq)

Usage:
    from data_pipeline.connectors.rss_news_scraper import news_scraper

    # Fetch latest headlines
    articles = news_scraper.fetch_all()

    # Get unprocessed headlines for Groq sentiment
    queue = news_scraper.get_sentiment_queue(max_items=50)

    # Mark as processed after Groq runs
    news_scraper.mark_processed(article_ids)

    # Start background scheduler (every 15 minutes)
    news_scraper.start_scheduler()
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import threading
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import feedparser
import requests
from loguru import logger

UTC = timezone.utc

# ── RSS Feed sources ───────────────────────────────────────────────────────────

RSS_FEEDS = [
    {
        "name":     "Reuters Business",
        "url":      "https://feeds.reuters.com/reuters/businessNews",
        "priority": 1,
    },
    {
        "name":     "ForexLive",
        "url":      "https://www.forexlive.com/feed/",
        "priority": 1,
    },
    {
        "name":     "FXStreet",
        "url":      "https://www.fxstreet.com/rss/news",
        "priority": 2,
    },
    {
        "name":     "Investing.com FX",
        "url":      "https://www.investing.com/rss/news_301.rss",
        "priority": 2,
    },
    {
        "name":     "DailyFX",
        "url":      "https://www.dailyfx.com/feeds/all",
        "priority": 2,
    },
    {
        "name":     "MarketWatch FX",
        "url":      "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        "priority": 3,
    },
]

# ── G10 currency keyword mapping ───────────────────────────────────────────────
# Maps keywords found in headlines → currency codes

CURRENCY_KEYWORDS: Dict[str, List[str]] = {
    "USD": [
        "dollar", "usd", "fed", "federal reserve", "fomc", "powell",
        "us economy", "u.s.", "united states", "treasury", "nonfarm",
        "non-farm", "payroll", "cpi us", "us inflation", "us gdp",
    ],
    "EUR": [
        "euro", "eur", "ecb", "european central bank", "lagarde",
        "eurozone", "euro zone", "germany", "france", "italy", "spain",
        "eu economy", "european",
    ],
    "GBP": [
        "pound", "sterling", "gbp", "boe", "bank of england", "bailey",
        "uk economy", "britain", "british", "united kingdom", "brexit",
    ],
    "JPY": [
        "yen", "jpy", "boj", "bank of japan", "ueda", "kuroda",
        "japan economy", "japanese", "nikkei", "boj rate",
    ],
    "CHF": [
        "franc", "chf", "snb", "swiss national bank", "swiss",
        "switzerland",
    ],
    "CAD": [
        "loonie", "cad", "boc", "bank of canada", "macklem",
        "canada", "canadian", "oil price", "crude",
    ],
    "AUD": [
        "aussie", "aud", "rba", "reserve bank of australia", "bullock",
        "australia", "australian", "iron ore", "china gdp",
    ],
    "NZD": [
        "kiwi", "nzd", "rbnz", "reserve bank of new zealand", "orr",
        "new zealand", "nz economy",
    ],
    "SEK": [
        "krona", "sek", "riksbank", "sweden", "swedish",
    ],
    "NOK": [
        "krone", "nok", "norges bank", "norway", "norwegian", "oil",
    ],
}

# High-impact headline keywords — these get priority in sentiment queue
HIGH_IMPACT_KEYWORDS = {
    "rate decision", "interest rate", "rate hike", "rate cut", "rate hold",
    "nonfarm payroll", "non-farm", "cpi", "inflation", "gdp", "recession",
    "fomc", "ecb meeting", "boe meeting", "boj meeting", "rba meeting",
    "emergency", "crisis", "intervention", "default", "sanctions",
    "tariff", "trade war",
}


# ── Article dataclass ──────────────────────────────────────────────────────────

class Article:
    """Single news article from RSS feed."""

    __slots__ = (
        "article_id", "title", "summary", "url", "source",
        "published_utc", "currencies", "is_high_impact", "sentiment_processed",
    )

    def __init__(
        self,
        title:        str,
        summary:      str,
        url:          str,
        source:       str,
        published_utc: datetime,
    ) -> None:
        self.title           = title.strip()
        self.summary         = summary.strip()[:500]   # cap summary length
        self.url             = url
        self.source          = source
        self.published_utc   = published_utc
        self.currencies      = _extract_currencies(title + " " + summary)
        self.is_high_impact  = _is_high_impact(title)
        self.sentiment_processed = False

        # Deterministic ID from URL (deduplication key)
        self.article_id = hashlib.md5(url.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id":     self.article_id,
            "title":          self.title,
            "summary":        self.summary,
            "url":            self.url,
            "source":         self.source,
            "published_utc":  self.published_utc,
            "currencies":     json.dumps(self.currencies),
            "is_high_impact": self.is_high_impact,
        }

    def __repr__(self) -> str:
        return (
            f"Article({self.source} | {self.published_utc.strftime('%H:%M')} | "
            f"{self.currencies} | {self.title[:60]})"
        )


# ── Helper functions ───────────────────────────────────────────────────────────

def _extract_currencies(text: str) -> List[str]:
    """Extract G10 currency codes mentioned in text."""
    text_lower = text.lower()
    found: Set[str] = set()
    for currency, keywords in CURRENCY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.add(currency)
    return sorted(found)


def _is_high_impact(title: str) -> bool:
    """Check if headline mentions high-impact events."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in HIGH_IMPACT_KEYWORDS)


def _parse_rss_date(entry: Any) -> datetime:
    """
    Parse RSS entry date → UTC datetime.

    RSS uses RFC 2822 format e.g. "Fri, 07 Mar 2026 14:30:00 +0000"
    feedparser normalises this to a time.struct_time in UTC.
    """
    # feedparser provides published_parsed as UTC struct_time
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=UTC)
        except Exception:
            pass

    # Fallback: parse raw published string
    if hasattr(entry, "published") and entry.published:
        try:
            dt = parsedate_to_datetime(entry.published)
            return dt.astimezone(UTC)
        except Exception:
            pass

    # Last resort: now
    return datetime.now(UTC)


def _clean_html(text: str) -> str:
    """Strip HTML tags from summary text."""
    return re.sub(r"<[^>]+>", " ", text or "").strip()


# ── Main scraper class ─────────────────────────────────────────────────────────

class RSSNewsScraper:
    """
    Multi-source RSS news scraper for G10 FX sentiment pipeline.

    Fetches headlines every 15 minutes, deduplicates by URL hash,
    filters for G10-relevant content, stores to DB + Redis queue.

    Redis keys:
        news:queue          → list of article_ids pending Groq processing
        news:{article_id}   → article JSON (TTL 48h)
    """

    FETCH_INTERVAL_S = 900    # 15 minutes
    REDIS_TTL_S      = 172800  # 48 hours

    def __init__(self) -> None:
        self._seen_ids: Set[str] = set()   # in-memory dedup cache
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; G10Bot/1.0; +https://example.com)"
        })

    # =========================================================================
    # Fetch methods
    # =========================================================================

    def fetch_feed(self, feed_config: Dict[str, Any]) -> List[Article]:
        """
        Fetch and parse a single RSS feed.

        Returns list of new Article objects (not seen before).
        """
        name = feed_config["name"]
        url  = feed_config["url"]

        try:
            # Use feedparser with requests session for better header control
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as exc:
            logger.warning(f"RSS fetch failed [{name}]: {exc}")
            return []

        articles: List[Article] = []
        cutoff = datetime.now(UTC) - timedelta(hours=48)

        for entry in feed.entries:
            try:
                title    = getattr(entry, "title", "")
                summary  = _clean_html(getattr(entry, "summary", ""))
                link     = getattr(entry, "link", "")
                pub_time = _parse_rss_date(entry)

                # Skip old articles
                if pub_time < cutoff:
                    continue

                # Skip if no G10 content
                currencies = _extract_currencies(title + " " + summary)
                if not currencies:
                    continue

                article = Article(
                    title=title,
                    summary=summary,
                    url=link,
                    source=name,
                    published_utc=pub_time,
                )

                # Skip duplicates
                if article.article_id in self._seen_ids:
                    continue

                articles.append(article)
                self._seen_ids.add(article.article_id)

            except Exception as exc:
                logger.debug(f"Entry parse error [{name}]: {exc}")
                continue

        logger.debug(f"RSS [{name}]: {len(articles)} new articles")
        return articles

    def fetch_all(self) -> List[Article]:
        """
        Fetch all configured RSS feeds.

        Returns combined list sorted newest → oldest.
        Stores to TimescaleDB and Redis queue.
        """
        all_articles: List[Article] = []

        for feed in RSS_FEEDS:
            articles = self.fetch_feed(feed)
            all_articles.extend(articles)
            time.sleep(0.5)   # polite delay between feeds

        if not all_articles:
            logger.debug("RSS | No new articles")
            return []

        # Sort newest first
        all_articles.sort(key=lambda a: a.published_utc, reverse=True)

        # Store
        self._store_db(all_articles)
        self._store_redis(all_articles)

        high_impact = [a for a in all_articles if a.is_high_impact]
        logger.info(
            f"RSS | {len(all_articles)} new articles | "
            f"{len(high_impact)} high-impact | "
            f"currencies: {sorted(set(c for a in all_articles for c in a.currencies))}"
        )

        return all_articles

    # =========================================================================
    # Sentiment queue interface (for Groq)
    # =========================================================================

    def get_sentiment_queue(
        self,
        max_items:      int  = 50,
        high_impact_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get unprocessed articles for Groq sentiment analysis.

        Called by Analyst 5 (daily LLaMA sentiment) every cycle.

        Returns list of article dicts sorted by:
            1. is_high_impact (high first)
            2. published_utc (newest first)
        """
        try:
            r = self._get_redis()
            if r is None:
                return self._queue_from_db(max_items, high_impact_only)

            # Pop from Redis queue
            items = []
            for _ in range(max_items):
                article_id = r.lpop("news:queue")
                if article_id is None:
                    break
                data = r.get(f"news:{article_id}")
                if data:
                    try:
                        items.append(json.loads(data))
                    except Exception:
                        pass

            return items

        except Exception as exc:
            logger.error(f"get_sentiment_queue failed: {exc}")
            return self._queue_from_db(max_items, high_impact_only)

    def mark_processed(self, article_ids: List[str], scores: Optional[Dict[str, float]] = None) -> None:
        """Mark articles as sentiment-processed, optionally storing Groq scores."""
        if not article_ids:
            return
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text
            scores = scores or {}
            with db.connection() as conn:
                for aid in article_ids:
                    score = scores.get(str(aid))
                    conn.execute(
                        text("""
                            UPDATE news_articles
                            SET processed = TRUE,
                                sentiment_score = COALESCE(:score, sentiment_score)
                            WHERE id = :aid
                        """),
                        {"score": score, "aid": int(aid)},
                    )
        except Exception as exc:
            logger.error(f"mark_processed failed: {exc}")

    def get_recent_headlines(
        self,
        currency:   Optional[str] = None,
        hours_back: int = 24,
        limit:      int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent headlines for a currency (used by Analyst 5).

        Args:
            currency:   Filter by currency e.g. "USD" (None = all)
            hours_back: Look back window
            limit:      Max articles to return
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text

            if currency:
                sql = text("""
                    SELECT id, title, summary, source,
                           published_at, currencies, impact_level,
                           sentiment_score
                    FROM news_articles
                    WHERE published_at >= :cutoff
                      AND :currency = ANY(currencies)
                    ORDER BY (impact_level = 'HIGH') DESC, published_at DESC
                    LIMIT :limit
                """)
                params = {"cutoff": cutoff, "currency": currency, "limit": limit}
            else:
                sql = text("""
                    SELECT id, title, summary, source,
                           published_at, currencies, impact_level,
                           sentiment_score
                    FROM news_articles
                    WHERE published_at >= :cutoff
                    ORDER BY (impact_level = 'HIGH') DESC, published_at DESC
                    LIMIT :limit
                """)
                params = {"cutoff": cutoff, "limit": limit}

            with db.connection() as conn:
                result = conn.execute(sql, params)
                return [dict(row) for row in result.fetchall()]

        except Exception as exc:
            logger.error(f"get_recent_headlines failed: {exc}")
            return []

    # =========================================================================
    # Storage
    # =========================================================================

    def _store_db(self, articles: List[Article]) -> int:
        """
        Store articles to TimescaleDB news_articles table.

        Actual schema:
            published_at    TIMESTAMPTZ
            source          VARCHAR(50)
            title           TEXT
            summary         TEXT
            url             TEXT
            currencies      VARCHAR(3)[]   ← Postgres array, not JSONB
            sentiment_score FLOAT          ← filled later by Groq
            impact_level    VARCHAR(10)    ← 'HIGH' or 'LOW'
        Dedup key: url (unique constraint expected or we use DO NOTHING on url)
        """
        if not articles:
            return 0
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text

            rows = [
                {
                    "published_at":  a.published_utc,
                    "source":        a.source[:50],
                    "title":         a.title,
                    "summary":       a.summary or "",
                    "url":           a.url,
                    # Postgres array literal: ARRAY['USD','EUR']
                    "currencies":    a.currencies,
                    "impact_level":  "HIGH" if a.is_high_impact else "LOW",
                }
                for a in articles
            ]

            stored = 0
            with db.connection() as conn:
                for row in rows:
                    try:
                        conn.execute(
                            text("""
                                INSERT INTO news_articles
                                    (published_at, source, title, summary,
                                     url, currencies, impact_level, processed)
                                SELECT :published_at, :source, :title, :summary,
                                       :url,
                                       CAST(:currencies AS VARCHAR(3)[]),
                                       :impact_level, FALSE
                                WHERE NOT EXISTS (
                                    SELECT 1 FROM news_articles WHERE url = :url
                                )
                            """),
                            {**row, "currencies": "{" + ",".join(row["currencies"]) + "}"},
                        )
                        stored += 1
                    except Exception as row_exc:
                        logger.debug(f"Row skip: {row_exc}")
                        continue

            logger.debug(f"RSS DB | {stored}/{len(rows)} articles stored")
            return stored
        except Exception as exc:
            logger.error(f"RSS DB store failed: {exc}")
            return 0

    def _store_redis(self, articles: List[Article]) -> None:
        """Cache articles to Redis and add to sentiment queue."""
        try:
            r = self._get_redis()
            if r is None:
                return

            pipe = r.pipeline()
            for article in articles:
                data = json.dumps({
                    "article_id":    article.article_id,
                    "title":         article.title,
                    "summary":       article.summary,
                    "source":        article.source,
                    "published_utc": article.published_utc.isoformat(),
                    "currencies":    article.currencies,
                    "is_high_impact": article.is_high_impact,
                })
                # Store article data
                pipe.set(f"news:{article.article_id}", data, ex=self.REDIS_TTL_S)
                # Add to sentiment queue (high-impact articles pushed to front)
                if article.is_high_impact:
                    pipe.lpush("news:queue", article.article_id)
                else:
                    pipe.rpush("news:queue", article.article_id)

            pipe.execute()
        except Exception as exc:
            logger.debug(f"Redis store failed: {exc}")

    def _queue_from_db(
        self,
        max_items: int,
        high_impact_only: bool,
    ) -> List[Dict[str, Any]]:
        """Fallback: get unprocessed articles from DB."""
        try:
            from data_pipeline.storage.timeseries_db import db
            from sqlalchemy import text

            cutoff = datetime.now(UTC) - timedelta(hours=24)
            sql = text("""
                SELECT id, title, summary, source,
                       published_at, currencies, impact_level
                FROM news_articles
                WHERE processed = FALSE
                  AND published_at >= :cutoff
                  AND (:high_impact_only = FALSE OR impact_level = 'HIGH')
                ORDER BY (impact_level = 'HIGH') DESC, published_at DESC
                LIMIT :limit
            """)
            with db.connection() as conn:
                result = conn.execute(sql, {
                    "cutoff": cutoff,
                    "high_impact_only": high_impact_only,
                    "limit": max_items,
                })
                return [dict(row) for row in result.fetchall()]
        except Exception as exc:
            logger.error(f"DB queue fallback failed: {exc}")
            return []

    # =========================================================================
    # Redis helper
    # =========================================================================

    def _get_redis(self):
        """Lazy Redis connection."""
        try:
            import redis, os
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            pwd  = os.getenv("REDIS_PASSWORD") or None
            return redis.Redis(host=host, port=port, password=pwd,
                               decode_responses=True)
        except Exception:
            return None

    # =========================================================================
    # Scheduler
    # =========================================================================

    def start_scheduler(self) -> None:
        """
        Background thread that fetches RSS every 15 minutes.
        Call once at bot startup.
        """
        def _run():
            logger.info("RSS scheduler started — fetching every 15min")
            while True:
                try:
                    self.fetch_all()
                except Exception as exc:
                    logger.error(f"RSS scheduler error: {exc}")
                time.sleep(self.FETCH_INTERVAL_S)

        t = threading.Thread(target=_run, name="rss-scheduler", daemon=True)
        t.start()
        logger.info("RSS news scheduler started ✅")


# ── Module singleton ───────────────────────────────────────────────────────────
news_scraper = RSSNewsScraper()