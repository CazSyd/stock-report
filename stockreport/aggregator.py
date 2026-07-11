"""Collect news per topic: parallel fan-out over (topic, source) tasks,
then time-filter, dedupe, sort, and cap."""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from . import sources
from .config import AppConfig
from .models import MARKET_LABEL, MARKET_TOPIC, NewsItem, TopicResult

log = logging.getLogger(__name__)

MAX_WORKERS = 8
GOOGLE_FALLBACK_HOURS = 30 * 24  # Google News search window when the fallback is enabled

_TRACKING_PREFIXES = ("utm_", "guce", "fbclid", "gclid", "ncid", "cmpid", "soc_src", "soc_trk")


def collect_all(cfg: AppConfig) -> list[TopicResult]:
    """Fetch every (topic, source) pair concurrently; sources fail independently.

    A topic whose sources ALL failed gets TopicResult.error set, so the report
    can distinguish an outage from a genuinely quiet news day.
    """
    fallback_enabled = cfg.news.fallback_max_articles > 0
    # give Google a wide window so the quiet-ticker fallback has a pool to draw
    # from (yfinance/Yahoo RSS always return the latest ~20 regardless of age)
    google_hours = max(cfg.news.lookback_hours, GOOGLE_FALLBACK_HOURS) if fallback_enabled else cfg.news.lookback_hours

    tasks: list[tuple[str, str, object]] = []
    for feed_url in cfg.news.market_feeds:
        try:
            label = urlparse(feed_url).netloc or feed_url
        except ValueError:
            label = feed_url
        tasks.append(
            (
                MARKET_TOPIC,
                f"feed {label}",
                lambda u=feed_url: sources.fetch_market_feed(u, cfg.news.request_timeout_seconds),
            )
        )
    for ticker in cfg.tickers:
        tasks.append((ticker, "yfinance", lambda t=ticker: sources.fetch_yfinance_news(t)))
        tasks.append(
            (
                ticker,
                "google_rss",
                lambda t=ticker: sources.fetch_google_news_rss(
                    t, google_hours, cfg.news.request_timeout_seconds
                ),
            )
        )
        tasks.append(
            (
                ticker,
                "yahoo_rss",
                lambda t=ticker: sources.fetch_yahoo_ticker_rss(t, cfg.news.request_timeout_seconds),
            )
        )

    # Submission order doubles as dedup priority: yfinance items (richest
    # summaries) come first within each ticker.
    per_topic: dict[str, list[NewsItem]] = {MARKET_TOPIC: []}
    for ticker in cfg.tickers:
        per_topic.setdefault(ticker, [])
    attempts: dict[str, int] = {}
    failures: dict[str, int] = {}
    for topic, _name, _fn in tasks:
        attempts[topic] = attempts.get(topic, 0) + 1

    aliases_by_ticker: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        alias_futures = (
            {t: pool.submit(sources.fetch_company_aliases, t) for t in cfg.tickers}
            if cfg.news.require_ticker_mention
            else {}
        )
        futures = [(topic, name, pool.submit(fn)) for topic, name, fn in tasks]
        try:
            for topic, name, future in futures:
                try:
                    fetched = future.result()
                except Exception as exc:
                    failures[topic] = failures.get(topic, 0) + 1
                    # Yahoo's per-ticker RSS is known-flaky; don't warn about it
                    level = logging.DEBUG if name == "yahoo_rss" else logging.WARNING
                    log.log(level, "%s: %s failed: %s", topic, name, exc)
                    continue
                per_topic[topic].extend(fetched)
                log.info("%s: %s returned %d items", topic, name, len(fetched))
            for ticker, future in alias_futures.items():
                try:
                    aliases_by_ticker[ticker] = future.result()
                except Exception as exc:
                    log.debug("%s: company-name lookup failed: %s", ticker, exc)
        except BaseException:
            # Ctrl+C etc.: don't let queued fetches keep running for minutes
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    now = datetime.now(timezone.utc)
    results: list[TopicResult] = []
    for topic in [MARKET_TOPIC, *cfg.tickers]:
        label = MARKET_LABEL if topic == MARKET_TOPIC else topic
        if attempts.get(topic) and failures.get(topic, 0) == attempts[topic]:
            results.append(
                TopicResult(
                    topic=topic,
                    label=label,
                    error=f"News fetch failed: all {attempts[topic]} sources errored (network problem?)",
                )
            )
            continue
        topic_items = per_topic[topic]
        if topic != MARKET_TOPIC and cfg.news.require_ticker_mention:
            pattern = build_relevance_pattern(topic, aliases_by_ticker.get(topic, []))
            relevant = filter_relevant(topic_items, pattern)
            if len(relevant) < len(topic_items):
                log.info(
                    "%s: filtered out %d of %d items that do not mention the ticker or company",
                    topic,
                    len(topic_items) - len(relevant),
                    len(topic_items),
                )
            topic_items = relevant
        # Time-filter BEFORE dedupe, so a stale out-of-window copy of a story
        # can never shadow a fresh in-window copy of the same headline.
        merged = dedupe(filter_recent(topic_items, cfg.news.lookback_hours, now))
        merged.sort(key=_recency_key)
        capped = merged[: cfg.news.max_articles_per_topic]
        is_fallback = False
        if not capped and topic != MARKET_TOPIC and fallback_enabled:
            # no fixed window: just the most recent items known for this ticker
            wider = dedupe(topic_items)
            wider.sort(key=_recency_key)
            capped = wider[: cfg.news.fallback_max_articles]
            if capped:
                is_fallback = True
                log.info(
                    "%s: no news within %dh; showing the %d most recent items found",
                    topic,
                    cfg.news.lookback_hours,
                    len(capped),
                )
        if len(merged) > len(capped) and not is_fallback:
            log.info("%s: capped %d items to %d", topic, len(merged), len(capped))
        results.append(TopicResult(topic=topic, label=label, items=capped, is_fallback=is_fallback))
    return results


def _recency_key(item: NewsItem):
    return (item.published is None, -(item.published.timestamp() if item.published else 0))


def build_relevance_pattern(ticker: str, aliases: list[str]) -> re.Pattern:
    """The symbol matches case-sensitively (an insensitive 'A' or 'IT' would hit
    ordinary words); company-name aliases match case-insensitively.
    A dashed symbol like BRK-B also matches its BRK.B spelling, and an
    exchange-suffixed symbol like U96.SI also matches its bare base (SGX:U96)."""
    symbol = "[-.]".join(re.escape(part) for part in ticker.split("-"))
    parts = [rf"\b{symbol}\b"]
    if "." in ticker:
        base = ticker.split(".")[0]
        if len(base) >= 2:
            base_pattern = "[-.]".join(re.escape(part) for part in base.split("-"))
            parts.append(rf"\b{base_pattern}\b")
    for alias in aliases:
        alias = alias.strip()
        if len(alias) >= 3 and alias.upper() != ticker:
            parts.append(rf"(?i:\b{re.escape(alias)}\b)")
    return re.compile("|".join(parts))


def filter_relevant(items: list[NewsItem], pattern: re.Pattern) -> list[NewsItem]:
    return [i for i in items if pattern.search(i.title) or pattern.search(i.summary)]


def dedupe(items: list[NewsItem]) -> list[NewsItem]:
    """Drop items whose normalized URL or title was already seen; first wins."""
    seen: set[str] = set()
    out: list[NewsItem] = []
    for item in items:
        keys = [k for k in (_norm_url(item.url), "t:" + _norm_title(item.title)) if k and k != "t:"]
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        out.append(item)
    return out


def filter_recent(items: list[NewsItem], lookback_hours: int, now: datetime) -> list[NewsItem]:
    """Keep items inside the lookback window; undated items are kept."""
    cutoff = now - timedelta(hours=lookback_hours)
    return [i for i in items if i.published is None or i.published >= cutoff]


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _norm_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not k.lower().startswith(_TRACKING_PREFIXES)
    ]
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.params,
            urlencode(query),
            "",  # drop fragment
        )
    )
