"""News fetchers. Each returns list[NewsItem]; exceptions propagate to the
aggregator, which isolates per-source failures."""
from __future__ import annotations

import logging
import math
import random
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import quote_plus, urlparse

import feedparser
import requests

from .models import NewsItem

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _http_get(url: str, timeout: int) -> bytes:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.content


class _TagStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    stripper = _TagStripper()
    try:
        stripper.feed(text)
        stripper.close()
    except Exception:
        return " ".join(text.split())
    return " ".join("".join(stripper.parts).split())


def _entry_datetime(entry) -> datetime | None:
    # feedparser normalizes *_parsed struct_times to UTC
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
    for key in ("published", "updated"):
        value = entry.get(key)
        if not value:
            continue
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            continue
        if dt is not None:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _parse_feed(raw: bytes, origin: str, default_source: str = "") -> list[NewsItem]:
    parsed = feedparser.parse(raw)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"unparseable feed: {getattr(parsed, 'bozo_exception', 'unknown error')}")
    feed_title = ""
    if parsed.feed:
        feed_title = _strip_html(parsed.feed.get("title", "")).strip()
    items: list[NewsItem] = []
    for entry in parsed.entries:
        title = _strip_html(entry.get("title", "")).strip()
        url = (entry.get("link") or "").strip()
        if not title or not url:
            continue
        source = ""
        entry_source = entry.get("source")
        if isinstance(entry_source, dict):
            source = (entry_source.get("title") or "").strip()
        if not source:
            source = feed_title or default_source
        # Google News style titles carry the publisher as a suffix
        if source and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")].rstrip()
        summary = _strip_html(entry.get("summary") or entry.get("description") or "")
        items.append(
            NewsItem(
                title=title,
                url=url,
                source=source,
                published=_entry_datetime(entry),
                summary=summary,
                origin=origin,
            )
        )
    return items


def fetch_yfinance_news(ticker: str, count: int = 20) -> list[NewsItem]:
    import yfinance as yf

    stock = yf.Ticker(ticker)
    try:
        raw_items = stock.get_news(count=count)
    except (TypeError, AttributeError):
        raw_items = stock.news
    items = []
    for raw in raw_items or []:
        item = _normalize_yf_item(raw)
        if item is not None:
            items.append(item)
    return items


def _normalize_yf_item(raw) -> NewsItem | None:
    if not isinstance(raw, dict):
        return None
    content = raw.get("content")
    if isinstance(content, dict):
        # yfinance >= 0.2.50 nested schema
        title = _strip_html(content.get("title") or "")
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl"):
            candidate = content.get(key)
            if isinstance(candidate, dict) and candidate.get("url"):
                url = candidate["url"].strip()
                break
        provider = content.get("provider")
        source = ""
        if isinstance(provider, dict):
            source = (provider.get("displayName") or "").strip()
        published = _parse_iso_datetime(content.get("pubDate") or content.get("displayTime"))
        summary = _strip_html(content.get("summary") or content.get("description") or "")
    else:
        # pre-0.2.50 flat schema
        title = _strip_html(raw.get("title") or "")
        url = (raw.get("link") or "").strip()
        source = (raw.get("publisher") or "").strip()
        published = None
        ts = raw.get("providerPublishTime")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0:
            try:
                published = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, OSError, OverflowError):
                published = None
        summary = ""
    if not title or not url:
        return None
    return NewsItem(
        title=title,
        url=url,
        source=source or "Yahoo Finance",
        published=published,
        summary=summary,
        origin="yfinance",
    )


def _parse_iso_datetime(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fetch_google_news_rss(ticker: str, lookback_hours: int, timeout: int) -> list[NewsItem]:
    days = max(1, math.ceil(lookback_hours / 24))
    # exchange-suffixed symbols (D05.SI) confuse Google's search; query the base
    query_symbol = ticker.split(".")[0] or ticker
    query = f'"{query_symbol}" stock when:{days}d'
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    time.sleep(random.uniform(0.2, 0.5))  # jitter so concurrent ticker queries don't hit Google at once
    items = _parse_feed(_http_get(url, timeout), origin="google_rss", default_source="Google News")
    # Google News RSS descriptions are just the headline wrapped in a link, not a summary
    return [replace(item, summary="") for item in items]


_NAME_NOISE = re.compile(
    r"\s+(?:\(publ\)|(?:incorporated|corporation|company|holdings?|limited|group|inc|corp|co|ltd"
    r"|plc|ab|ag|nv|sa|asa|oyj|publ|class\s+[a-c])\.?,?)\s*$",
    re.IGNORECASE,
)

# never worth matching on their own as a company "anchor" word
_GENERIC_FIRST_WORDS = {
    "general", "american", "national", "united", "first", "global", "grand",
    "international", "standard", "advanced", "pacific", "digital", "royal",
    "energy", "capital", "financial", "industrial", "consolidated", "new",
}


def fetch_company_aliases(ticker: str) -> list[str]:
    """Company-name aliases for relevance matching; [] on any failure
    (matching then falls back to the ticker symbol alone)."""
    import yfinance as yf

    stock = yf.Ticker(ticker)
    try:
        try:
            info = stock.get_info() or {}
        except AttributeError:
            info = stock.info or {}
    except Exception as exc:
        log.debug("%s: could not fetch company info: %s", ticker, exc)
        return []
    return _derive_aliases({info.get("shortName"), info.get("longName")})


def _derive_aliases(names) -> list[str]:
    aliases: list[str] = []
    for raw in names:
        name = " ".join((raw or "").split())
        if not name:
            continue
        if name.lower().startswith("the "):
            name = name[4:]
        _add_alias(aliases, name)
        stripped = name
        for _ in range(4):  # "X Semiconductors AB (publ)" -> ... -> "X Semiconductors"
            shorter = _NAME_NOISE.sub("", stripped).strip(" ,.")
            if shorter == stripped:
                break
            stripped = shorter
            _add_alias(aliases, stripped)
        # distinctive anchor word: "Sembcorp Industries" -> "Sembcorp" (headlines
        # rarely spell out the full legal name)
        tokens = stripped.split()
        if len(tokens) > 1:
            first = tokens[0].strip(",.&")
            if len(first) >= 4 and first.isalpha() and first.lower() not in _GENERIC_FIRST_WORDS:
                _add_alias(aliases, first)
    return aliases


def _add_alias(aliases: list[str], name: str) -> None:
    if len(name) >= 3 and name not in aliases:
        aliases.append(name)


def fetch_yahoo_ticker_rss(ticker: str, timeout: int) -> list[NewsItem]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US"
    return _parse_feed(_http_get(url, timeout), origin="yahoo_rss", default_source="Yahoo Finance")


def fetch_market_feed(url: str, timeout: int) -> list[NewsItem]:
    items = _parse_feed(_http_get(url, timeout), origin="market_feed")
    # Google News feed descriptions are concatenated cluster headlines, not summaries
    if urlparse(url).netloc.endswith("news.google.com"):
        items = [replace(item, summary="") for item in items]
    return items
