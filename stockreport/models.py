"""Shared data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

MARKET_TOPIC = "MARKET"
MARKET_LABEL = "Market Overview"


@dataclass(frozen=True)
class NewsItem:
    title: str
    url: str
    source: str
    published: datetime | None  # tz-aware UTC, or None if the feed date was unparseable
    summary: str = ""
    origin: str = ""  # fetcher id: yfinance | google_rss | yahoo_rss | market_feed


@dataclass
class TopicResult:
    topic: str  # MARKET_TOPIC or a ticker symbol
    label: str  # section heading in the report
    items: list[NewsItem] = field(default_factory=list)
    prompt: str = ""  # user message sent (or that would be sent) to the model
    summary_md: str = ""  # model output
    model: str = ""  # the model that actually answered (fallbacks may differ from the config)
    error: str | None = None  # per-topic failure (fetch or summarize), rendered into the report
    dropped: int = 0  # trailing items that did not fit the model context budget
    is_fallback: bool = False  # no news in the lookback window; items are the most recent found
