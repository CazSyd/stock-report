from datetime import datetime, timedelta, timezone

from stockreport.models import NewsItem


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def make_item(
    title="Some headline",
    url="https://example.com/article",
    source="Example Wire",
    published=None,
    hours_ago=None,
    summary="",
    origin="test",
) -> NewsItem:
    # ages are relative to the real clock: collect_all filters against
    # datetime.now(), so a frozen reference would rot as days pass
    if hours_ago is not None:
        published = now_utc() - timedelta(hours=hours_ago)
    return NewsItem(
        title=title, url=url, source=source, published=published, summary=summary, origin=origin
    )
