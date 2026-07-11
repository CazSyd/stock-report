from datetime import datetime, timedelta, timezone

from stockreport.models import NewsItem

FIXED_NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def make_item(
    title="Some headline",
    url="https://example.com/article",
    source="Example Wire",
    published=None,
    hours_ago=None,
    summary="",
    origin="test",
) -> NewsItem:
    if hours_ago is not None:
        published = FIXED_NOW - timedelta(hours=hours_ago)
    return NewsItem(
        title=title, url=url, source=source, published=published, summary=summary, origin=origin
    )
