from datetime import datetime, timezone

import pytest

from stockreport.sources import (
    _derive_aliases,
    _entry_datetime,
    _normalize_yf_item,
    _parse_feed,
    _parse_iso_datetime,
    _strip_html,
)

GOOGLE_STYLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>"AAPL" stock - Google News</title>
<item>
<title>Apple hits record high - Reuters</title>
<link>https://news.google.com/rss/articles/abc?oc=5</link>
<pubDate>Fri, 10 Jul 2026 08:00:00 GMT</pubDate>
<source url="https://www.reuters.com">Reuters</source>
<description>&lt;a href="https://x"&gt;Apple hits record high&lt;/a&gt;</description>
</item>
<item>
<title>Entry without a link is skipped</title>
</item>
</channel></rss>"""

PLAIN_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Example Wire</title>
<item>
<title>Stocks rally on rate hopes</title>
<link>https://example.com/rally?utm_source=rss</link>
<pubDate>Fri, 10 Jul 2026 09:30:00 GMT</pubDate>
<description>The S&amp;P 500 rose &lt;b&gt;1.2%&lt;/b&gt; on Friday.</description>
</item>
</channel></rss>"""


def test_parse_feed_google_style():
    items = _parse_feed(GOOGLE_STYLE_RSS, origin="google_rss")
    assert len(items) == 1  # link-less entry skipped
    item = items[0]
    assert item.title == "Apple hits record high"  # publisher suffix stripped
    assert item.source == "Reuters"  # from <source>, not the feed title
    assert item.published == datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    assert item.origin == "google_rss"


def test_parse_feed_uses_feed_title_as_source():
    items = _parse_feed(PLAIN_RSS, origin="market_feed")
    assert items[0].source == "Example Wire"
    assert items[0].summary == "The S&P 500 rose 1.2% on Friday."  # HTML stripped, entities decoded


def test_parse_feed_garbage_raises():
    with pytest.raises(ValueError):
        _parse_feed(b"this is not xml or a feed at all", origin="test")


def test_strip_html_collapses_whitespace():
    assert _strip_html("<p>Hello &amp;\n  <b>world</b></p>") == "Hello & world"
    assert _strip_html("") == ""


def test_entry_datetime_string_fallback_normalizes_to_utc():
    # regression: non-UTC offsets used to be kept and later printed as "UTC"
    entry = {"published": "Fri, 10 Jul 2026 21:30:00 -0400"}
    dt = _entry_datetime(entry)
    assert dt == datetime(2026, 7, 11, 1, 30, tzinfo=timezone.utc)
    assert dt.utcoffset().total_seconds() == 0


def test_entry_datetime_unparseable_is_none():
    assert _entry_datetime({"published": "sometime yesterday"}) is None
    assert _entry_datetime({}) is None


def test_parse_iso_datetime_normalizes_offset_to_utc():
    dt = _parse_iso_datetime("2026-07-10T21:30:00-04:00")
    assert dt == datetime(2026, 7, 11, 1, 30, tzinfo=timezone.utc)
    assert _parse_iso_datetime("2026-07-11T08:00:00Z") == datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)
    assert _parse_iso_datetime("garbage") is None
    assert _parse_iso_datetime(None) is None


def test_normalize_yf_item_nested_schema():
    raw = {
        "id": "x",
        "content": {
            "title": "Apple <b>expands</b>\nbuybacks",
            "summary": "<p>Board approved.</p>",
            "pubDate": "2026-07-11T08:00:00Z",
            "provider": {"displayName": "Reuters"},
            "canonicalUrl": {"url": "https://example.com/a "},
            "clickThroughUrl": None,
        },
    }
    item = _normalize_yf_item(raw)
    assert item.title == "Apple expands buybacks"  # HTML + newline cleaned (regression)
    assert item.url == "https://example.com/a"
    assert item.source == "Reuters"
    assert item.summary == "Board approved."
    assert item.published == datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)


def test_normalize_yf_item_clickthrough_fallback():
    raw = {"content": {"title": "T", "canonicalUrl": {}, "clickThroughUrl": {"url": "https://e.com/b"}}}
    assert _normalize_yf_item(raw).url == "https://e.com/b"


def test_normalize_yf_item_flat_schema():
    raw = {"title": "T", "publisher": "P", "link": "https://e.com/c", "providerPublishTime": 1783500000}
    item = _normalize_yf_item(raw)
    assert item.source == "P"
    assert item.published == datetime.fromtimestamp(1783500000, tz=timezone.utc)


def test_normalize_yf_item_rejects_unusable():
    assert _normalize_yf_item({"content": {"title": "", "canonicalUrl": {"url": "https://x"}}}) is None
    assert _normalize_yf_item({"title": "T"}) is None  # no url
    assert _normalize_yf_item("not a dict") is None


def test_normalize_yf_item_bool_timestamp_ignored():
    raw = {"title": "T", "link": "https://e.com", "providerPublishTime": True}
    assert _normalize_yf_item(raw).published is None


def test_derive_aliases_strips_legal_suffixes():
    aliases = _derive_aliases({"NVIDIA Corporation", "Apple Inc.", "The Coca-Cola Company", None, ""})
    assert "NVIDIA" in aliases
    assert "Apple" in aliases
    assert "Coca-Cola" in aliases  # "The" prefix and "Company" suffix both stripped
    assert "NVIDIA Corporation" in aliases  # full name kept too


def test_derive_aliases_multi_stage():
    assert "X" not in _derive_aliases({"X Holdings Inc"})  # too short to be safe
    aliases = _derive_aliases({"Fortinet Holdings Inc"})
    assert "Fortinet Holdings" in aliases and "Fortinet" in aliases


def test_derive_aliases_international_suffixes():
    # regression: "AB (publ)" and partial names left aliases that never matched headlines
    aliases = _derive_aliases({"Sivers Semiconductors AB (publ)", "Sivers Semiconductors AB"})
    assert "Sivers Semiconductors" in aliases
    assert "Sivers" in aliases  # anchor word
    aliases = _derive_aliases({"Sembcorp Industries Ltd", "Sembcorp Ind"})
    assert "Sembcorp" in aliases


def test_derive_aliases_generic_anchor_words_excluded():
    aliases = _derive_aliases({"General Motors Company"})
    assert "General Motors" in aliases
    assert "General" not in aliases  # would match half the news
