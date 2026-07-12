import pytest
from conftest import FIXED_NOW, make_item

import stockreport.aggregator as aggregator
from stockreport.aggregator import (
    _norm_url,
    build_relevance_pattern,
    collect_all,
    dedupe,
    filter_recent,
    filter_relevant,
)
from stockreport.config import AppConfig, NewsConfig
from stockreport.models import MARKET_TOPIC


def test_dedupe_same_url_different_tracking_params():
    items = [
        make_item(title="A", url="https://e.com/story?utm_source=x&id=1"),
        make_item(title="B", url="https://E.com/story/?id=1&utm_medium=y#frag"),
    ]
    assert [i.title for i in dedupe(items)] == ["A"]


def test_dedupe_same_title_first_wins():
    items = [
        make_item(title="Apple Hits High!", url="https://a.com/1", origin="yfinance"),
        make_item(title="apple hits high", url="https://b.com/2", origin="google_rss"),
    ]
    kept = dedupe(items)
    assert len(kept) == 1
    assert kept[0].origin == "yfinance"  # submission order is the priority


def test_dedupe_distinct_kept_in_order():
    items = [make_item(title=f"T{i}", url=f"https://e.com/{i}") for i in range(3)]
    assert dedupe(items) == items


def test_filter_recent_boundaries():
    kept = filter_recent(
        [make_item(hours_ago=23), make_item(hours_ago=25), make_item(published=None)],
        lookback_hours=24,
        now=FIXED_NOW,
    )
    assert len(kept) == 2
    assert kept[0].published is not None  # the 23h-old item survived
    assert kept[1].published is None  # undated items are kept


def test_norm_url():
    assert _norm_url("https://Ex.com/Path/?utm_source=a&x=1#f") == "https://ex.com/Path?x=1"


def _cfg(tickers=("NVDA",), feeds=("https://feeds.example.com/m",), **news_kwargs):
    news_kwargs.setdefault("require_ticker_mention", False)  # legacy tests use generic titles
    news_kwargs.setdefault("market_relevance_filter", False)  # pool sizing tested explicitly
    return AppConfig(
        tickers=list(tickers),
        news=NewsConfig(market_feeds=list(feeds), **news_kwargs),
    )


def test_collect_all_isolates_source_failures(monkeypatch):
    monkeypatch.setattr(aggregator.sources, "fetch_market_feed", lambda u, t: [make_item(title="M", url="https://m.com/1", hours_ago=1)])
    monkeypatch.setattr(aggregator.sources, "fetch_yfinance_news", lambda t: [make_item(title="Y", url="https://y.com/1", hours_ago=2)])
    monkeypatch.setattr(aggregator.sources, "fetch_google_news_rss", lambda t, lh, to: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(aggregator.sources, "fetch_yahoo_ticker_rss", lambda t, to: [])
    results = collect_all(_cfg())
    assert [r.topic for r in results] == [MARKET_TOPIC, "NVDA"]
    assert results[0].error is None and [i.title for i in results[0].items] == ["M"]
    assert results[1].error is None and [i.title for i in results[1].items] == ["Y"]


def test_collect_all_all_sources_failed_sets_error(monkeypatch):
    # regression: total outage used to be indistinguishable from a quiet news day
    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    for name in ("fetch_market_feed", "fetch_yfinance_news", "fetch_google_news_rss", "fetch_yahoo_ticker_rss"):
        monkeypatch.setattr(aggregator.sources, name, fail)
    results = collect_all(_cfg())
    assert all(r.error and "fetch failed" in r.error for r in results)
    assert all(not r.items for r in results)


def test_collect_all_filters_before_dedupe(monkeypatch):
    # regression: a stale (30h) yfinance copy used to shadow a fresh (5h) copy
    # of the same headline, and the story then vanished entirely
    stale = make_item(title="Apple announces X", url="https://y.com/old", hours_ago=30, origin="yfinance")
    fresh = make_item(title="Apple announces X", url="https://g.com/new", hours_ago=5, origin="google_rss")
    monkeypatch.setattr(aggregator.sources, "fetch_market_feed", lambda u, t: [])
    monkeypatch.setattr(aggregator.sources, "fetch_yfinance_news", lambda t: [stale])
    monkeypatch.setattr(aggregator.sources, "fetch_google_news_rss", lambda t, lh, to: [fresh])
    monkeypatch.setattr(aggregator.sources, "fetch_yahoo_ticker_rss", lambda t, to: [])
    results = collect_all(_cfg())
    ticker_items = results[1].items
    assert [i.url for i in ticker_items] == ["https://g.com/new"]


def test_collect_all_sorts_newest_first_and_caps(monkeypatch):
    items = [  # distinct sources so the market source-diversity cap stays out of the way
        make_item(title="old", url="https://e.com/1", hours_ago=3, source="A"),
        make_item(title="undated", url="https://e.com/2", published=None, source="B"),
        make_item(title="newest", url="https://e.com/3", hours_ago=1, source="C"),
        make_item(title="mid", url="https://e.com/4", hours_ago=2, source="D"),
    ]
    monkeypatch.setattr(aggregator.sources, "fetch_market_feed", lambda u, t: list(items))
    results = collect_all(_cfg(tickers=(), max_articles_per_topic=3))
    market = results[0]
    assert [i.title for i in market.items] == ["newest", "mid", "old"]  # undated sorts last, then capped


def test_relevance_pattern():
    pattern = build_relevance_pattern("NVDA", ["Nvidia"])
    assert pattern.search("New Buy Rating for Nvidia (NVDA), the Technology Giant")
    assert pattern.search("NVIDIA Corp beats estimates")  # alias is case-insensitive
    assert pattern.search("Chips rally as $NVDA soars")
    assert not pattern.search("CoreWeave Stock Sank 11% After Meta Unveiled a Cloud Business Plan")
    assert not pattern.search("Goldman Sachs Says Optical Networking Is AI's Next Opportunity")
    assert not pattern.search("nvda in lowercase is not a headline mention")  # symbol stays case-sensitive


def test_relevance_pattern_dashed_symbol_matches_dot_spelling():
    pattern = build_relevance_pattern("BRK-B", [])
    assert pattern.search("BRK.B edges higher") and pattern.search("BRK-B edges higher")


def test_relevance_pattern_suffixed_symbol_matches_base():
    # regression: exchange-suffixed symbols never appear verbatim in headlines
    pattern = build_relevance_pattern("U96.SI", ["Sembcorp"])
    assert pattern.search("Sembcorp Industries (SGX:U96) goes ex-dividend")
    assert pattern.search("Sembcorp concludes Alinta Energy acquisition")
    pattern = build_relevance_pattern("D05.SI", [])
    assert pattern.search("$DBS (D05.SG)$ is the only stock I never fear")  # bare base D05
    assert pattern.search("D05.SI hits a record high")


def test_filter_relevant_checks_title_and_summary():
    pattern = build_relevance_pattern("NVDA", ["Nvidia"])
    items = [
        make_item(title="Chip sector roundup", summary="Analysts weigh in on Nvidia results"),
        make_item(title="Dividend King stock is a screaming buy", summary="Above-average yield"),
    ]
    kept = filter_relevant(items, pattern)
    assert [i.title for i in kept] == ["Chip sector roundup"]


def test_collect_all_relevance_filter_applies_to_tickers_only(monkeypatch):
    # regression: unrelated "related news" items used to pollute ticker sections
    ticker_items = [
        make_item(title="New Buy Rating for Nvidia", url="https://e.com/1", hours_ago=1),
        make_item(title="CoreWeave Stock Sank 11% After Meta Unveiled a Cloud Plan", url="https://e.com/2", hours_ago=2),
        make_item(title="Chip roundup", url="https://e.com/3", hours_ago=3, summary="Commentary on NVDA earnings"),
    ]
    monkeypatch.setattr(aggregator.sources, "fetch_company_aliases", lambda t: ["Nvidia"])
    monkeypatch.setattr(
        aggregator.sources,
        "fetch_market_feed",
        lambda u, t: [make_item(title="General market news", url="https://m.com/1", hours_ago=1)],
    )
    monkeypatch.setattr(aggregator.sources, "fetch_yfinance_news", lambda t: list(ticker_items))
    monkeypatch.setattr(aggregator.sources, "fetch_google_news_rss", lambda t, lh, to: [])
    monkeypatch.setattr(aggregator.sources, "fetch_yahoo_ticker_rss", lambda t, to: [])
    results = collect_all(_cfg(require_ticker_mention=True))
    assert [i.url for i in results[1].items] == ["https://e.com/1", "https://e.com/3"]
    assert [i.title for i in results[0].items] == ["General market news"]  # market is not filtered


def test_collect_all_alias_failure_falls_back_to_symbol(monkeypatch):
    def alias_fail(t):
        raise RuntimeError("info endpoint blocked")

    monkeypatch.setattr(aggregator.sources, "fetch_company_aliases", alias_fail)
    monkeypatch.setattr(aggregator.sources, "fetch_market_feed", lambda u, t: [])
    monkeypatch.setattr(
        aggregator.sources,
        "fetch_yfinance_news",
        lambda t: [
            make_item(title="NVDA hits a record", url="https://e.com/1", hours_ago=1),
            make_item(title="Nvidia-only mention", url="https://e.com/2", hours_ago=2),
        ],
    )
    monkeypatch.setattr(aggregator.sources, "fetch_google_news_rss", lambda t, lh, to: [])
    monkeypatch.setattr(aggregator.sources, "fetch_yahoo_ticker_rss", lambda t, to: [])
    results = collect_all(_cfg(require_ticker_mention=True))
    assert [i.url for i in results[1].items] == ["https://e.com/1"]  # symbol match still works


def _patch_ticker_sources(monkeypatch, items):
    monkeypatch.setattr(aggregator.sources, "fetch_market_feed", lambda u, t: [])
    monkeypatch.setattr(aggregator.sources, "fetch_yfinance_news", lambda t: list(items))
    monkeypatch.setattr(aggregator.sources, "fetch_google_news_rss", lambda t, lh, to: [])
    monkeypatch.setattr(aggregator.sources, "fetch_yahoo_ticker_rss", lambda t, to: [])


def test_collect_all_quiet_ticker_falls_back_to_most_recent(monkeypatch):
    # no 24h news -> the 2 most recent items are shown, however old
    _patch_ticker_sources(
        monkeypatch,
        [
            make_item(title="Ancient story", url="https://e.com/1", hours_ago=2000),
            make_item(title="Newest old story", url="https://e.com/2", hours_ago=50),
            make_item(title="Older story", url="https://e.com/3", hours_ago=90),
        ],
    )
    results = collect_all(_cfg())
    ticker = results[1]
    assert ticker.is_fallback is True
    assert [i.url for i in ticker.items] == ["https://e.com/2", "https://e.com/3"]


def test_collect_all_no_fallback_when_fresh_news_exists(monkeypatch):
    _patch_ticker_sources(monkeypatch, [make_item(title="Fresh", url="https://e.com/1", hours_ago=2)])
    ticker = collect_all(_cfg())[1]
    assert ticker.is_fallback is False
    assert [i.title for i in ticker.items] == ["Fresh"]


def test_collect_all_fallback_disabled(monkeypatch):
    _patch_ticker_sources(monkeypatch, [make_item(title="Old", url="https://e.com/1", hours_ago=50)])
    ticker = collect_all(_cfg(fallback_max_articles=0))[1]
    assert ticker.is_fallback is False
    assert ticker.items == []


def test_collect_all_market_never_falls_back(monkeypatch):
    monkeypatch.setattr(
        aggregator.sources,
        "fetch_market_feed",
        lambda u, t: [make_item(title="Old market piece", url="https://m.com/1", hours_ago=50)],
    )
    market = collect_all(_cfg(tickers=()))[0]
    assert market.is_fallback is False
    assert market.items == []


def test_diversify_by_source_caps_publisher_bursts():
    burst = [make_item(title=f"Listicle {i}", url=f"https://e.com/{i}", source="Insider Monkey") for i in range(10)]
    other = [make_item(title="Fed news", url="https://e.com/fed", source="Reuters")]
    kept = aggregator.diversify_by_source(burst + other, per_source_cap=3)
    assert len([i for i in kept if i.source == "Insider Monkey"]) == 3
    assert [i.title for i in kept][:3] == ["Listicle 0", "Listicle 1", "Listicle 2"]  # order preserved
    assert any(i.source == "Reuters" for i in kept)


def test_collect_all_market_pool_is_source_diverse(monkeypatch):
    # regression: a syndication burst from one publisher filled the whole pool
    burst = [
        make_item(title=f"Burst {i}", url=f"https://m.com/b{i}", hours_ago=1, source="Insider Monkey")
        for i in range(25)
    ]
    others = [
        make_item(title=f"Real {i}", url=f"https://m.com/r{i}", hours_ago=2 + i, source=f"Outlet {i}")
        for i in range(5)
    ]
    monkeypatch.setattr(aggregator.sources, "fetch_market_feed", lambda u, t: burst + others)
    market = collect_all(_cfg(tickers=(), market_relevance_filter=True, market_source_cap=3))[0]
    assert len([i for i in market.items if i.source == "Insider Monkey"]) == 3
    assert all(f"Real {i}" in [x.title for x in market.items] for i in range(5))


def test_collect_all_market_keeps_candidate_pool(monkeypatch):
    items = [
        make_item(title=f"T{i}", url=f"https://m.com/{i}", hours_ago=1, source=f"Outlet {i % 10}")
        for i in range(30)
    ]
    monkeypatch.setattr(aggregator.sources, "fetch_market_feed", lambda u, t: list(items))
    with_ranking = collect_all(_cfg(tickers=(), market_relevance_filter=True, market_candidate_pool=12))[0]
    assert len(with_ranking.items) == 12  # configured pool for the LLM to rank
    without_ranking = collect_all(_cfg(tickers=(), market_relevance_filter=False))[0]
    assert len(without_ranking.items) == 5  # straight to the section cap


def test_collect_all_malformed_feed_url_does_not_crash(monkeypatch):
    monkeypatch.setattr(aggregator.sources, "fetch_yfinance_news", lambda t: [])
    monkeypatch.setattr(aggregator.sources, "fetch_google_news_rss", lambda t, lh, to: [])
    monkeypatch.setattr(aggregator.sources, "fetch_yahoo_ticker_rss", lambda t, to: [])
    results = collect_all(_cfg(feeds=("https://[bad-bracket/rss",)))
    assert results[0].topic == MARKET_TOPIC  # no ValueError from urlparse at task-build time
