"""Load and validate config.yaml into typed config objects."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import MARKET_TOPIC

log = logging.getLogger(__name__)


class ConfigError(Exception):
    """Config file missing or invalid; the message is user-facing."""


DEFAULT_MARKET_FEEDS = [
    "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
    "https://finance.yahoo.com/news/rssindex",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
]

# NVIDIA-hosted free endpoints have dedicated first-party capacity, so they
# tend to stay up when the shared community free pools are saturated.
DEFAULT_FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]


@dataclass
class OpenRouterConfig:
    model: str = "google/gemma-4-31b-it:free"  # only :free models are allowed
    timeout_seconds: int = 120
    temperature: float = 0.3
    max_tokens: int = 4000  # completion cap per topic; reasoning models spend thinking tokens from this too
    context_tokens: int = 32768  # prompt budgeting (current free models all offer >= 32k)
    # tried in order when the primary model is saturated (free tiers 429 often)
    fallback_models: list[str] = field(default_factory=lambda: list(DEFAULT_FALLBACK_MODELS))
    # last resort: discover whatever :free models are live right now and try those
    dynamic_fallback: bool = True


@dataclass
class NewsConfig:
    lookback_hours: int = 24
    max_articles_per_topic: int = 5
    max_chars_per_article: int = 1500
    request_timeout_seconds: int = 15
    require_ticker_mention: bool = True  # drop ticker-section items that never mention the ticker/company
    fallback_max_articles: int = 2  # quiet tickers: show this many most-recent items instead (0 disables)
    market_relevance_filter: bool = True  # LLM-screen market headlines for actual market relevance
    market_candidate_pool: int = 30  # market items offered to the LLM ranking (covers the whole day)
    market_source_cap: int = 2  # max market-pool items from any single publisher (stops feed bursts)
    market_feeds: list[str] = field(default_factory=lambda: list(DEFAULT_MARKET_FEEDS))


@dataclass
class AppConfig:
    tickers: list[str]
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    output_dir: str = "reports"
    base_dir: Path = field(default_factory=Path.cwd)  # anchor for a relative output_dir


def load_config(path: Path) -> AppConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping (key: value pairs)")

    tickers = _load_tickers(raw)
    openrouter = _load_openrouter(_section(raw, "openrouter"))
    news = _load_news(_section(raw, "news"))
    report = _section(raw, "report")
    output_dir = report.get("output_dir", "reports")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("'report.output_dir' must be a non-empty string")

    return AppConfig(
        tickers=tickers,
        openrouter=openrouter,
        news=news,
        output_dir=output_dir.strip(),
        base_dir=path.resolve().parent,
    )


def normalize_tickers(entries, source: str) -> list[str]:
    """Shared by config loading and the --tickers CLI override."""
    if not entries:
        raise ConfigError(f"No ticker symbols found in {source}")
    tickers: list[str] = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ConfigError(f"Invalid ticker entry in {source}: {entry!r}")
        symbol = entry.strip().upper()
        if symbol == MARKET_TOPIC:
            raise ConfigError(
                f"'{MARKET_TOPIC}' is reserved for the market overview section and cannot be used as a ticker"
            )
        if symbol not in tickers:
            tickers.append(symbol)
    return tickers


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a mapping (key: value pairs)")
    return value


def _load_tickers(raw: dict) -> list[str]:
    tickers_raw = raw.get("tickers")
    if not isinstance(tickers_raw, list) or not tickers_raw:
        raise ConfigError("'tickers' must be a non-empty list of ticker symbols")
    return normalize_tickers(tickers_raw, "config 'tickers'")


def _load_openrouter(section: dict) -> OpenRouterConfig:
    cfg = OpenRouterConfig()
    model = _free_model(section.get("model", cfg.model), "openrouter.model")
    fallbacks_raw = section.get("fallback_models", cfg.fallback_models)
    if not isinstance(fallbacks_raw, list):
        raise ConfigError("'openrouter.fallback_models' must be a list of model ids (may be empty)")
    fallback_models = [_free_model(m, "openrouter.fallback_models") for m in fallbacks_raw]
    dynamic_fallback = section.get("dynamic_fallback", cfg.dynamic_fallback)
    if not isinstance(dynamic_fallback, bool):
        raise ConfigError("'openrouter.dynamic_fallback' must be true or false")
    temperature = section.get("temperature", cfg.temperature)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ConfigError(f"'openrouter.temperature' must be a number between 0 and 2, got {temperature!r}")
    return OpenRouterConfig(
        model=model,
        timeout_seconds=_positive(
            section.get("timeout_seconds", cfg.timeout_seconds), "openrouter.timeout_seconds"
        ),
        temperature=float(temperature),
        max_tokens=_positive(section.get("max_tokens", cfg.max_tokens), "openrouter.max_tokens"),
        context_tokens=_positive(
            section.get("context_tokens", cfg.context_tokens), "openrouter.context_tokens"
        ),
        fallback_models=fallback_models,
        dynamic_fallback=dynamic_fallback,
    )


def _free_model(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string")
    model = value.strip()
    if not model.endswith(":free"):
        raise ConfigError(
            f"'{name}' must be a free model (ending in ':free'), got '{model}'. "
            "This project only uses free OpenRouter models; browse them at "
            "https://openrouter.ai/models?max_price=0"
        )
    return model


def _load_news(section: dict) -> NewsConfig:
    cfg = NewsConfig()
    feeds = section.get("market_feeds", cfg.market_feeds)
    if not isinstance(feeds, list) or not all(isinstance(f, str) and f.strip() for f in feeds):
        raise ConfigError("'news.market_feeds' must be a list of URLs")
    require_mention = section.get("require_ticker_mention", cfg.require_ticker_mention)
    if not isinstance(require_mention, bool):
        raise ConfigError("'news.require_ticker_mention' must be true or false")
    market_filter = section.get("market_relevance_filter", cfg.market_relevance_filter)
    if not isinstance(market_filter, bool):
        raise ConfigError("'news.market_relevance_filter' must be true or false")
    return NewsConfig(
        require_ticker_mention=require_mention,
        market_relevance_filter=market_filter,
        lookback_hours=_positive(section.get("lookback_hours", cfg.lookback_hours), "news.lookback_hours"),
        max_articles_per_topic=_positive(
            section.get("max_articles_per_topic", cfg.max_articles_per_topic), "news.max_articles_per_topic"
        ),
        max_chars_per_article=_positive(
            section.get("max_chars_per_article", cfg.max_chars_per_article), "news.max_chars_per_article"
        ),
        request_timeout_seconds=_positive(
            section.get("request_timeout_seconds", cfg.request_timeout_seconds), "news.request_timeout_seconds"
        ),
        fallback_max_articles=_whole(
            section.get("fallback_max_articles", cfg.fallback_max_articles),
            "news.fallback_max_articles",
            minimum=0,
        ),
        market_candidate_pool=_positive(
            section.get("market_candidate_pool", cfg.market_candidate_pool), "news.market_candidate_pool"
        ),
        market_source_cap=_positive(
            section.get("market_source_cap", cfg.market_source_cap), "news.market_source_cap"
        ),
        market_feeds=[f.strip() for f in feeds],
    )


def _positive(value, name: str) -> int:
    return _whole(value, name, minimum=1)


def _whole(value, name: str, minimum: int) -> int:
    invalid = (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not value.is_integer())
        or int(value) < minimum
    )
    if invalid:
        raise ConfigError(f"'{name}' must be a whole number >= {minimum}, got {value!r}")
    return int(value)
