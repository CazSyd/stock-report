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

DEFAULT_OLLAMA_OPTIONS = {"temperature": 0.3, "num_ctx": 8192}


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "gemma3:12b"
    timeout_seconds: int = 300
    keep_alive: str | int | float = "10m"  # duration string, or plain seconds (Ollama accepts both)
    options: dict = field(default_factory=lambda: dict(DEFAULT_OLLAMA_OPTIONS))


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
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
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
    ollama = _load_ollama(_section(raw, "ollama"))
    news = _load_news(_section(raw, "news"))
    report = _section(raw, "report")
    output_dir = report.get("output_dir", "reports")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("'report.output_dir' must be a non-empty string")

    return AppConfig(
        tickers=tickers,
        ollama=ollama,
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


def _load_ollama(section: dict) -> OllamaConfig:
    cfg = OllamaConfig()
    host = section.get("host", cfg.host)
    model = section.get("model", cfg.model)
    if not isinstance(host, str) or not host.strip():
        raise ConfigError("'ollama.host' must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("'ollama.model' must be a non-empty string")
    keep_alive = section.get("keep_alive", cfg.keep_alive)
    if isinstance(keep_alive, bool) or not isinstance(keep_alive, (str, int, float)):
        raise ConfigError("'ollama.keep_alive' must be a duration string like \"10m\" or a number of seconds")
    options = dict(DEFAULT_OLLAMA_OPTIONS)
    user_options = section.get("options")
    if user_options is not None:
        if not isinstance(user_options, dict):
            raise ConfigError("'ollama.options' must be a mapping")
        options.update(user_options)
    _coerce_option(options, "num_ctx", int)
    _coerce_option(options, "temperature", float)
    return OllamaConfig(
        host=host.strip().rstrip("/"),
        model=model.strip(),
        timeout_seconds=_positive(section.get("timeout_seconds", cfg.timeout_seconds), "ollama.timeout_seconds"),
        keep_alive=keep_alive,
        options=options,
    )


def _coerce_option(options: dict, key: str, kind) -> None:
    """The Ollama server type-checks options, so a quoted number in YAML must be
    coerced here or every chat call fails with a 400 later."""
    if key not in options:
        return
    try:
        options[key] = kind(options[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'ollama.options.{key}' must be a {kind.__name__}, got {options[key]!r}") from exc
    if key == "num_ctx" and options[key] <= 0:
        raise ConfigError(f"'ollama.options.num_ctx' must be positive, got {options[key]!r}")


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
