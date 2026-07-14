import pytest

from stockreport.config import DEFAULT_MARKET_FEEDS, ConfigError, load_config, normalize_tickers


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_minimal_config_gets_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "tickers: [aapl, MSFT, aapl]\n"))
    assert cfg.tickers == ["AAPL", "MSFT"]  # uppercased and deduped
    assert cfg.openrouter.model.endswith(":free")
    assert cfg.openrouter.temperature == 0.3
    assert cfg.openrouter.max_tokens == 4000
    assert cfg.openrouter.context_tokens == 32768
    assert cfg.news.market_feeds == DEFAULT_MARKET_FEEDS
    assert cfg.news.max_articles_per_topic == 5
    assert cfg.news.require_ticker_mention is True
    assert cfg.news.fallback_max_articles == 2
    assert cfg.news.market_relevance_filter is True
    assert cfg.news.market_candidate_pool == 30
    assert cfg.news.market_source_cap == 2
    assert cfg.output_dir == "reports"
    assert cfg.base_dir == tmp_path.resolve()


def test_openrouter_partial_section_keeps_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "tickers: [A]\nopenrouter:\n  temperature: 0.7\n"))
    assert cfg.openrouter.temperature == 0.7
    assert cfg.openrouter.context_tokens == 32768


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path):
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(write(tmp_path, "tickers: [unclosed\n"))


def test_non_mapping_root(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- a\n- list\n"))


def test_empty_tickers(tmp_path):
    with pytest.raises(ConfigError, match="tickers"):
        load_config(write(tmp_path, "tickers: []\n"))


def test_market_is_reserved(tmp_path):
    with pytest.raises(ConfigError, match="reserved"):
        load_config(write(tmp_path, "tickers: [AAPL, market]\n"))


def test_fractional_number_rejected(tmp_path):
    # regression: 0.5 used to pass validation and truncate to 0
    with pytest.raises(ConfigError, match="lookback_hours"):
        load_config(write(tmp_path, "tickers: [A]\nnews:\n  lookback_hours: 0.5\n"))
    with pytest.raises(ConfigError, match="max_articles_per_topic"):
        load_config(write(tmp_path, "tickers: [A]\nnews:\n  max_articles_per_topic: 1.9\n"))


def test_bool_rejected_as_number(tmp_path):
    with pytest.raises(ConfigError, match="timeout_seconds"):
        load_config(write(tmp_path, "tickers: [A]\nopenrouter:\n  timeout_seconds: true\n"))


def test_paid_model_rejected(tmp_path):
    # this project only uses free OpenRouter models
    with pytest.raises(ConfigError, match="free"):
        load_config(write(tmp_path, 'tickers: [A]\nopenrouter:\n  model: "openai/gpt-5.2"\n'))


def test_free_model_accepted(tmp_path):
    cfg = load_config(
        write(tmp_path, 'tickers: [A]\nopenrouter:\n  model: "meta-llama/llama-3.3-70b-instruct:free"\n')
    )
    assert cfg.openrouter.model == "meta-llama/llama-3.3-70b-instruct:free"


def test_fallback_models_must_be_free(tmp_path):
    with pytest.raises(ConfigError, match="free"):
        load_config(
            write(tmp_path, 'tickers: [A]\nopenrouter:\n  fallback_models: ["openai/gpt-5.2"]\n')
        )


def test_dynamic_fallback_default_and_validation(tmp_path):
    cfg = load_config(write(tmp_path, "tickers: [A]\n"))
    assert cfg.openrouter.dynamic_fallback is True
    with pytest.raises(ConfigError, match="dynamic_fallback"):
        load_config(write(tmp_path, "tickers: [A]\nopenrouter:\n  dynamic_fallback: sometimes\n"))


def test_fallback_models_default_and_empty(tmp_path):
    cfg = load_config(write(tmp_path, "tickers: [A]\n"))
    assert cfg.openrouter.fallback_models  # non-empty default chain
    assert all(m.endswith(":free") for m in cfg.openrouter.fallback_models)
    cfg = load_config(write(tmp_path, "tickers: [A]\nopenrouter:\n  fallback_models: []\n"))
    assert cfg.openrouter.fallback_models == []


def test_temperature_validated(tmp_path):
    with pytest.raises(ConfigError, match="temperature"):
        load_config(write(tmp_path, "tickers: [A]\nopenrouter:\n  temperature: 3.5\n"))
    with pytest.raises(ConfigError, match="temperature"):
        load_config(write(tmp_path, 'tickers: [A]\nopenrouter:\n  temperature: "hot"\n'))


def test_fallback_max_articles_zero_disables(tmp_path):
    cfg = load_config(write(tmp_path, "tickers: [A]\nnews:\n  fallback_max_articles: 0\n"))
    assert cfg.news.fallback_max_articles == 0
    with pytest.raises(ConfigError, match="fallback_max_articles"):
        load_config(write(tmp_path, "tickers: [A]\nnews:\n  fallback_max_articles: -1\n"))


def test_require_ticker_mention_must_be_bool(tmp_path):
    with pytest.raises(ConfigError, match="require_ticker_mention"):
        load_config(write(tmp_path, "tickers: [A]\nnews:\n  require_ticker_mention: maybe\n"))


def test_market_relevance_filter_must_be_bool(tmp_path):
    with pytest.raises(ConfigError, match="market_relevance_filter"):
        load_config(write(tmp_path, "tickers: [A]\nnews:\n  market_relevance_filter: 5\n"))


def test_market_pool_settings_validated(tmp_path):
    cfg = load_config(write(tmp_path, "tickers: [A]\nnews:\n  market_candidate_pool: 15\n  market_source_cap: 5\n"))
    assert cfg.news.market_candidate_pool == 15
    assert cfg.news.market_source_cap == 5
    with pytest.raises(ConfigError, match="market_candidate_pool"):
        load_config(write(tmp_path, "tickers: [A]\nnews:\n  market_candidate_pool: 0\n"))
    with pytest.raises(ConfigError, match="market_source_cap"):
        load_config(write(tmp_path, "tickers: [A]\nnews:\n  market_source_cap: 0\n"))


def test_market_feeds_must_be_list(tmp_path):
    with pytest.raises(ConfigError, match="market_feeds"):
        load_config(write(tmp_path, "tickers: [A]\nnews:\n  market_feeds: notalist\n"))


def test_normalize_tickers_rejects_blank():
    with pytest.raises(ConfigError):
        normalize_tickers(["AAPL", "  "], "--tickers")


def test_normalize_tickers_dedupes_preserving_order():
    assert normalize_tickers(["nvda", "AAPL", "NVDA"], "x") == ["NVDA", "AAPL"]
