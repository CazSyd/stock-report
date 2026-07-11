import re
from types import SimpleNamespace

import pytest
from conftest import make_item

import stockreport.summarizer as summarizer
from stockreport.config import NewsConfig, OllamaConfig
from stockreport.models import MARKET_LABEL, MARKET_TOPIC, TopicResult
from stockreport.summarizer import (
    PreflightError,
    _model_names,
    build_user_message,
    preflight,
    summarize_topic,
)


def _result(topic="NVDA", n_items=3, summary_chars=100):
    items = [
        make_item(
            title=f"Headline {i}",
            url=f"https://e.com/{i}",
            hours_ago=i + 1,
            summary="x" * summary_chars,
        )
        for i in range(n_items)
    ]
    label = MARKET_LABEL if topic == MARKET_TOPIC else topic
    return TopicResult(topic=topic, label=label, items=items)


def test_build_user_message_subject_and_count():
    msg, dropped = build_user_message(_result(), NewsConfig(), OllamaConfig(), "2026-07-11")
    assert "Topic: the stock NVDA" in msg
    assert "(3 items)" in msg
    assert dropped == 0
    assert "[1] Headline 0" in msg and "[3] Headline 2" in msg

    msg, _ = build_user_message(_result(topic=MARKET_TOPIC), NewsConfig(), OllamaConfig(), "2026-07-11")
    assert "Topic: the overall stock market" in msg


def test_build_user_message_fallback_header():
    result = _result()
    result.is_fallback = True
    msg, _ = build_user_message(result, NewsConfig(), OllamaConfig(), "2026-07-11")
    assert "The 3 most recent news items found" in msg
    assert "none were published within the last 24 hours" in msg


def test_build_user_message_drops_to_fit_budget():
    ollama_cfg = OllamaConfig(options={"num_ctx": 1300})  # budget floors at MIN_BUDGET_CHARS
    msg, dropped = build_user_message(_result(n_items=10, summary_chars=600), NewsConfig(), ollama_cfg, "2026-07-11")
    included = 10 - dropped
    assert 0 < dropped < 10
    assert f"({included} items)" in msg  # header reflects what was actually included
    assert f"[{included}]" in msg and f"[{included + 1}]" not in msg


def test_build_user_message_truncates_single_oversized_item():
    # regression: the first block used to be exempt from the budget entirely
    ollama_cfg = OllamaConfig(options={"num_ctx": 1300})
    news_cfg = NewsConfig(max_chars_per_article=50_000)
    msg, dropped = build_user_message(_result(n_items=2, summary_chars=20_000), news_cfg, ollama_cfg, "2026-07-11")
    assert dropped == 1
    assert len(msg) <= summarizer.MIN_BUDGET_CHARS + 200  # header + truncated block


def test_build_user_message_truncates_snippets():
    news_cfg = NewsConfig(max_chars_per_article=10)
    msg, _ = build_user_message(_result(n_items=1, summary_chars=100), news_cfg, OllamaConfig(), "2026-07-11")
    assert "x" * 10 in msg and "x" * 11 not in msg


def test_model_names_handles_both_shapes():
    assert _model_names({"models": [{"name": "a:latest"}, {"model": "b:7b"}]}) == {"a:latest", "b:7b"}
    listing = SimpleNamespace(models=[SimpleNamespace(model="c:12b")])
    assert _model_names(listing) == {"c:12b"}


class _FakeClient:
    listing = {"models": [{"name": "gemma3:12b"}, {"name": "tiny:latest"}]}

    def __init__(self, host=None, timeout=None):
        pass

    def list(self):
        if isinstance(self.listing, Exception):
            raise self.listing
        return self.listing


def test_preflight_ok_exact_and_latest(monkeypatch):
    monkeypatch.setattr(summarizer.ollama, "Client", _FakeClient)
    preflight(OllamaConfig(model="gemma3:12b"))  # exact
    preflight(OllamaConfig(model="tiny"))  # resolves to tiny:latest


def test_preflight_missing_model(monkeypatch):
    monkeypatch.setattr(summarizer.ollama, "Client", _FakeClient)
    with pytest.raises(PreflightError, match=r"ollama pull nosuch"):
        preflight(OllamaConfig(model="nosuch"))


def test_preflight_unreachable(monkeypatch):
    class DownClient(_FakeClient):
        listing = ConnectionError("refused")

    monkeypatch.setattr(summarizer.ollama, "Client", DownClient)
    with pytest.raises(PreflightError, match="not reachable"):
        preflight(OllamaConfig())


def test_summarize_topic_dict_response_and_think_stripping():
    class Chat:
        def chat(self, **kwargs):
            return {"message": {"content": "<think>internal\nreasoning</think>\n- Real summary"}}

    text = summarize_topic(Chat(), OllamaConfig(), "msg")
    assert text == "- Real summary"


def test_summarize_topic_object_response():
    class Chat:
        def chat(self, **kwargs):
            return SimpleNamespace(message=SimpleNamespace(content="  text  "))

    assert summarize_topic(Chat(), OllamaConfig(), "msg") == "text"


def test_summarize_topic_passes_fresh_context_per_call():
    captured = []

    class Chat:
        def chat(self, **kwargs):
            captured.append(kwargs["messages"])
            return {"message": {"content": "ok"}}

    client = Chat()
    summarize_topic(client, OllamaConfig(), "first topic")
    summarize_topic(client, OllamaConfig(), "second topic")
    # each topic is an independent two-message conversation (system + user)
    assert all(len(m) == 2 and m[0]["role"] == "system" for m in captured)
    assert captured[0][1]["content"] == "first topic"
    assert captured[1][1]["content"] == "second topic"
