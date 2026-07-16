import pytest
from conftest import make_item

import stockreport.summarizer as summarizer
from stockreport.config import NewsConfig, OpenRouterConfig
from stockreport.models import MARKET_LABEL, MARKET_TOPIC, TopicResult
from stockreport.summarizer import (
    API_KEY_ENV,
    OpenRouterClient,
    PreflightError,
    build_user_message,
    preflight,
    rank_market_items,
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


# ---------- prompt building ----------


def test_build_user_message_subject_and_count():
    msg, dropped = build_user_message(_result(), NewsConfig(), OpenRouterConfig(), "2026-07-11")
    assert "Topic: the stock NVDA" in msg
    assert "(3 items)" in msg
    assert dropped == 0
    assert "[1] Headline 0" in msg and "[3] Headline 2" in msg

    msg, _ = build_user_message(_result(topic=MARKET_TOPIC), NewsConfig(), OpenRouterConfig(), "2026-07-11")
    assert "Topic: the overall stock market" in msg


def test_build_user_message_fallback_header():
    result = _result()
    result.is_fallback = True
    msg, _ = build_user_message(result, NewsConfig(), OpenRouterConfig(), "2026-07-11")
    assert "The 3 most recent news items found" in msg
    assert "none were published within the last 24 hours" in msg


def test_build_user_message_drops_to_fit_budget():
    llm_cfg = OpenRouterConfig(context_tokens=1300)  # budget floors at MIN_BUDGET_CHARS
    msg, dropped = build_user_message(_result(n_items=10, summary_chars=600), NewsConfig(), llm_cfg, "2026-07-11")
    included = 10 - dropped
    assert 0 < dropped < 10
    assert f"({included} items)" in msg  # header reflects what was actually included
    assert f"[{included}]" in msg and f"[{included + 1}]" not in msg


def test_build_user_message_truncates_single_oversized_item():
    llm_cfg = OpenRouterConfig(context_tokens=1300)
    news_cfg = NewsConfig(max_chars_per_article=50_000)
    msg, dropped = build_user_message(_result(n_items=2, summary_chars=20_000), news_cfg, llm_cfg, "2026-07-11")
    assert dropped == 1
    assert len(msg) <= summarizer.MIN_BUDGET_CHARS + 200  # header + truncated block


def test_build_user_message_truncates_snippets():
    news_cfg = NewsConfig(max_chars_per_article=10)
    msg, _ = build_user_message(_result(n_items=1, summary_chars=100), news_cfg, OpenRouterConfig(), "2026-07-11")
    assert "x" * 10 in msg and "x" * 11 not in msg


# ---------- HTTP client ----------


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise summarizer.requests.HTTPError(f"HTTP {self.status_code}")


def _chat_response(content, finish_reason="stop"):
    return FakeResponse(200, {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]})


def test_client_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return _chat_response("  hello  ")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    cfg = OpenRouterConfig()
    client = OpenRouterClient(cfg, "sk-test")
    assert client.chat("sys", "usr") == ("hello", cfg.model)
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"].endswith(":free")
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert captured["json"]["temperature"] == 0.3
    assert captured["json"]["max_tokens"] == cfg.max_tokens
    assert captured["timeout"] == 120


def test_client_temperature_override(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json=json)
        return _chat_response("ok")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    OpenRouterClient(OpenRouterConfig(), "k").chat("s", "u", temperature=0.0)
    assert captured["json"]["temperature"] == 0.0


def test_client_retries_on_429_then_succeeds(monkeypatch):
    responses = [
        FakeResponse(429, {"error": {"message": "rate limited"}}, headers={"Retry-After": "0"}),
        _chat_response("recovered"),
    ]
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return responses[len(calls) - 1]

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig()
    assert OpenRouterClient(cfg, "k").chat("s", "u") == ("recovered", cfg.model)
    assert len(calls) == 2


def test_client_does_not_retry_client_errors(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(400, {"error": {"message": "bad model id"}})

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    cfg = OpenRouterConfig(fallback_models=[], dynamic_fallback=False)
    with pytest.raises(RuntimeError, match="bad model id"):
        OpenRouterClient(cfg, "k").chat("s", "u")
    assert len(calls) == 1  # 400 is not retried


def test_client_falls_back_to_next_model_when_saturated(monkeypatch):
    # regression: free endpoints 429 with "Provider returned error" when saturated;
    # the client must switch models instead of failing the topic
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "primary/model:free":
            return FakeResponse(429, {"error": {"message": "Provider returned error"}}, headers={"Retry-After": "0"})
        return _chat_response("from fallback")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig(model="primary/model:free", fallback_models=["backup/model:free"])
    assert OpenRouterClient(cfg, "k").chat("s", "u") == ("from fallback", "backup/model:free")
    # primary tried MAX_ATTEMPTS times, then the fallback once
    assert models_called == ["primary/model:free"] * summarizer.MAX_ATTEMPTS + ["backup/model:free"]


def test_client_falls_back_on_truncated_response(monkeypatch):
    # regression: a reasoning model that hit max_tokens returned its cut-off
    # thinking transcript as the "summary" (the LMT incident)
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "primary/model:free":
            return _chat_response("We need to produce a concise summary. Let's parse each item...", "length")
        return _chat_response("- Proper summary")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    cfg = OpenRouterConfig(model="primary/model:free", fallback_models=["backup/model:free"])
    assert OpenRouterClient(cfg, "k").chat("s", "u") == ("- Proper summary", "backup/model:free")
    # truncation is not a rate limit: the model is not marked saturated, just skipped this call
    assert models_called == ["primary/model:free", "backup/model:free"]


def test_client_raises_when_all_responses_truncated(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _chat_response("truncated thinking...", "length")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    cfg = OpenRouterConfig(model="a/b:free", fallback_models=[], dynamic_fallback=False)
    with pytest.raises(RuntimeError, match="token limit"):
        OpenRouterClient(cfg, "k").chat("s", "u")


def _embedded_error_response(
    code=502,
    message="Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (33/32)",
):
    # OpenRouter relays upstream provider failures as HTTP 200 + error body
    return FakeResponse(200, {"error": {"message": message, "code": code}})


def test_client_retries_embedded_error_body_then_succeeds(monkeypatch):
    # regression: a 200-with-error-body crashed the topic with "unexpected
    # OpenRouter response shape" instead of being retried like an HTTP 502
    responses = [_embedded_error_response(), _chat_response("recovered")]
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return responses[len(calls) - 1]

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig(dynamic_fallback=False)
    assert OpenRouterClient(cfg, "k").chat("s", "u") == ("recovered", cfg.model)
    assert len(calls) == 2


def test_client_falls_back_when_embedded_error_persists(monkeypatch):
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "primary/model:free":
            return _embedded_error_response()
        return _chat_response("from fallback")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig(
        model="primary/model:free", fallback_models=["backup/model:free"], dynamic_fallback=False
    )
    assert OpenRouterClient(cfg, "k").chat("s", "u") == ("from fallback", "backup/model:free")
    # the embedded 502 is retried like an HTTP 502, then the next model takes over
    assert models_called == ["primary/model:free"] * summarizer.MAX_ATTEMPTS + ["backup/model:free"]


def test_client_embedded_429_marks_model_saturated(monkeypatch):
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "primary/model:free":
            return _embedded_error_response(code=429, message="rate limited upstream")
        return _chat_response("ok")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig(
        model="primary/model:free", fallback_models=["backup/model:free"], dynamic_fallback=False
    )
    client = OpenRouterClient(cfg, "k")
    client.chat("s", "u")
    client.chat("s", "u")
    # only the first chat touched the primary; the second went straight to the backup
    assert models_called.count("primary/model:free") == summarizer.MAX_ATTEMPTS
    assert models_called[-1] == "backup/model:free"


@pytest.mark.parametrize(
    "payload, text",
    [
        ({"object": "chat.completion", "choices": []}, ""),  # JSON, but nothing usable in it
        (None, "<html>bad gateway</html>"),  # not JSON at all
    ],
)
def test_client_falls_back_on_unexpected_response_shape(monkeypatch, payload, text):
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "primary/model:free":
            return FakeResponse(200, payload, text=text)
        return _chat_response("from fallback")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    cfg = OpenRouterConfig(
        model="primary/model:free", fallback_models=["backup/model:free"], dynamic_fallback=False
    )
    assert OpenRouterClient(cfg, "k").chat("s", "u") == ("from fallback", "backup/model:free")
    # garbage is not a rate limit: no same-model retry, straight to the next model
    assert models_called == ["primary/model:free", "backup/model:free"]


def test_client_raises_when_every_model_returns_garbage(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse(200, {"object": "chat.completion"})

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    cfg = OpenRouterConfig(model="a/b:free", fallback_models=[], dynamic_fallback=False)
    with pytest.raises(RuntimeError, match="unexpected OpenRouter response shape"):
        OpenRouterClient(cfg, "k").chat("s", "u")


def test_client_falls_back_on_mid_generation_error(monkeypatch):
    # the provider can also die mid-generation: the error lands on the choice
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "primary/model:free":
            return FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": None},
                            "finish_reason": "error",
                            "error": {"message": "provider disconnected", "code": 502},
                        }
                    ]
                },
            )
        return _chat_response("from fallback")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    cfg = OpenRouterConfig(
        model="primary/model:free", fallback_models=["backup/model:free"], dynamic_fallback=False
    )
    assert OpenRouterClient(cfg, "k").chat("s", "u") == ("from fallback", "backup/model:free")
    assert models_called == ["primary/model:free", "backup/model:free"]


def test_client_falls_back_on_empty_content(monkeypatch):
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "primary/model:free":
            return _chat_response("   ")
        return _chat_response("real answer")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    cfg = OpenRouterConfig(
        model="primary/model:free", fallback_models=["backup/model:free"], dynamic_fallback=False
    )
    assert OpenRouterClient(cfg, "k").chat("s", "u") == ("real answer", "backup/model:free")
    assert models_called == ["primary/model:free", "backup/model:free"]


def test_client_skips_saturated_model_on_later_calls(monkeypatch):
    # once a model has 429'd out, later calls in the same run go straight to the fallback
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "primary/model:free":
            return FakeResponse(429, {"error": {"message": "Provider returned error"}}, headers={"Retry-After": "0"})
        return _chat_response("ok")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig(model="primary/model:free", fallback_models=["backup/model:free"])
    client = OpenRouterClient(cfg, "k")
    client.chat("s", "u")
    client.chat("s", "u")
    primary_calls = models_called.count("primary/model:free")
    assert primary_calls == summarizer.MAX_ATTEMPTS  # only the first chat touched the primary
    assert models_called[-1] == "backup/model:free"


def test_client_raises_when_all_models_saturated(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse(429, {"error": {"message": "Provider returned error"}}, headers={"Retry-After": "0"})

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig(model="a/b:free", fallback_models=["c/d:free"], dynamic_fallback=False)
    with pytest.raises(RuntimeError, match="429"):
        OpenRouterClient(cfg, "k").chat("s", "u")


def _catalog_response():
    return FakeResponse(
        200,
        {
            "data": [
                {"id": "huge/model:free", "context_length": 1_000_000},
                {"id": "nvidia/nemotron-3.5-content-safety:free", "context_length": 128_000},  # excluded
                {"id": "big/model:free", "context_length": 262_144},
                {"id": "tiny/model:free", "context_length": 8_192},  # below context_tokens
                {"id": "paid/model", "context_length": 128_000},  # not :free
            ]
        },
    )


def test_client_dynamic_fallback_discovers_live_models(monkeypatch):
    models_called = []

    def fake_post(url, json=None, headers=None, timeout=None):
        models_called.append(json["model"])
        if json["model"] == "huge/model:free":
            return _chat_response("from discovered model")
        return FakeResponse(429, {"error": {"message": "Provider returned error"}}, headers={"Retry-After": "0"})

    gets = []

    def fake_get(url, **kwargs):
        gets.append(url)
        return _catalog_response()

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.requests, "get", fake_get)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig(model="a/b:free", fallback_models=["c/d:free"])
    client = OpenRouterClient(cfg, "k")

    assert client.chat("s", "u") == ("from discovered model", "huge/model:free")
    # configured chain first, then discovered models (content-safety, tiny, paid excluded)
    assert models_called == ["a/b:free"] * 2 + ["c/d:free"] * 2 + ["huge/model:free"]
    assert len(gets) == 1

    # second call: chain is known-saturated, goes straight to the discovered model,
    # and the catalog is not fetched again
    client.chat("s", "u")
    assert models_called[-1] == "huge/model:free"
    assert models_called.count("a/b:free") == 2
    assert len(gets) == 1


def test_client_dynamic_fallback_catalog_failure_is_safe(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse(429, {"error": {"message": "Provider returned error"}}, headers={"Retry-After": "0"})

    def fake_get(url, **kwargs):
        raise summarizer.requests.ConnectionError("catalog down")

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.requests, "get", fake_get)
    monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
    cfg = OpenRouterConfig(model="a/b:free", fallback_models=[])
    with pytest.raises(RuntimeError, match="429"):
        OpenRouterClient(cfg, "k").chat("s", "u")


# ---------- preflight ----------


def test_preflight_missing_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(PreflightError, match=API_KEY_ENV):
        preflight(OpenRouterConfig())


def test_preflight_rejected_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-bad")
    monkeypatch.setattr(
        summarizer.requests, "get", lambda *a, **k: FakeResponse(401, {"error": {"message": "unauthorized"}})
    )
    with pytest.raises(PreflightError, match="rejected"):
        preflight(OpenRouterConfig())


def test_preflight_ok(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-good")
    monkeypatch.setattr(summarizer.requests, "get", lambda *a, **k: FakeResponse(200, {"data": {}}))
    preflight(OpenRouterConfig())  # no exception


def test_preflight_unreachable(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-good")

    def boom(*args, **kwargs):
        raise summarizer.requests.ConnectionError("dns failure")

    monkeypatch.setattr(summarizer.requests, "get", boom)
    with pytest.raises(PreflightError, match="Could not reach"):
        preflight(OpenRouterConfig())


# ---------- summarize / rank ----------


class FakeChat:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, system, user, temperature=None, max_tokens=None):
        self.calls.append(
            {"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.content, "fake/model:free"


def test_summarize_topic_strips_thinking_block():
    client = FakeChat("<think>internal\nreasoning</think>\n- Real summary")
    text, model = summarize_topic(client, "msg")
    assert text == "- Real summary"
    assert model == "fake/model:free"  # reported so the section can credit the real model
    assert client.calls[0]["system"] == summarizer.SYSTEM_PROMPT
    assert client.calls[0]["user"] == "msg"


def test_summarize_topic_fresh_context_per_call():
    client = FakeChat("ok")
    summarize_topic(client, "first topic")
    summarize_topic(client, "second topic")
    # each topic is an independent system+user exchange
    assert [c["user"] for c in client.calls] == ["first topic", "second topic"]
    assert all(c["system"] == summarizer.SYSTEM_PROMPT for c in client.calls)


def test_rank_market_items_preserves_model_ranking():
    client = FakeChat('{"ranked": [3, 1, "2", 99, 1]}')
    items = [make_item(title=f"T{i}", url=f"https://e.com/{i}") for i in range(4)]
    kept = rank_market_items(client, items)
    # coerced, deduped, out-of-range dropped; the model's order IS the ranking
    assert [i.title for i in kept] == ["T2", "T0", "T1"]
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["max_tokens"] == summarizer.RANK_MAX_TOKENS  # reasoning headroom


def test_rank_market_items_handles_json_code_fence():
    client = FakeChat('Here you go:\n```json\n{"ranked": [2]}\n```')
    items = [make_item(title=f"T{i}", url=f"https://e.com/{i}") for i in range(3)]
    assert [i.title for i in rank_market_items(client, items)] == ["T1"]


def test_rank_market_items_regex_fallback_on_bad_json():
    client = FakeChat("Most important is 4, then 2.")
    items = [make_item(title=f"T{i}", url=f"https://e.com/{i}") for i in range(4)]
    assert [i.title for i in rank_market_items(client, items)] == ["T3", "T1"]


def test_rank_market_items_empty_response():
    client = FakeChat('{"ranked": []}')
    items = [make_item(title="T0", url="https://e.com/0")]
    assert rank_market_items(client, items) == []


def test_rank_market_items_finds_ranked_fragment_in_prose():
    # reasoning models sometimes wrap the JSON in rambling text
    client = FakeChat("Let me think about each item. " * 20 + 'Final answer: {"ranked": [3, 1]}')
    items = [make_item(title=f"T{i}", url=f"https://e.com/{i}") for i in range(4)]
    assert [i.title for i in rank_market_items(client, items)] == ["T2", "T0"]


def test_rank_market_items_never_digit_harvests_essays():
    # regression: a leaked reasoning transcript mentions every item number ([1],
    # [2], ...) and the old digit-regex turned the screen into a no-op
    essay = " ".join(f"Item [{i}] discusses something about the news of the day." for i in range(1, 31))
    client = FakeChat(essay)
    items = [make_item(title=f"T{i}", url=f"https://e.com/{i}") for i in range(30)]
    assert rank_market_items(client, items) == []  # cli falls back to newest items
