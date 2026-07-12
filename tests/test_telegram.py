import pytest

import stockreport.telegram as tg


class FakeResponse:
    def __init__(self, status_code, payload=None, text="ok"):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


# ---------- markdown -> Telegram HTML ----------


def test_conversion_headings_meta_bullets_links():
    md = (
        "# Daily Stock Report - 2026-07-12\n"
        "\n"
        "_Generated 10:00 | Default model: x_\n"
        "\n"
        "## Market Overview\n"
        "\n"
        "_Summarized by a/b:free_\n"
        "\n"
        "- **Fed:** rates & more\n"
        "- [A \\[big\\] day](https://e.com/a_b?x=1&y=2) - Reuters\n"
    )
    out = tg.markdown_to_telegram_html(md)
    assert "<b>Daily Stock Report - 2026-07-12</b>" in out
    assert "<b>Market Overview</b>" in out
    assert "<i>Summarized by a/b:free</i>" in out
    assert "• <b>Fed:</b> rates &amp; more" in out
    assert '<a href="https://e.com/a_b?x=1&amp;y=2">A [big] day</a>' in out
    assert "\x00" not in out  # all link placeholders restored


def test_conversion_never_styles_inside_urls():
    md = "- [t](https://e.com/a_b_c) and [u](<https://e.com/d(e)_f>)"
    out = tg.markdown_to_telegram_html(md)
    assert "<i>" not in out and out.count("<a href=") == 2
    assert "a_b_c" in out  # url untouched


def test_conversion_escapes_html_specials():
    out = tg.markdown_to_telegram_html("AT&T <beats> estimates")
    assert out == "AT&amp;T &lt;beats&gt; estimates"


def _section(i, lines=40):
    # a report-like section ending in a Sources list with a unique marker link
    return (
        f"## Section {i}\n\nSummary takeaway for section {i}.\n"
        + f"- bullet content for section {i}\n" * lines
        + f"\n**Sources**\n\n- [end-{i}](https://e.com/{i}) - Wire\n"
    )


def test_chunking_respects_cap_and_never_loses_content():
    md = "# Title\n\n_meta line_\n\n" + "".join(_section(i, lines=80) for i in range(4))
    chunks = tg._chunk_report(md)
    assert len(chunks) > 1
    assert all(len(c) <= tg.MAX_MESSAGE_CHARS for c in chunks)
    sent_lines = [line for chunk in chunks for line in chunk.split("\n") if line.strip()]
    expected = [
        line for line in tg.markdown_to_telegram_html(md).split("\n") if line.strip()
    ]
    assert sent_lines == expected  # nothing lost or reordered


def test_sections_never_straddle_chunks():
    # regression: a section's Sources used to start in the next message
    md = "# Title\n\n_meta line_\n\n" + "".join(_section(i) for i in range(6))
    chunks = tg._chunk_report(md)
    assert len(chunks) > 1
    for i in range(6):
        containing = [c for c in chunks if f"<b>Section {i}</b>" in c]
        assert len(containing) == 1
        assert f"end-{i}" in containing[0]  # heading, summary, and sources travel together


def test_oversized_section_splits_at_its_sources_marker():
    md = _section(0, lines=200)  # single section well over the cap
    chunks = tg._chunk_report(md)
    assert len(chunks) >= 2
    assert all(len(c) <= tg.MAX_MESSAGE_CHARS for c in chunks)
    assert any(c.startswith("<b>Sources</b>") for c in chunks)  # split lands at the marker


# ---------- sending ----------


def test_send_report_posts_html_messages(monkeypatch):
    monkeypatch.setenv(tg.TOKEN_ENV, "tok123")
    monkeypatch.setenv(tg.CHAT_ENV, "42")
    posted = []

    def fake_post(url, json=None, timeout=None):
        posted.append((url, json))
        return FakeResponse(200)

    monkeypatch.setattr(tg.requests, "post", fake_post)
    monkeypatch.setattr(tg.time, "sleep", lambda s: None)
    count = tg.send_report("# Title\n\nhello world")
    assert count == 1 == len(posted)
    url, payload = posted[0]
    assert url == f"{tg.API_BASE}/bottok123/sendMessage"
    assert payload["chat_id"] == "42"
    assert payload["parse_mode"] == "HTML"
    assert payload["disable_web_page_preview"] is True
    assert "<b>Title</b>" in payload["text"]


def test_send_message_falls_back_to_plain_text_on_400(monkeypatch):
    posted = []

    def fake_post(url, json=None, timeout=None):
        posted.append(json)
        return FakeResponse(400 if len(posted) == 1 else 200, text="can't parse entities")

    monkeypatch.setattr(tg.requests, "post", fake_post)
    tg._send_message("tok", "42", "<b>Title</b>\nAT&amp;T rallies")
    assert len(posted) == 2
    assert "parse_mode" not in posted[1]
    assert posted[1]["text"] == "Title\nAT&T rallies"  # tags stripped, entities unescaped


def test_send_message_raises_on_persistent_error(monkeypatch):
    monkeypatch.setattr(tg.requests, "post", lambda *a, **k: FakeResponse(403, text="bot blocked"))
    with pytest.raises(tg.NotifyError, match="403"):
        tg._send_message("tok", "42", "text")


def test_require_env_missing(monkeypatch):
    monkeypatch.delenv(tg.TOKEN_ENV, raising=False)
    monkeypatch.delenv(tg.CHAT_ENV, raising=False)
    with pytest.raises(tg.NotifyError, match=tg.TOKEN_ENV):
        tg.require_env()
