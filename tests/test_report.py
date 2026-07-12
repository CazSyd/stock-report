from datetime import datetime, timezone

from conftest import make_item

from stockreport.models import MARKET_LABEL, MARKET_TOPIC, TopicResult
from stockreport.report import _source_line, default_output_path, render_report, write_report

GENERATED = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)


def _render(results, dry_run=False):
    return render_report(results, model="test-model", generated_at=GENERATED, lookback_hours=24, dry_run=dry_run)


def test_render_sections_in_order():
    md = _render(
        [
            TopicResult(MARKET_TOPIC, MARKET_LABEL, items=[make_item()], summary_md="Market summary."),
            TopicResult("AAPL", "AAPL", items=[make_item(url="https://e.com/2")], summary_md="AAPL summary."),
        ]
    )
    assert md.index("## Market Overview") < md.index("## AAPL")
    assert "Market summary." in md and "AAPL summary." in md
    assert "**Sources**" in md


def test_render_no_news_notes():
    md = _render(
        [
            TopicResult(MARKET_TOPIC, MARKET_LABEL),
            TopicResult("ZZZZ", "ZZZZ"),
        ]
    )
    assert "_No news found in the last 24 hours._" in md
    assert "check that 'ZZZZ' is a valid ticker symbol" in md


def test_render_error_still_lists_sources():
    result = TopicResult("AAPL", "AAPL", items=[make_item()], error="Summary failed: ResponseError: boom")
    md = _render([result])
    assert "_Summary failed: ResponseError: boom_" in md
    assert "**Sources**" in md


def test_render_dry_run_uses_fence_longer_than_prompt_backticks():
    # regression: a prompt containing ```` used to break the fixed 4-backtick fence
    result = TopicResult("AAPL", "AAPL", items=[make_item()], prompt="line\n````\nmore")
    md = _render([result], dry_run=True)
    assert "`````text" in md and md.count("`````") == 2


def test_render_fallback_note():
    result = TopicResult("D05.SI", "D05.SI", items=[make_item()], summary_md="s", is_fallback=True)
    md = _render([result])
    assert "_No news in the last 24 hours; showing the 1 most recent item(s) found._" in md
    assert "\ns\n" in md  # summary still rendered after the note


def test_render_section_credits_answering_model():
    result = TopicResult(
        "AAPL", "AAPL", items=[make_item()], summary_md="s", model="nvidia/nemotron-3-super-120b-a12b:free"
    )
    md = _render([result])
    assert "_Summarized by nvidia/nemotron-3-super-120b-a12b:free_" in md
    assert md.index("Summarized by") < md.index("\ns\n")  # insert sits above the summary


def test_render_dropped_note():
    result = TopicResult("AAPL", "AAPL", items=[make_item()], summary_md="s", dropped=2)
    md = _render([result])
    assert "The last 2 source(s) listed above did not fit the model context" in md


def test_render_empty_summary_note():
    md = _render([TopicResult("AAPL", "AAPL", items=[make_item()], summary_md="")])
    assert "_The model returned an empty summary._" in md


def test_source_line_escapes_markdown_specials():
    # regression: backslashes/asterisks used to break the Sources links
    item = make_item(title="Retail piles into \"meme\\", url="https://e.com/a", hours_ago=1)
    line = _source_line(item)
    assert "meme\\\\" in line  # backslash escaped
    item2 = make_item(title="*UPDATE* Fed [holds] rates_now", url="https://e.com/b")
    line2 = _source_line(item2)
    assert r"\*UPDATE\* Fed \[holds\] rates\_now" in line2


def test_source_line_wraps_url_with_parens():
    item = make_item(title="T", url="https://e.com/a(b)")
    assert "(<https://e.com/a(b)>)" in _source_line(item)


def test_source_line_prints_utc():
    item = make_item(title="T", url="https://e.com/a", published=datetime(2026, 7, 10, 21, 30, tzinfo=timezone.utc))
    assert "2026-07-10 21:30 UTC" in _source_line(item)


def test_default_output_path_dry_run_suffix(tmp_path):
    assert default_output_path(tmp_path, GENERATED, dry_run=False).name == "2026-07-11.md"
    assert default_output_path(tmp_path, GENERATED, dry_run=True).name == "2026-07-11.dry-run.md"


def test_write_report_creates_dirs_and_utf8(tmp_path):
    target = tmp_path / "nested" / "deep" / "r.md"
    write_report("# Report — with em-dash and ‘quotes’\n", target)
    assert target.read_text(encoding="utf-8") == "# Report — with em-dash and ‘quotes’\n"
