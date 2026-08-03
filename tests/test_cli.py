from datetime import datetime

import pytest
from conftest import make_item

import stockreport.cli as cli
from stockreport.models import MARKET_LABEL, MARKET_TOPIC, TopicResult
from stockreport.summarizer import PreflightError


@pytest.fixture()
def config_file(tmp_path):
    reports = tmp_path / "out"
    path = tmp_path / "config.yaml"
    path.write_text(
        f"tickers: [AAPL]\nreport:\n  output_dir: {reports.as_posix()}\n", encoding="utf-8"
    )
    return path


def _canned_results():
    return [
        TopicResult(MARKET_TOPIC, MARKET_LABEL, items=[make_item(hours_ago=1, summary="market news")]),
        TopicResult("AAPL", "AAPL", items=[make_item(url="https://e.com/aapl", hours_ago=2, summary="aapl news")]),
    ]


def test_missing_config_exits_1(tmp_path):
    assert cli.main(["--config", str(tmp_path / "nope.yaml")]) == cli.EXIT_CONFIG


def test_bad_tickers_override_exits_1(config_file):
    assert cli.main(["--config", str(config_file), "--tickers", "MARKET"]) == cli.EXIT_CONFIG


def test_preflight_failure_exits_2(config_file, monkeypatch):
    def fail(cfg):
        raise PreflightError("down")

    monkeypatch.setattr(cli.summarizer, "preflight", fail)
    assert cli.main(["--config", str(config_file)]) == cli.EXIT_PREFLIGHT


def test_dry_run_writes_report_and_exits_0(config_file, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _canned_results())
    out = tmp_path / "report.md"
    assert cli.main(["--config", str(config_file), "--dry-run", "--output", str(out)]) == cli.EXIT_OK
    content = out.read_text(encoding="utf-8")
    assert "## Market Overview" in content and "## AAPL" in content
    assert "this prompt would be sent" in content


def test_market_only_drops_all_tickers(config_file, tmp_path, monkeypatch):
    seen = {}

    def capture(cfg):
        seen["tickers"] = cfg.tickers
        return [TopicResult(MARKET_TOPIC, MARKET_LABEL, items=[make_item(hours_ago=1)])]

    monkeypatch.setattr(cli, "collect_all", capture)
    out = tmp_path / "r.md"
    code = cli.main(["--config", str(config_file), "--dry-run", "--market-only", "--output", str(out)])
    assert code == cli.EXIT_OK
    assert seen["tickers"] == []  # no ticker topics are fetched at all
    content = out.read_text(encoding="utf-8")
    assert "## Market Overview" in content
    assert "## AAPL" not in content  # config ticker absent from the report


def test_market_only_overrides_tickers_flag(config_file, monkeypatch):
    seen = {}

    def capture(cfg):
        seen["tickers"] = cfg.tickers
        return []

    monkeypatch.setattr(cli, "collect_all", capture)
    cli.main(["--config", str(config_file), "--dry-run", "--market-only", "--tickers", "NVDA,MSFT"])
    assert seen["tickers"] == []


def test_tickers_override_dedupes(config_file, monkeypatch):
    seen = {}

    def capture(cfg):
        seen["tickers"] = cfg.tickers
        return []

    monkeypatch.setattr(cli, "collect_all", capture)
    cli.main(["--config", str(config_file), "--dry-run", "--tickers", "aapl,AAPL,msft"])
    assert seen["tickers"] == ["AAPL", "MSFT"]  # regression: duplicates used to survive


def test_summarize_failure_exits_3(config_file, monkeypatch):
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _canned_results())
    monkeypatch.setattr(cli.summarizer, "preflight", lambda cfg: None)
    monkeypatch.setattr(cli.summarizer, "make_client", lambda cfg: object())

    def boom(client, msg):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(cli.summarizer, "summarize_topic", boom)
    assert cli.main(["--config", str(config_file)]) == cli.EXIT_PARTIAL


def test_failed_run_does_not_clobber_existing_report(config_file, tmp_path, monkeypatch):
    # regression: an all-errors rerun used to overwrite the day's good report
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _canned_results())
    monkeypatch.setattr(cli.summarizer, "preflight", lambda cfg: None)
    monkeypatch.setattr(cli.summarizer, "make_client", lambda cfg: object())
    monkeypatch.setattr(
        cli.summarizer, "summarize_topic", lambda c, m: (_ for _ in ()).throw(RuntimeError("x"))
    )
    out_dir = tmp_path / "out"
    today = f"{datetime.now().astimezone():%Y-%m-%d}"
    good = out_dir / f"{today}.md"
    good.parent.mkdir(parents=True)
    good.write_text("precious good report", encoding="utf-8")

    assert cli.main(["--config", str(config_file)]) == cli.EXIT_PARTIAL
    assert good.read_text(encoding="utf-8") == "precious good report"
    partial = out_dir / f"{today}.partial.md"
    assert partial.exists()
    assert "Summary failed" in partial.read_text(encoding="utf-8")


def test_successful_run_exits_0(config_file, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _canned_results())
    monkeypatch.setattr(cli.summarizer, "preflight", lambda cfg: None)
    monkeypatch.setattr(cli.summarizer, "make_client", lambda cfg: object())
    monkeypatch.setattr(
        cli.summarizer, "summarize_topic", lambda c, m: ("- Bullet summary", "backup/model:free")
    )
    out = tmp_path / "full.md"
    assert cli.main(["--config", str(config_file), "--output", str(out)]) == cli.EXIT_OK
    content = out.read_text(encoding="utf-8")
    assert "- Bullet summary" in content
    assert "_Summarized by backup/model:free_" in content  # sections credit the answering model


def _market_results(n=8):
    items = [make_item(title=f"M{i}", url=f"https://m.com/{i}", hours_ago=1) for i in range(n)]
    return [TopicResult(MARKET_TOPIC, MARKET_LABEL, items=items)]


def _patch_llm(monkeypatch):
    monkeypatch.setattr(cli.summarizer, "preflight", lambda cfg: None)
    monkeypatch.setattr(cli.summarizer, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli.summarizer, "summarize_topic", lambda c, m: ("- summary", "fake/model:free"))


def test_market_ranking_filters_and_caps(config_file, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _market_results())
    _patch_llm(monkeypatch)
    # model says: M6 most important, then M2, then M4
    monkeypatch.setattr(
        cli.summarizer, "rank_market_items", lambda c, items: [items[6], items[2], items[4]]
    )
    out = tmp_path / "r.md"
    assert cli.main(["--config", str(config_file), "--output", str(out)]) == cli.EXIT_OK
    content = out.read_text(encoding="utf-8")
    assert content.index("M6") < content.index("M2") < content.index("M4")  # importance order kept
    assert "M0" not in content and "M7" not in content  # screened out


def test_market_ranking_failure_falls_back_to_newest(config_file, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _market_results())
    _patch_llm(monkeypatch)

    def boom(c, items):
        raise RuntimeError("model endpoint unavailable")

    monkeypatch.setattr(cli.summarizer, "rank_market_items", boom)
    out = tmp_path / "r.md"
    assert cli.main(["--config", str(config_file), "--output", str(out)]) == cli.EXIT_OK
    content = out.read_text(encoding="utf-8")
    assert "M0" in content and "M4" in content  # newest 5 kept
    assert "M5" not in content


def test_dry_run_never_calls_ranking(config_file, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _market_results())

    def must_not_run(c, items):
        raise AssertionError("ranking must not run in dry-run mode")

    monkeypatch.setattr(cli.summarizer, "rank_market_items", must_not_run)
    out = tmp_path / "r.md"
    assert cli.main(["--config", str(config_file), "--dry-run", "--output", str(out)]) == cli.EXIT_OK


def test_telegram_env_missing_exits_2_before_fetching(config_file, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)  # ignore the developer's real .env
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    def must_not_fetch(cfg):
        raise AssertionError("news must not be fetched when telegram env is missing")

    monkeypatch.setattr(cli, "collect_all", must_not_fetch)
    assert cli.main(["--config", str(config_file), "--dry-run", "--telegram"]) == cli.EXIT_PREFLIGHT


def test_telegram_delivery_sends_headline_digest(config_file, tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/stock-report")  # as on GitHub Actions
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _canned_results())
    sent = {}
    monkeypatch.setattr(cli.telegram, "send_report", lambda md: sent.update(md=md) or 2)
    out = tmp_path / "r.md"
    code = cli.main(["--config", str(config_file), "--dry-run", "--telegram", "--output", str(out)])
    assert code == cli.EXIT_OK
    assert "## Market Overview" in sent["md"]  # the headline news is pushed...
    assert "## AAPL" not in sent["md"]  # ...the ticker sections are not
    assert "(https://github.com/someone/stock-report/releases/tag/latest)" in sent["md"]
    assert "## AAPL" in out.read_text(encoding="utf-8")  # the report itself still has everything


def test_telegram_digest_without_repo_env_omits_link(config_file, tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)  # local run: nothing public to link to
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _canned_results())
    sent = {}
    monkeypatch.setattr(cli.telegram, "send_report", lambda md: sent.update(md=md) or 1)
    out = tmp_path / "r.md"
    code = cli.main(["--config", str(config_file), "--dry-run", "--telegram", "--output", str(out)])
    assert code == cli.EXIT_OK
    assert "releases/tag/latest" not in sent["md"]
    assert "no public link for this run" in sent["md"]


def test_telegram_failure_exits_4(config_file, tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(cli, "collect_all", lambda cfg: _canned_results())

    def boom(md):
        raise cli.telegram.NotifyError("bot blocked by user")

    monkeypatch.setattr(cli.telegram, "send_report", boom)
    out = tmp_path / "r.md"
    code = cli.main(["--config", str(config_file), "--dry-run", "--telegram", "--output", str(out)])
    assert code == cli.EXIT_NOTIFY
    assert out.exists()  # the report itself was still written


def test_fetch_failure_topic_skips_llm_and_exits_3(config_file, tmp_path, monkeypatch):
    failed = TopicResult("AAPL", "AAPL", error="News fetch failed: all 3 sources errored")
    monkeypatch.setattr(cli, "collect_all", lambda cfg: [failed])
    monkeypatch.setattr(cli.summarizer, "preflight", lambda cfg: None)
    monkeypatch.setattr(cli.summarizer, "make_client", lambda cfg: object())
    called = []
    monkeypatch.setattr(cli.summarizer, "summarize_topic", lambda c, m: called.append(1))
    out = tmp_path / "r.md"
    assert cli.main(["--config", str(config_file), "--output", str(out)]) == cli.EXIT_PARTIAL
    assert not called  # no LLM call for a fetch-failed topic
    assert "News fetch failed" in out.read_text(encoding="utf-8")
