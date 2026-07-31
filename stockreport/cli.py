"""Command-line entry point: python -m stockreport"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from . import __version__, summarizer, telegram
from .aggregator import collect_all
from .config import AppConfig, ConfigError, load_config, normalize_tickers
from .models import MARKET_TOPIC, TopicResult
from .report import default_output_path, render_report, render_telegram_digest, write_report
from .summarizer import PreflightError

log = logging.getLogger("stockreport")

EXIT_OK = 0
EXIT_CONFIG = 1
EXIT_PREFLIGHT = 2
EXIT_PARTIAL = 3
EXIT_NOTIFY = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockreport",
        description="Daily market + ticker news report summarized by a free OpenRouter model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: ./config.yaml, else the one next to this package)",
    )
    parser.add_argument("--tickers", help="Comma-separated ticker override, e.g. AAPL,MSFT")
    parser.add_argument("--model", help="Override openrouter.model from the config (must end in :free)")
    parser.add_argument("--output", type=Path, help="Explicit output file path for the report")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch news and build prompts but skip the LLM; writes a .dry-run.md report",
    )
    parser.add_argument(
        "--market-only",
        action="store_true",
        help="Skip all ticker sections and report only the market overview",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Deliver the report to Telegram (needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _resolve_config_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    cwd_config = Path("config.yaml")
    if cwd_config.is_file():
        return cwd_config
    return Path(__file__).resolve().parent.parent / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = _build_parser().parse_args(argv)
    load_dotenv()  # .env in the working directory, if any

    try:
        cfg = load_config(_resolve_config_path(args.config))
        if args.tickers:
            entries = [t for t in args.tickers.split(",") if t.strip()]
            cfg.tickers = normalize_tickers(entries, "--tickers")
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG
    if args.model:
        cfg.openrouter.model = args.model.strip()
    if args.market_only:
        cfg.tickers = []
    load_dotenv(cfg.base_dir / ".env")  # .env next to the config; real env vars win

    if args.telegram:
        try:
            telegram.require_env()  # fail fast, before any fetching or API spend
        except telegram.NotifyError as exc:
            log.error("%s", exc)
            return EXIT_PREFLIGHT

    if not args.dry_run:
        try:
            summarizer.preflight(cfg.openrouter)
        except PreflightError as exc:
            log.error("%s", exc)
            return EXIT_PREFLIGHT
        log.info("OpenRouter preflight OK (model %s)", cfg.openrouter.model)

    if cfg.tickers:
        log.info("Collecting news: market overview + %d tickers (%s)", len(cfg.tickers), ", ".join(cfg.tickers))
    else:
        log.info("Collecting news: market overview only")
    results = collect_all(cfg)

    generated_at = datetime.now().astimezone()
    report_date = f"{generated_at:%Y-%m-%d}"

    client = None if args.dry_run else summarizer.make_client(cfg.openrouter)
    for result in results:
        if result.error:
            log.warning("%s: %s", result.label, result.error)
            continue
        if not result.items:
            log.warning("%s: no news found, skipping summarization", result.label)
            continue
        if result.topic == MARKET_TOPIC and cfg.news.market_relevance_filter:
            result.items = _apply_market_triage(result, cfg, client, args.dry_run)
        result.prompt, result.dropped = summarizer.build_user_message(
            result, cfg.news, cfg.openrouter, report_date
        )
        if result.dropped:
            log.info(
                "%s: dropped %d of %d items to fit the model context",
                result.label,
                result.dropped,
                len(result.items),
            )
        if args.dry_run:
            continue
        log.info("Summarizing %s (%d items)...", result.label, len(result.items) - result.dropped)
        started = time.monotonic()
        try:
            result.summary_md, result.model = summarizer.summarize_topic(client, result.prompt)
        except Exception as exc:
            result.error = f"Summary failed: {exc.__class__.__name__}: {exc}"
            log.error("%s: %s", result.label, result.error)
        else:
            log.info("%s: summarized in %.0fs", result.label, time.monotonic() - started)

    if args.dry_run:
        log.info("Dry run: skipped the LLM; the report contains the prompts that would be sent")

    any_error = any(r.error for r in results)
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.base_dir / output_dir
    output_path = args.output or default_output_path(output_dir, generated_at, args.dry_run)
    if any_error and args.output is None and output_path.exists():
        # don't clobber an existing (possibly good) report with a failed run
        output_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
        log.warning("Some topics failed; writing to %s to keep the existing report intact", output_path.name)

    markdown = render_report(
        results, cfg.openrouter.model, generated_at, cfg.news.lookback_hours, dry_run=args.dry_run
    )
    try:
        write_report(markdown, output_path)
    except OSError as exc:
        log.error("Could not write the report to %s: %s", output_path, exc)
        return EXIT_CONFIG
    log.info("Report written to %s", output_path.resolve())

    if args.telegram:
        # push only the headline news; the ticker sections live in the report
        digest = render_telegram_digest(
            results, generated_at, cfg.news.lookback_hours, _report_url(), dry_run=args.dry_run
        )
        try:
            sent = telegram.send_report(digest)
        except Exception as exc:
            log.error("Telegram delivery failed: %s", exc)
            return EXIT_NOTIFY
        log.info("Headline digest delivered to Telegram in %d message(s)", sent)

    return EXIT_PARTIAL if any_error else EXIT_OK


def _report_url() -> str | None:
    """Public URL of the freshest full report: the rolling 'latest' release
    page, which the workflow refreshes right after this run. GITHUB_REPOSITORY
    is set by GitHub Actions; local runs have no public copy to link to."""
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    return f"https://github.com/{repo}/releases/tag/latest" if repo else None


def _apply_market_triage(result: TopicResult, cfg: AppConfig, client, dry_run: bool):
    """Ask the model to drop non-market candidates and rank the rest by
    importance, then keep the top max_articles_per_topic. Any failure falls
    back to the newest items so the section is never lost."""
    cap = cfg.news.max_articles_per_topic
    if dry_run:
        log.info("%s: relevance ranking skipped in dry run", result.label)
        return result.items[:cap]
    try:
        ranked = summarizer.rank_market_items(client, result.items)
    except Exception as exc:
        log.warning("%s: relevance ranking failed (%s); keeping the newest items", result.label, exc)
        return result.items[:cap]
    if not ranked:
        log.warning("%s: relevance ranking kept nothing; keeping the newest items", result.label)
        return result.items[:cap]
    log.info(
        "%s: model kept %d of %d candidates as market-relevant; showing the top %d by importance",
        result.label,
        len(ranked),
        len(result.items),
        min(cap, len(ranked)),
    )
    return ranked[:cap]
