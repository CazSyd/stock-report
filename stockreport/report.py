"""Assemble the markdown report and write it as UTF-8."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .models import MARKET_TOPIC, NewsItem, TopicResult

_MD_SPECIALS = re.compile(r"([\\\[\]`*_<>])")


def default_output_path(output_dir: str | Path, generated_at: datetime, dry_run: bool) -> Path:
    suffix = ".dry-run.md" if dry_run else ".md"
    return Path(output_dir) / f"{generated_at:%Y-%m-%d}{suffix}"


def render_report(
    results: list[TopicResult],
    model: str,
    generated_at: datetime,
    lookback_hours: int,
    dry_run: bool = False,
) -> str:
    lines: list[str] = [f"# Daily Stock Report - {generated_at:%Y-%m-%d}", ""]
    mode = " | DRY RUN (no model was called)" if dry_run else ""
    lines.append(
        f"_Generated {generated_at:%Y-%m-%d %H:%M} local time | Default model: {model} | "
        f"Lookback: {lookback_hours}h{mode}_"
    )
    lines.append("")

    for result in results:
        lines.append(f"## {result.label}")
        lines.append("")
        if result.is_fallback and result.items:
            lines.append(
                f"_No news in the last {lookback_hours} hours; showing the "
                f"{len(result.items)} most recent item(s) found._"
            )
            lines.append("")
        if result.error:
            lines.append(f"_{result.error}_")
        elif not result.items:
            note = f"_No news found in the last {lookback_hours} hours._"
            if result.topic != MARKET_TOPIC:
                note = (
                    f"_No news found in the last {lookback_hours} hours "
                    f"(if this persists, check that '{result.topic}' is a valid ticker symbol)._"
                )
            lines.append(note)
        elif dry_run:
            # fence must be longer than any backtick run inside the prompt
            fence = "`" * max(4, _longest_backtick_run(result.prompt) + 1)
            lines.append("_Dry run - this prompt would be sent to the model:_")
            lines.append("")
            lines.append(fence + "text")
            lines.append(result.prompt.rstrip())
            lines.append(fence)
        else:
            if result.model:
                lines.append(f"_Summarized by {result.model}_")
                lines.append("")
            lines.append(result.summary_md or "_The model returned an empty summary._")
        if result.items:
            lines.append("")
            lines.append("**Sources**")
            lines.append("")
            for item in result.items:
                lines.append(_source_line(item))
            if result.dropped:
                lines.append("")
                lines.append(
                    f"_The last {result.dropped} source(s) listed above did not fit the model "
                    "context and were not part of the summary._"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _longest_backtick_run(text: str) -> int:
    return max((len(run) for run in re.findall(r"`+", text)), default=0)


def _source_line(item: NewsItem) -> str:
    title = _MD_SPECIALS.sub(r"\\\1", item.title)
    url = item.url
    if any(ch in url for ch in "() "):
        url = f"<{url}>"
    when = f", {item.published.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC" if item.published else ""
    return f"- [{title}]({url}) - {item.source or 'unknown'}{when}"


def write_report(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8", newline="\n")
    return output_path
