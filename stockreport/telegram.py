"""Deliver the report to a Telegram chat as properly formatted messages.

Telegram does not render Markdown text; it accepts its own HTML subset
(<b>, <i>, <a>, ...). The report is converted line by line and split into
messages under the 4096-character cap, preferring section boundaries.
"""
from __future__ import annotations

import html
import logging
import os
import re
import time

import requests

log = logging.getLogger(__name__)

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"
API_BASE = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 4000  # official cap is 4096; keep a margin


class NotifyError(Exception):
    """Telegram delivery cannot proceed; the message says how to fix it."""


def require_env() -> tuple[str, str]:
    token = os.environ.get(TOKEN_ENV, "").strip()
    chat_id = os.environ.get(CHAT_ENV, "").strip()
    if not token or not chat_id:
        raise NotifyError(
            f"{TOKEN_ENV} and {CHAT_ENV} must be set to deliver via Telegram. "
            "Create a bot with @BotFather, send it one message, then read your chat id "
            "from https://api.telegram.org/bot<TOKEN>/getUpdates (see README)."
        )
    return token, chat_id


def send_report(markdown: str) -> int:
    """Convert the report to Telegram HTML and send it; returns the message count."""
    token, chat_id = require_env()
    chunks = _chunk_report(markdown)
    for position, chunk in enumerate(chunks):
        if position:
            time.sleep(1)  # Telegram allows roughly one message per second per chat
        _send_message(token, chat_id, chunk)
    return len(chunks)


# ---------- markdown -> Telegram HTML ----------

_BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")
_LINK = re.compile(r"\[((?:[^\]\\]|\\.)+)\]\(<?([^)>\s]+)>?\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_ESCAPE = re.compile(r"\\([\\\[\]`*_<>-])")


def markdown_to_telegram_html(markdown: str) -> str:
    out: list[str] = []
    for line in markdown.split("\n"):
        stripped = line.strip()
        if line.startswith("## "):
            out.append(f"<b>{_inline(line[3:])}</b>")
        elif line.startswith("# "):
            out.append(f"<b>{_inline(line[2:])}</b>")
        elif match := _BULLET.match(line):
            out.append(f"{match.group(1)}• {_inline(match.group(2))}")
        elif len(stripped) > 2 and stripped.startswith("_") and stripped.endswith("_"):
            out.append(f"<i>{_inline(stripped[1:-1])}</i>")
        else:
            out.append(_inline(line))
    return "\n".join(out)


def _inline(text: str) -> str:
    """Escape HTML and convert links/bold. Links are stashed first so that
    underscores or asterisks inside URLs can never be misparsed as styling."""
    links: list[str] = []

    def stash(match: re.Match) -> str:
        label = _MD_ESCAPE.sub(r"\1", match.group(1))
        url = html.escape(match.group(2), quote=True)
        links.append(f'<a href="{url}">{html.escape(label, quote=False)}</a>')
        return f"\x00{len(links) - 1}\x00"

    text = _LINK.sub(stash, text)
    text = html.escape(text, quote=False)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _MD_ESCAPE.sub(r"\1", text)
    for index, link in enumerate(links):
        text = text.replace(f"\x00{index}\x00", link)
    return text


def _chunk_report(markdown: str) -> list[str]:
    """Sections are never split across messages: the report is divided at its
    '## ' headings first, and whole sections are packed into messages. Only a
    section that is by itself larger than the cap gets split internally - and
    then preferentially at its 'Sources' marker so each piece reads naturally."""
    blocks = [markdown_to_telegram_html(section) for section in _split_sections(markdown)]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > MAX_MESSAGE_CHARS:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_block(block))
            continue
        candidate = f"{current}\n\n{block}" if current else block
        if current and len(candidate) > MAX_MESSAGE_CHARS:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


def _split_sections(markdown: str) -> list[str]:
    """The report header, then one block per '## ' section."""
    sections: list[list[str]] = [[]]
    for line in markdown.split("\n"):
        if line.startswith("## "):
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(s).strip("\n") for s in sections if any(line.strip() for line in s)]


def _split_block(block: str) -> list[str]:
    """Split one oversized section: at its Sources marker first, then at line
    boundaries, then hard (a single monster line cannot be helped)."""
    pieces: list[str] = []
    current = ""
    for line in block.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if current and (line == "<b>Sources</b>" or len(candidate) > MAX_MESSAGE_CHARS):
            pieces.append(current)
            current = line
        else:
            current = candidate
        while len(current) > MAX_MESSAGE_CHARS:
            pieces.append(current[:MAX_MESSAGE_CHARS])
            current = current[MAX_MESSAGE_CHARS:]
    if current.strip():
        pieces.append(current)
    return pieces


# ---------- sending ----------


def _send_message(token: str, chat_id: str, text_html: str) -> None:
    url = f"{API_BASE}/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code == 429:
        retry_after = 3
        try:
            retry_after = int(response.json()["parameters"]["retry_after"])
        except Exception:
            pass
        time.sleep(min(retry_after, 30))
        response = requests.post(url, json=payload, timeout=30)
    if response.status_code == 400:
        # formatting rejected: never lose content - resend as plain text
        log.warning("Telegram rejected the formatting (%s); resending as plain text", response.text[:150])
        plain = {key: value for key, value in payload.items() if key != "parse_mode"}
        plain["text"] = _strip_tags(text_html)
        response = requests.post(url, json=plain, timeout=30)
    if response.status_code >= 400:
        raise NotifyError(f"Telegram sendMessage failed ({response.status_code}): {response.text[:200]}")


def _strip_tags(text_html: str) -> str:
    return html.unescape(re.sub(r"</?(?:b|i|a)(?:\s[^>]*)?>", "", text_html))
