"""Ollama preflight and per-topic summarization.

Each topic gets ONE stateless chat() call with a fresh two-message
conversation (system + user), so every ticker and the market base run in
an independent context.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import timezone

import ollama

from .config import NewsConfig, OllamaConfig
from .models import MARKET_TOPIC, NewsItem, TopicResult

log = logging.getLogger(__name__)


class PreflightError(Exception):
    """Ollama unavailable or model missing; the message says how to fix it."""


SYSTEM_PROMPT = """You are a financial news analyst writing one section of a daily stock report.
You will be given a numbered list of recent news items about a single topic
(either the overall market or one stock ticker).

Write a concise Markdown summary that keeps the main point of every distinct
piece of news:
- Begin with a 1-2 sentence overall takeaway for the topic.
- Then write one bullet per distinct story. Merge items that cover the same event.
- Each bullet: the key facts and why they matter, in at most 2 sentences.
- Preserve concrete numbers (prices, percentages, dates) exactly as given.
- Use only information from the provided items. Do not speculate, do not add
  disclaimers, and do not write a preamble or a closing line.
"""

# Reserved out of num_ctx for the system prompt, message template, and the reply
PROMPT_OVERHEAD_TOKENS = 1200
CHARS_PER_TOKEN = 4  # rough heuristic for English news text
HEADER_ALLOWANCE_CHARS = 200
MIN_BUDGET_CHARS = 2000


def make_client(cfg: OllamaConfig) -> ollama.Client:
    return ollama.Client(host=cfg.host, timeout=cfg.timeout_seconds)


def preflight(cfg: OllamaConfig) -> None:
    client = ollama.Client(host=cfg.host, timeout=10)
    try:
        listing = client.list()
    except Exception as exc:
        raise PreflightError(
            f"Ollama is not reachable at {cfg.host} ({exc.__class__.__name__}: {exc}). "
            "Install it from https://ollama.com/download and make sure it is running, "
            "or use --dry-run to skip summarization."
        ) from exc
    names = _model_names(listing)
    wanted = {cfg.model, f"{cfg.model}:latest" if ":" not in cfg.model else cfg.model}
    if not wanted & names:
        available = ", ".join(sorted(names)) or "none"
        raise PreflightError(
            f"Model '{cfg.model}' is not available in Ollama (installed: {available}). "
            f"Run: ollama pull {cfg.model}"
        )


def _model_names(listing) -> set[str]:
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models")
    names: set[str] = set()
    for m in models or []:
        name = getattr(m, "model", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.add(name)
    return names


def build_user_message(
    result: TopicResult, news_cfg: NewsConfig, ollama_cfg: OllamaConfig, report_date: str
) -> tuple[str, int]:
    """Return (user_message, dropped_count); drops items past the char budget."""
    num_ctx = int(ollama_cfg.options.get("num_ctx", 4096))
    raw_budget = (num_ctx - PROMPT_OVERHEAD_TOKENS) * CHARS_PER_TOKEN
    char_budget = max(MIN_BUDGET_CHARS, raw_budget)
    if raw_budget < MIN_BUDGET_CHARS:
        log.warning(
            "num_ctx=%d leaves almost no room for news text; the prompt may exceed the model's context window",
            num_ctx,
        )

    subject = "the overall stock market" if result.topic == MARKET_TOPIC else f"the stock {result.topic}"
    blocks: list[str] = []
    used = HEADER_ALLOWANCE_CHARS
    for index, item in enumerate(result.items, start=1):
        snippet = item.summary[: news_cfg.max_chars_per_article].strip()
        published = f"{item.published.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC" if item.published else "unknown"
        block = f"\n[{index}] {item.title}\nSource: {item.source or 'unknown'} | Published: {published}\n"
        if snippet:
            block += snippet + "\n"
        if used + len(block) > char_budget:
            if not blocks:  # a single oversized item is truncated, not exempted
                blocks.append(block[: char_budget - used])
            break
        used += len(block)
        blocks.append(block)

    dropped = len(result.items) - len(blocks)
    if result.is_fallback:
        recency = (
            f"The {len(blocks)} most recent news items found "
            f"(none were published within the last {news_cfg.lookback_hours} hours):"
        )
    else:
        recency = f"News items from the last {news_cfg.lookback_hours} hours ({len(blocks)} items):"
    header = f"Topic: {subject}\nReport date: {report_date}\n{recency}\n"
    return header + "".join(blocks), dropped


def summarize_topic(client: ollama.Client, cfg: OllamaConfig, user_message: str) -> str:
    response = client.chat(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options=cfg.options,
        keep_alive=cfg.keep_alive,
    )
    text = _response_content(response)
    # Thinking models (e.g. qwen3) may prepend a reasoning block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


RANK_SYSTEM_PROMPT = """You screen and rank headlines for the market-overview section of a daily stock report.
You will be given a numbered list of news items.

First, discard items that do not matter to financial markets or investors:
lifestyle and social-media trends, personal-finance advice and individual money
stories, product reviews, human-interest pieces, and daily service journalism
such as "best CD rates today" or "mortgage rates today".

Then rank the remaining items by importance to markets, most important first.
Importance means the breadth and size of the likely market impact: macroeconomic
data and central-bank decisions, wars, sanctions and geopolitics with market
consequences, index-level moves, major M&A, and mega-cap company news all rank
above routine single-stock analyst notes or small-cap items.

Respond with JSON only, in the form {"ranked": [7, 1, 4]} listing the numbers
of the relevant items from most to least important."""


def rank_market_items(client: ollama.Client, cfg: OllamaConfig, items: list[NewsItem]) -> list[NewsItem]:
    """One cheap JSON-mode call: drop non-market items and rank the rest by
    importance. Returns the kept items most-important-first."""
    lines = []
    for index, item in enumerate(items, start=1):
        line = f"[{index}] {item.title}"
        if item.summary:
            line += f" - {item.summary[:200]}"
        lines.append(line)
    response = client.chat(
        model=cfg.model,
        messages=[
            {"role": "system", "content": RANK_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        options={**cfg.options, "temperature": 0.0},
        keep_alive=cfg.keep_alive,
        format="json",
    )
    content = re.sub(r"<think>.*?</think>", "", _response_content(response), flags=re.DOTALL)
    try:
        numbers = json.loads(content).get("ranked", [])
    except (json.JSONDecodeError, AttributeError):
        numbers = re.findall(r"\d+", content)
    kept_indices: list[int] = []
    for number in numbers if isinstance(numbers, list) else []:
        try:
            index = int(number)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= len(items) and index not in kept_indices:
            kept_indices.append(index)
    # the model's order IS the ranking - do not re-sort
    return [items[index - 1] for index in kept_indices]


def _response_content(response) -> str:
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if content is None and isinstance(response, dict):
        content = (response.get("message") or {}).get("content")
    return (content or "").strip()
