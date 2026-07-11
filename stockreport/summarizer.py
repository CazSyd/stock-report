"""Ollama preflight and per-topic summarization.

Each topic gets ONE stateless chat() call with a fresh two-message
conversation (system + user), so every ticker and the market base run in
an independent context.
"""
from __future__ import annotations

import logging
import re
from datetime import timezone

import ollama

from .config import NewsConfig, OllamaConfig
from .models import MARKET_TOPIC, TopicResult

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
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if content is None and isinstance(response, dict):
        content = (response.get("message") or {}).get("content")
    text = (content or "").strip()
    # Thinking models (e.g. qwen3) may prepend a reasoning block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text
