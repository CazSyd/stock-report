"""OpenRouter preflight and per-topic summarization.

Each topic gets ONE stateless chat completion (fresh system + user messages),
so every ticker and the market base run in an independent context.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import timezone

import requests

from .config import NewsConfig, OpenRouterConfig
from .models import MARKET_TOPIC, NewsItem, TopicResult

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
MAX_ATTEMPTS = 2  # per model; after that we fall back to the next model in the chain
DYNAMIC_TRY_LIMIT = 5  # discovered models to attempt per call once the configured chain is exhausted


class PreflightError(Exception):
    """OpenRouter unavailable or the API key is missing/invalid; the message says how to fix it."""


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

RANK_SYSTEM_PROMPT = """You screen and rank headlines for the market-overview section of a daily stock report.
You will be given a numbered list of news items. These feeds mix real market
news with general-interest content, and MOST items do not qualify. Be strict.

First, discard every item that does not matter to financial markets or
investors: lifestyle and social-media trends, health scares, travel and
rankings pieces, personal-finance advice and individual money stories or
anecdotes, product reviews, human-interest pieces, and daily service
journalism such as "best CD rates today" or "mortgage rates today".

Then rank the remaining items by importance to markets, most important first.
Importance means the breadth and size of the likely market impact: macroeconomic
data and central-bank decisions, wars, sanctions and geopolitics with market
consequences, index-level moves, major M&A, and mega-cap company news all rank
above routine single-stock analyst notes or small-cap items.

Respond with JSON only, in the form {"ranked": [7, 1, 4]} listing the numbers
of the relevant items from most to least important. List AT MOST 8 numbers.
Never include an item just to fill the list."""

# Reserved out of context_tokens (besides max_tokens) for the system prompt and template
OVERHEAD_RESERVE_TOKENS = 500
# The ranking answer is tiny, but reasoning models spend their token budget on
# thinking first - give them headroom or the JSON gets truncated
RANK_MAX_TOKENS = 4000
CHARS_PER_TOKEN = 4  # rough heuristic for English news text
HEADER_ALLOWANCE_CHARS = 200
MIN_BUDGET_CHARS = 2000


class OpenRouterClient:
    def __init__(self, cfg: OpenRouterConfig, api_key: str):
        self.cfg = cfg
        self.api_key = api_key
        self._saturated: set[str] = set()  # models that 429'd out earlier in this run
        self._discovered: list[str] | None = None  # live catalog, fetched at most once per run

    def chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        """Return (content, model_that_answered).

        Try the configured chain, then dynamically discovered free models.
        Free-tier endpoints saturate regularly (429 'Provider returned error'),
        so switching models recovers where retrying the same one would not.
        A model that exhausted its retries is skipped for the rest of the run."""
        models = self._candidate_models()
        last_error: RuntimeError | None = None
        for position, model in enumerate(models):
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
            }
            try:
                response = _post_with_retry(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    payload,
                    self.api_key,
                    self.cfg.timeout_seconds,
                )
            except RuntimeError as exc:
                last_error = exc
                if "(429)" in str(exc):
                    self._saturated.add(model)
                if position + 1 < len(models):
                    log.warning("Model %s failed (%s); falling back to %s", model, exc, models[position + 1])
                continue
            data = response.json()
            try:
                choice = data["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"unexpected OpenRouter response shape: {str(data)[:300]}") from exc
            if choice.get("finish_reason") == "length":
                # a truncated answer is garbage (often a cut-off reasoning
                # transcript) - treat it as a failure and try the next model
                last_error = RuntimeError(f"{model} hit the token limit before finishing its answer")
                if position + 1 < len(models):
                    log.warning(
                        "Model %s returned a truncated response; falling back to %s",
                        model,
                        models[position + 1],
                    )
                continue
            if model != self.cfg.model:
                log.info("Fallback model %s answered", model)
            return (content or "").strip(), model
        raise last_error or RuntimeError("no models configured")

    def _candidate_models(self) -> list[str]:
        chain = [self.cfg.model] + [m for m in self.cfg.fallback_models if m != self.cfg.model]
        candidates = [m for m in chain if m not in self._saturated]
        if self.cfg.dynamic_fallback:
            extra = [
                m
                for m in self._discover_free_models()
                if m not in self._saturated and m not in chain
            ]
            candidates += extra[:DYNAMIC_TRY_LIMIT]
        # everything known is saturated: retry the configured chain anyway
        return candidates or chain

    def _discover_free_models(self) -> list[str]:
        if self._discovered is not None:
            return self._discovered
        self._discovered = []  # cache even on failure; don't refetch every call
        try:
            response = requests.get(f"{OPENROUTER_BASE_URL}/models", timeout=20)
            response.raise_for_status()
            entries = response.json().get("data", [])
        except Exception as exc:
            log.warning("Could not fetch the OpenRouter catalog for dynamic fallback: %s", exc)
            return self._discovered
        free = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and str(entry.get("id", "")).endswith(":free")
            and "content-safety" not in entry["id"]  # moderation model, can't summarize
            and int(entry.get("context_length") or 0) >= self.cfg.context_tokens
        ]
        # context length as a weak quality/capacity proxy for ordering
        free.sort(key=lambda entry: int(entry.get("context_length") or 0), reverse=True)
        self._discovered = [entry["id"] for entry in free]
        log.info("Dynamic fallback: discovered %d live free models", len(self._discovered))
        return self._discovered


def _post_with_retry(url: str, payload: dict, api_key: str, timeout: int) -> requests.Response:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "DailyStockReport",
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            if attempt < MAX_ATTEMPTS:
                delay = 2.0**attempt
                log.warning("OpenRouter request error (%s); retrying in %.0fs", exc, delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        if response.status_code < 400:
            return response
        message = _error_message(response)
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < MAX_ATTEMPTS:
            delay = _retry_delay(response, attempt)
            log.warning(
                "OpenRouter returned %d (%s); retrying in %.0fs", response.status_code, message, delay
            )
            time.sleep(delay)
            continue
        raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {message}")
    raise RuntimeError("OpenRouter request failed: retries exhausted")


def _error_message(response: requests.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except Exception:
        return (response.text or "")[:300]


def _retry_delay(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "")
    try:
        return min(max(float(retry_after), 0.0), 60.0)
    except (TypeError, ValueError):
        return min(2.0 ** attempt * 2, 60.0)


def preflight(cfg: OpenRouterConfig) -> None:
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise PreflightError(
            f"{API_KEY_ENV} is not set. Create a key at https://openrouter.ai/settings/keys "
            "and put it in a .env file next to config.yaml (see .env.example)."
        )
    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise PreflightError(f"Could not reach OpenRouter: {exc}") from exc
    if response.status_code == 401:
        raise PreflightError(
            f"{API_KEY_ENV} was rejected by OpenRouter (401). Check the key in your .env file."
        )
    if response.status_code >= 400:
        raise PreflightError(
            f"OpenRouter key check failed ({response.status_code}): {_error_message(response)}"
        )


def make_client(cfg: OpenRouterConfig) -> OpenRouterClient:
    return OpenRouterClient(cfg, os.environ.get(API_KEY_ENV, "").strip())


def build_user_message(
    result: TopicResult, news_cfg: NewsConfig, llm_cfg: OpenRouterConfig, report_date: str
) -> tuple[str, int]:
    """Return (user_message, dropped_count); drops items past the char budget."""
    raw_budget = (llm_cfg.context_tokens - llm_cfg.max_tokens - OVERHEAD_RESERVE_TOKENS) * CHARS_PER_TOKEN
    char_budget = max(MIN_BUDGET_CHARS, raw_budget)
    if raw_budget < MIN_BUDGET_CHARS:
        log.warning(
            "context_tokens=%d leaves almost no room for news text; the prompt may exceed the model's context",
            llm_cfg.context_tokens,
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


def summarize_topic(client: OpenRouterClient, user_message: str) -> tuple[str, str]:
    """Return (summary_markdown, model_that_answered)."""
    text, model = client.chat(SYSTEM_PROMPT, user_message)
    # Reasoning models may prepend a thinking block
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip(), model


def rank_market_items(client: OpenRouterClient, items: list[NewsItem]) -> list[NewsItem]:
    """One cheap call: drop non-market items and rank the rest by importance.
    Returns the kept items most-important-first."""
    lines = []
    for index, item in enumerate(items, start=1):
        line = f"[{index}] {item.title}"
        if item.summary:
            line += f" - {item.summary[:200]}"
        lines.append(line)
    text, _ = client.chat(RANK_SYSTEM_PROMPT, "\n".join(lines), temperature=0.0, max_tokens=RANK_MAX_TOKENS)
    content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    content = _strip_code_fences(content).strip()
    numbers = _extract_ranked_numbers(content)
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


def _extract_ranked_numbers(content: str) -> list:
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data.get("ranked", [])
    except json.JSONDecodeError:
        pass
    # a "ranked": [...] fragment anywhere in the text (reasoning models ramble)
    match = re.search(r'"ranked"\s*:\s*\[([^\]]*)', content)
    if match:
        return re.findall(r"\d+", match.group(1))
    # a bare short answer like "2, 5 and 7" is fine, but never digit-harvest an
    # essay: a leaked reasoning transcript mentions every item number and would
    # turn the screen into a no-op
    if len(content) <= 200:
        return re.findall(r"\d+", content)
    log.warning("ranking response was not parseable JSON; treating as no ranking")
    return []


def _strip_code_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    return match.group(1) if match else text
