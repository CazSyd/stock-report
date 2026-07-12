# DailyStockReport

Generates a daily markdown briefing of breaking market news plus news for the tickers you
care about, summarized by a local LLM running in [Ollama](https://ollama.com).

How it works:

1. Tickers are read from `config.yaml`.
2. News is fetched concurrently, with no API keys needed:
   - **Market overview** (always included): Google News Business, Yahoo Finance, CNBC, MarketWatch feeds.
   - **Per ticker**: Yahoo Finance news (via `yfinance`), Google News search, Yahoo per-ticker RSS.
3. Items are deduplicated, filtered to the lookback window (default 24h), and capped per topic.
4. Each topic (every ticker + the market base) gets its **own fresh LLM session** — one
   independent Ollama chat call — so no single context window has to hold everything.
5. All summaries are compiled into a single report: `reports/YYYY-MM-DD.md`, with source
   links under every section.

## Setup

```powershell
# 1. Python environment (uv reads pyproject.toml)
uv sync

# 2. Ollama + model (one-time)
# Install from https://ollama.com/download, then:
ollama pull gemma3:12b
```

## Usage

```powershell
uv run stockreport
```

Options:

| Flag | Effect |
|---|---|
| `--dry-run` | Fetch news and build prompts but skip the LLM; writes `reports/<date>.dry-run.md` showing exactly what would be sent to the model. Works without Ollama installed. |
| `--tickers AAPL,TSLA` | Override the ticker list from the config for this run. |
| `--model llama3.1:8b` | Override the model for this run. |
| `--config path\to\file.yaml` | Use a different config file. |
| `--output path\to\report.md` | Write the report to an explicit path. |

Exit codes: `0` success, `1` config error, `2` Ollama unreachable / model not pulled,
`3` report written but at least one topic failed to summarize (the report notes which).

## Configuration

Edit `config.yaml`:

- `tickers` — Yahoo Finance symbol format (e.g. `BRK-B`, not `BRK.B`). Non-US
  listings need their exchange suffix: `D05.SI` (SGX), `SIVE.ST` (Stockholm),
  `7203.T` (Tokyo) — a bare `D05` is not a valid Yahoo symbol and returns nothing.
- `ollama.model` — any model you have pulled; `ollama list` shows what's installed.
- `ollama.options.num_ctx` — context window. The news prompt is automatically budgeted to
  fit it; raise it (e.g. 16384) if you want more articles per topic considered.
- `news.max_articles_per_topic` — top N most recent items per section (default 5).
- `news.require_ticker_mention` — when `true` (default), a ticker's section only
  keeps items whose title or snippet mentions the ticker symbol or company name;
  Yahoo's "related news" otherwise drags in adjacent stories. Set `false` for
  broader coverage.
- `news.market_relevance_filter` — when `true` (default), an extra quick model
  call screens the Market Overview candidates: items that don't affect markets
  (lifestyle trends, personal-finance advice columns, "best CD rates today"
  service posts) are dropped, and the rest are ranked by market importance —
  macro/central-bank news, geopolitics with market impact, and major M&A rank
  above routine single-stock notes. The section shows the top items in that
  order. Runs only in real runs; `--dry-run` shows the newest items unscreened.
- `news.market_candidate_pool` / `news.market_source_cap` — how many candidates
  the market ranking chooses from (default 30), and how many of those any single
  publisher may contribute (default 2, so one outlet's syndication burst can't
  crowd out the pool).
- `news.fallback_max_articles` — when a ticker has no news inside the lookback
  window, its section shows this many most-recent items instead (however old),
  clearly labeled. `0` disables the fallback. Useful for quiet or non-US
  tickers that don't make daily English-language news.
- `news.lookback_hours`, `news.max_chars_per_article` — control how much news
  each topic's session receives.
- `news.market_feeds` — RSS feeds for the market overview section.

## Tests

```powershell
uv run pytest -q
```

The suite (70 tests) covers config validation, feed parsing, yfinance schema
normalization, dedup/time-filtering, prompt context budgeting, report rendering,
and CLI exit codes. No network or Ollama needed.

## Notes

- The first summarization call after a while is slow (30–90s) because Ollama loads the
  model into VRAM; `keep_alive: "10m"` keeps it resident for the rest of the run.
- Rerunning on the same day overwrites that day's report.
- A ticker with no news in the lookback window gets a note instead of a summary (and no
  LLM call is made for it).
