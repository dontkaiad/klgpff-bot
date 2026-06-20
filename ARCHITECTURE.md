# Architecture

Technical reference for `claude_tg_bot.py` — a Telegram bot wrapping the Anthropic Claude API with model routing, prompt caching, and cost tracking.

## Components

```
Telegram → python-telegram-bot (polling)
         → handler (cmd_* / handle_message)
         → classifier (Haiku) ─┐
                                ├→ Anthropic Messages API
         → main model ─────────┘
         → reply (chunks + optional .txt document)
         → file-backed state (facts / prompts / models / usage / outputs)
```

The bot is single-process, single-event-loop, polling-based. State is on disk; nothing is held only in memory across requests.

## Three-tier model router

Before each user-initiated turn, `classify_message(chat_id, user_text)` calls `claude-haiku-4-5` with `max_tokens=4` and a bilingual (RU+EN) system prompt that constrains the response to a single token: `opus`, `sonnet`, or `haiku`.

Routing rules:

| Bucket | Intent | Model |
|---|---|---|
| `opus` | Write a new scene from scratch | `claude-opus-4-8` |
| `sonnet` | Continue / rewrite / extend / edit existing text | `claude-sonnet-4-6` |
| `haiku` | Discussion, questions, short replies, meta-talk | `claude-haiku-4-5` |

Output is parsed by substring match against the three aliases; on any error or malformed response the bot defaults to `haiku`. Every classifier call is metered and counted toward `/cost`.

A per-chat manual override is persisted in `models/<chat_id>.txt`:

- `auto` (default) — classifier picks per message
- `opus` / `sonnet` / `haiku` — pinned, classifier skipped

`handle_message` logs which path was taken on every turn:

```
[chat 12345] [router → opus] (claude-opus-4-8) | msg: '...'
[chat 12345] [manual → sonnet] (claude-sonnet-4-6) | msg: '...'
[chat 12345] [scene → opus] (claude-opus-4-8) | desc: '...'
```

## Prompt caching

The Anthropic API supports ephemeral prompt caching: a block marked with `cache_control: {"type": "ephemeral"}` is cached for ~5 minutes; subsequent reads of that block within the window cost 10% of the normal input rate.

The bot wraps the system prompt + facts in a single cacheable block:

```python
system=[
    {
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }
]
```

`build_system_prompt()` concatenates the user-defined system prompt with a `## Факты о персонажах и мире` section so both pieces sit inside one cache block. On long sessions this is the single biggest cost lever — every turn after the first within the cache window pays 10% on the system+facts portion.

## History compression

`/summarize` calls Haiku with the entire transcript and a fixed instruction to compress it into ≤500 words preserving characters, plot beats, tone, and key facts. The conversation history is then replaced with a synthetic two-message exchange:

```python
[
    {"role": "user", "content": "Что произошло до этого момента?"},
    {"role": "assistant", "content": summary},
]
```

Future turns build from this short summary instead of the full log. The summarization itself is metered and counted toward `/cost`.

## Cost tracking

Every API call (main, classifier, summarizer) increments per-chat counters via `add_usage(chat_id, model, response.usage)`. Counters are stored in `usage/<chat_id>.json` keyed by model id, with four buckets: `input`, `output`, `cache_read`, `cache_write`.

`/cost` resolves these counters against a static `PRICING` table (USD per 1M tokens) and emits per-model breakdowns plus a chat total. The table covers regular input/output rates plus 5-minute cache write (input × 1.25) and cache read (input × 0.10):

```python
PRICING = {
    "claude-opus-4-8":      {"in": 15.0, "out": 75.0, "cw": 18.75, "cr": 1.50},
    "claude-sonnet-4-6":    {"in":  3.0, "out": 15.0, "cw":  3.75, "cr": 0.30},
    "claude-haiku-4-5":   {"in":  0.80, "out":  4.0, "cw":  1.00, "cr": 0.08},
}
```

The reported cache-hit ratio (`cache_read / (input + cache_read)`) is a quick sanity check that the caching block is actually engaging.

## Persistence

In-memory `chat_data` from python-telegram-bot is intentionally not used. `PicklePersistence` flushes only on a configurable interval and on clean shutdown — restarts during the flush window silently lose state, which is unacceptable for things like a system prompt the user just edited.

All mutable per-chat state lives in plain files written synchronously on each mutation:

| Path | Type | Content |
|---|---|---|
| `facts/<user_id>.json` | JSON list | Character/world facts (user-scoped, not chat-scoped) |
| `prompts/<chat_id>.txt` | UTF-8 text | System prompt for this chat |
| `models/<chat_id>.txt` | UTF-8 text | One of `auto`, `opus`, `sonnet`, `haiku` |
| `usage/<chat_id>.json` | JSON dict | `{model_id: {input, output, cache_read, cache_write}}` |
| `outputs/<chat_id>/*.txt` | UTF-8 text | Auto-saved long replies |

Conversation history (`conversations: dict[int, list[dict]]`) is kept in memory only — it represents the live message stream and is naturally rebuildable; persisting it would not survive `MAX_HISTORY` trimming anyway.

## Auto-save and `/scene`

Replies longer than `LONG_REPLY_THRESHOLD = 1500` characters are saved to `outputs/<chat_id>/<YYYYMMDD-HHMMSS>.txt` and re-sent as a Telegram document after the inline text. This handles two pain points: long replies get visually buried in chat scrollback, and Telegram's per-message 4096-char limit forces multi-message splits that fragment a single scene.

`/scene <description>` is a thin wrapper that:

1. Wraps the description in a fixed long-form template (≥1500 words, use facts, atmospheric).
2. Forces `model_override=MODELS["opus"]`, bypassing the router.
3. Routes through the same `_generate_and_reply` path, so auto-save fires naturally on the resulting long reply.

## Message lifecycle

```
update arrives
 ├─ allow-list check
 ├─ command? → cmd_*
 └─ free text →
     1. append to history, trim to MAX_HISTORY*2 messages
     2. classify (or read pinned model)
     3. _generate_and_reply
        a. load_prompt + load_facts → build cached system block
        b. resolve model
        c. messages.create(...)
        d. add_usage(chat_id, model, response.usage)
        e. split_by_paragraphs → reply_text per chunk
        f. if len > 1500: save_output → reply_document
```

Errors in the main API call are caught and reported to the user as text; classifier errors fall back to `haiku` silently. Telegram-side errors (e.g. message-too-long) are avoided by paragraph-aware chunking in `split_by_paragraphs`.

## Configuration

Read from `.env` next to the script (loader is hand-rolled, no dotenv dependency):

| Env | Default | Meaning |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | required | BotFather token |
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `ALLOWED_USERS` | empty (open) | Comma-separated Telegram user IDs |
| `CLAUDE_MODEL` | `claude-opus-4-8` | Fallback model when mode is `auto` and `/regenerate` is invoked |
| `MAX_HISTORY` | `25` | Turns retained in context |
| `MAX_TOKENS` | `8000` | `max_tokens` for main generation |
| `SYSTEM_PROMPT` | built-in | Default system prompt for new chats |

## Dependencies

- `python-telegram-bot` (v20+) — Telegram polling, command/message handlers, `BotCommand`, `set_my_commands`
- `anthropic` — official Python SDK, used for `messages.create` with `system` blocks and `cache_control`

No database, no Redis, no message broker. Process state is the filesystem.
