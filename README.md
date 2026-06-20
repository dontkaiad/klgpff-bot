# klgpff — Kai Lark's Guilty Pleasure Fanfiction Bot

Telegram bot for literary prose and explicit fanfiction via the Anthropic Claude API directly — no claude.ai filters. With character-fact memory, smart model routing, prompt caching and cost tracking.

## Features

- Writes long atmospheric scenes via the Claude API directly
- **Smart router** powered by a Haiku classifier: new scenes → Opus, edits/continuations → Sonnet, chat → Haiku. Manual override available.
- **Prompt caching** — system+facts use `cache_control: ephemeral`, repeat turns within 5 min pay 10% of normal input cost
- **Token & cost accounting** per chat: `/cost` reports per-model usage broken down into input / output / cache
- **Auto-save** for replies over 1500 chars to `outputs/<chat>/<timestamp>.txt`, also sent as a Telegram document
- Character-fact memory on disk — survives restarts
- Per-chat system prompt: set, extend, persists across restarts
- History compression (`/summarize`) via Haiku — slashes input cost on long sessions
- User whitelist by Telegram ID

## Commands

| Command | Description |
|---|---|
| `/new` | Reset conversation context |
| `/scene <desc>` | Long scene via Opus (1500+ words, file) |
| `/regenerate [edit]` | Regenerate last reply (optional inline correction) |
| `/summarize` | Compress history via Haiku |
| `/last` | Last saved file |
| `/remember <fact>` | Save a character fact |
| `/forget <n>` | Forget fact by number |
| `/forget all` | Forget all facts |
| `/facts` | Show all facts |
| `/system` | Show current system prompt |
| `/setsystem <text>` | Replace system prompt |
| `/addprompt <text>` | Append to current prompt |
| `/model` | Current mode |
| `/model auto` | Classifier decides (default) |
| `/model opus\|sonnet\|haiku` | Pin model, classifier off |
| `/cost` | Costs for this chat |
| `/help` | Command list |

## Install

```bash
pip install python-telegram-bot anthropic
```

Create `.env` next to the script:

```
TELEGRAM_BOT_TOKEN=your_botfather_token
ANTHROPIC_API_KEY=sk-ant-...
ALLOWED_USERS=your_telegram_id
CLAUDE_MODEL=claude-opus-4-8     # default when mode is auto
MAX_HISTORY=25                           # turns kept in context
MAX_TOKENS=8000                          # max_tokens for generation
```

Run:

```bash
python claude_tg_bot.py
```

## Storage layout

| Folder | Contents |
|---|---|
| `facts/<user_id>.json` | User-scoped facts |
| `prompts/<chat_id>.txt` | Per-chat system prompt |
| `models/<chat_id>.txt` | Model mode (auto/opus/sonnet/haiku) |
| `usage/<chat_id>.json` | Token counters per model |
| `outputs/<chat_id>/*.txt` | Auto-saved long replies |

All of the above are in `.gitignore` — private content never leaks.

## Requirements

- Python 3.9+
- Anthropic API key
- Telegram bot token (from [@BotFather](https://t.me/BotFather))

## ⚙️ Engineering highlights

- **Cost-aware model routing** — a Haiku classifier picks Opus / Sonnet / Haiku per task, so expensive models run only where they earn it
- **RAG character memory** — Voyage embeddings + Qdrant, semantic retrieval instead of context-stuffing
- **Raw API over a managed platform** — full control of the system prompt, model selection, and generation params
- **File-based persistence, streaming generation, graceful degradation** — small footprint, debuggable, falls back instead of failing
- **Dockerized, auto-deployed** via GitHub Actions on push to `main`

Full engineering writeup → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Why

Because claude.ai filters explicit content, and fanfiction has to get written.
