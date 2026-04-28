# klgpff — Kai Lark's Guilty Pleasure Fanfiction Bot

Telegram-бот для художественной прозы и explicit-фанфиков через Anthropic Claude API напрямую — без фильтров claude.ai. С памятью фактов о персонажах, умным роутингом моделей, кешированием промпта и учётом расходов.

## Что умеет

- Пишет длинные атмосферные сцены через Claude API напрямую
- **Умный роутер** на Haiku-классификаторе: новые сцены → Opus, редактирование/продолжение → Sonnet, обсуждение → Haiku. Можно зафиксировать модель вручную.
- **Prompt caching** — system+факты помечены `cache_control: ephemeral`, повторные ходы в течение 5 минут платят 10% от input
- **Учёт токенов и денег** per-chat: `/cost` показывает расходы по моделям с разбивкой на input/output/cache
- **Авто-сохранение** ответов длиннее 1500 символов в `outputs/<chat>/<timestamp>.txt` + отправка файлом в чат
- Память фактов о персонажах на диске — не забывает между сессиями
- Per-chat system prompt: задаёшь, дополняешь, переживает рестарт
- Сжатие истории (`/summarize`) через Haiku — экономит токены на длинных диалогах
- Whitelist пользователей по Telegram ID

## Команды

| Команда | Описание |
|---|---|
| `/new` | Сбросить контекст диалога |
| `/scene <описание>` | Длинная сцена через Opus (1500+ слов, файл) |
| `/regenerate [правка]` | Перегенерить последний ответ (с опц. правкой) |
| `/summarize` | Сжать историю через Haiku |
| `/last` | Последний сохранённый файл |
| `/remember <факт>` | Запомнить факт о персонаже |
| `/forget <номер>` | Забыть факт по номеру |
| `/forget all` | Забыть все факты |
| `/facts` | Показать все факты |
| `/system` | Текущий system prompt |
| `/setsystem <текст>` | Заменить system prompt |
| `/addprompt <текст>` | Добавить к существующему промпту |
| `/model` | Текущий режим |
| `/model auto` | Классификатор решает (дефолт) |
| `/model opus\|sonnet\|haiku` | Зафиксировать модель, классификатор off |
| `/cost` | Расходы по этому чату |
| `/help` | Список команд |

## Установка

```bash
pip install python-telegram-bot anthropic
```

Создай `.env` рядом со скриптом:

```
TELEGRAM_BOT_TOKEN=токен_от_botfather
ANTHROPIC_API_KEY=sk-ant-...
ALLOWED_USERS=твой_telegram_id
CLAUDE_MODEL=claude-opus-4-20250514     # дефолт когда режим auto
MAX_HISTORY=25                           # сколько ходов держать в контексте
MAX_TOKENS=8000                          # max_tokens для генерации
```

Запуск:

```bash
python claude_tg_bot.py
```

## Структура хранения

| Папка | Что внутри |
|---|---|
| `facts/<user_id>.json` | Факты пользователя |
| `prompts/<chat_id>.txt` | System prompt чата |
| `models/<chat_id>.txt` | Режим модели (auto/opus/sonnet/haiku) |
| `usage/<chat_id>.json` | Счётчики токенов по моделям |
| `outputs/<chat_id>/*.txt` | Авто-сохранённые длинные ответы |

Все эти папки в `.gitignore` — приватный контент не утечёт.

## Требования

- Python 3.9+
- Anthropic API key
- Telegram bot token (от [@BotFather](https://t.me/BotFather))

## Зачем

Потому что claude.ai режет explicit контент, а писать фанфики надо.

---

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
CLAUDE_MODEL=claude-opus-4-20250514     # default when mode is auto
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

## Why

Because claude.ai filters explicit content, and fanfiction has to get written.
