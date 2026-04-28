# klgpff — Kai Lark's Guilty Pleasure Fanfiction Bot

Telegram-бот для длинной художественной прозы и фанфиков через Anthropic Claude API. С памятью фактов о персонажах, умным роутингом моделей, prompt caching и учётом расходов per-chat.

Проект изначально про prompt engineering на длинных творческих сессиях: как держать контекст, как не платить лишнего за Opus и как разруливать модель под тип запроса.

## Архитектура

- **Three-tier роутер.** Перед каждым ходом Haiku-классификатор с двуязычным (RU+EN) промптом возвращает одно слово: `opus` / `sonnet` / `haiku`. Новая сцена с нуля → Opus, продолжить/переписать/отредактировать → Sonnet, обсуждение и мета-разговор → Haiku. Ручной override через `/model` если нужно зафиксировать.
- **Prompt caching.** System prompt + факты идут блоком с `cache_control: ephemeral`. Повторные ходы в течение 5 минут платят 10% от обычного input — на 20+ ходах разница в разы.
- **History compression.** `/summarize` вызывает Haiku, сжимает диалог в краткое содержание и заменяет историю синтетическим обменом. Стоит копейки, дальше input в разы меньше.
- **Per-chat cost tracking.** `/cost` показывает токены и доллары по моделям с разбивкой input / output / cache write / cache read. Считается из `response.usage`.
- **Persistence на диске.** Промпт, режим модели, факты, расходы и сохранённые ответы лежат в файлах. `chat_data` в памяти не используем — он не переживает рестарт.
- **Авто-сохранение длинных ответов.** Всё что больше 1500 символов сохраняется в `outputs/<chat>/<timestamp>.txt` и приходит в чат файлом — длинная сцена не теряется в скролле.

## Команды

| Команда | Описание |
|---|---|
| `/new` | Сбросить контекст диалога |
| `/scene <описание>` | Длинная сцена через Opus (1500+ слов, файл) |
| `/regenerate [правка]` | Перегенерить последний ответ (с опц. правкой) |
| `/summarize` | Сжать историю через Haiku |
| `/last` | Последний сохранённый файл |
| `/remember <факт>` | Запомнить факт о персонаже |
| `/forget <номер>` / `/forget all` | Забыть факт / все факты |
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

`.env` рядом со скриптом:

```
TELEGRAM_BOT_TOKEN=токен_от_botfather
ANTHROPIC_API_KEY=sk-ant-...
ALLOWED_USERS=твой_telegram_id
CLAUDE_MODEL=claude-opus-4-20250514     # дефолт когда режим auto
MAX_HISTORY=25                           # ходов в контексте
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

Всё это в `.gitignore`.

## Требования

- Python 3.9+
- Anthropic API key
- Telegram bot token (от [@BotFather](https://t.me/BotFather))

## Зачем

Веб-интерфейс не очень подходит для длинных творческих сессий: контекст рвётся, ответы обрываются, моделью не покрутишь, токены никто не считает. API даёт всё это, но на Opus экономика быстро становится грустной — отсюда роутер, кеширование и компрессия истории. Бот — это рабочая площадка чтобы отлаживать prompt engineering на длинной форме, а заодно писать то, что хочется писать.

---

# klgpff — Kai Lark's Guilty Pleasure Fanfiction Bot

Telegram bot for long-form fiction and fanfiction via the Anthropic Claude API. With character-fact memory, smart model routing, prompt caching, and per-chat cost tracking.

The project is primarily a prompt-engineering sandbox for long creative sessions: how to keep context, how to avoid overpaying for Opus, and how to route requests to the right model.

## Architecture

- **Three-tier router.** Before every turn a Haiku classifier with a bilingual (RU+EN) prompt returns one word: `opus` / `sonnet` / `haiku`. New scenes → Opus, continuations / rewrites / edits → Sonnet, discussion and meta-talk → Haiku. Manual override via `/model` when you need to pin a model.
- **Prompt caching.** System prompt + facts go as a block with `cache_control: ephemeral`. Repeat turns within 5 min pay 10% of normal input — on 20+ turn sessions the difference compounds.
- **History compression.** `/summarize` calls Haiku, condenses the dialog into a short summary and replaces history with a synthetic exchange. Cheap call, big input savings on long sessions.
- **Per-chat cost tracking.** `/cost` reports tokens and dollars per model, broken down into input / output / cache write / cache read. Sourced from `response.usage`.
- **On-disk persistence.** Prompt, model mode, facts, usage, and saved replies live in files. In-memory `chat_data` is avoided — it doesn't survive restarts.
- **Auto-saving long replies.** Anything over 1500 chars is written to `outputs/<chat>/<timestamp>.txt` and sent as a Telegram document, so long scenes don't get lost in chat scrollback.

## Commands

| Command | Description |
|---|---|
| `/new` | Reset context |
| `/scene <desc>` | Long scene via Opus (1500+ words, file) |
| `/regenerate [edit]` | Regenerate last reply (optional inline correction) |
| `/summarize` | Compress history via Haiku |
| `/last` | Last saved file |
| `/remember <fact>` | Save a character fact |
| `/forget <n>` / `/forget all` | Forget by number / all |
| `/facts` | Show all facts |
| `/system` | Show current system prompt |
| `/setsystem <text>` | Replace system prompt |
| `/addprompt <text>` | Append to current prompt |
| `/model` | Current mode |
| `/model auto` | Classifier decides (default) |
| `/model opus\|sonnet\|haiku` | Pin a model, classifier off |
| `/cost` | Costs for this chat |
| `/help` | Command list |

## Install

```bash
pip install python-telegram-bot anthropic
```

`.env` next to the script:

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

All of the above live in `.gitignore`.

## Requirements

- Python 3.9+
- Anthropic API key
- Telegram bot token (from [@BotFather](https://t.me/BotFather))

## Why

The web UI isn't great for long creative sessions: context drops, replies get cut off, you can't switch models, and there's no token accounting. The API solves all that, but Opus economics get painful fast — hence the router, caching, and history compression. The bot is a working surface for prompt engineering on long-form output, and a place to actually write the things I want to write.
