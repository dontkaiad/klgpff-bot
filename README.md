# klgpff — Kai Lark's Guilty Pleasure Fanfiction Bot

Telegram-бот для написания художественной прозы и explicit/NSFW-фанфиков через Anthropic Claude API напрямую — без фильтров claude.ai. С памятью фактов о персонажах между сессиями.

## Что умеет

- Пишет длинные атмосферные тексты через Claude API напрямую
- Работает на `claude-opus-4-20250514` — лучшее качество прозы, но ест много токенов (оно того стоит)
- Хранит факты о персонажах на диске — не забывает между сессиями
- Разбивает длинные ответы по абзацам, не по символам
- Настраиваемый system prompt прямо из чата
- Whitelist пользователей по Telegram ID

## Команды

| Команда | Описание |
|---|---|
| `/new` | Сбросить контекст диалога |
| `/remember <факт>` | Запомнить факт о персонаже |
| `/forget <номер>` | Забыть факт по номеру |
| `/forget all` | Забыть все факты |
| `/facts` | Показать все факты |
| `/system` | Текущий system prompt |
| `/setsystem <текст>` | Заменить system prompt |
| `/addprompt <текст>` | Добавить к существующему промпту |

## Установка

```bash
pip install python-telegram-bot anthropic
```

Создай `.env` рядом со скриптом:

```
TELEGRAM_BOT_TOKEN=токен_от_botfather
ANTHROPIC_API_KEY=sk-ant-...
ALLOWED_USERS=твой_telegram_id
CLAUDE_MODEL=claude-opus-4-20250514
```

Запуск:

```bash
python claude_tg_bot.py
```

## Требования

- Python 3.9+
- Anthropic API key
- Telegram bot token (от [@BotFather](https://t.me/BotFather))

## Зачем

Потому что claude.ai режет explicit контент, а писать фанфики надо.

---

# klgpff — Kai Lark's Guilty Pleasure Fanfiction Bot

Telegram bot for writing literary prose and explicit/NSFW fanfiction via the Anthropic Claude API directly — no claude.ai filters. With persistent character-fact memory across sessions.

## Features

- Writes long atmospheric texts via the Claude API directly
- Runs on `claude-opus-4-20250514` — best prose quality, eats a lot of tokens (worth it)
- Stores character facts on disk — doesn't forget between sessions
- Splits long replies by paragraphs, not by characters
- System prompt configurable from chat
- User whitelist by Telegram ID

## Commands

| Command | Description |
|---|---|
| `/new` | Reset conversation context |
| `/remember <fact>` | Save a character fact |
| `/forget <n>` | Forget fact by number |
| `/forget all` | Forget all facts |
| `/facts` | Show all facts |
| `/system` | Show current system prompt |
| `/setsystem <text>` | Replace system prompt |
| `/addprompt <text>` | Append to current prompt |

## Install

```bash
pip install python-telegram-bot anthropic
```

Create `.env` next to the script:

```
TELEGRAM_BOT_TOKEN=your_botfather_token
ANTHROPIC_API_KEY=sk-ant-...
ALLOWED_USERS=your_telegram_id
CLAUDE_MODEL=claude-opus-4-20250514
```

Run:

```bash
python claude_tg_bot.py
```

## Requirements

- Python 3.9+
- Anthropic API key
- Telegram bot token (from [@BotFather](https://t.me/BotFather))

## Why

Because claude.ai filters explicit content, and fanfiction has to get written.
