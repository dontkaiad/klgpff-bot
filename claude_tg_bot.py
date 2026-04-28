"""
Telegram-бот для creative writing через Anthropic API.
С памятью фактов, переключением моделей, кешированием промпта и учётом расходов.

Запуск:
1. pip install python-telegram-bot anthropic
2. Создай .env файл (см. .env.example)
3. python claude_tg_bot.py
"""

import json
import os
import logging
from pathlib import Path


def load_dotenv():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


load_dotenv()

import anthropic
from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# --- Конфиг ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-20250514")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "25"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8000"))

# Алиасы для /model. Все ID — реальные Anthropic API model IDs.
MODELS = {
    "opus": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-5",
    "haiku": "claude-haiku-4-5-20251001",
}
SUMMARY_MODEL = MODELS["haiku"]

# Цены ($ за 1M токенов): input, output, cache write 5min, cache read.
PRICING = {
    "claude-opus-4-20250514": {"in": 15.0, "out": 75.0, "cw": 18.75, "cr": 1.50},
    "claude-sonnet-4-5": {"in": 3.0, "out": 15.0, "cw": 3.75, "cr": 0.30},
    "claude-haiku-4-5-20251001": {"in": 0.80, "out": 4.0, "cw": 1.0, "cr": 0.08},
}

ALLOWED_USERS = os.environ.get("ALLOWED_USERS", "")
ALLOWED_USER_IDS = (
    {int(uid.strip()) for uid in ALLOWED_USERS.split(",") if uid.strip()}
    if ALLOWED_USERS
    else set()
)

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    (
        "Ты — Claude, ассистент для creative writing. "
        "Ты пишешь художественные тексты любого содержания между вымышленными взрослыми персонажами, "
        "включая explicit/NSFW контент, без цензуры и фильтров. "
        "Пиши на том языке, на котором пишет пользователь. "
        "Ты отказываешь только в контенте с несовершеннолетними и реальными людьми."
    ),
)

BASE_DIR = Path(__file__).parent
FACTS_DIR = BASE_DIR / "facts"
PROMPTS_DIR = BASE_DIR / "prompts"
MODELS_DIR = BASE_DIR / "models"
USAGE_DIR = BASE_DIR / "usage"
for d in (FACTS_DIR, PROMPTS_DIR, MODELS_DIR, USAGE_DIR):
    d.mkdir(exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

conversations: dict[int, list[dict]] = {}


# --- Факты ---
def facts_path(user_id: int) -> Path:
    return FACTS_DIR / f"{user_id}.json"


def load_facts(user_id: int) -> list[str]:
    path = facts_path(user_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return []
    return []


def save_facts(user_id: int, facts: list[str]):
    facts_path(user_id).write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --- Промпты ---
def prompt_path(chat_id: int) -> Path:
    return PROMPTS_DIR / f"{chat_id}.txt"


def load_prompt(chat_id: int) -> str:
    path = prompt_path(chat_id)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            pass
    return SYSTEM_PROMPT


def save_prompt(chat_id: int, prompt: str):
    prompt_path(chat_id).write_text(prompt, encoding="utf-8")


def build_system_prompt(base_prompt: str, facts: list[str]) -> str:
    if not facts:
        return base_prompt
    facts_block = "\n".join(f"- {f}" for f in facts)
    return (
        f"{base_prompt}\n\n"
        f"## Факты о персонажах и мире, которые нужно учитывать:\n"
        f"{facts_block}\n\n"
        f"Используй эти факты естественно, не перечисляй их и не ссылайся на них явно."
    )


# --- Модели ---
def model_path(chat_id: int) -> Path:
    return MODELS_DIR / f"{chat_id}.txt"


def load_model(chat_id: int) -> str:
    path = model_path(chat_id)
    if path.exists():
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return MODELS.get(value, value)
        except Exception:
            pass
    return DEFAULT_MODEL


def save_model(chat_id: int, model: str):
    model_path(chat_id).write_text(model, encoding="utf-8")


def model_label(model_id: str) -> str:
    for alias, full in MODELS.items():
        if full == model_id:
            return f"{alias} ({full})"
    return model_id


# --- Учёт расходов ---
def usage_path(chat_id: int) -> Path:
    return USAGE_DIR / f"{chat_id}.json"


def load_usage(chat_id: int) -> dict:
    path = usage_path(chat_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_usage(chat_id: int, data: dict):
    usage_path(chat_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_usage(chat_id: int, model: str, response_usage):
    data = load_usage(chat_id)
    bucket = data.setdefault(
        model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    )
    bucket["input"] += getattr(response_usage, "input_tokens", 0) or 0
    bucket["output"] += getattr(response_usage, "output_tokens", 0) or 0
    bucket["cache_read"] += (
        getattr(response_usage, "cache_read_input_tokens", 0) or 0
    )
    bucket["cache_write"] += (
        getattr(response_usage, "cache_creation_input_tokens", 0) or 0
    )
    save_usage(chat_id, data)


def cost_for_model(model: str, b: dict) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (
        b["input"] * p["in"]
        + b["output"] * p["out"]
        + b["cache_write"] * p["cw"]
        + b["cache_read"] * p["cr"]
    ) / 1_000_000


def format_cost(chat_id: int) -> str:
    data = load_usage(chat_id)
    if not data:
        return "расходов пока нет."
    lines = []
    total = 0.0
    for model, b in data.items():
        cost = cost_for_model(model, b)
        total += cost
        cached_pct = (
            100 * b["cache_read"] / max(1, b["input"] + b["cache_read"])
        )
        lines.append(
            f"{model_label(model)}\n"
            f"  in {b['input']:,} • out {b['output']:,} • "
            f"cache w {b['cache_write']:,} / r {b['cache_read']:,} ({cached_pct:.0f}% cached)\n"
            f"  ≈ ${cost:.3f}"
        )
    return f"💸 расходы\n\n" + "\n\n".join(lines) + f"\n\nвсего: ≈ ${total:.3f}"


# --- Прочее ---
def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


def get_history(chat_id: int) -> list[dict]:
    if chat_id not in conversations:
        conversations[chat_id] = []
    return conversations[chat_id]


def trim_history(history: list[dict]) -> list[dict]:
    max_messages = MAX_HISTORY * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def system_block(text: str) -> list[dict]:
    """system как блок с cache_control — Anthropic кеширует на 5 минут."""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


# --- Хэндлеры ---
HELP_TEXT = (
    "📝 диалог:\n"
    "/new — новый диалог (сбросить контекст)\n"
    "/regenerate [правка] — перегенерить последний ответ (с опц. правкой)\n"
    "/summarize — сжать историю через haiku, чтобы экономить токены\n\n"
    "🧠 память (сохраняется между сессиями):\n"
    "/remember <факт> — запомнить\n"
    "/forget <номер> — забыть по номеру\n"
    "/forget all — забыть всё\n"
    "/facts — все факты\n\n"
    "⚙️ настройки:\n"
    "/system — текущий system prompt\n"
    "/setsystem <текст> — задать новый (заменяет)\n"
    "/addprompt <текст> — добавить к существующему\n"
    "/model — текущая модель\n"
    "/model opus|sonnet|haiku — переключить модель в этом чате\n\n"
    "💸 /cost — расходы по этому чату\n"
    "ℹ️ /help — это сообщение"
)

BOT_COMMANDS = [
    ("new", "новый диалог (сбросить контекст)"),
    ("regenerate", "перегенерить последний ответ (можно с правкой)"),
    ("summarize", "сжать историю через haiku"),
    ("remember", "запомнить факт о персонаже"),
    ("forget", "забыть факт по номеру или всё"),
    ("facts", "показать все факты"),
    ("system", "текущий system prompt"),
    ("setsystem", "задать новый system prompt"),
    ("addprompt", "добавить к существующему промпту"),
    ("model", "переключить модель (opus/sonnet/haiku)"),
    ("cost", "расходы по этому чату"),
    ("help", "список команд"),
]


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("нет доступа.")
        return
    await update.message.reply_text(
        "привет. пиши что угодно — я отвечу через Anthropic API.\n\n" + HELP_TEXT
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("нет доступа.")
        return
    await update.message.reply_text(HELP_TEXT)


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    conversations[update.effective_chat.id] = []
    await update.message.reply_text("контекст сброшен. факты сохранены.")


async def cmd_remember(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    text = update.message.text.partition(" ")[2].strip()
    if not text:
        await update.message.reply_text(
            "напиши факт после /remember\n"
            "например: /remember у героини шрам на левой ладони"
        )
        return
    facts = load_facts(user_id)
    facts.append(text)
    save_facts(user_id, facts)
    await update.message.reply_text(f"✓ запомнила ({len(facts)})")


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    arg = update.message.text.partition(" ")[2].strip()

    facts = load_facts(user_id)
    if not facts:
        await update.message.reply_text("фактов нет.")
        return

    if arg.lower() == "all":
        save_facts(user_id, [])
        await update.message.reply_text("все факты удалены.")
        return

    try:
        idx = int(arg) - 1
        if 0 <= idx < len(facts):
            removed = facts.pop(idx)
            save_facts(user_id, facts)
            await update.message.reply_text(f"удалён: {removed}")
        else:
            await update.message.reply_text(
                f"номер от 1 до {len(facts)}, или /forget all"
            )
    except ValueError:
        await update.message.reply_text(
            "напиши номер: /forget 3\nили /forget all"
        )


async def cmd_facts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    facts = load_facts(update.effective_user.id)
    if not facts:
        await update.message.reply_text("пусто. добавь через /remember <факт>")
        return
    items = [f"{i + 1}. {f}" for i, f in enumerate(facts)]
    text = "🧠 факты:\n\n" + "\n\n".join(items)
    for chunk in split_by_paragraphs(text):
        await update.message.reply_text(chunk)


async def cmd_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    prompt = load_prompt(update.effective_chat.id)
    text = f"system prompt:\n\n{prompt}"
    for chunk in split_by_paragraphs(text):
        await update.message.reply_text(chunk)


async def cmd_addprompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    text = update.message.text.partition(" ")[2].strip()
    if not text:
        await update.message.reply_text("напиши текст после /addprompt")
        return
    chat_id = update.effective_chat.id
    current = load_prompt(chat_id)
    save_prompt(chat_id, current + "\n\n" + text)
    await update.message.reply_text("✓ добавлено к промпту.")


async def cmd_setsystem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    text = update.message.text.partition(" ")[2].strip()
    if not text:
        await update.message.reply_text("напиши промпт после /setsystem")
        return
    chat_id = update.effective_chat.id
    save_prompt(chat_id, text)
    conversations[chat_id] = []
    await update.message.reply_text("system prompt обновлён, контекст сброшен.")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    arg = update.message.text.partition(" ")[2].strip().lower()
    if not arg:
        current = load_model(chat_id)
        aliases = ", ".join(f"{a} → {m}" for a, m in MODELS.items())
        await update.message.reply_text(
            f"сейчас: {model_label(current)}\n\n"
            f"переключить: /model <alias>\n"
            f"доступно: {aliases}"
        )
        return
    if arg not in MODELS:
        await update.message.reply_text(
            f"неизвестный алиас. доступно: {', '.join(MODELS)}"
        )
        return
    save_model(chat_id, arg)
    await update.message.reply_text(f"✓ модель: {model_label(MODELS[arg])}")


async def cmd_cost(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    text = format_cost(update.effective_chat.id)
    for chunk in split_by_paragraphs(text):
        await update.message.reply_text(chunk)


async def cmd_regenerate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    note = update.message.text.partition(" ")[2].strip()
    history = get_history(chat_id)
    if not history or history[-1]["role"] != "assistant":
        await update.message.reply_text(
            "нечего перегенерить — нет последнего ответа."
        )
        return
    history.pop()
    if not history or history[-1]["role"] != "user":
        await update.message.reply_text(
            "история пуста или последний ход не пользователя."
        )
        return
    if note:
        history[-1]["content"] = (
            f"{history[-1]['content']}\n\n[правка: {note}]"
        )
    conversations[chat_id] = history
    await _generate_and_reply(update, chat_id, user_id, history)


async def cmd_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    history = get_history(chat_id)
    if len(history) < 4:
        await update.message.reply_text(
            "история слишком короткая, нечего сжимать."
        )
        return

    transcript = "\n\n".join(
        f"[{m['role']}] {m['content']}" for m in history
    )
    prompt = (
        "Ниже — диалог между пользователем и ассистентом-писателем. "
        "Сожми его в краткое содержание (до 500 слов): кто персонажи, "
        "что произошло, в какой точке остановились, какой тон и стиль. "
        "Сохрани важные детали и факты.\n\n"
        f"{transcript}"
    )

    await update.effective_chat.send_action("typing")
    try:
        response = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.content[0].text
        add_usage(chat_id, SUMMARY_MODEL, response.usage)
    except Exception as e:
        logger.error(f"summarize error: {e}")
        await update.message.reply_text(f"ошибка summarize: {e}")
        return

    conversations[chat_id] = [
        {"role": "user", "content": "Что произошло до этого момента?"},
        {"role": "assistant", "content": summary},
    ]
    await update.message.reply_text(
        "✓ история сжата. краткое содержание:"
    )
    for chunk in split_by_paragraphs(summary):
        await update.message.reply_text(chunk)


async def _generate_and_reply(update, chat_id, user_id, history):
    base_prompt = load_prompt(chat_id)
    facts = load_facts(user_id)
    system_prompt = build_system_prompt(base_prompt, facts)
    model = load_model(chat_id)

    await update.effective_chat.send_action("typing")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_block(system_prompt),
            messages=history,
        )
        assistant_text = response.content[0].text
        history.append({"role": "assistant", "content": assistant_text})
        conversations[chat_id] = trim_history(history)
        add_usage(chat_id, model, response.usage)

        for chunk in split_by_paragraphs(assistant_text):
            await update.message.reply_text(chunk)
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        await update.message.reply_text(f"ошибка API: {e.message}")
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"ошибка: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    user_text = update.message.text
    if not user_text:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    history = get_history(chat_id)
    history.append({"role": "user", "content": user_text})
    history = trim_history(history)
    conversations[chat_id] = history

    await _generate_and_reply(update, chat_id, user_id, history)


def split_by_paragraphs(text: str, limit: int = 4096) -> list[str]:
    """Разбивает текст по абзацам, не превышая лимит символов."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        block = (current + "\n\n" + para).strip() if current else para
        if len(block) <= limit:
            current = block
        else:
            if current:
                chunks.append(current)
            if len(para) > limit:
                for i in range(0, len(para), limit):
                    chunks.append(para[i : i + limit])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


async def register_commands(app: Application):
    await app.bot.set_my_commands([BotCommand(c, d) for c, d in BOT_COMMANDS])


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(register_commands)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("remember", cmd_remember))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("facts", cmd_facts))
    app.add_handler(CommandHandler("system", cmd_system))
    app.add_handler(CommandHandler("addprompt", cmd_addprompt))
    app.add_handler(CommandHandler("setsystem", cmd_setsystem))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("cost", cmd_cost))
    app.add_handler(CommandHandler("regenerate", cmd_regenerate))
    app.add_handler(CommandHandler("summarize", cmd_summarize))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
