"""
Telegram-бот для creative writing через Anthropic API.
С памятью фактов, переключением моделей, кешированием промпта и учётом расходов.

Запуск:
1. pip install python-telegram-bot anthropic
2. Создай .env файл (см. .env.example)
3. python claude_tg_bot.py
"""

import asyncio
import json
import os
import logging
import time
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
import rag
from tg_logger import send_log
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
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")  # для RAG (rag.py)
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
# История режется пачками, а не по одному сообщению: при скользящем окне префикс
# менялся бы на каждом ходу и prompt cache не попадал бы никогда. Дошли до
# MAX_HISTORY пар → оставляем HISTORY_KEEP последних пар и дальше растём заново.
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "8"))
HISTORY_KEEP = int(os.environ.get("HISTORY_KEEP", "4"))
# TTL prompt cache: "5m" или "1h". При паузах между ходами >5 мин (читаешь сцену,
# пишешь правку) 1h выгоднее: запись дороже, но кеш не протухает.
CACHE_TTL = os.environ.get("CACHE_TTL", "1h")
# RAG: сколько фактов тянуть на один запрос и сколько отдать в промпт максимум.
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "30"))
RAG_MAX_FACTS = int(os.environ.get("RAG_MAX_FACTS", "50"))
# Пока фактов не больше этого порога — отдаём все без поиска (0 = RAG всегда).
FACTS_FULL_LIMIT = int(os.environ.get("FACTS_FULL_LIMIT", "0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32000"))

# Алиасы для /model → конкретные API model ID. ID берутся из .env, чтобы
# менять версию модели без правки кода.
MODELS = {
    "opus": os.environ.get("MODEL_OPUS", "claude-opus-5-5"),
    "sonnet": os.environ.get("MODEL_SONNET", "claude-sonnet-5-5"),
    "haiku": os.environ.get("MODEL_HAIKU", "claude-haiku-5-5"),
}
SUMMARY_MODEL = MODELS["haiku"]
CLASSIFIER_MODEL = MODELS["haiku"]

# Цены ($ за 1M токенов): input, output, cache write 5min, cache read.
# Ключи — алиасы (opus/sonnet/haiku), а не ID: при смене версии в .env
# цены не отвалятся.
PRICING = {
    # Цена записи в кеш считается из "in" (1.25x для 5m, 2x для 1h) в cost_for_model.
    # Opus 5.5: cache read = 5% от input. Sonnet/Haiku — сверить на pricing-странице.
    "opus": {"in": 4.0, "out": 20.0, "cw": 5.0, "cr": 0.20},
    "sonnet": {"in": 2.0, "out": 10.0, "cw": 2.5, "cr": 0.20},
    "haiku": {"in": 0.10, "out": 0.50, "cw": 0.125, "cr": 0.01},
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
# Бэкапы фактов живут ВНЕ facts/, в отдельном volume — чтобы reset/пересборка
# рабочей папки facts/ не уносила единственную копию (источник правды — сервер).
FACTS_BACKUP_DIR = BASE_DIR / "backups" / "facts"
for d in (FACTS_DIR, PROMPTS_DIR, MODELS_DIR, USAGE_DIR):
    d.mkdir(exist_ok=True)
FACTS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
# Заглушаем болтливые библиотеки (getUpdates-поллинг, HTTP-строки), чтобы
# в логе были видны наши собственные события.
for _noisy in ("httpx", "httpcore", "telegram", "telegram.ext", "apscheduler", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _tag(chat_id: int) -> str:
    return f"[chat {chat_id}]"


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def response_text(message) -> str:
    """Склеивает только text-блоки ответа. У моделей с thinking первым блоком
    может идти ThinkingBlock без .text."""
    return "".join(
        b.text for b in message.content if getattr(b, "type", "") == "text"
    )


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

conversations: dict[int, list[dict]] = {}


# --- Факты ---
def facts_path(user_id: int) -> Path:
    return FACTS_DIR / f"{user_id}.json"


def facts_backup_path(user_id: int) -> Path:
    return FACTS_BACKUP_DIR / f"{user_id}.json"


def _read_facts_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"failed to read facts from {path}: {e}")
        return []


def load_facts(user_id: int) -> list[str]:
    return _read_facts_file(facts_path(user_id))


def _dump_facts(facts: list[str]) -> str:
    return json.dumps(facts, ensure_ascii=False, indent=2)


def save_facts(user_id: int, facts: list[str]):
    facts_path(user_id).write_text(_dump_facts(facts), encoding="utf-8")
    # Зеркалируем в backups/ вне facts/ — переживает reset/пересборку рабочей папки.
    try:
        facts_backup_path(user_id).write_text(_dump_facts(facts), encoding="utf-8")
        logger.info(f"[user {user_id}] 💾 backup saved ({len(facts)} facts)")
    except Exception as e:
        logger.warning(f"facts backup failed for user {user_id}: {e}")


def restore_facts_from_backups():
    """На старте: если рабочий facts/<id>.json пуст/отсутствует, а в backups/
    есть непустая копия — восстановить файл и переиндексировать в pgvector."""
    for bpath in sorted(FACTS_BACKUP_DIR.glob("*.json")):
        try:
            user_id = int(bpath.stem)
        except ValueError:
            continue
        backup_facts = _read_facts_file(bpath)
        if not backup_facts:
            continue  # пустой бэкап — восстанавливать нечего
        if load_facts(user_id):
            continue  # рабочий файл на месте и непустой — не трогаем
        facts_path(user_id).write_text(_dump_facts(backup_facts), encoding="utf-8")
        logger.info(
            f"[user {user_id}] ♻️ facts restored from backup ({len(backup_facts)} facts)"
        )
        try:
            rag.index_facts(user_id, backup_facts)
        except Exception as e:
            logger.warning(f"RAG reindex after restore skipped for user {user_id}: {e}")


def reindex_user_facts(user_id: int):
    """Переиндексирует факты юзера в pgvector. Тихо логирует при недоступности RAG."""
    try:
        rag.index_facts(user_id, load_facts(user_id))
    except Exception as e:
        logger.warning(f"RAG reindex skipped for user {user_id}: {e}")


def retrieval_queries(history: list[dict]) -> list[str]:
    """Два запроса вместо одного. Просьба вроде «перепиши» или «ещё» сама по
    себе ни на какие факты не похожа, поэтому второй запрос — контекст: прошлые
    реплики юзера и начало последнего ответа (там уже названы герои и место)."""
    users = [m["content"] for m in history if m["role"] == "user"]
    current = users[-1][:1500] if users else ""
    prev_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    context = "\n\n".join(users[-3:-1] + ([prev_assistant[:800]] if prev_assistant else []))
    return [q for q in (current, context[:2000]) if q.strip()]


def get_relevant_facts(user_id: int, history: list[dict]):
    """Факты для текущего запроса. Возвращает (facts, source), где source —
    человекочитаемое описание источника для лога. Fallback на все факты:
    при недоступности БД с векторами, пустом запросе или ненаполненном индексе."""
    all_facts = load_facts(user_id)
    if not all_facts:
        return [], "no facts stored"
    if FACTS_FULL_LIMIT and len(all_facts) <= FACTS_FULL_LIMIT:
        return all_facts, f"all (<= {FACTS_FULL_LIMIT})"
    queries = retrieval_queries(history)
    if not queries:
        return all_facts, "all (no query)"
    try:
        hits = rag.search_facts_multi(
            user_id, queries, top_k=RAG_TOP_K, limit=RAG_MAX_FACTS
        )
        if hits:
            return hits, f"semantic ({len(queries)} queries)"
        # БД доступна, но факты ещё не проиндексированы (напр. первый
        # запуск после деплоя) — наполняем индекс и используем все факты.
        reindex_user_facts(user_id)
        return all_facts, "fallback — index empty, backfilled"
    except Exception as e:
        logger.warning(f"RAG search failed, fallback to all facts: {e}")
        return all_facts, "fallback — vector DB unavailable"


# --- Промпты ---
def prompt_path(chat_id: int) -> Path:
    return PROMPTS_DIR / f"{chat_id}.txt"


def load_prompt(chat_id: int) -> str:
    path = prompt_path(chat_id)
    if not path.exists():
        return SYSTEM_PROMPT
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"failed to load prompt for chat {chat_id} from {path}: {e}")
        return SYSTEM_PROMPT


def save_prompt(chat_id: int, prompt: str):
    prompt_path(chat_id).write_text(prompt, encoding="utf-8")


def build_canon_block(facts: list[str]) -> str:
    """Канон для текущего запроса. Идёт в хвост последнего user-сообщения, а не
    в system: system и история остаются стабильным префиксом для prompt cache,
    а факты (меняются от запроса к запросу) стоят рядом с самой просьбой."""
    if not facts:
        return ""
    facts_block = "\n".join(f"- {f}" for f in facts)
    return (
        "<canon>\n"
        "Канон персонажей и мира — обязателен к соблюдению. Любая написанная "
        "сцена должна соответствовать ему: ни один факт не должен быть нарушен, "
        "а все релевантные сцене факты должны быть отражены в повествовании. "
        "Если канон не покрывает деталь — не выдумывай противоречащую ему.\n\n"
        f"{facts_block}\n"
        "</canon>\n\n"
        "Вплетай эти детали в текст естественно — не выводи их списком и не "
        "называй внутри сцены «фактами» или «каноном»."
    )


def build_messages(history: list[dict], canon: str) -> list[dict]:
    """Копия истории для API: канон дописан к последнему user-сообщению, а на
    последнем ответе ассистента стоит cache-breakpoint (кешируется всё до него).
    Сама история в памяти остаётся чистой — канон в неё не пишется."""
    msgs = [dict(m) for m in history]
    if not msgs:
        return msgs
    last = msgs[-1]
    if last["role"] == "user":
        text = last["content"] if isinstance(last["content"], str) else ""
        if canon:
            text = f"{text}\n\n{canon}"
        last["content"] = text
    if len(msgs) >= 2 and msgs[-2]["role"] == "assistant":
        prev = msgs[-2]
        msgs[-2] = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": prev["content"],
                    "cache_control": cache_control(),
                }
            ],
        }
    return msgs


# --- Модели ---
def model_path(chat_id: int) -> Path:
    return MODELS_DIR / f"{chat_id}.txt"


VALID_MODES = ("auto", "opus", "sonnet", "haiku")


def load_mode(chat_id: int) -> str:
    """Returns one of VALID_MODES. Default 'auto'."""
    path = model_path(chat_id)
    if path.exists():
        try:
            v = path.read_text(encoding="utf-8").strip().lower()
            if v in VALID_MODES:
                return v
        except Exception:
            pass
    return "auto"


def save_mode(chat_id: int, mode: str):
    model_path(chat_id).write_text(mode, encoding="utf-8")


def load_model(chat_id: int) -> str:
    """Resolve mode → concrete model id. 'auto' falls back to DEFAULT_MODEL."""
    mode = load_mode(chat_id)
    if mode == "auto":
        return DEFAULT_MODEL
    return MODELS[mode]


# Деприкейтнутые ID, под которыми писался usage до перехода на новые версии.
# Нужны чтобы /cost не терял исторические расходы после ребренда модели.
LEGACY_MODEL_ALIASES = {
    "claude-opus-4-20250514": "opus",
    "claude-sonnet-4-20250514": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
    "claude-opus-4-8": "opus",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5": "haiku",
}


def model_alias(model_id: str) -> str:
    """Reverse MODELS: concrete id → alias (opus/sonnet/haiku). Резолвит и
    текущие ID из MODELS, и деприкейтнутые legacy-ID. '' если неизвестен."""
    for alias, full in MODELS.items():
        if full == model_id:
            return alias
    return LEGACY_MODEL_ALIASES.get(model_id, "")


def model_label(model_id: str) -> str:
    alias = model_alias(model_id)
    return f"{alias} ({model_id})" if alias else model_id


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


def usage_bucket(response_usage) -> dict:
    """Извлекает токены из Anthropic usage в плоский bucket-словарь."""
    return {
        "input": getattr(response_usage, "input_tokens", 0) or 0,
        "output": getattr(response_usage, "output_tokens", 0) or 0,
        "cache_read": getattr(response_usage, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(response_usage, "cache_creation_input_tokens", 0) or 0,
    }


def add_usage(chat_id: int, model: str, response_usage):
    data = load_usage(chat_id)
    bucket = data.setdefault(
        model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    )
    for k, v in usage_bucket(response_usage).items():
        bucket[k] += v
    save_usage(chat_id, data)


def cost_for_model(model: str, b: dict) -> float:
    # usage пишется по model_id; цены — по алиасу, поэтому маппим id → alias.
    p = PRICING.get(model_alias(model))
    if not p:
        return 0.0
    # запись в кеш: 1.25x input для TTL 5m, 2x input для TTL 1h
    cw = p["in"] * (2.0 if CACHE_TTL == "1h" else 1.25)
    return (
        b["input"] * p["in"]
        + b["output"] * p["out"]
        + b["cache_write"] * cw
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
        history = history[-(HISTORY_KEEP * 2):]
    # Anthropic API requires the first message to be from the user.
    while history and history[0]["role"] != "user":
        history = history[1:]
    return history


def cache_control() -> dict:
    cc = {"type": "ephemeral"}
    if CACHE_TTL == "1h":
        cc["ttl"] = "1h"
    return cc


def system_block(text: str) -> list[dict]:
    """system как блок с cache_control. Он статичный (без фактов), поэтому
    кеш попадает на каждом ходу."""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": cache_control(),
        }
    ]


# --- Хэндлеры ---
HELP_TEXT = (
    "📝 диалог:\n"
    "/new — новый диалог (сбросить контекст)\n"
    "/scene <описание> — длинная сцена через opus (1500+ слов)\n"
    "/regenerate [правка] — перегенерить последний ответ (с опц. правкой)\n"
    "/summarize — сжать историю через haiku, чтобы экономить токены\n"
    "\n"
    "🧠 память (сохраняется между сессиями):\n"
    "/remember <факт> — запомнить\n"
    "/forget <номер> — забыть по номеру\n"
    "/forget all — забыть всё\n"
    "/facts — все факты\n\n"
    "⚙️ настройки:\n"
    "/system — текущий system prompt\n"
    "/setsystem <текст> — задать новый (заменяет)\n"
    "/addprompt <текст> — добавить к существующему\n"
    "/model — текущий режим\n"
    "/model auto — классификатор решает (дефолт)\n"
    "/model opus|sonnet|haiku — всегда эта модель, классификатор off\n\n"
    "💸 /cost — расходы по этому чату\n"
    "ℹ️ /help — это сообщение"
)

BOT_COMMANDS = [
    ("new", "новый диалог (сбросить контекст)"),
    ("scene", "длинная сцена через opus (1500+ слов)"),
    ("regenerate", "перегенерить последний ответ (можно с правкой)"),
    ("summarize", "сжать историю через haiku"),
    ("remember", "запомнить факт о персонаже"),
    ("forget", "забыть факт по номеру или всё"),
    ("facts", "показать все факты"),
    ("system", "текущий system prompt"),
    ("setsystem", "задать новый system prompt"),
    ("addprompt", "добавить к существующему промпту"),
    ("model", "режим модели (auto/opus/sonnet/haiku)"),
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
    reindex_user_facts(user_id)
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
        reindex_user_facts(user_id)
        await update.message.reply_text("все факты удалены.")
        return

    try:
        idx = int(arg) - 1
        if 0 <= idx < len(facts):
            removed = facts.pop(idx)
            save_facts(user_id, facts)
            reindex_user_facts(user_id)
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
        mode = load_mode(chat_id)
        if mode == "auto":
            body = "режим: auto — классификатор решает"
        else:
            body = f"режим: {mode} ({MODELS[mode]}) — всегда эта модель"
        await update.message.reply_text(
            f"{body}\n\n"
            "переключить: /model <auto|opus|sonnet|haiku>"
        )
        return
    if arg not in VALID_MODES:
        await update.message.reply_text(
            f"доступно: {', '.join(VALID_MODES)}"
        )
        return
    save_mode(chat_id, arg)
    if arg == "auto":
        await update.message.reply_text("✓ режим: auto (классификатор)")
    else:
        await update.message.reply_text(
            f"✓ режим: {arg} ({MODELS[arg]}) — классификатор пропускается"
        )


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
    logger.info(f"{_tag(chat_id)} 🔁 regenerate" + (f' (edit: "{note[:60]}")' if note else ""))
    await _generate_and_reply(update, chat_id, user_id, history)


async def cmd_scene(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    desc = update.message.text.partition(" ")[2].strip()
    if not desc:
        await update.message.reply_text(
            "напиши описание после /scene\n"
            "пример: /scene первая встреча героев в портовом городе"
        )
        return
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    wrapped = (
        "Напиши длинную атмосферную сцену. Используй все факты о персонажах. "
        f"Минимум 1500 слов. Описание сцены: {desc}"
    )
    history = get_history(chat_id)
    history.append({"role": "user", "content": wrapped})
    history = trim_history(history)
    conversations[chat_id] = history

    tag = _tag(chat_id)
    logger.info(f'{tag} 📩 /scene "{desc[:80]}"')
    logger.info(f"{tag} 📌 scene → opus ({MODELS['opus']})")
    await _generate_and_reply(
        update, chat_id, user_id, history, model_override=MODELS["opus"]
    )


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
        summary = response_text(response)
        add_usage(chat_id, SUMMARY_MODEL, response.usage)
    except Exception as e:
        logger.error(f"summarize error: {e}")
        await send_log(f"🔴 klgpff summarize error: {e!r}")
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


async def _generate_and_reply(update, chat_id, user_id, history, model_override=None):
    tag = _tag(chat_id)
    base_prompt = load_prompt(chat_id)
    facts, fact_source = get_relevant_facts(user_id, history)
    canon = build_canon_block(facts)
    model = model_override or load_model(chat_id)
    facts_chars = len(canon)

    logger.info(f"{tag} 🔍 RAG: {len(facts)} facts ({fact_source})")
    # get_relevant_facts() гасит исключение БД внутри (sync, без await) и
    # сообщает о деградации через fact_source — логируем её здесь, в async-точке.
    if "vector DB unavailable" in fact_source:
        await send_log(
            f"⚠️ klgpff RAG fallback: vector DB unavailable ({len(facts)} facts used)"
        )
    logger.info(
        f"{tag} 🧠 prompt: system {len(base_prompt)} chars + canon {facts_chars} chars, "
        f"history {len(history)} msgs"
    )
    logger.info(f"{tag} ✍️ generating...")

    await update.effective_chat.send_action("typing")

    try:
        # Streaming обязателен: при больших max_tokens (32000) генерация
        # может занять >10 мин, и SDK без stream бросает ошибку.
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_block(base_prompt),
            messages=build_messages(history, canon),
        ) as stream:
            final = stream.get_final_message()
        assistant_text = response_text(final)
        add_usage(chat_id, model, final.usage)

        if not assistant_text.strip():
            # Пустой текст при stop_reason=max_tokens значит, что бюджет ушёл
            # на размышления. В history не пишем, reply_text('') не зовём.
            u = usage_bucket(final.usage)
            logger.warning(
                f"{tag} ⚠️ empty response: stop_reason={final.stop_reason}, usage={u}"
            )
            await send_log(
                f"⚠️ klgpff empty response: stop_reason={final.stop_reason} "
                f"usage={u} model={model}"
            )
            await update.message.reply_text(
                f"пустой ответ (stop_reason={final.stop_reason})"
            )
            return

        history.append({"role": "assistant", "content": assistant_text})
        conversations[chat_id] = trim_history(history)

        chunks = split_by_paragraphs(assistant_text)
        for chunk in chunks:
            await update.message.reply_text(chunk)

        u = usage_bucket(final.usage)
        alias = model_alias(model) or model
        req_cost = cost_for_model(model, u)
        logger.info(
            f"{tag} ✅ done: {len(assistant_text)} chars → "
            f"{len(chunks)} messages | "
            f"{alias}: {_fmt_tokens(u['input'])} in / {_fmt_tokens(u['output'])} out / "
            f"cache r {u['cache_read']} w {u['cache_write']} | "
            f"${req_cost:.2f}"
        )
    except anthropic.APIError as e:
        logger.error(f"{tag} ❌ Anthropic API error: {e}")
        await send_log(f"🔴 klgpff generate API error: {e!r}")
        await update.message.reply_text(f"ошибка API: {e.message}")
    except Exception as e:
        logger.error(f"{tag} ❌ error: {e}")
        await send_log(f"🔴 klgpff generate error: {e!r}")
        await update.message.reply_text(f"ошибка: {str(e)}")


CLASSIFIER_PROMPT = (
    "You are a router for a creative-writing chat bot. The user writes in Russian "
    "or English. Classify the user's message and reply with EXACTLY ONE word: "
    "opus, sonnet, or haiku. No quotes, no punctuation, no explanation.\n\n"
    "Rules:\n"
    "- opus — request to WRITE A NEW scene/fanfic from scratch.\n"
    "  RU triggers: «напиши», «придумай», «создай сцену», «сочини», «напиши сцену».\n"
    "  EN triggers: \"write\", \"create\", \"make a scene\", \"come up with\".\n"
    "- sonnet — continue / rewrite / extend / edit existing text.\n"
    "  RU triggers: «продолжи», «перепиши», «допиши», «измени», «отредактируй».\n"
    "  EN triggers: \"continue\", \"rewrite\", \"extend\", \"edit\", \"change\".\n"
    "- haiku — everything else: discussion, questions, short replies, meta-talk "
    "about characters, planning, clarifications.\n\n"
    "Output: opus | sonnet | haiku"
)


async def classify_message(chat_id: int, user_text: str) -> str:
    """Returns 'opus' | 'sonnet' | 'haiku'. Defaults to 'haiku' on any error.
    async только ради await send_log в фолбэке; сам вызов client остаётся
    синхронным, как и в остальном коде."""
    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=4,
            system=CLASSIFIER_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )
        add_usage(chat_id, CLASSIFIER_MODEL, response.usage)
        result = response_text(response).strip().lower()
        for alias in ("opus", "sonnet", "haiku"):
            if alias in result:
                return alias
    except Exception as e:
        logger.warning(f"classify_message fallback to haiku: {e}")
        await send_log(f"⚠️ klgpff classifier fallback→haiku: {e!r}")
    return "haiku"


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    doc = update.message.document
    if not doc or not doc.file_name or not doc.file_name.lower().endswith(".txt"):
        return

    try:
        file = await doc.get_file()
        file_bytes = await file.download_as_bytearray()
        user_text = file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"document read error: {e}")
        await send_log(f"🔴 klgpff document read error: {e!r}")
        await update.message.reply_text(f"не смогла прочитать файл: {e}")
        return

    caption = update.message.caption or ""
    if caption:
        user_text = f"{caption}\n\n{user_text}"

    await _process_user_text(update, user_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    user_text = update.message.text
    if not user_text:
        return

    await _process_user_text(update, user_text)


async def _process_user_text(update: Update, user_text: str):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    history = get_history(chat_id)
    history.append({"role": "user", "content": user_text})
    history = trim_history(history)
    conversations[chat_id] = history

    tag = _tag(chat_id)
    logger.info(f'{tag} 📩 "{user_text[:80]}"')

    mode = load_mode(chat_id)
    if mode == "auto":
        alias = await classify_message(chat_id, user_text)
        source = "router"
    else:
        alias = mode
        source = "manual"
    model_id = MODELS[alias]
    icon = "🧭" if source == "router" else "📌"
    logger.info(f"{tag} {icon} {source} → {alias} ({model_id})")

    await _generate_and_reply(
        update, chat_id, user_id, history, model_override=model_id
    )


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


HEARTBEAT_PATH = "/tmp/heartbeat"
HEARTBEAT_INTERVAL = 30


async def heartbeat_loop():
    """Раз в HEARTBEAT_INTERVAL сек пишет unix-время в HEARTBEAT_PATH — внешний
    healthcheck считает бота живым, пока файл свежий. Ошибка записи не должна
    ронять бота: тело цикла под try/except, sleep — снаружи (на ошибке всё
    равно ждём интервал, без busy-loop)."""
    while True:
        try:
            Path(HEARTBEAT_PATH).write_text(str(int(time.time())), encoding="utf-8")
        except Exception as e:
            logger.warning(f"heartbeat write failed: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def register_commands(app: Application):
    await app.bot.set_my_commands([BotCommand(c, d) for c, d in BOT_COMMANDS])
    await send_log("🚀 klgpff поднялся")
    # PTB-tracked task: держит strong-ref (без GC-гонки голого create_task) и
    # отменяется при shutdown'е приложения.
    app.create_task(heartbeat_loop())


async def error_handler(update, context):
    """Глобальный хук PTB: ловит ИСКЛЮЧЕНИЯ, не пойманные в хендлерах.
    Локальные try/except (generate/summary/classifier) гасят свои ошибки и
    логируют сами — сюда они не доходят, дублей нет. Шлём тип+текст, не трейс."""
    await send_log(f"🔴 klgpff error: {context.error!r}")


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
    app.add_handler(CommandHandler("scene", cmd_scene))
    app.add_handler(MessageHandler(filters.Document.MimeType("text/plain"), handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    restore_facts_from_backups()
    logger.info("🚀 бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
