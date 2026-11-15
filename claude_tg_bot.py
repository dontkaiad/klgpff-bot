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
import re
import logging
import time
from datetime import datetime, timedelta, timezone
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
import extract
import rag
from tg_logger import alert_error, alert_warn, exc_where, send_alert
from ops_transient import TransientTracker, is_transient
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
FACTS_FULL_LIMIT = int(os.environ.get("FACTS_FULL_LIMIT", "150"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32000"))
# Память сцен: сколько чанков сохранённых сцен подмешивать к запросу.
# Память сцен подмешивается, только когда включена (/context scenes on): хроника сюжета из
# выжимок (STORY_*) и, если SCENE_TOP_K > 0, ещё фрагменты сцен. Фрагменты по умолчанию выключены.
SCENE_TOP_K = int(os.environ.get("SCENE_TOP_K", "0"))
STORY_MAX_SCENES = int(os.environ.get("STORY_MAX_SCENES", "30"))
STORY_MAX_CHARS = int(os.environ.get("STORY_MAX_CHARS", "12000"))
# Выжимка сцены при сохранении (/done, /keep): один вызов SUMMARY_MODEL, до 150 слов.
SUMMARY_MAX_TOKENS = 800
SUMMARY_MAX_CHARS = 1500  # страховка: ответ длиннее обрезается
SUMMARY_MAX_INPUT_CHARS = 60000
# Вспомогательные вызовы (выжимка, извлечение фактов) идут на Haiku 5.5, у которого размышление по
# умолчанию включено и его токены входят в max_tokens: на извлечении весь лимит уходил на размышление
# и JSON не появлялся (stop_reason=max_tokens, 15.11.2026). По умолчанию размышление выключено.
# AUX_THINKING=adaptive возвращает поведение модели по умолчанию.
AUX_THINKING = os.environ.get("AUX_THINKING", "disabled").strip().lower()
# Защита от дорогих запросов. Дневной лимит считается по расходам за сегодня
# (дата по UTC+3, данные те же, что у /cost); 0 или меньше выключает его.
DAILY_LIMIT_USD = float(os.environ.get("DAILY_LIMIT_USD", "2.00"))
# Обычное сообщение (не команда) длиннее этого в модель не уходит: длинный текст
# это обычно сцена, вставленная частями, и каждая часть стоила бы запроса Opus.
MAX_INPUT_CHARS = 3500
# Части сцены, пересланные боту от него самого: сколько хранить без новых частей.
FORWARD_BUFFER_TTL = 30 * 60
# Режим /import и буфер частей: потолок буфера в символах и размер принимаемого файла.
IMPORT_BUFFER_MAX_CHARS = 60000
IMPORT_DOC_MAX_BYTES = 200 * 1024
# Извлечение кандидатов в факты (extract.py): после /done и после каждой сцены,
# которую написал бот (ответ от EXTRACT_MIN_WORDS слов). В факты попадает только по /accept.
EXTRACT_ENABLED = os.environ.get("EXTRACT_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")
EXTRACT_MIN_WORDS = int(os.environ.get("EXTRACT_MIN_WORDS", "300"))
SIM_DEDUP = float(os.environ.get("SIM_DEDUP", "0.92"))  # косинус, с которого кандидат считается дублем
EXTRACT_MAX_INPUT_CHARS = 60000
EXTRACT_MAX_TOKENS = int(os.environ.get("EXTRACT_MAX_TOKENS", "3000"))  # ответ обрезался на 1500
# Окно вставки: Telegram режет длинную вставку на много сообщений. После отказа по порогу
# MAX_INPUT_CHARS следующие обычные сообщения этого же юзера в течение PASTE_WINDOW_SECONDS
# (скользящее: каждое продлевает) тоже не идут в модель, иначе хвост короче порога уходит в Opus.
# 0 выключает окно. Два обычных сообщения с интервалом меньше PASTE_FAST_SECONDS открывают его же.
PASTE_WINDOW_SECONDS = float(os.environ.get("PASTE_WINDOW_SECONDS", "15"))
PASTE_FAST_SECONDS = 1.0

# Алиасы для /model → конкретные API model ID. ID берутся из .env, чтобы
# менять версию модели без правки кода.
MODELS = {
    "opus": os.environ.get("MODEL_OPUS", "claude-opus-5-5"),
    "sonnet": os.environ.get("MODEL_SONNET", "claude-sonnet-5-5"),
    "haiku": os.environ.get("MODEL_HAIKU", "claude-haiku-5-5"),
}
CLASSIFIER_MODEL = MODELS["haiku"]
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL") or MODELS["haiku"]
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL") or MODELS["haiku"]

# Цены ($ за 1M токенов): input, output, cache write 5min, cache read.
# Ключи — алиасы (opus/sonnet/haiku), а не ID: при смене версии в .env
# цены не отвалятся.
PRICING = {
    # Цена записи в кеш считается из "in" (1.25x для 5m, 2x для 1h) в cost_for_model.
    # Сверено с pricing-страницей Anthropic, ноябрь 2026: Opus 5.5 cache read $0.20,
    # Sonnet 5.5 $0.10 (снижен с $0.20 в октябре), Haiku 5.5 $0.01 при промпте до 100K
    # токенов (дороже в 5 раз выше этого порога, нашего размера это не касается).
    "opus": {"in": 4.0, "out": 20.0, "cw": 5.0, "cr": 0.20},
    "sonnet": {"in": 2.0, "out": 10.0, "cw": 2.5, "cr": 0.10},
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


# Закреплённый факт начинается с маркера ("📌 у героини шрам"). Он попадает в
# выдачу всегда, когда включён поиск, и не занимает места в RAG_MAX_FACTS.
# Маркер служебный: в канон и в текст для эмбеддинга уходит факт без него.
PIN_MARK = "📌"
PINNED_WARN = 40  # больше закреплённых: канон раздувается мимо RAG_MAX_FACTS


def is_pinned(fact: str) -> bool:
    return fact.lstrip().startswith(PIN_MARK)


def strip_pin(fact: str) -> str:
    """Факт без маркера (и без variation selector/пробелов после него)."""
    s = fact.lstrip()
    if s.startswith(PIN_MARK):
        s = s[len(PIN_MARK):].lstrip("️ \t")
    return s


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
            rag.index_facts(user_id, [strip_pin(f) for f in backup_facts])
        except Exception as e:
            logger.warning(f"RAG reindex after restore skipped for user {user_id}: {e}")


def reindex_user_facts(user_id: int):
    """Переиндексирует факты юзера в pgvector. Тихо логирует при недоступности RAG."""
    try:
        rag.index_facts(user_id, [strip_pin(f) for f in load_facts(user_id)])
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
    raw = load_facts(user_id)
    if not raw:
        return [], "no facts stored"
    all_facts = [strip_pin(f) for f in raw]  # маркер 📌 дальше не идёт никуда
    if FACTS_FULL_LIMIT and len(all_facts) <= FACTS_FULL_LIMIT:
        return all_facts, f"all (<= {FACTS_FULL_LIMIT})"
    # Поиск включён. Закреплённые идут первыми в авторском порядке и не считаются
    # в RAG_MAX_FACTS: их исключаем из поиска по индексу, а не вычитаем потом.
    pinned_idx = [i for i, f in enumerate(raw) if is_pinned(f)]
    pinned = [all_facts[i] for i in pinned_idx]
    if len(pinned) > PINNED_WARN:
        logger.warning(
            f"[user {user_id}] закреплённых фактов {len(pinned)} (> {PINNED_WARN}): "
            f"канон растёт мимо RAG_MAX_FACTS"
        )
    queries = retrieval_queries(history)
    if not queries:
        return all_facts, "all (no query)"
    if len(pinned) == len(raw):
        return pinned, "all pinned"
    try:
        hits = rag.search_facts_multi(
            user_id,
            queries,
            top_k=RAG_TOP_K,
            limit=RAG_MAX_FACTS,
            exclude=set(pinned_idx),
        )
        if hits:
            # Индекс мог устареть после ручной правки файла: чистим маркер и
            # убираем дубли с закреплёнными по тексту.
            seen = set(pinned)
            extra = []
            for h in (strip_pin(x) for x in hits):
                if h not in seen:
                    seen.add(h)
                    extra.append(h)
            suffix = f" + {len(pinned)} pinned" if pinned else ""
            return pinned + extra, f"semantic ({len(queries)} queries){suffix}"
        # БД доступна, но факты ещё не проиндексированы (напр. первый
        # запуск после деплоя) — наполняем индекс и используем все факты.
        reindex_user_facts(user_id)
        return all_facts, "fallback — index empty, backfilled"
    except Exception as e:
        logger.warning(f"RAG search failed, fallback to all facts: {e}")
        return all_facts, "fallback — vector DB unavailable"


def scenes_flag_path(user_id: int) -> Path:
    return MODELS_DIR / f"{user_id}.scenes"


def scenes_enabled(user_id: int) -> bool:
    """Подмешивание памяти сцен включено для юзера? По умолчанию нет: в запрос идут
    только промпт и факты. Состояние в файле models/<user_id>.scenes (как режим /model),
    переживает перезапуск."""
    try:
        return scenes_flag_path(user_id).read_text(encoding="utf-8").strip() == "on"
    except Exception:
        return False


def set_scenes_enabled(user_id: int, on: bool):
    scenes_flag_path(user_id).write_text("on" if on else "off", encoding="utf-8")


def get_memory_scenes(user_id: int, history: list[dict]) -> list:
    """Фрагменты сохранённых сцен для текущего запроса: [(номер, текст)].
    Те же поисковые запросы, что у фактов. Память сцен выключена (по умолчанию),
    сцен нет, поиск отключён или упал: [] и запрос уходит как раньше (при сбое
    warning в лог, без текста запроса)."""
    if SCENE_TOP_K < 1 or not scenes_enabled(user_id):
        return []
    queries = retrieval_queries(history)
    if not queries:
        return []
    try:
        return rag.search_scenes(user_id, queries, top_k=SCENE_TOP_K)
    except Exception as e:
        logger.warning(f"[user {user_id}] память сцен: поиск не удался, без справки: {e!r}")
        return []


def build_scenes_block(fragments: list) -> str:
    """Справка из сохранённых сцен. Идёт в хвост последнего user-сообщения сразу
    после канона. Это не канон: при расхождении прав канон."""
    if not fragments:
        return ""
    body = "\n\n".join(f"[сцена {n}]\n{text}" for n, text in fragments)
    return (
        "<memory_scenes>\n"
        "Фрагменты ранее сохранённых сцен автора. Это не канон. Если они "
        "противоречат <canon>, прав канон. Используй для преемственности "
        "(события, интонация, детали отношений), не пересказывай и не копируй "
        "дословно.\n\n"
        f"{body}\n"
        "</memory_scenes>"
    )


def select_chronicle(items: list):
    """Хроника для блока: самые свежие выжимки, что помещаются в STORY_MAX_SCENES сцен и
    STORY_MAX_CHARS символов. items: [(номер, выжимка)] по порядку сохранения. Возвращает
    (выбранные в том же порядке, сколько ранних не вошло)."""
    chosen, used = [], 0
    for number, summary in reversed(items):
        size = len(f"{number}. {summary}") + 1
        if len(chosen) >= STORY_MAX_SCENES or used + size > STORY_MAX_CHARS:
            break
        chosen.append((number, summary))
        used += size
    chosen.reverse()
    return chosen, len(items) - len(chosen)


def build_story_block(items: list) -> str:
    """Хроника сюжета: выжимки сохранённых сцен по порядку сохранения. Идёт в хвост
    последнего user-сообщения после канона. Нет выжимок: пусто, запрос как раньше."""
    chosen, skipped = select_chronicle(items)
    if not chosen:
        return ""
    note = f"(ранних сцен не вошло в хронику: {skipped})\n" if skipped else ""
    body = "\n".join(f"{n}. {summary}" for n, summary in chosen)
    return (
        "<story_so_far>\n"
        "Хроника ранее сохранённых сцен автора по порядку. Это не канон. Если она противоречит "
        "<canon>, прав канон. Продолжай сюжет с учётом событий и развития отношений, не "
        "повторяй сцены и не пересказывай их.\n"
        f"{note}\n"
        f"{body}\n"
        "</story_so_far>"
    )


def get_story(user_id: int) -> list:
    """Выжимки для хроники: [(номер, выжимка)]. Память сцен выключена (по умолчанию), лимиты
    нулевые, сцен или выжимок нет, БД упала: [] и запрос уходит как раньше."""
    if STORY_MAX_SCENES < 1 or STORY_MAX_CHARS < 1 or not scenes_enabled(user_id):
        return []
    try:
        return rag.story_chronicle(user_id)
    except Exception as e:
        logger.warning(f"[user {user_id}] хроника: не удалось получить выжимки, без неё: {e!r}")
        return []


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


def build_messages(history: list[dict], canon: str, scenes: str = "", story: str = "") -> list[dict]:
    """Копия истории для API: канон, за ним хроника сюжета (story_so_far) и справка из
    сцен (memory_scenes) дописаны к последнему user-сообщению, на последнем ответе ассистента стоит
    cache-breakpoint (кешируется всё до него). Сама история в памяти остаётся
    чистой — канон и сцены в неё не пишутся."""
    msgs = [dict(m) for m in history]
    if not msgs:
        return msgs
    last = msgs[-1]
    if last["role"] == "user":
        text = last["content"] if isinstance(last["content"], str) else ""
        for block in (canon, story, scenes):  # порядок: канон, хроника, фрагменты
            if block:
                text = f"{text}\n\n{block}"
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
    """Returns one of VALID_MODES. Default 'opus' (сцены пишет Opus, роутер
    включается явно через /model auto)."""
    path = model_path(chat_id)
    if path.exists():
        try:
            v = path.read_text(encoding="utf-8").strip().lower()
            if v in VALID_MODES:
                return v
        except Exception:
            pass
    return "opus"


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


TZ_MSK = timezone(timedelta(hours=3))
DAILY_KEEP_DAYS = 60  # сколько дней истории расходов хранить в _daily


def today_key() -> str:
    """Сегодняшняя дата по UTC+3, ключ в usage["_daily"]."""
    return datetime.now(TZ_MSK).strftime("%Y-%m-%d")


def _zero_bucket() -> dict:
    return {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


def add_usage(chat_id: int, model: str, response_usage):
    """Пишет токены в общий счётчик по модели и в разбивку по дням (ключ "_daily":
    дата -> модель -> bucket). Дневная разбивка нужна для DAILY_LIMIT_USD; суммы
    по моделям, которые читает /cost, лежат как раньше на верхнем уровне."""
    data = load_usage(chat_id)
    delta = usage_bucket(response_usage)
    day = data.setdefault("_daily", {}).setdefault(today_key(), {})
    for bucket in (data.setdefault(model, _zero_bucket()), day.setdefault(model, _zero_bucket())):
        for k, v in delta.items():
            bucket[k] += v
    for old in sorted(data["_daily"])[:-DAILY_KEEP_DAYS]:
        del data["_daily"][old]
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


def spent_today(chat_id: int) -> float:
    """Расходы чата за сегодня (по UTC+3), $. До появления _daily (старые данные)
    считается нулём: день деплоя начинается с чистого счёта."""
    day = load_usage(chat_id).get("_daily", {}).get(today_key(), {})
    return sum(cost_for_model(model, b) for model, b in day.items())


def limit_line(chat_id: int) -> str:
    spent = spent_today(chat_id)
    if DAILY_LIMIT_USD <= 0:
        return f"сегодня (UTC+3): ≈ ${spent:.2f}, дневной лимит выключен"
    return f"сегодня (UTC+3): ≈ ${spent:.2f} из ${DAILY_LIMIT_USD:.2f} (DAILY_LIMIT_USD)"


def over_daily_limit(chat_id: int) -> bool:
    """Дневной лимит исчерпан? (без ответа пользователю: для фоновых вызовов)"""
    return DAILY_LIMIT_USD > 0 and spent_today(chat_id) >= DAILY_LIMIT_USD


async def block_if_over_limit(update, chat_id: int) -> bool:
    """True и ответ пользователю, если дневной лимит исчерпан: вызывать API нельзя.
    Проверка стоит перед каждым входом в модель (классификатор тоже считается)."""
    if DAILY_LIMIT_USD <= 0:
        return False
    spent = spent_today(chat_id)
    if spent < DAILY_LIMIT_USD:
        return False
    logger.warning(
        f"{_tag(chat_id)} 🛑 daily limit reached: ${spent:.2f} of ${DAILY_LIMIT_USD:.2f}, API not called"
    )
    await update.message.reply_text(
        f"дневной лимит ${DAILY_LIMIT_USD:.2f} исчерпан (потрачено ${spent:.2f}), "
        "поднять можно в .env: DAILY_LIMIT_USD"
    )
    return True


def format_cost(chat_id: int) -> str:
    data = load_usage(chat_id)
    if not any(not k.startswith("_") for k in data):
        return "расходов пока нет.\n\n" + limit_line(chat_id)
    lines = []
    total = 0.0
    for model, b in data.items():
        if model.startswith("_"):  # служебные ключи (_daily), не модели
            continue
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
    return (
        "💸 расходы\n\n"
        + "\n\n".join(lines)
        + f"\n\nвсего: ≈ ${total:.3f}\n"
        + limit_line(chat_id)
    )


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
    "\n"
    "🧠 память (сохраняется между сессиями):\n"
    "/remember <факт> — запомнить (📌 в начале закрепляет: факт всегда в промпте)\n"
    "/forget <номер> — забыть по номеру\n"
    "/forget all — забыть всё\n"
    "/facts — все факты с номерами для /forget\n\n"
    "📥 загрузка текста (без модели, без расходов):\n"
    "/import — режим приёма: присылай части, пересылки или .txt/.md файл\n"
    "/done — сохранить принятое как сцену и найти кандидатов в факты\n"
    "/cancel — выйти из режима без сохранения\n"
    "/accept <номера|all> — принять кандидатов в факты\n"
    "/extract <номер> — повторить поиск кандидатов в факты для сохранённой сцены\n\n"
    "🎬 память сцен (только вручную, по умолчанию не подмешивается):\n"
    "/keep — сохранить сцену (ответом на сообщение бота; пересланные боту части сцены склеит; иначе последний ответ)\n"
    "/scenes — сохранённые сцены с номерами; /scenes N — выжимка сцены N\n"
    "/unkeep <номер> — удалить сцену\n\n"
    "⚙️ настройки:\n"
    "/context — что уйдёт в модель; /context scenes on|off — память сцен\n"
    "/addprompt <текст> — добавить к существующему\n"
    "/model — текущий режим\n"
    "/model auto — классификатор решает\n"
    "/model opus|sonnet|haiku — всегда эта модель, классификатор off (дефолт: opus)\n\n"
    "💸 /cost — расходы по этому чату, за сегодня и дневной лимит\n"
    "ℹ️ /help — это сообщение"
)

BOT_COMMANDS = [
    ("new", "новый диалог (сбросить контекст)"),
    ("scene", "длинная сцена через opus (1500+ слов)"),
    ("regenerate", "перегенерить последний ответ (можно с правкой)"),
    ("remember", "запомнить факт о персонаже"),
    ("forget", "забыть факт по номеру или всё"),
    ("facts", "показать все факты с номерами"),
    ("keep", "сохранить сцену в память (ответом на сообщение бота)"),
    ("scenes", "сохранённые сцены; /scenes N: выжимка"),
    ("unkeep", "удалить сохранённую сцену по номеру"),
    ("import", "режим приёма: части сцены или .txt, в конце /done"),
    ("done", "сохранить принятый текст как сцену"),
    ("cancel", "выйти из режима /import без сохранения"),
    ("accept", "принять кандидатов в факты: номера или all"),
    ("extract", "повторить поиск кандидатов в факты для сцены N"),
    ("context", "что уйдёт в модель; scenes on|off"),
    ("addprompt", "добавить к существующему промпту"),
    ("model", "режим модели (по умолчанию opus; auto включает роутер)"),
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
    clear_import(update.effective_user.id)  # буфер частей и режим /import
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


FACTS_MSG_LIMIT = 3800  # символов в одном сообщении со списком (лимит Telegram 4096)


def facts_messages(facts: list[str], limit: int = FACTS_MSG_LIMIT) -> list[str]:
    """Нумерованный список фактов, нарезанный на сообщения по границам фактов.

    Номер это позиция строки в facts/<id>.json с единицы, ровно то, что принимает
    /forget. Факт не режется, пока он сам помещается в limit; один факт длиннее
    limit режется по символам (остаток без номера), чтобы ничего не потерять.
    Закреплённые факты идут как хранятся, с маркером."""
    chunks: list[str] = []
    cur = ""
    for i, fact in enumerate(facts):
        item = f"{i + 1}. {fact}"
        while len(item) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(item[:limit])
            item = item[limit:]
        block = f"{cur}\n\n{item}" if cur else item
        if len(block) <= limit:
            cur = block
        else:
            chunks.append(cur)
            cur = item
    if cur:
        chunks.append(cur)
    if chunks:
        chunks[0] = "🧠 факты:\n\n" + chunks[0]
    return chunks


async def cmd_facts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    facts = load_facts(update.effective_user.id)
    if not facts:
        await update.message.reply_text("пусто. добавь через /remember <факт>")
        return
    for chunk in facts_messages(facts):
        await update.message.reply_text(chunk)


# --- Части сцены, пересланные боту от него самого ---
# Пересланные от бота сообщения в модель не идут никогда: бот складывает их в
# буфер (в памяти процесса, свой на пользователя), /keep без ответа сохраняет
# склеенную сцену. Буфер живёт FORWARD_BUFFER_TTL без новых частей, сбрасывается
# по /new и после сохранения.
forward_buffers: dict = {}  # user_id -> {"parts": [(дата, seq, текст)], "hashes", "seq", "chars", "last", "import"}
# user_id, у которых режим /import погас по таймауту, а юзер ещё не уведомлён.
import_expired: set = set()


def _now() -> float:
    return time.time()


def _live_buffer(user_id: int, create: bool = False):
    b = forward_buffers.get(user_id)
    if b is not None and _now() - b["last"] > FORWARD_BUFFER_TTL:
        forward_buffers.pop(user_id)
        if b.get("import"):
            import_expired.add(user_id)
        b = None
    if b is None and create:
        b = forward_buffers[user_id] = {
            "parts": [], "hashes": set(), "seq": 0, "chars": 0, "last": _now(), "import": False,
        }
    return b


def import_active(user_id: int) -> bool:
    b = _live_buffer(user_id)
    return bool(b and b["import"])


def clear_import(user_id: int):
    """Выход из режима /import и очистка буфера (по /new, /cancel, /done, /keep)."""
    forward_buffers.pop(user_id, None)
    import_expired.discard(user_id)


async def notify_import_expired(update, user_id: int) -> bool:
    """Если режим /import погас по таймауту, сообщает об этом один раз и возвращает
    True: текущее сообщение не обрабатывается (иначе часть фанфика ушла бы в модель)."""
    if user_id not in import_expired:
        return False
    import_expired.discard(user_id)
    await update.message.reply_text(
        "режим /import выключен: 30 минут без новых частей, буфер очищен. "
        "это сообщение не обработано; начни заново: /import"
    )
    return True


def forwarded_from_bot(msg, bot_id: int):
    """Дата исходного сообщения (unix), если сообщение переслано от самого бота,
    иначе None. Новый формат: forward_origin (MessageOriginUser.sender_user), его
    отдают установленные версии PTB (22.x). Старый forward_from/forward_date в них
    уже не приходит, проверка оставлена на случай другой версии."""
    origin = getattr(msg, "forward_origin", None)
    if origin is not None:
        user = getattr(origin, "sender_user", None)
        if getattr(origin, "type", "") == "user" and user is not None and user.id == bot_id:
            return origin.date.timestamp()
        return None
    user = getattr(msg, "forward_from", None)
    if user is not None and user.id == bot_id:
        date = getattr(msg, "forward_date", None) or getattr(msg, "date", None)
        return date.timestamp() if date else 0.0
    return None


def add_part(user_id: int, ts: float, text: str):
    """Кладёт часть в буфер. Возвращает (число частей, статус): "new", "dup" (тот же
    текст уже есть) или "full" (не помещается в IMPORT_BUFFER_MAX_CHARS)."""
    b = _live_buffer(user_id, create=True)
    b["last"] = _now()
    text = text.strip()
    h = rag.scene_hash(text)
    if h in b["hashes"]:
        return len(b["parts"]), "dup"
    if b["chars"] + len(text) > IMPORT_BUFFER_MAX_CHARS:
        return len(b["parts"]), "full"
    b["hashes"].add(h)
    b["seq"] += 1
    b["chars"] += len(text)
    b["parts"].append((ts, b["seq"], text))
    return len(b["parts"]), "new"


def buffered_parts(user_id: int) -> list:
    """Тексты частей по дате исходного сообщения; при равной дате (даты с точностью
    до секунды) по порядку прихода."""
    b = _live_buffer(user_id)
    if not b:
        return []
    return [t for _, _, t in sorted(b["parts"], key=lambda x: (x[0], x[1]))]


async def accept_part(update, user_id: int, ts: float, text: str, import_mode: bool):
    """Часть в буфер и короткий ответ без модели."""
    n, status = add_part(user_id, ts, text)
    logger.info(f"[user {user_id}] 📥 part {status}, {n} in buffer" + (" (import)" if import_mode else ""))
    if status == "new":
        reply = f"часть {n}, {len(text.split())} слов" if import_mode else f"часть {n} принята"
    elif status == "dup":
        reply = f"эта часть уже принята (всего {n})"
    else:
        reply = (
            f"буфер ограничен {IMPORT_BUFFER_MAX_CHARS} символов, эта часть не принята и дальше не принимаю. "
            "/done сохранит накопленное, /cancel сбросит"
        )
    await update.message.reply_text(reply)


# --- Память сцен ---
# Префикс запроса /scene: из request для /scenes он вырезается, иначе у всех
# сцен из /scene в списке была бы одна и та же начальная фраза.
SCENE_PROMPT_PREFIX = (
    "Напиши длинную атмосферную сцену. Используй все факты о персонажах. "
    "Минимум 1500 слов. Описание сцены: "
)
SCENE_LIST_REQUEST_CHARS = 60

# Ответ бота, ушедший в Telegram несколькими сообщениями: message_id каждой части
# → полный текст ответа. Нужен, чтобы /keep ответом на любую часть взял сцену
# целиком. Только в памяти процесса (как и история): после рестарта /keep берёт
# одно сообщение. Строка общая для всех частей, копий нет.
REPLY_TEXTS_MAX = 200
reply_texts: dict = {}


def remember_reply_parts(chat_id: int, message_ids: list, full_text: str):
    for mid in message_ids:
        reply_texts[(chat_id, mid)] = full_text
    while len(reply_texts) > REPLY_TEXTS_MAX:
        reply_texts.pop(next(iter(reply_texts)))


def request_for(history: list[dict], text: str) -> str:
    """Запрос юзера, на который бот ответил этим текстом: user-сообщение прямо
    перед ответом из истории. Нет такого ответа в истории: первые 60 символов
    самого текста."""
    for i in range(len(history) - 1, 0, -1):
        m = history[i]
        if m["role"] == "assistant" and m["content"] == text:
            prev = history[i - 1]
            if prev["role"] == "user" and isinstance(prev["content"], str):
                req = prev["content"]
                if req.startswith(SCENE_PROMPT_PREFIX):
                    req = req[len(SCENE_PROMPT_PREFIX):]
                return " ".join(req.split())
            break
    return " ".join(text.split())[:60]


async def cmd_keep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    replied = getattr(update.message, "reply_to_message", None)
    n_parts = 0
    if replied is None and import_active(user_id):
        await update.message.reply_text("идёт режим /import: заверши командой /done (или /cancel).")
        return
    if replied is None:
        parts = buffered_parts(user_id)
        n_parts = len(parts)
    if replied is not None:
        if not (replied.from_user and replied.from_user.is_bot):
            await update.message.reply_text("ответь на сообщение бота со сценой.")
            return
        text = reply_texts.get((chat_id, replied.message_id)) or replied.text or ""
    elif n_parts:
        text = "\n\n".join(parts)
    else:
        text = next(
            (
                m["content"]
                for m in reversed(get_history(chat_id))
                if m["role"] == "assistant" and isinstance(m["content"], str)
            ),
            "",
        )
    if not text.strip():
        await update.message.reply_text(
            "нечего сохранять: ответь командой /keep на сообщение со сценой."
        )
        return
    request = request_for(get_history(chat_id), text)
    try:
        number, created = rag.save_scene(user_id, request, text)
    except Exception as e:
        logger.warning(f"[user {user_id}] /keep: сохранить не удалось: {e!r}")
        await alert_warn("/keep: сохранить не удалось", e)
        await update.message.reply_text(
            "не удалось сохранить: БД или Voyage недоступны. попробуй позже."
        )
        return
    logger.info(
        f"[user {user_id}] 🎬 scene {number} {'saved' if created else 'duplicate'}"
        f" ({len(text.split())} words{f', {n_parts} parts' if n_parts else ''})"
    )
    if n_parts:
        forward_buffers.pop(user_id, None)  # при сбое выше буфер остаётся для повтора
    suffix = f" (частей: {n_parts})" if n_parts else ""
    if created:
        await update.message.reply_text(f"✓ сохранена: сцена {number}{suffix}")
    else:
        await update.message.reply_text(f"уже сохранена: сцена {number}{suffix}")
    await summarize_saved_scene(update, chat_id, user_id, number, created, text)


def scenes_messages(rows: list, limit: int = FACTS_MSG_LIMIT) -> list[str]:
    """Нумерованный список сцен «N. дата · request · слов», нарезанный на
    сообщения по границам строк (строка не рвётся). rows: [(created_at, request,
    слов, есть_выжимка)] в порядке номеров; номер это позиция с единицы, то же число
    принимает /unkeep и /scenes N."""
    lines = []
    for i, (created, request, words, has_summary) in enumerate(rows):
        req = " ".join(request.split())[:SCENE_LIST_REQUEST_CHARS]
        mark = "выжимка есть" if has_summary else "без выжимки"
        lines.append(f"{i + 1}. {created:%d.%m.%Y} · {req} · {words} сл. · {mark}")
    chunks: list[str] = []
    cur = ""
    for line in lines:
        block = f"{cur}\n{line}" if cur else line
        if len(block) <= limit:
            cur = block
        else:
            chunks.append(cur)
            cur = line
    if cur:
        chunks.append(cur)
    if chunks:
        chunks[0] = "🎬 сцены:\n\n" + chunks[0]
    return chunks


async def cmd_scenes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    arg = update.message.text.partition(" ")[2].strip()
    if arg:
        # /scenes N: выжимка сцены N (без полного текста), без вызова модели
        try:
            number = int(arg)
        except ValueError:
            await update.message.reply_text("напиши номер из /scenes: /scenes 3")
            return
        try:
            info = rag.get_scene_summary(update.effective_user.id, number)
        except Exception as e:
            logger.warning(f"/scenes {number}: БД недоступна: {e!r}")
            await update.message.reply_text("память сцен недоступна. попробуй позже.")
            return
        if info is None:
            await update.message.reply_text(f"сцены {number} нет, список: /scenes")
            return
        created, request, summary, words = info
        req = " ".join(request.split())[:SCENE_LIST_REQUEST_CHARS]
        head = f"сцена {number} · {created:%d.%m.%Y} · {req} · {words} сл."
        body = summary or "выжимки нет: сцена сохранена без неё"
        await update.message.reply_text(f"{head}\n\n📖 выжимка (проверь, нет ли выдумок):\n{body}")
        return
    try:
        rows = rag.list_scenes(update.effective_user.id)
    except Exception as e:
        logger.warning(f"/scenes: БД недоступна: {e!r}")
        await update.message.reply_text("память сцен недоступна. попробуй позже.")
        return
    if not rows:
        await update.message.reply_text("пусто. сохрани сцену командой /keep")
        return
    for chunk in scenes_messages(rows):
        await update.message.reply_text(chunk)


async def cmd_unkeep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    arg = update.message.text.partition(" ")[2].strip()
    try:
        number = int(arg)
    except ValueError:
        await update.message.reply_text("напиши номер из /scenes: /unkeep 3")
        return
    try:
        deleted = rag.delete_scene(user_id, number)
    except Exception as e:
        logger.warning(f"[user {user_id}] /unkeep: БД недоступна: {e!r}")
        await update.message.reply_text("память сцен недоступна. попробуй позже.")
        return
    if deleted:
        await update.message.reply_text(
            f"удалена: сцена {number}. номера после неё сдвинулись, см. /scenes"
        )
    else:
        await update.message.reply_text(f"сцены {number} нет, список: /scenes")


# --- Режим /import, извлечение кандидатов в факты, /accept, /context ---
async def cmd_import(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    b = _live_buffer(user_id, create=True)  # просроченный буфер заменяется новым
    import_expired.discard(user_id)  # новый /import: старое уведомление о таймауте не нужно
    b["import"] = True
    b["last"] = _now()
    n = len(b["parts"])
    await update.message.reply_text(
        "присылай части или .txt файл, в конце /done"
        + (f"\n(в буфере уже частей: {n})" if n else "")
    )


async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    active = import_active(user_id)
    if await notify_import_expired(update, user_id):
        return
    if not active:
        await update.message.reply_text("режим /import не включён. начни с /import")
        return
    parts = buffered_parts(user_id)
    if not parts:
        clear_import(user_id)
        await update.message.reply_text("буфер пуст, импорт завершён.")
        return
    text = "\n\n".join(parts)
    request = " ".join(text.split())[:60]
    try:
        number, created = rag.save_scene(user_id, request, text)
    except Exception as e:
        logger.warning(f"[user {user_id}] /done: сохранить не удалось: {e!r}")
        await alert_warn("/done: сохранить не удалось", e)
        await update.message.reply_text(
            "не удалось сохранить: БД или Voyage недоступны. буфер цел, повтори /done позже."
        )
        return
    clear_import(user_id)
    words = len(text.split())
    logger.info(
        f"[user {user_id}] 📥 import done: scene {number} {'saved' if created else 'duplicate'}"
        f" ({words} words, {len(parts)} parts)"
    )
    head = "✓ сохранена" if created else "уже сохранена"
    await update.message.reply_text(f"{head}: сцена {number} (частей: {len(parts)}, слов: {words})")
    await summarize_saved_scene(update, chat_id, user_id, number, created, text)
    extracted, cost = await run_extraction(
        update, chat_id, user_id, text, notify_empty=True, scene_number=number, notify_failure=True
    )
    logger.info(f"[user {user_id}] 📥 import extract: extracted={extracted} extract_cost=${cost:.4f}")


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    active = import_active(user_id)
    if await notify_import_expired(update, user_id):
        return
    b = forward_buffers.get(user_id)
    k = len(b["parts"]) if b else 0
    if not active and not k:
        await update.message.reply_text("нечего отменять.")
        return
    clear_import(user_id)
    await update.message.reply_text(f"импорт отменён, буфер очищен (частей: {k}). ничего не сохранено.")


def aux_call_kwargs() -> dict:
    """Параметры размышления для вспомогательных вызовов (см. AUX_THINKING)."""
    return {} if AUX_THINKING == "adaptive" else {"thinking": {"type": "disabled"}}


def describe_response(resp) -> str:
    """Служебные поля ответа для лога: stop_reason, токены, типы блоков (text/thinking).
    Без текста ответа."""
    u = usage_bucket(resp.usage)
    blocks = ",".join(getattr(b, "type", "?") for b in resp.content) or "none"
    return f"stop={getattr(resp, 'stop_reason', None)} in={u['input']} out={u['output']} blocks={blocks}"


def extraction_cut_message(scene_number) -> str:
    """Подсказка при оборванном ответе извлечения: после /done буфера уже нет, поэтому
    ведём на /extract (у сцены, которую ещё не сохранили, сначала /keep)."""
    if scene_number:
        return f"извлечение фактов оборвалось, попробуй /extract {scene_number}"
    return "извлечение фактов оборвалось: сохрани сцену командой /keep и запусти /extract <номер>"


async def run_extraction(
    update, chat_id: int, user_id: int, text: str, notify_empty: bool = False,
    scene_number=None, notify_failure: bool = False,
):
    """Один вызов EXTRACT_MODEL: кандидаты в факты и противоречия канону. Идёт после
    отправки сцены, результат отдельным сообщением без LLM. Возвращает (сколько
    кандидатов показано, стоимость $). Никогда не бросает: любой сбой это warning
    в лог и тишина для пользователя. В факты ничего не пишет, только /accept."""
    cost = 0.0
    if not EXTRACT_ENABLED:
        return 0, cost
    try:
        if over_daily_limit(chat_id):
            logger.warning(f"{_tag(chat_id)} 🛑 extraction skipped: daily limit reached")
            return 0, cost
        facts = [strip_pin(f) for f in load_facts(user_id)]
        user_msg = extract.build_extract_input(facts, text, EXTRACT_MAX_INPUT_CHARS)
        resp = await asyncio.to_thread(
            lambda: client.messages.create(
                model=EXTRACT_MODEL,
                max_tokens=EXTRACT_MAX_TOKENS,
                system=extract.EXTRACT_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                **aux_call_kwargs(),
            )
        )
        add_usage(chat_id, EXTRACT_MODEL, resp.usage)  # идёт в /cost и в дневной лимит
        logger.info(f"{_tag(chat_id)} extraction call: {describe_response(resp)}")
        cost = cost_for_model(EXTRACT_MODEL, usage_bucket(resp.usage))
        raw = response_text(resp)
        truncated = getattr(resp, "stop_reason", None) == "max_tokens"
        parsed = extract.parse_extraction(raw)
        if parsed is None and truncated:
            # Ответ оборвался по max_tokens: берём полные элементы JSON. Повторный вызов
            # автоматически не делаем (это ещё деньги), пользователю говорим один раз.
            parsed = extract.salvage_truncated(raw)
            if parsed is None:
                logger.warning(
                    f"{_tag(chat_id)} extraction: ответ обрезан (max_tokens={EXTRACT_MAX_TOKENS}), "
                    "достать нечего"
                )
                await update.message.reply_text(extraction_cut_message(scene_number))
                return 0, cost
            logger.warning(
                f"{_tag(chat_id)} extraction: ответ обрезан (max_tokens={EXTRACT_MAX_TOKENS}), "
                f"взяты полные элементы: {len(parsed[0])} фактов, {len(parsed[1])} противоречий"
            )
        if parsed is None:
            logger.warning(f"{_tag(chat_id)} extraction: ответ модели не разобран, пропуск")
            if notify_failure:
                await update.message.reply_text("извлечение фактов не получилось: ответ модели не разобран, подробности в логе")
            return 0, cost
        new_facts, conflicts = parsed
        pending = [t for _, t in rag.pending_candidates(user_id)]
        new_facts = extract.dedupe_normalized(new_facts, facts + pending)
        if new_facts:
            try:
                new_facts = await asyncio.to_thread(
                    rag.filter_similar, user_id, new_facts, pending, SIM_DEDUP
                )
            except Exception as e:
                logger.warning(f"{_tag(chat_id)} extraction: проверка близости пропущена: {e!r}")
        numbered = rag.add_candidates(user_id, new_facts) if new_facts else []
        if numbered or conflicts:
            await update.message.reply_text(extract.candidates_message(numbered, conflicts))
        elif notify_empty:
            await update.message.reply_text("новых фактов в тексте не найдено.")
        return len(numbered), cost
    except Exception as e:
        logger.warning(f"{_tag(chat_id)} extraction failed: {e!r}")
        if notify_failure:
            try:
                await update.message.reply_text("извлечение фактов не получилось, подробности в логе")
            except Exception:
                pass
        return 0, cost


SUMMARY_SYSTEM = (
    "Тебе дан текст сцены. Напиши выжимку до 150 слов: кто участвует, где и когда, что произошло, "
    "как изменились отношения и состояние героев, какие линии остались открытыми. "
    "Нейтральные формулировки, без откровенных подробностей. "
    "Не выдумывай ничего, чего нет в тексте. Ответ: только текст выжимки, без заголовков и пояснений."
)


async def make_summary(chat_id: int, text: str):
    """Выжимка сцены одним вызовом SUMMARY_MODEL. Возвращает (статус, текст, стоимость $):
    "ok", "skipped" (исчерпан дневной лимит) или "failed" (вызов упал, ответ обрезан или
    пуст). Никогда не бросает. Стоимость идёт в /cost и в дневной лимит."""
    cost = 0.0
    try:
        if over_daily_limit(chat_id):
            logger.warning(f"{_tag(chat_id)} 🛑 summary skipped: daily limit reached")
            return "skipped", None, cost
        resp = await asyncio.to_thread(
            lambda: client.messages.create(
                model=SUMMARY_MODEL,
                max_tokens=SUMMARY_MAX_TOKENS,
                system=SUMMARY_SYSTEM,
                messages=[{"role": "user", "content": text[:SUMMARY_MAX_INPUT_CHARS]}],
                **aux_call_kwargs(),
            )
        )
        add_usage(chat_id, SUMMARY_MODEL, resp.usage)
        logger.info(f"{_tag(chat_id)} summary call: {describe_response(resp)}")
        cost = cost_for_model(SUMMARY_MODEL, usage_bucket(resp.usage))
        if getattr(resp, "stop_reason", None) == "max_tokens":
            logger.warning(f"{_tag(chat_id)} summary: ответ обрезан (max_tokens={SUMMARY_MAX_TOKENS})")
            return "failed", None, cost
        summary = response_text(resp).strip()
        if not summary:
            logger.warning(f"{_tag(chat_id)} summary: пустой ответ")
            return "failed", None, cost
        return "ok", summary[:SUMMARY_MAX_CHARS], cost
    except Exception as e:
        logger.warning(f"{_tag(chat_id)} summary failed: {e!r}")
        return "failed", None, cost


async def summarize_saved_scene(update, chat_id: int, user_id: int, number: int, created: bool, text: str):
    """Выжимка для только что сохранённой сцены (или для дубля, у которого её ещё нет).
    Сцена к этому моменту уже сохранена: любой сбой это строка «выжимка не получилась» и
    warning, сохранение не страдает. Лимит исчерпан: молча (warning)."""
    try:
        if not created:
            info = rag.get_scene_summary(user_id, number)
            if info is None or info[2]:
                return  # сцены нет или выжимка уже есть
        status, summary, cost = await make_summary(chat_id, text)
        if status == "skipped":
            return
        if status == "ok":
            if rag.set_summary(user_id, number, summary):
                logger.info(f"[user {user_id}] 📖 summary scene {number}: ok, {len(summary)} chars, ${cost:.4f}")
                return
            logger.warning(f"[user {user_id}] summary: сцена {number} не найдена при записи")
    except Exception as e:
        logger.warning(f"[user {user_id}] summary: запись не удалась: {e!r}")
    await update.message.reply_text("выжимка не получилась")


async def cmd_extract(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/extract <номер>: повторно запускает извлечение кандидатов в факты для уже сохранённой
    сцены (номер как в /scenes). Один вызов EXTRACT_MODEL, результат как после /done."""
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    arg = update.message.text.partition(" ")[2].strip()
    try:
        number = int(arg)
    except ValueError:
        await update.message.reply_text("напиши номер сцены из /scenes: /extract 1")
        return
    try:
        text = rag.get_scene_text(user_id, number)
    except Exception as e:
        logger.warning(f"[user {user_id}] /extract {number}: БД недоступна: {e!r}")
        await update.message.reply_text("память сцен недоступна. попробуй позже.")
        return
    if text is None:
        await update.message.reply_text(f"сцены {number} нет, список: /scenes")
        return
    if not EXTRACT_ENABLED:
        await update.message.reply_text("извлечение фактов выключено (EXTRACT_ENABLED=false).")
        return
    if await block_if_over_limit(update, chat_id):
        return
    await update.message.reply_text(f"извлекаю кандидатов в факты из сцены {number}...")
    extracted, cost = await run_extraction(
        update, chat_id, user_id, text, notify_empty=True, scene_number=number, notify_failure=True
    )
    logger.info(f"[user {user_id}] 🧠 /extract scene {number}: extracted={extracted} extract_cost=${cost:.4f}")


def _fact_numbers(first: int, k: int) -> str:
    return f"№ {first}" if k == 1 else f"№ {first}-{first + k - 1}"


async def cmd_accept(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    arg = update.message.text.partition(" ")[2].strip().lower()
    usage = "напиши номера из списка кандидатов: /accept 1 3 5 или /accept all"
    if not arg:
        await update.message.reply_text(usage)
        return
    nums = None
    if arg != "all":
        try:
            nums = [int(t) for t in re.split(r"[,\s]+", arg) if t]
        except ValueError:
            await update.message.reply_text(usage)
            return
        if not nums or any(n < 1 for n in nums):
            await update.message.reply_text(usage)
            return
    try:
        claimed, already, missing = rag.claim_candidates(user_id, nums)
    except Exception as e:
        logger.warning(f"[user {user_id}] /accept: БД недоступна: {e!r}")
        await update.message.reply_text("кандидаты недоступны: БД не отвечает. попробуй позже.")
        return
    lines = []
    if claimed:
        facts = load_facts(user_id)
        first = len(facts) + 1
        facts.extend(t for _, t in claimed)
        try:
            save_facts(user_id, facts)
        except Exception as e:
            rag.release_candidates(user_id, [n for n, _ in claimed])  # кандидаты не потеряны
            logger.warning(f"[user {user_id}] /accept: запись фактов не удалась: {e!r}")
            await update.message.reply_text("не удалось записать факты, кандидаты остались. повтори позже.")
            return
        reindex_user_facts(user_id)  # тот же путь, что у /remember: индекс и бэкап файла
        lines.append(f"✓ добавлено фактов: {len(claimed)} (в /facts {_fact_numbers(first, len(claimed))})")
        lines += [f"{first + i}. {t[:80]}" for i, (_, t) in enumerate(claimed)]
    if already:
        lines.append("уже приняты раньше: " + ", ".join(map(str, already)))
    if missing:
        lines.append("нет таких кандидатов: " + ", ".join(map(str, missing)) + " (список живёт 24 часа)")
    if not lines:
        lines.append("нет ожидающих кандидатов.")
    await update.message.reply_text("\n".join(lines))


def context_report(user_id: int, chat_id: int) -> str:
    """Что уйдёт в модель при следующем запросе, одними числами: без текстов
    промпта и фактов, без вызова LLM."""
    raw = load_facts(user_id)
    total = len(raw)
    if not total:
        facts_line = "нет"
    elif FACTS_FULL_LIMIT and total <= FACTS_FULL_LIMIT:
        facts_line = f"все ({total})"
    else:
        facts_line = f"{sum(1 for f in raw if is_pinned(f))} закреплённых + поиск до {RAG_MAX_FACTS} из {total}"
    on = scenes_enabled(user_id)
    try:
        rows = rag.list_scenes(user_id)
        items = rag.story_chronicle(user_id)
        saved = str(len(rows))
        story_line = (
            f"хроника сюжета: {len(items)} сцен, {sum(len(s) for _, s in items)} симв.; "
            f"без выжимки: {len(rows) - len(items)}"
        )
        used = len(select_chronicle(items)[0])
        if on and used < len(items):
            story_line += f"; в запрос войдут последние {used}"
        if not on:
            story_line += " (не подмешивается: память сцен выкл)"
    except Exception:
        saved, story_line = "н/д", "хроника сюжета: н/д"
    scenes_line = ("вкл" if on else "выкл") + f", сохранено сцен: {saved}"
    fragments_line = f"до {SCENE_TOP_K}" if SCENE_TOP_K >= 1 else "выкл (SCENE_TOP_K=0)"
    mode = load_mode(chat_id)
    mode_line = "auto (роутер)" if mode == "auto" else f"{mode} ({MODELS[mode]})"
    lines = [
        "🧭 что уйдёт в модель при следующем запросе:",
        f"system-промпт: {len(load_prompt(chat_id))} симв.",
        f"факты: {facts_line}",
        f"память сцен: {scenes_line}",
        story_line,
        f"фрагменты сцен: {fragments_line}",
        f"режим модели: {mode_line}",
        f"извлечение фактов из сцен: {'вкл' if EXTRACT_ENABLED else 'выкл'}",
        limit_line(chat_id),
    ]
    if import_active(user_id):
        lines.append("режим /import: идёт (запросы к модели не уходят)")
    return "\n".join(lines)


async def cmd_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    args = update.message.text.partition(" ")[2].strip().lower().split()
    if args:
        if len(args) == 2 and args[0] == "scenes" and args[1] in ("on", "off"):
            set_scenes_enabled(user_id, args[1] == "on")
            await update.message.reply_text(
                "✓ память сцен включена: в запрос после канона идёт хроника сюжета (выжимки "
                "сохранённых сцен по порядку)."
                + (f" Фрагменты сцен: до {SCENE_TOP_K}." if SCENE_TOP_K >= 1 else "")
                if args[1] == "on"
                else "✓ память сцен выключена: в запрос идут только промпт и факты."
            )
            return
        await update.message.reply_text("использование: /context или /context scenes on|off")
        return
    await update.message.reply_text(context_report(user_id, update.effective_chat.id))


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
    if await block_if_over_limit(update, chat_id):
        return
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
    if await block_if_over_limit(update, chat_id):
        return
    wrapped = f"{SCENE_PROMPT_PREFIX}{desc}"
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


async def _generate_and_reply(update, chat_id, user_id, history, model_override=None):
    tag = _tag(chat_id)
    if await block_if_over_limit(update, chat_id):
        return
    base_prompt = load_prompt(chat_id)
    facts, fact_source = get_relevant_facts(user_id, history)
    canon = build_canon_block(facts)
    fragments = get_memory_scenes(user_id, history)
    scenes_block = build_scenes_block(fragments)
    story_items = get_story(user_id)
    story_block = build_story_block(story_items)
    model = model_override or load_model(chat_id)
    facts_chars = len(canon)

    logger.info(f"{tag} 🔍 RAG: {len(facts)} facts ({fact_source})")
    # get_relevant_facts() гасит исключение БД внутри (sync, без await) и
    # сообщает о деградации через fact_source — логируем её здесь, в async-точке.
    if "vector DB unavailable" in fact_source:
        await alert_warn("RAG fallback: vector DB unavailable", body=f"использовано фактов: {len(facts)}")
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
            messages=build_messages(history, canon, scenes_block, story_block),
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
            await alert_warn("empty response", body=f"stop_reason={final.stop_reason} · model={model} · usage={u}")
            await update.message.reply_text(
                f"пустой ответ (stop_reason={final.stop_reason})"
            )
            return

        history.append({"role": "assistant", "content": assistant_text})
        conversations[chat_id] = trim_history(history)

        chunks = split_by_paragraphs(assistant_text)
        sent_ids = []
        for chunk in chunks:
            sent = await update.message.reply_text(chunk)
            if getattr(sent, "message_id", None) is not None:
                sent_ids.append(sent.message_id)
        remember_reply_parts(chat_id, sent_ids, assistant_text)

        # Сцена от бота: после отправки, отдельным сообщением, предлагаем кандидатов в факты.
        extracted, extract_cost = 0, 0.0
        if EXTRACT_ENABLED and len(assistant_text.split()) >= EXTRACT_MIN_WORDS:
            extracted, extract_cost = await run_extraction(update, chat_id, user_id, assistant_text)

        u = usage_bucket(final.usage)
        alias = model_alias(model) or model
        req_cost = cost_for_model(model, u)
        logger.info(
            f"{tag} ✅ done: {len(assistant_text)} chars → "
            f"{len(chunks)} messages | "
            f"{alias}: {_fmt_tokens(u['input'])} in / {_fmt_tokens(u['output'])} out / "
            f"cache r {u['cache_read']} w {u['cache_write']} | "
            f"${req_cost:.2f} | scenes={len(fragments)} | "
            f"story={len(select_chronicle(story_items)[0]) if story_block else 0} | "
            f"extracted={extracted} extract_cost=${extract_cost:.4f}"
        )
    except anthropic.APIError as e:
        logger.error(f"{tag} ❌ Anthropic API error: {e}")
        await alert_error("generate: ошибка API", e)
        await update.message.reply_text(f"ошибка API: {e.message}")
    except Exception as e:
        logger.error(f"{tag} ❌ error: {e}")
        await alert_error("generate: ошибка", e)
        await update.message.reply_text(f"ошибка: {str(e)}")


CLASSIFIER_PROMPT = (
    "You are a router for a Russian/English creative-writing chat bot. Reply with EXACTLY ONE word: opus, sonnet, or haiku. No quotes, no punctuation, no explanation.\n\n"
    "Decide by what the REPLY must contain, not by the verbs used. Messages are casual and often have no verb (\"короткая сцена про них\", \"ещё одну\", \"а теперь в баре\").\n\n"
    "- opus — the reply is a NEW piece of prose: a scene, story, chapter or fragment written from scratch. Includes bare noun requests (\"сцена про X\", \"короткая сцена\", \"ещё одну сцену\"), a new situation to write about, \"придумай\", \"напиши\", \"write\".\n"
    "- sonnet — the reply is prose built on existing text: continue, rewrite, shorten, extend, change style / POV / tone, fix, add a detail (\"продолжай\", \"перепиши короче\", \"добавь деталь\", \"дальше\", \"а потом\", \"измени\").\n"
    "- haiku — ONLY when the reply is NOT prose to be read as a story: questions about characters or facts, planning, brainstorming, meta-talk, greetings, short clarifications.\n\n"
    "When unsure between haiku and anything else, choose sonnet. Never choose haiku for a request that could produce a scene.\n\n"
    "Output: opus | sonnet | haiku"
)


async def classify_message(chat_id: int, user_text: str) -> str:
    """Returns 'opus' | 'sonnet' | 'haiku'. Defaults to 'sonnet' on any error
    and when the reply is empty or names none of the three.
    async только ради await alert_warn в фолбэке; сам вызов client остаётся
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
        logger.warning(f"classify_message fallback to sonnet: {e}")
        await alert_warn("classifier fallback → sonnet", e)
    return "sonnet"


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Документы никогда не идут в модель. В режиме /import .txt и .md (UTF-8, до
    200 КБ) становятся частью буфера; вне режима файл не принимается (раньше .txt
    уходил прямо в Opus, мимо порога длины)."""
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    doc = update.message.document
    name = ((doc.file_name if doc else "") or "").lower()
    is_text_doc = name.endswith((".txt", ".md"))
    active = import_active(user_id)
    if await notify_import_expired(update, user_id):
        return
    if not active:
        if is_text_doc:
            logger.info(f"[user {user_id}] 📄 document outside /import: not sent to model")
            await update.message.reply_text(
                "файл не отправлен в модель, чтобы не тратить деньги. "
                "чтобы загрузить текст: /import, затем файл, затем /done"
            )
        return
    if not is_text_doc:
        await update.message.reply_text("в режиме /import принимаю только .txt и .md")
        return
    if doc.file_size and doc.file_size > IMPORT_DOC_MAX_BYTES:
        await update.message.reply_text(
            f"файл больше {IMPORT_DOC_MAX_BYTES // 1024} КБ ({doc.file_size // 1024} КБ). "
            "раздели его на части или перешли текстом"
        )
        return
    try:
        file = await doc.get_file()
        data = bytes(await file.download_as_bytearray())
    except Exception as e:
        logger.error(f"document read error: {e}")
        await alert_error("document read error", e)
        await update.message.reply_text(f"не смогла прочитать файл: {e}")
        return
    if len(data) > IMPORT_DOC_MAX_BYTES:
        await update.message.reply_text(f"файл больше {IMPORT_DOC_MAX_BYTES // 1024} КБ, не принят")
        return
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        await update.message.reply_text("файл не в UTF-8: пересохрани его в UTF-8 и пришли снова")
        return
    if not text.strip():
        await update.message.reply_text("файл пустой")
        return
    await accept_part(update, user_id, update.message.date.timestamp(), text, True)


# user_id -> {"until": конец окна (unix), "count": сколько сообщений отклонено, "last": дата
# предыдущего обычного сообщения}. Время берётся из даты сообщения Telegram, а не из момента
# обработки: пока бот ждёт ответ модели, очередь стоит, и по часам обработки окно бы не работало.
paste_state: dict = {}


def paste_check(user_id: int, ts: float, too_long: bool) -> str:
    """Что делать с обычным текстовым сообщением: "pass" (в модель), "reject_long"
    (длиннее порога, ответ пользователю), "reject_paste" (вторая вставка подряд, ответ
    пользователю) или "swallow" (внутри окна, молча). Отклонение открывает или продлевает окно."""
    if PASTE_WINDOW_SECONDS <= 0:
        return "reject_long" if too_long else "pass"
    st = paste_state.setdefault(user_id, {"until": 0.0, "count": 0, "last": None})
    fast = st["last"] is not None and 0 <= ts - st["last"] < PASTE_FAST_SECONDS
    st["last"] = ts
    if ts < st["until"]:
        st["until"] = ts + PASTE_WINDOW_SECONDS
        st["count"] += 1
        return "swallow"
    if too_long or fast:
        st["until"] = ts + PASTE_WINDOW_SECONDS
        st["count"] = 1
        return "reject_long" if too_long else "reject_paste"
    return "pass"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    user_text = update.message.text
    if not user_text:
        return

    user_id = update.effective_user.id
    import_mode = import_active(user_id)
    if await notify_import_expired(update, user_id):
        return
    ts = forwarded_from_bot(update.message, context.bot.id)
    if import_mode:
        # Режим /import: любой текст (и пересылка) молча в буфер, модель не зовётся,
        # порог MAX_INPUT_CHARS не действует.
        when = ts if ts is not None else update.message.date.timestamp()
        await accept_part(update, user_id, when, user_text, True)
        return
    if ts is not None:
        # Часть сцены, пересланная от самого бота: в модель не идёт, только буфер.
        await accept_part(update, user_id, ts, user_text, False)
        return

    date = getattr(update.message, "date", None)
    verdict = paste_check(
        user_id, date.timestamp() if date else _now(), len(user_text) > MAX_INPUT_CHARS
    )
    if verdict == "swallow":
        # Продолжение вставки: молча, одна строка в лог со счётчиком, без текста.
        logger.info(
            f"[user {user_id}] paste window: message swallowed "
            f"(#{paste_state[user_id]['count']}, {len(user_text)} chars)"
        )
        return
    if verdict == "reject_long":
        logger.warning(
            f"[user {user_id}] long text not sent to model: {len(user_text)} chars > {MAX_INPUT_CHARS}; "
            f"paste window {PASTE_WINDOW_SECONDS:g}s opened"
        )
        await update.message.reply_text(
            "длинный текст не отправлен в модель, чтобы не тратить деньги; "
            "для сохранения сцены: /import, части (или .txt), /done"
        )
        return
    if verdict == "reject_paste":
        logger.warning(
            f"[user {user_id}] several messages in a row not sent to model; "
            f"paste window {PASTE_WINDOW_SECONDS:g}s opened"
        )
        await update.message.reply_text(
            "несколько сообщений подряд не отправлены в модель, чтобы не тратить деньги; "
            "для сохранения сцены: /import, части (или .txt), /done"
        )
        return

    await _process_user_text(update, user_text)


async def _process_user_text(update: Update, user_text: str):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if await block_if_over_limit(update, chat_id):
        return  # до истории и классификатора: в историю запрос не попадает
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


async def log_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Одна строка в лог на каждую команду: имя и user_id, без текста сообщения.
    Стоит в группе -1 и ничего не блокирует: хендлеры команд в группе 0 отработают
    как обычно. Ловит и неизвестные команды, и команды от чужих пользователей."""
    msg = update.effective_message
    user = update.effective_user
    if not msg or not msg.text:
        return
    name = msg.text.split(maxsplit=1)[0].lstrip("/").split("@")[0].lower()
    logger.info(f"⌨️ command /{name} user={user.id if user else '?'}")


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
transient = TransientTracker()   # счётчик сетевых сбоев Telegram: /tmp/ops_transient.json
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
        try:                                       # «восстановлено» после тихих минут
            action = transient.tick()
            if action:
                await send_alert(*action)
        except Exception as e:
            logger.warning(f"transient tick failed: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def register_commands(app: Application):
    # Меню команд обновляется при каждом старте, но оно не стоит работы бота: если Telegram не
    # ответил (TimedOut на set_my_commands бывает, когда сам Telegram подвисает на этом методе),
    # исключение из post_init убивало процесс, и docker перезапускал его по кругу. Теперь
    # остаётся прежнее меню, бот работает, в ops-группу уходит одно предупреждение (только класс).
    try:
        await app.bot.set_my_commands([BotCommand(c, d) for c, d in BOT_COMMANDS])
    except Exception as e:
        logger.warning("set_my_commands failed, the command menu was not refreshed: %r", e)
        await alert_warn("меню команд не обновлено (set_my_commands), бот работает", e)
    # PTB-tracked task: держит strong-ref (без GC-гонки голого create_task) и
    # отменяется при shutdown'е приложения.
    app.create_task(heartbeat_loop())


async def error_handler(update, context):
    """Глобальный хук PTB: ловит ИСКЛЮЧЕНИЯ, не пойманные в хендлерах, и обрывы long polling.
    Локальные try/except (generate/summary/classifier) гасят свои ошибки и логируют сами:
    сюда они не доходят, дублей нет. Сетевые сбои Telegram только считаются (ops_transient),
    алерт один на всплеск. Остальное: класс исключения и место в коде, без текста и без трейса
    (полный текст в docker logs)."""
    err = context.error
    if is_transient(err):
        logger.warning(f"transient telegram/network error: {err!r}")
        action = transient.record(err)
        if action:
            await send_alert(*action)
        return
    logger.error(f"unhandled error: {err!r}")
    await alert_error("необработанная ошибка", err, exc_where(err))


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(register_commands)
        .build()
    )

    # группа -1: логирование всех команд до основных хендлеров (группа 0)
    app.add_handler(MessageHandler(filters.COMMAND, log_command), group=-1)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("remember", cmd_remember))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("facts", cmd_facts))
    app.add_handler(CommandHandler("keep", cmd_keep))
    app.add_handler(CommandHandler("scenes", cmd_scenes))
    app.add_handler(CommandHandler("unkeep", cmd_unkeep))
    app.add_handler(CommandHandler("import", cmd_import))
    app.add_handler(CommandHandler("done", cmd_done))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("accept", cmd_accept))
    app.add_handler(CommandHandler("extract", cmd_extract))
    app.add_handler(CommandHandler("context", cmd_context))
    app.add_handler(CommandHandler("addprompt", cmd_addprompt))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("cost", cmd_cost))
    app.add_handler(CommandHandler("regenerate", cmd_regenerate))
    app.add_handler(CommandHandler("scene", cmd_scene))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    restore_facts_from_backups()
    logger.info("🚀 бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
