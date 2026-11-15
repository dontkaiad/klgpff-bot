"""Алерты в Telegram-топик klgpff. Только отправка (sendMessage), без команд и без polling.

Тот же бот и тот же механизм, что раньше: TG_LOG_BOT_TOKEN, TG_LOG_CHAT_ID, TG_LOG_THREAD_KLGPFF.
Текст собирается по общему шаблону ops-группы (как ops/ops-notify.sh в heylark-infra):
«<эмодзи> <b>🤖 klgpff</b> · слово», строка-заголовок, при необходимости строка курсивом; parse_mode HTML.

В алерты НЕ попадает пользовательский текст: ни сообщения, ни сцены, ни факты, ни repr исключения
(он содержит текст запроса и ответ API). Только класс исключения, код ответа, место в коде и счётчики.
Полный текст ошибки остаётся в `docker logs` (его пишут сами обработчики через logger).
"""
import html
import logging
import os
import time
import traceback
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)

TOKEN = os.environ.get("TG_LOG_BOT_TOKEN")
CHAT_ID = os.environ.get("TG_LOG_CHAT_ID")
THREAD_ID = os.environ.get("TG_LOG_THREAD_KLGPFF")

enabled = all((TOKEN, CHAT_ID, THREAD_ID))
if not enabled:
    log.warning("tg_logger: disabled, missing TG_LOG_* env vars")

API = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
LABEL = "🤖 klgpff"            # эмодзи и имя проекта из ops/topics.env (klgpff: топик 2)
DEDUP_SECONDS = 60.0           # одинаковый алерт не чаще раза в минуту
_last_sent: dict = {}


def format_alert(emoji: str, word: str, title: str = "", body: str = "") -> str:
    """Шапка, заголовок, курсивная строка: как error/warn/recovered в ops-notify.sh."""
    msg = f"{emoji} <b>{LABEL}</b> · {word}"
    if title:
        msg += "\n" + html.escape(title, quote=False)
    if body:
        msg += "\n<i>" + html.escape(body, quote=False) + "</i>"
    return msg


def exc_label(exc: BaseException) -> str:
    """Класс исключения (и HTTP-статус, если он есть). Без текста: в нём бывает запрос пользователя."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    return f"{name} (HTTP {status})" if isinstance(status, int) else name


def exc_where(exc: BaseException) -> str:
    """Файл:строка:функция последнего кадра трейса. Путь, не данные."""
    try:
        frames = traceback.extract_tb(exc.__traceback__)
        if not frames:
            return ""
        f = frames[-1]
        return f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}"
    except Exception:  # noqa: BLE001 — место в коде не стоит сбоя алерта
        return ""


async def _post(payload: dict) -> Optional[int]:
    """Один sendMessage. Возвращает message_id, если Telegram принял, иначе None. Не бросает исключений."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                API, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.warning("tg_logger: telegram %s: %s", resp.status, body)
                    return None
                data = await resp.json()
                return int(data.get("result", {}).get("message_id", 0)) or -1
    except Exception as e:
        # текст исключения aiohttp может содержать URL, а в нём токен бота
        log.warning("tg_logger: send failed (%s)", type(e).__name__)
        return None


async def send_alert(emoji: str, word: str, title: str = "", body: str = "", *,
                     silent: bool = False) -> Optional[int]:
    """Алерт по шаблону. silent=True без звука (так шлёт тестовый прогон scripts/ops_test.py).
    Одинаковый текст не чаще раза в DEDUP_SECONDS. Возвращает message_id или None."""
    if not enabled:
        return None
    text = format_alert(emoji, word, title, body)
    now = time.monotonic()
    prev = _last_sent.get(text)
    if prev is not None and now - prev < DEDUP_SECONDS:
        return None
    _last_sent[text] = now
    if len(_last_sent) > 200:           # не копить память на разных текстах
        for k in [k for k, t in _last_sent.items() if now - t >= DEDUP_SECONDS]:
            _last_sent.pop(k, None)
    payload = {
        "chat_id": CHAT_ID,
        "message_thread_id": THREAD_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if silent:
        payload["disable_notification"] = True
    return await _post(payload)


async def alert_error(title: str, exc: Optional[BaseException] = None, body: str = "") -> None:
    """🔴 error: заголовок + класс исключения (и body, например место в коде)."""
    parts = [p for p in (exc_label(exc) if exc is not None else "", body) if p]
    await send_alert("🔴", "error", title, " · ".join(parts))


async def alert_warn(title: str, exc: Optional[BaseException] = None, body: str = "") -> None:
    """🟠 warning: деградация, бот продолжает работать."""
    parts = [p for p in (exc_label(exc) if exc is not None else "", body) if p]
    await send_alert("🟠", "warning", title, " · ".join(parts))
