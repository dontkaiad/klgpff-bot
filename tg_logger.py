import os
import logging
import aiohttp

log = logging.getLogger(__name__)

TOKEN = os.environ.get("TG_LOG_BOT_TOKEN")
CHAT_ID = os.environ.get("TG_LOG_CHAT_ID")
THREAD_ID = os.environ.get("TG_LOG_THREAD_KLGPFF")

enabled = all((TOKEN, CHAT_ID, THREAD_ID))
if not enabled:
    log.warning("tg_logger: disabled, missing TG_LOG_* env vars")

API = f"https://api.telegram.org/bot{TOKEN}/sendMessage"


async def send_log(text: str) -> None:
    if not enabled:
        return
    payload = {
        "chat_id": CHAT_ID,
        "message_thread_id": THREAD_ID,
        "text": text,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                API, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.warning("tg_logger: telegram %s: %s", resp.status, body)
    except Exception as e:
        log.warning("tg_logger: send failed: %s", e)
