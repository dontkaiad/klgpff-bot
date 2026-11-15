"""Живая проверка алертов: аналог OPS_TEST из heylark-infra/ops/ops-notify.sh, для Python.

Шлёт настоящие алерты (error, warning, recovered) в настоящий топик БЕЗ ЗВУКА, ждёт TTL секунд и удаляет
каждое сообщение через deleteMessage. Токен, chat_id и топик берутся из окружения (TG_LOG_BOT_TOKEN,
TG_LOG_CHAT_ID, TG_LOG_THREAD_KLGPFF), как в боте. message_id живёт только в памяти, токен не печатается.
Код выхода 3, если хоть одно сообщение не отправилось или не удалилось.

    TG_LOG_BOT_TOKEN=... TG_LOG_CHAT_ID=... TG_LOG_THREAD_KLGPFF=2 python scripts/ops_test.py [TTL]
"""
import asyncio
import os
import sys

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_logger  # noqa: E402


async def main(ttl: float) -> int:
    if not tg_logger.enabled:
        print("OPS_TEST: TG_LOG_BOT_TOKEN / TG_LOG_CHAT_ID / TG_LOG_THREAD_KLGPFF are not all set", file=sys.stderr)
        return 3
    cases = [
        ("🔴", "error", "ops-test: необработанная ошибка", "KeyError · claude_tg_bot.py:1 in handler"),
        ("🟠", "warning", "ops-test: Telegram API нестабилен: 5 сетевых ошибок за 10 мин", "NetworkError ×4, TimedOut ×1"),
        ("🟢", "recovered", "ops-test: Telegram/сеть снова в норме (тихо 10 мин)", "за время сбоя 5 ошибок: NetworkError ×4, TimedOut ×1"),
    ]
    ids = []
    for emoji, word, title, body in cases:
        mid = await tg_logger.send_alert(emoji, word, title, body, silent=True)
        if mid:
            ids.append(mid)
    failures = len(cases) - len(ids)
    print(f"OPS_TEST: sent {len(ids)}/{len(cases)} silent messages to thread={tg_logger.THREAD_ID}")
    await asyncio.sleep(ttl)
    deleted = 0
    url = f"https://api.telegram.org/bot{tg_logger.TOKEN}/deleteMessage"
    async with aiohttp.ClientSession() as session:
        for mid in ids:
            ok = False
            try:
                async with session.post(url, json={"chat_id": tg_logger.CHAT_ID, "message_id": mid},
                                        timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    ok = resp.status == 200 and (await resp.json()).get("ok") is True
            except Exception as e:  # noqa: BLE001 — текст исключения aiohttp может содержать URL с токеном
                print(f"OPS_TEST: delete error ({type(e).__name__})", file=sys.stderr)
            if ok:
                deleted += 1
            else:
                failures += 1
                print("OPS_TEST: DELETE FAILED: a message is still in the topic", file=sys.stderr)
    print(f"OPS_TEST: deleted {deleted}/{len(ids)} after {ttl:g}s, failures={failures}")
    return 3 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 8.0)))
