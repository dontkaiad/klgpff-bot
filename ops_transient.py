"""Транзиентные сбои Telegram/сети: не алерт на каждую ошибку, а счётчик и один алерт на всплеск.

PTB отдаёт в error_handler и обрывы long polling (NetworkError, TimedOut, RetryAfter): их бывает
несколько в час, ничего не ломается, бот переподключается сам. Поэтому:
  * такая ошибка только считается (по классам, по суткам UTC) и пишется в /tmp/ops_transient.json;
    суточные счётчики читает «пульс» host-монитора (monitoring/monitor.sh в heylark-infra);
  * один алерт «Telegram API нестабилен», когда за 10 минут набралось 5 и больше;
  * «восстановлено», когда после этого прошло 10 тихих минут (проверяет tick() из heartbeat-цикла).
Никакого текста исключений: только имена классов и числа.
"""
import json
import os
import time
from collections import Counter, deque
from typing import Callable, Optional, Tuple

COUNTERS_PATH = "/tmp/ops_transient.json"
WINDOW_SECONDS = 600.0          # окно, в котором считаем всплеск
BURST_COUNT = 5                 # столько ошибок в окне = один алерт
QUIET_SECONDS = 600.0           # столько тишины до «восстановлено»

# Имена классов по MRO: TimedOut и BadGateway-подобные подпадают через NetworkError.
TRANSIENT_NAMES = frozenset({
    "NetworkError", "TimedOut", "RetryAfter", "TelegramNetworkError", "TelegramServerError",
    "TelegramRetryAfter", "ServerDisconnectedError", "ClientConnectorError", "ClientOSError",
    "ConnectionResetError", "TimeoutError", "ReadTimeout", "ConnectTimeout", "ReadError",
    "ConnectError", "RemoteProtocolError", "PoolTimeout",
})
# В PTB BadRequest наследует NetworkError, но это не сбой сети, а наш баг (длинное сообщение, битая
# разметка): такие ошибки всегда алертим, даже если их класс стоит под NetworkError.
NEVER_TRANSIENT = frozenset({"BadRequest", "Forbidden", "InvalidToken", "Conflict", "ChatMigrated"})
TRANSIENT_MARKERS = ("Bad Gateway", "Request timeout error", "Read timed out")

Action = Tuple[str, str, str, str]      # (эмодзи, слово шаблона, заголовок, курсивная строка)


def is_transient(exc: BaseException) -> bool:
    if any(c.__name__ in NEVER_TRANSIENT for c in type(exc).__mro__):
        return False
    if any(c.__name__ in TRANSIENT_NAMES for c in type(exc).__mro__):
        return True
    text = str(exc)[:300]               # только для сравнения, наружу не уходит
    return any(m in text for m in TRANSIENT_MARKERS)


def _top(counts: Counter, n: int = 3) -> str:
    return ", ".join(f"{name} ×{c}" for name, c in counts.most_common(n))


class TransientTracker:
    def __init__(self, path: str = COUNTERS_PATH, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        self.events: deque = deque()            # (время, класс) за последнее окно
        self.alerted = False
        self.last_event = 0.0
        self.incident = Counter()               # классы с начала всплеска
        self.day: Optional[str] = None
        self.counts: Counter = Counter()
        self.prev_day: Optional[str] = None
        self.prev_counts: Counter = Counter()
        self._load()

    # ---------- счётчики по суткам ----------
    def _today(self, now: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(now))

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.day, self.prev_day = d.get("day"), d.get("prev_day")
            self.counts, self.prev_counts = Counter(d.get("counts") or {}), Counter(d.get("prev_counts") or {})
        except Exception:  # noqa: BLE001 — нет файла или битый: начинаем с нуля
            pass

    def _roll(self, now: float) -> bool:
        """Перейти на новые сутки UTC. True, если сутки сменились (файл надо обновить)."""
        today = self._today(now)
        if self.day is None:
            self.day = today
            return False
        if today != self.day:
            self.prev_day, self.prev_counts = self.day, self.counts
            self.day, self.counts = today, Counter()
            return True
        return False

    def _save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"day": self.day, "counts": dict(self.counts),
                           "prev_day": self.prev_day, "prev_counts": dict(self.prev_counts)}, fh)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 — счётчик не стоит сбоя бота
            pass

    # ---------- события ----------
    def record(self, exc: BaseException) -> Optional[Action]:
        """Учесть транзиентную ошибку. Возвращает алерт, когда всплеск впервые достиг порога."""
        now = self.clock()
        name = type(exc).__name__
        self._roll(now)
        self.counts[name] += 1
        self._save()
        self.events.append((now, name))
        while self.events and now - self.events[0][0] > WINDOW_SECONDS:
            self.events.popleft()
        self.last_event = now
        if self.alerted:
            self.incident[name] += 1
            return None
        if len(self.events) >= BURST_COUNT:
            self.alerted = True
            self.incident = Counter(n for _, n in self.events)
            return ("🟠", "warning", f"Telegram API нестабилен: {len(self.events)} сетевых ошибок за 10 мин",
                    _top(self.incident))
        return None

    def tick(self) -> Optional[Action]:
        """Зовётся из heartbeat-цикла. «Восстановлено» после QUIET_SECONDS тишины."""
        now = self.clock()
        if self._roll(now):
            self._save()
        if self.alerted and now - self.last_event >= QUIET_SECONDS:
            self.alerted = False
            total = sum(self.incident.values())
            top = _top(self.incident)
            self.incident = Counter()
            self.events.clear()
            return ("🟢", "recovered", f"Telegram/сеть снова в норме (тихо {int(QUIET_SECONDS // 60)} мин)",
                    f"за время сбоя {total} ошибок: {top}")
        return None
