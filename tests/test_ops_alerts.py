"""Алерты klgpff: общий шаблон ops-группы, без пользовательского текста, транзиентные только считаются.

Запуск: TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m pytest tests/test_ops_alerts.py
Сети нет: отправка подменена.
"""
import asyncio
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1:a")
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
import httpx  # noqa: E402
import telegram.error as tgerr  # noqa: E402

import claude_tg_bot as bot  # noqa: E402
import ops_transient  # noqa: E402
import tg_logger  # noqa: E402

SECRET = "ЛИЧНАЯ СЦЕНА: тайный текст пользователя sk-ant-SECRET"


def run(coro):
    return asyncio.run(coro)


class FormatTests(unittest.TestCase):
    def test_template_shape(self):
        self.assertEqual(
            tg_logger.format_alert("🟠", "warning", "RAG fallback", "использовано фактов: 3"),
            "🟠 <b>🤖 klgpff</b> · warning\nRAG fallback\n<i>использовано фактов: 3</i>",
        )
        self.assertEqual(tg_logger.format_alert("🔴", "error", "x"), "🔴 <b>🤖 klgpff</b> · error\nx")

    def test_html_is_escaped(self):
        self.assertNotIn("<script>", tg_logger.format_alert("🔴", "error", "<script>", "a & b"))
        self.assertIn("a &amp; b", tg_logger.format_alert("🔴", "error", "t", "a & b"))

    def test_exception_label_is_the_class_not_the_text(self):
        self.assertEqual(tg_logger.exc_label(ValueError(SECRET)), "ValueError")
        err = anthropic.APIStatusError(
            SECRET, response=httpx.Response(529, request=httpx.Request("POST", "http://x")), body=None)
        label = tg_logger.exc_label(err)
        self.assertEqual(label, "APIStatusError (HTTP 529)")
        self.assertNotIn("ЛИЧНАЯ", label)

    def test_where_is_a_path_not_data(self):
        try:
            raise RuntimeError(SECRET)
        except RuntimeError as e:
            where = tg_logger.exc_where(e)
        self.assertRegex(where, r"^test_ops_alerts\.py:\d+ in test_where_is_a_path_not_data$")
        self.assertNotIn("ЛИЧНАЯ", where)


class SendTests(unittest.TestCase):
    def setUp(self):
        self.sent = []

        async def fake_post(payload):
            self.sent.append(payload)
            return 77

        for p in (mock.patch.object(tg_logger, "enabled", True),
                  mock.patch.object(tg_logger, "CHAT_ID", "-100"),
                  mock.patch.object(tg_logger, "THREAD_ID", "2"),
                  mock.patch.object(tg_logger, "_post", fake_post),
                  mock.patch.dict(tg_logger._last_sent, clear=True)):
            p.start()
            self.addCleanup(p.stop)

    def test_payload_has_html_topic_and_sound(self):
        self.assertEqual(run(tg_logger.alert_error("generate: ошибка", ValueError(SECRET))), None)
        p = self.sent[0]
        self.assertEqual((p["chat_id"], p["message_thread_id"], p["parse_mode"]), ("-100", "2", "HTML"))
        self.assertEqual(p["text"], "🔴 <b>🤖 klgpff</b> · error\ngenerate: ошибка\n<i>ValueError</i>")
        self.assertNotIn("disable_notification", p)
        self.assertNotIn("ЛИЧНАЯ", p["text"])

    def test_silent_and_message_id(self):
        mid = run(tg_logger.send_alert("🟢", "recovered", "ok", silent=True))
        self.assertEqual(mid, 77)
        self.assertIs(self.sent[0]["disable_notification"], True)

    def test_same_alert_not_twice_in_a_minute(self):
        run(tg_logger.alert_warn("x"))
        run(tg_logger.alert_warn("x"))
        run(tg_logger.alert_warn("y"))
        self.assertEqual(len(self.sent), 2)

    def test_disabled_sends_nothing(self):
        with mock.patch.object(tg_logger, "enabled", False):
            run(tg_logger.alert_error("x", ValueError("a")))
        self.assertEqual(self.sent, [])


class TransientTests(unittest.TestCase):
    def tracker(self):
        self.now = [1_000_000_000.0]
        self.path = os.path.join(tempfile.mkdtemp(), "t.json")
        return ops_transient.TransientTracker(self.path, clock=lambda: self.now[0])

    def test_classification(self):
        for e in (tgerr.NetworkError("x"), tgerr.TimedOut("x"), tgerr.RetryAfter(3),
                  httpx.ReadError("x"), ConnectionResetError(), RuntimeError("Bad Gateway")):
            self.assertTrue(ops_transient.is_transient(e), repr(e))
        # BadRequest наследует NetworkError в PTB, но это не сбой сети
        self.assertTrue(issubclass(tgerr.BadRequest, tgerr.NetworkError))
        for e in (ValueError("x"), KeyError("x"), tgerr.BadRequest("x"), tgerr.Forbidden("x"),
                  tgerr.Conflict("x"), tgerr.InvalidToken("x"),
                  anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))):
            self.assertFalse(ops_transient.is_transient(e), repr(e))

    def test_four_errors_count_but_do_not_alert_fifth_does_once(self):
        t = self.tracker()
        for i in range(4):
            self.assertIsNone(t.record(tgerr.NetworkError("x")))
            self.now[0] += 30
        action = t.record(tgerr.TimedOut("x"))
        self.assertEqual(action[:2], ("🟠", "warning"))
        self.assertIn("5 сетевых ошибок за 10 мин", action[2])
        self.assertIn("NetworkError ×4", action[3])
        for _ in range(10):                       # дальше в том же всплеске молчим
            self.assertIsNone(t.record(tgerr.NetworkError("x")))

    def test_slow_errors_never_alert(self):
        t = self.tracker()
        for _ in range(30):                       # по одной в 3 минуты: в окне не больше 4
            self.assertIsNone(t.record(tgerr.NetworkError("x")))
            self.now[0] += 180

    def test_recovered_after_ten_quiet_minutes_once(self):
        t = self.tracker()
        for _ in range(5):
            t.record(tgerr.NetworkError("x"))
        self.now[0] += 599
        self.assertIsNone(t.tick())
        self.now[0] += 2
        action = t.tick()
        self.assertEqual(action[:2], ("🟢", "recovered"))
        self.assertIn("5 ошибок", action[3])
        self.assertIsNone(t.tick())
        for _ in range(5):                        # новый всплеск после восстановления снова алертит
            a = t.record(tgerr.NetworkError("x"))
        self.assertIsNotNone(a)

    def test_counters_file_for_the_pulse(self):
        t = self.tracker()
        t.record(tgerr.NetworkError("x"))
        t.record(tgerr.NetworkError("x"))
        t.record(tgerr.TimedOut("x"))
        d = json.load(open(self.path))
        self.assertEqual(d["counts"], {"NetworkError": 2, "TimedOut": 1})
        self.assertIsNone(d["prev_day"])
        self.now[0] += 86400                      # сутки сменились
        t.record(tgerr.NetworkError("x"))
        d = json.load(open(self.path))
        self.assertEqual(d["prev_counts"], {"NetworkError": 2, "TimedOut": 1})
        self.assertEqual(d["counts"], {"NetworkError": 1})
        self.assertEqual(sum(d["prev_counts"].values()), 3)   # так читает monitor.sh
        t2 = ops_transient.TransientTracker(self.path, clock=lambda: self.now[0])   # рестарт процесса
        self.assertEqual(t2.counts, {"NetworkError": 1})

    def test_no_exception_text_in_actions_or_file(self):
        t = self.tracker()
        out = []
        for _ in range(5):
            out.append(t.record(tgerr.NetworkError(SECRET)))
        blob = json.dumps(out, ensure_ascii=False) + open(self.path).read()
        self.assertNotIn("ЛИЧНАЯ", blob)
        self.assertNotIn("SECRET", blob)


class ErrorHandlerTests(unittest.TestCase):
    def setUp(self):
        self.t = ops_transient.TransientTracker(os.path.join(tempfile.mkdtemp(), "t.json"))
        self.sent = []
        self.errs = []

        async def fake_send(*a, **k):
            self.sent.append(a)

        async def fake_error(title, exc=None, body=""):
            self.errs.append((title, tg_logger.exc_label(exc), body))

        for p in (mock.patch.object(bot, "transient", self.t),
                  mock.patch.object(bot, "send_alert", fake_send),
                  mock.patch.object(bot, "alert_error", fake_error)):
            p.start()
            self.addCleanup(p.stop)

    def handle(self, err):
        run(bot.error_handler(None, SimpleNamespace(error=err)))

    def test_transient_is_counted_not_alerted(self):
        for _ in range(4):
            self.handle(tgerr.NetworkError(SECRET))
        self.assertEqual((self.sent, self.errs), ([], []))
        self.assertEqual(sum(self.t.counts.values()), 4)
        self.handle(tgerr.NetworkError(SECRET))
        self.assertEqual(len(self.sent), 1)                 # пятая даёт один алерт всплеска
        self.assertEqual(self.errs, [])

    def test_other_errors_alert_with_class_and_place_only(self):
        try:
            raise KeyError(SECRET)
        except KeyError as e:
            self.handle(e)
        (title, label, body), = self.errs
        self.assertEqual((title, label), ("необработанная ошибка", "KeyError"))
        self.assertRegex(body, r"^test_ops_alerts\.py:\d+ in test_other_errors_alert")
        self.assertNotIn("ЛИЧНАЯ", json.dumps(self.errs, ensure_ascii=False))

    def test_startup_ping_is_gone(self):
        src = Path(bot.__file__).read_text(encoding="utf-8")
        self.assertNotIn("поднялся", src)
        self.assertNotIn("send_log", src)
        self.assertIsNone(re.search(r"alert_\w+\([^)]*!r", src))   # repr исключения в алерты не идёт


class RegisterCommandsTests(unittest.TestCase):
    """post_init не должен ронять процесс, если Telegram не отвечает на set_my_commands."""

    def _run(self, set_cmds):
        app = SimpleNamespace(bot=SimpleNamespace(set_my_commands=set_cmds),
                              create_task=mock.MagicMock(side_effect=lambda coro: coro.close()))
        warns = []

        async def fake_warn(title, exc=None, body=""):
            warns.append((title, exc, body))

        with mock.patch.object(bot, "alert_warn", fake_warn):
            asyncio.run(bot.register_commands(app))
        return app, warns

    def test_timeout_does_not_kill_startup(self):
        async def boom(_cmds):
            raise tgerr.TimedOut("Timed out")
        app, warns = self._run(boom)
        app.create_task.assert_called_once()                 # heartbeat запущен, бот работает
        self.assertEqual(len(warns), 1)
        self.assertIsInstance(warns[0][1], tgerr.TimedOut)   # в алерт уходит класс, не текст

    def test_success_is_silent_and_starts_heartbeat(self):
        async def ok(_cmds):
            return True
        app, warns = self._run(ok)
        app.create_task.assert_called_once()
        self.assertEqual(warns, [])


if __name__ == "__main__":
    unittest.main()
