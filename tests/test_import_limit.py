"""Импорт сцены пересылкой (без LLM) и защита от дорогих запросов: дневной лимит,
порог длины входа, /cost. Без сети: Anthropic, Voyage и Postgres подменены.

Тексты нейтральные («часть N», «w<номер>»): настоящих сцен в репозитории нет.

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import asyncio
import datetime
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1:a")
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from telegram import MessageOriginHiddenUser, MessageOriginUser, User  # noqa: E402

import claude_tg_bot as bot  # noqa: E402
import rag  # noqa: E402
from test_scenes import FakeConn, FakeStore, FakeStream, paragraph, paras_of  # noqa: E402

BOT_ID = 5000
ALLOWED = (111, 222)
OUTSIDER = 999
T0 = datetime.datetime(2026, 11, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)


def at(seconds):
    return T0 + datetime.timedelta(seconds=seconds)


def origin_from(user_id, date, is_bot=True):
    return MessageOriginUser(date=date, sender_user=User(id=user_id, first_name="x", is_bot=is_bot))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.client = mock.Mock()
        self.store = FakeStore()
        self.now = 1_000_000.0
        bot.conversations.clear()
        bot.reply_texts.clear()
        bot.forward_buffers.clear()
        for p in (
            mock.patch.object(bot, "ALLOWED_USER_IDS", set(ALLOWED)),
            mock.patch.object(bot, "USAGE_DIR", self.tmp),
            mock.patch.object(bot, "client", self.client),
            mock.patch.object(bot, "alert_warn", new=mock.AsyncMock()),
            mock.patch.object(bot, "alert_error", new=mock.AsyncMock()),
            mock.patch.object(bot, "_now", side_effect=lambda: self.now),
            mock.patch.object(bot, "DAILY_LIMIT_USD", 3.0),
            mock.patch.object(bot, "load_mode", return_value="auto"),  # роутер включён: он тоже не должен вызываться
            mock.patch.object(bot, "load_prompt", return_value="системный промпт"),
            mock.patch.object(bot, "get_relevant_facts", return_value=([], "none")),
            mock.patch.object(bot.rag, "search_scenes", return_value=[]),
        ):
            p.start()
            self.addCleanup(p.stop)
        final = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ответ")],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
            stop_reason="end_turn",
        )
        self.client.messages.stream.side_effect = lambda **kw: FakeStream(final)
        self.client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="opus")], usage=final.usage
        )

    def update(self, user_id, text, forward_origin=None, **extra):
        replies = []

        async def reply_text(t):
            replies.append(t)
            return SimpleNamespace(message_id=100 + len(replies))

        async def send_action(a):
            return None

        message = SimpleNamespace(
            text=text, reply_text=reply_text, reply_to_message=None, forward_origin=forward_origin,
            date=at(0), **extra,
        )
        upd = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=user_id, send_action=send_action),
            message=message,
        )
        return upd, replies

    def run_handler(self, handler, user_id, text, forward_origin=None, **extra):
        upd, replies = self.update(user_id, text, forward_origin, **extra)
        ctx = SimpleNamespace(bot=SimpleNamespace(id=BOT_ID))
        asyncio.run(handler(upd, ctx))
        return replies

    def forward(self, user_id, text, seconds, from_id=BOT_ID):
        return self.run_handler(bot.handle_message, user_id, text, origin_from(from_id, at(seconds)))

    def say(self, user_id, text):
        return self.run_handler(bot.handle_message, user_id, text)

    def assert_model_untouched(self):
        self.assertEqual(self.client.mock_calls, [])

    def spend(self, user_id, dollars):
        """Записывает расход за сегодня через настоящий add_usage (Opus: $20 за 1M out)."""
        out = int(dollars / 20 * 1_000_000)
        bot.add_usage(user_id, bot.MODELS["opus"], SimpleNamespace(
            input_tokens=0, output_tokens=out, cache_read_input_tokens=0, cache_creation_input_tokens=0))


# ------------------------------------------------------------ 1. пересылка
class ForwardImportTests(Base):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(bot.rag, "save_scene", side_effect=self.store.save_scene)
        p.start()
        self.addCleanup(p.stop)

    def test_parts_in_any_order_are_joined_by_original_date(self):
        for text, sec in (("часть C", 30), ("часть A", 10), ("часть B", 20)):
            self.forward(111, text, sec)
        out = self.run_handler(bot.cmd_keep, 111, "/keep")
        self.assertEqual(out, ["✓ сохранена: сцена 1 (частей: 3)"])
        self.assertEqual(self.store.scenes[111][0]["text"], "часть A\n\nчасть B\n\nчасть C")

    def test_forwarded_part_never_reaches_the_model_or_history(self):
        replies = self.forward(111, "часть 1", 10)
        self.assertEqual(replies, ["часть 1 принята"])
        self.assert_model_untouched()  # ни stream, ни роутер (create)
        self.assertEqual(bot.conversations, {})

    def test_acknowledgements_count_parts(self):
        self.assertEqual(self.forward(111, "часть A", 10), ["часть 1 принята"])
        self.assertEqual(self.forward(111, "часть B", 20), ["часть 2 принята"])

    def test_duplicate_part_is_not_added_twice(self):
        self.forward(111, "часть A", 10)
        self.assertEqual(self.forward(111, "часть A", 10), ["эта часть уже принята (всего 1)"])
        self.forward(111, "часть B", 20)
        self.run_handler(bot.cmd_keep, 111, "/keep")
        self.assertEqual(self.store.scenes[111][0]["text"], "часть A\n\nчасть B")

    def test_same_second_dates_keep_arrival_order(self):
        for text in ("часть A", "часть B", "часть C"):
            self.forward(111, text, 10)
        self.run_handler(bot.cmd_keep, 111, "/keep")
        self.assertEqual(self.store.scenes[111][0]["text"], "часть A\n\nчасть B\n\nчасть C")

    def test_long_forwarded_part_is_accepted_not_blocked_by_length_guard(self):
        long_part = paragraph(0, 900)  # > 3500 символов
        self.assertGreater(len(long_part), bot.MAX_INPUT_CHARS)
        self.assertEqual(self.forward(111, long_part, 10), ["часть 1 принята"])

    def test_keep_clears_buffer_then_old_behaviour(self):
        self.forward(111, "часть A", 10)
        self.run_handler(bot.cmd_keep, 111, "/keep")
        out = self.run_handler(bot.cmd_keep, 111, "/keep")  # буфер пуст, истории нет
        self.assertIn("нечего сохранять", out[0])

    def test_keep_with_empty_buffer_keeps_last_assistant_message(self):
        h = bot.get_history(111)
        h += [{"role": "user", "content": "запрос 1"}, {"role": "assistant", "content": "сцена 1"}]
        out = self.run_handler(bot.cmd_keep, 111, "/keep")
        self.assertEqual(out, ["✓ сохранена: сцена 1"])  # без «частей»

    def test_new_clears_buffer(self):
        self.forward(111, "часть A", 10)
        self.run_handler(bot.cmd_new, 111, "/new")
        self.assertNotIn(111, bot.forward_buffers)
        self.assertIn("нечего сохранять", self.run_handler(bot.cmd_keep, 111, "/keep")[0])
        self.assertFalse(self.store.scenes)

    def test_buffer_expires_after_30_minutes(self):
        self.forward(111, "часть A", 10)
        self.now += bot.FORWARD_BUFFER_TTL + 1
        self.assertIn("нечего сохранять", self.run_handler(bot.cmd_keep, 111, "/keep")[0])
        self.assertEqual(self.forward(111, "часть B", 20), ["часть 1 принята"])  # новый буфер

    def test_each_new_part_extends_the_timeout(self):
        self.forward(111, "часть A", 10)
        self.now += bot.FORWARD_BUFFER_TTL - 10
        self.forward(111, "часть B", 20)
        self.now += bot.FORWARD_BUFFER_TTL - 10  # от первой части уже > 30 мин, от второй нет
        self.run_handler(bot.cmd_keep, 111, "/keep")
        self.assertEqual(self.store.scenes[111][0]["text"], "часть A\n\nчасть B")

    def test_storage_failure_keeps_buffer_for_retry(self):
        self.forward(111, "часть A", 10)
        self.store.fail = True
        self.assertIn("не удалось сохранить", self.run_handler(bot.cmd_keep, 111, "/keep")[0])
        self.store.fail = False
        self.assertEqual(self.run_handler(bot.cmd_keep, 111, "/keep"), ["✓ сохранена: сцена 1 (частей: 1)"])

    def test_buffers_are_per_user(self):
        self.forward(111, "часть A", 10)
        self.forward(222, "часть X", 10)
        self.run_handler(bot.cmd_keep, 111, "/keep")
        self.assertEqual(self.store.scenes[111][0]["text"], "часть A")
        self.assertIn(222, bot.forward_buffers)

    def test_outsider_gets_silence_and_no_buffer(self):
        self.assertEqual(self.forward(OUTSIDER, "часть A", 10), [])
        self.assertNotIn(OUTSIDER, bot.forward_buffers)
        self.assert_model_untouched()

    def test_forward_from_someone_else_behaves_as_before(self):
        with mock.patch.object(bot, "_process_user_text", new=mock.AsyncMock()) as proc:
            self.forward(111, "чужой текст", 10, from_id=12345)
        proc.assert_awaited_once()
        self.assertEqual(proc.await_args.args[1], "чужой текст")
        self.assertNotIn(111, bot.forward_buffers)

    def test_forward_with_hidden_origin_and_plain_text_behave_as_before(self):
        with mock.patch.object(bot, "_process_user_text", new=mock.AsyncMock()) as proc:
            self.run_handler(bot.handle_message, 111, "скрытый", MessageOriginHiddenUser(date=at(1), sender_user_name="x"))
            self.say(111, "обычный текст")
        self.assertEqual(proc.await_count, 2)
        self.assertNotIn(111, bot.forward_buffers)

    def test_legacy_forward_from_format_is_recognised(self):
        upd, replies = self.update(111, "часть A")
        upd.message.forward_origin = None
        upd.message.forward_from = SimpleNamespace(id=BOT_ID)
        upd.message.forward_date = at(5)
        asyncio.run(bot.handle_message(upd, SimpleNamespace(bot=SimpleNamespace(id=BOT_ID))))
        self.assertEqual(replies, ["часть 1 принята"])
        self.assert_model_untouched()

    def test_reply_to_bot_message_still_works_with_buffer_empty(self):
        upd, replies = self.update(111, "/keep")
        upd.message.reply_to_message = SimpleNamespace(message_id=1, text="сцена 1", from_user=SimpleNamespace(is_bot=True))
        asyncio.run(bot.cmd_keep(upd, None))
        self.assertEqual(replies, ["✓ сохранена: сцена 1"])


class ForwardImportRealStorageTests(Base):
    """Сохранение идёт через настоящий rag.save_scene: чанки и эмбеддинги (Voyage
    замокан), запись в фейковую БД."""

    def test_joined_scene_goes_through_chunking_and_embedding(self):
        log = []

        def on_sql(sql, params):
            if sql.startswith("SELECT id FROM scenes"):
                return []
            if sql.startswith("INSERT INTO scenes"):
                return [(7,)]
            if sql.startswith("SELECT count(*)"):
                return [(1,)]
            return []

        def embed(texts, input_type):
            log.append((input_type, len(texts)))
            return [[0.1] * 3 for _ in texts]

        parts = ["\n\n".join(paragraph(1000 * k + 100 * i, 150) for i in range(3)) for k in range(4)]
        with mock.patch.object(rag, "_connect", lambda: FakeConn(log, on_sql)), \
                mock.patch.object(rag, "_embed", embed):
            for k in (2, 0, 3, 1):  # вперемешку
                self.forward(111, parts[k], 10 + k)
            out = self.run_handler(bot.cmd_keep, 111, "/keep")
        self.assertEqual(out, ["✓ сохранена: сцена 1 (частей: 4)"])
        rows = next(e[2] for e in log if len(e) == 3 and e[0] == "executemany")
        self.assertGreater(len(rows), 1)
        expected = [p for part in parts for p in paras_of(part)]  # порядок дат: 0, 1, 2, 3
        self.assertEqual([p for r in rows for p in paras_of(r[3])], expected)
        self.assertIn(("document", len(rows)), [e for e in log if len(e) == 2])
        self.assert_model_untouched()


# ------------------------------------------------------- 2. защита от расходов
class DailyLimitTests(Base):
    def test_over_limit_model_is_not_called(self):
        self.spend(111, 3.5)
        out = self.say(111, "привет")
        self.assert_model_untouched()  # ни stream, ни классификатор
        self.assertEqual(len(out), 1)
        self.assertIn("дневной лимит $3.00 исчерпан (потрачено $3.50)", out[0])
        self.assertIn("DAILY_LIMIT_USD", out[0])
        self.assertEqual(bot.conversations.get(111, []), [])  # запрос в историю не попал

    def test_warning_is_logged(self):
        self.spend(111, 3.5)
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            self.say(111, "привет")
        self.assertTrue(any("daily limit" in l for l in cm.output), cm.output)

    def test_exactly_at_limit_blocks(self):
        self.spend(111, 3.0)
        self.say(111, "привет")
        self.assert_model_untouched()

    def test_under_limit_model_is_called(self):
        self.spend(111, 2.9)
        out = self.say(111, "привет")
        self.client.messages.create.assert_called_once()  # классификатор (режим auto)
        self.client.messages.stream.assert_called_once()
        self.assertEqual(out, ["ответ"])

    def test_spend_accumulates_through_real_add_usage(self):
        self.say(111, "привет")
        self.assertGreater(bot.spent_today(111), 0)

    def test_scene_and_regenerate_respect_the_limit(self):
        self.spend(111, 3.5)
        h = bot.get_history(111)
        h += [{"role": "user", "content": "запрос 1"}, {"role": "assistant", "content": "ответ 1"}]
        out1 = self.run_handler(bot.cmd_scene, 111, "/scene встреча")
        out2 = self.run_handler(bot.cmd_regenerate, 111, "/regenerate")
        self.assert_model_untouched()
        self.assertTrue(all("дневной лимит" in o[0] for o in (out1, out2)))
        self.assertEqual(h[-1]["role"], "assistant")  # история не тронута

    def test_document_respects_the_limit_too(self):
        self.spend(111, 3.5)
        asyncio.run(bot._process_user_text(self.update(111, "x")[0], "текст файла"))
        self.assert_model_untouched()

    def test_limit_is_per_chat(self):
        self.spend(111, 3.5)
        self.say(222, "привет")
        self.client.messages.stream.assert_called_once()

    def test_zero_limit_disables_the_check(self):
        self.spend(111, 50)
        with mock.patch.object(bot, "DAILY_LIMIT_USD", 0):
            self.say(111, "привет")
        self.client.messages.stream.assert_called_once()

    def test_yesterdays_spend_does_not_count(self):
        self.spend(111, 3.5)
        with mock.patch.object(bot, "today_key", return_value="2099-01-01"):
            self.assertEqual(bot.spent_today(111), 0.0)
            self.say(111, "привет")
        self.client.messages.stream.assert_called_once()

    def test_day_is_counted_in_utc_plus_3(self):
        class FakeDT(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.datetime(2026, 11, 15, 21, 30, tzinfo=datetime.timezone.utc).astimezone(tz)

        with mock.patch.object(bot, "datetime", FakeDT):
            self.assertEqual(bot.today_key(), "2026-11-16")  # 21:30 UTC уже следующий день по UTC+3

    def test_old_usage_file_without_daily_counts_as_zero_today(self):
        bot.save_usage(111, {bot.MODELS["opus"]: {"input": 0, "output": 10**9, "cache_read": 0, "cache_write": 0}})
        self.assertEqual(bot.spent_today(111), 0.0)
        self.say(111, "привет")
        self.client.messages.stream.assert_called_once()

    def test_add_usage_keeps_totals_and_adds_daily(self):
        self.spend(111, 1.0)
        self.spend(111, 1.0)
        data = bot.load_usage(111)
        model = bot.MODELS["opus"]
        day = data["_daily"][bot.today_key()][model]
        self.assertEqual(data[model]["output"], day["output"])
        self.assertAlmostEqual(bot.cost_for_model(model, day), 2.0, places=2)

    def test_daily_history_is_pruned(self):
        data = {"_daily": {f"2026-01-{d:02d}": {} for d in range(1, 29)}}
        data["_daily"].update({f"2026-02-{d:02d}": {} for d in range(1, 29)})
        data["_daily"].update({f"2026-03-{d:02d}": {} for d in range(1, 15)})
        bot.save_usage(111, data)
        self.spend(111, 0.01)
        self.assertLessEqual(len(bot.load_usage(111)["_daily"]), bot.DAILY_KEEP_DAYS)


class DefaultLimitTests(unittest.TestCase):
    def test_default_daily_limit_is_two_dollars(self):
        src = (Path(__file__).resolve().parent.parent / "claude_tg_bot.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("DAILY_LIMIT_USD", "2.00")', src)
        if "DAILY_LIMIT_USD" not in os.environ:
            self.assertEqual(bot.DAILY_LIMIT_USD, 2.0)


class CostCommandTests(Base):
    def test_cost_shows_today_and_limit(self):
        self.spend(111, 1.0)
        out = self.run_handler(bot.cmd_cost, 111, "/cost")
        text = "\n".join(out)
        self.assertIn("сегодня (UTC+3): ≈ $1.00 из $3.00", text)
        self.assertIn("DAILY_LIMIT_USD", text)
        self.assertNotIn("_daily", text)  # служебный ключ не выводится как «модель»

    def test_cost_without_usage_still_shows_the_limit(self):
        text = "\n".join(self.run_handler(bot.cmd_cost, 111, "/cost"))
        self.assertIn("расходов пока нет", text)
        self.assertIn("из $3.00", text)

    def test_cost_with_disabled_limit(self):
        with mock.patch.object(bot, "DAILY_LIMIT_USD", 0):
            self.assertIn("лимит выключен", "\n".join(self.run_handler(bot.cmd_cost, 111, "/cost")))


class InputSizeTests(Base):
    def test_long_message_does_not_call_the_model(self):
        text = "а" * (bot.MAX_INPUT_CHARS + 1)
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.say(111, text)
        self.assert_model_untouched()
        self.assertEqual(
            out,
            ["длинный текст не отправлен в модель, чтобы не тратить деньги; "
             "для сохранения сцены перешли её части боту и вызови /keep"],
        )
        self.assertNotIn(text[:50], "\n".join(cm.output))  # текст в лог не пишется
        self.assertEqual(bot.conversations, {})

    def test_message_at_the_threshold_goes_to_the_model(self):
        self.say(111, "а" * bot.MAX_INPUT_CHARS)
        self.client.messages.stream.assert_called_once()

    def test_threshold_is_3500(self):
        self.assertEqual(bot.MAX_INPUT_CHARS, 3500)

    def test_outsider_long_message_gets_silence(self):
        self.assertEqual(self.say(OUTSIDER, "а" * 5000), [])


if __name__ == "__main__":
    logging.disable(logging.NOTSET)
    unittest.main()
