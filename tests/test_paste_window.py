"""Окно вставки: хвост длинной вставки не уходит в модель. Время считается по дате
сообщения Telegram (а не по часам обработки). Без сети. Тексты нейтральные.

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import unittest
from unittest import mock

from support import Base2, at, bot, origin_from
from test_import_limit import BOT_ID, OUTSIDER

LONG = "а" * (bot.MAX_INPUT_CHARS + 1)
LONG_NOTICE = "длинный текст не отправлен в модель"
PASTE_NOTICE = "несколько сообщений подряд не отправлены в модель"
WINDOW = bot.PASTE_WINDOW_SECONDS


class PasteBase(Base2):
    def msg(self, text, seconds, user=111, **kw):
        return self.run_handler(bot.handle_message, user, text, date=at(seconds), **kw)


class PasteWindowTests(PasteBase):
    def test_default_window_is_15_seconds(self):
        self.assertEqual(WINDOW, 15.0)

    def test_tail_shorter_than_the_threshold_does_not_reach_the_model(self):
        self.assertIn(LONG_NOTICE, self.msg(LONG, 1000)[0])
        self.assertEqual(self.msg("короткий хвост вставки", 1001), [])  # молча
        self.assertEqual(self.client.mock_calls, [])  # ни stream, ни роутер, ни извлечение
        self.assertEqual(bot.conversations, {})

    def test_one_rejection_one_reply_for_the_whole_burst(self):
        replies = [self.msg(LONG, 1000 + i // 5) for i in range(14)] + [self.msg("хвост", 1003)]
        self.assertEqual(sum(len(r) for r in replies), 1)
        self.assertIn(LONG_NOTICE, replies[0][0])
        self.assertEqual(self.client.mock_calls, [])

    def test_window_is_sliding_each_rejection_extends_it(self):
        self.msg(LONG, 1000)
        for t in (1010, 1020, 1030):  # каждое в пределах 15 с от предыдущего, от первого уже 30 с
            self.assertEqual(self.msg("часть", t), [], t)
        self.assertEqual(self.client.mock_calls, [])
        self.assertEqual(self.msg("привет", 1046), ["ответ"])  # 16 с после последнего: окно закрылось
        self.client.messages.stream.assert_called_once()

    def test_message_after_a_pause_is_processed_normally(self):
        self.msg(LONG, 1000)
        out = self.msg("привет", 1000 + int(WINDOW) + 5)
        self.assertEqual(out, ["ответ"])
        self.client.messages.stream.assert_called_once()

    def test_window_edge(self):
        self.msg(LONG, 1000)
        self.assertEqual(self.msg("внутри", 1014), [])  # 14 < 15
        self.msg(LONG, 1100)
        self.assertEqual(self.msg("на границе", 1115), ["ответ"])  # ровно 15 с: окно уже закрыто

    def test_window_uses_message_dates_not_processing_time(self):
        # бот занят генерацией: обработка сообщений откладывается, их даты остаются близкими
        self.msg(LONG, 1000)
        self.now += 1000
        self.assertEqual(self.msg("хвост", 1002), [])
        self.assertEqual(self.client.mock_calls, [])

    def test_after_the_window_a_new_long_message_gets_a_new_reply(self):
        self.msg(LONG, 1000)
        self.assertIn(LONG_NOTICE, self.msg(LONG, 1100)[0])

    def test_commands_inside_the_window_work(self):
        self.msg(LONG, 1000)
        self.assertIn("расходов пока нет", "\n".join(self.run_handler(bot.cmd_cost, 111, "/cost")))
        self.assertIn("🧭", self.run_handler(bot.cmd_context, 111, "/context")[0])
        self.assertIn("присылай части", self.run_handler(bot.cmd_import, 111, "/import")[0])

    def test_import_mode_inside_the_window_takes_text_into_the_buffer(self):
        self.msg(LONG, 1000)
        self.run_handler(bot.cmd_import, 111, "/import")
        self.assertEqual(self.msg("часть A", 1001), ["часть 1, 2 слов"])
        self.assertEqual(self.client.mock_calls, [])

    def test_forwarded_bot_parts_inside_the_window_are_buffered(self):
        self.msg(LONG, 1000)
        out = self.run_handler(bot.handle_message, 111, "часть A", origin_from(BOT_ID, at(5)), date=at(1001))
        self.assertEqual(out, ["часть 1 принята"])

    def test_documents_inside_the_window_are_handled_as_documents(self):
        self.msg(LONG, 1000)
        out = self.send_doc(111, self.doc(b"x"), date=at(1001))
        self.assertIn("не отправлен в модель", out[0])  # обычный отказ вне /import, не тишина окна

    def test_window_is_per_user(self):
        self.msg(LONG, 1000)
        self.assertEqual(self.msg("привет", 1001, user=222), ["ответ"])

    def test_outsider_gets_silence_and_opens_no_window(self):
        self.assertEqual(self.msg(LONG, 1000, user=OUTSIDER), [])
        self.assertEqual(self.msg("привет", 1001, user=OUTSIDER), [])
        self.assertNotIn(OUTSIDER, bot.paste_state)
        self.assertEqual(self.client.mock_calls, [])

    def test_swallowed_messages_log_one_line_with_a_counter_and_no_text(self):
        secret = "СЕКРЕТНЫЙ ХВОСТ"
        self.msg(LONG, 1000)
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.msg(secret, 1001)
            self.msg(secret, 1002)
        lines = [l for l in cm.output if "paste window" in l]
        self.assertEqual(len(lines), 2)
        self.assertIn("#2", lines[0])
        self.assertIn("#3", lines[1])
        self.assertNotIn(secret, "\n".join(cm.output))

    def test_zero_disables_the_window(self):
        with mock.patch.object(bot, "PASTE_WINDOW_SECONDS", 0):
            self.assertIn(LONG_NOTICE, self.msg(LONG, 1000)[0])
            self.assertEqual(self.msg("привет", 1001), ["ответ"])


class FastBurstTests(PasteBase):
    def test_two_messages_within_a_second_are_a_paste(self):
        first = self.msg("первое", 1000)
        second = self.msg("второе", 1000)  # та же секунда, интервал меньше 1 с
        self.assertEqual(first, ["ответ"])  # первое уже ушло в модель, его не вернуть
        self.assertIn(PASTE_NOTICE, second[0])
        self.assertEqual(self.client.messages.stream.call_count, 1)

    def test_burst_continuation_is_silent_and_extends_the_window(self):
        self.msg("первое", 1000)
        self.assertEqual(len(self.msg("второе", 1000)), 1)  # один ответ на отклонение
        self.assertEqual(self.msg("третье", 1000), [])
        self.assertEqual(self.msg("четвёртое", 1012), [])  # окно продлено до 1015
        self.assertEqual(self.msg("пятое", 1026), [])  # от 1012 прошло 14 с: ещё внутри
        self.assertEqual(self.client.messages.stream.call_count, 1)

    def test_messages_one_second_apart_are_not_a_paste(self):
        self.assertEqual(self.msg("первое", 1000), ["ответ"])
        self.assertEqual(self.msg("второе", 1001), ["ответ"])
        self.assertEqual(self.client.messages.stream.call_count, 2)

    def test_fast_rule_ignores_commands_imports_and_forwards(self):
        self.run_handler(bot.cmd_context, 111, "/context")
        self.run_handler(bot.cmd_context, 111, "/context")
        self.assertEqual(self.msg("привет", 1000), ["ответ"])  # команды вставкой не считаются

    def test_import_mode_messages_do_not_count_as_a_burst(self):
        self.run_handler(bot.cmd_import, 111, "/import")
        self.msg("часть A", 1000)
        self.msg("часть B", 1000)
        self.run_handler(bot.cmd_cancel, 111, "/cancel")
        self.assertEqual(self.msg("привет", 1000), ["ответ"])  # последнее сообщение до него было в импорте

    def test_burst_notice_differs_from_the_long_notice(self):
        self.msg("первое", 1000)
        self.assertNotIn(LONG_NOTICE, self.msg("второе", 1000)[0])

    def test_incident_shape_fourteen_long_parts_and_a_tail(self):
        """20:48 МСК: 14 кусков по ~4000 символов и хвост короче порога в одну секунду."""
        replies = [self.msg("б" * 4000, 1000) for _ in range(14)] + [self.msg("хвост сцены " * 100, 1001)]
        self.assertEqual(sum(len(r) for r in replies), 1)
        self.assertEqual(self.client.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
