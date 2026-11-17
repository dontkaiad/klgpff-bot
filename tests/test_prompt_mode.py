"""Режим /prompt и /go: длинный запрос на генерацию одним сообщением в модель.
Без сети: Anthropic, Voyage и Postgres подменены. Тексты нейтральные («часть N», «а» * n):
настоящих запросов, сцен и фактов в репозитории и в логах тестов нет.

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from support import Base2, at, bot, origin_from
from test_import_limit import BOT_ID, OUTSIDER
from test_scenes import FakeStream

LIMIT = bot.PROMPT_BUFFER_MAX_CHARS


class PromptBase(Base2):
    def setUp(self):
        super().setUp()
        # Модули без своей очистки (test_scenes) увидели бы наш буфер просроченным
        # и унесли user_id в prompt_expired: чистим состояние и после теста.
        for state in (bot.forward_buffers, bot.import_expired, bot.prompt_expired, bot.paste_state):
            state.clear()
            self.addCleanup(state.clear)

    def start(self, user_id=111):
        return self.command(bot.cmd_prompt, user_id, "/prompt")

    def part(self, user_id, text, seconds=0):
        return self.run_handler(bot.handle_message, user_id, text, date=at(seconds))

    def go(self, user_id=111):
        return self.command(bot.cmd_go, user_id, "/go")

    def ready(self, *texts, user_id=111):
        self.start(user_id)
        for i, t in enumerate(texts):
            self.part(user_id, t, i + 1)

    def stream_calls(self):
        return self.client.messages.stream.call_args_list


class PromptCollectTests(PromptBase):
    def test_prompt_reply(self):
        out = self.start()
        self.assertEqual(len(out), 1)
        self.assertIn("/go", out[0])
        self.assertIn(str(LIMIT), out[0])
        self.assertTrue(bot.prompt_active(111))
        self.assertFalse(bot.import_active(111))

    def test_parts_are_buffered_without_calling_the_model(self):
        self.start()
        replies = [self.part(111, "абв"[k] * 100, k) for k in range(3)]
        self.assertIn("часть 1, 100 символов", replies[0][0])
        self.assertIn(f"из {LIMIT}", replies[2][0])
        self.assertEqual(self.client.mock_calls, [])  # ни stream, ни роутер, ни извлечение
        self.assertEqual(bot.conversations, {})

    def test_text_over_the_normal_threshold_is_accepted_in_prompt_mode(self):
        self.start()
        out = self.part(111, "а" * (bot.MAX_INPUT_CHARS + 90), 1)
        self.assertIn("часть 1", out[0])
        self.assertEqual(self.client.mock_calls, [])

    def test_forwarded_parts_use_the_original_date(self):
        self.start()
        self.run_handler(bot.handle_message, 111, "часть B", origin_from(BOT_ID, at(20)), date=at(500))
        self.run_handler(bot.handle_message, 111, "часть A", origin_from(BOT_ID, at(10)), date=at(600))
        self.assertEqual(bot.buffered_parts(111), ["часть A", "часть B"])
        self.assertEqual(self.client.mock_calls, [])

    def test_duplicate_part_is_not_added_twice(self):
        self.start()
        self.part(111, "часть A", 1)
        self.assertEqual(self.part(111, "часть A", 2), ["эта часть уже принята (всего 1)"])

    def test_txt_and_md_files_go_to_the_buffer_without_the_model(self):
        self.start()
        out = self.send_doc(111, self.doc("часть файла".encode(), name="story.md"), date=at(5))
        self.assertIn("часть 1", out[0])
        self.assertEqual(self.client.mock_calls, [])
        self.assertEqual(bot.buffered_parts(111), ["часть файла"])

    def test_other_file_types_are_refused_in_prompt_mode(self):
        self.start()
        out = self.send_doc(111, self.doc(b"x", name="pic.png"), date=at(5))
        self.assertEqual(out, ["в режиме /prompt принимаю только .txt и .md"])
        self.assertEqual(bot.buffered_parts(111), [])

    def test_file_outside_the_modes_points_to_both_routes(self):
        out = self.send_doc(111, self.doc(b"x"), date=at(5))
        self.assertIn("/prompt", out[0])
        self.assertIn("/go", out[0])
        self.assertIn("/import", out[0])
        self.assertEqual(self.client.mock_calls, [])

    def test_buffers_are_per_user(self):
        self.ready("часть A", user_id=111)
        self.ready("часть B", user_id=222)
        self.assertEqual(bot.buffered_parts(111), ["часть A"])
        self.assertEqual(bot.buffered_parts(222), ["часть B"])


class PromptBufferLimitTests(PromptBase):
    def test_limit_is_twenty_thousand(self):
        self.assertEqual(LIMIT, 20000)

    def test_buffer_may_be_filled_exactly_to_the_limit_counting_separators(self):
        self.start()
        self.part(111, "а" * 9999, 1)
        out = self.part(111, "б" * 9999, 2)  # 9999 + "\n\n" + 9999 == 20000
        self.assertIn("часть 2", out[0])
        self.assertEqual(len("\n\n".join(bot.buffered_parts(111))), LIMIT)

    def test_one_char_over_the_limit_is_refused(self):
        self.start()
        self.part(111, "а" * 10000, 1)
        out = self.part(111, "б" * 9999, 2)  # 10000 + 2 + 9999 == 20001
        self.assertIn(f"буфер запроса ограничен {LIMIT} символов", out[0])
        self.assertIn("/go", out[0])
        self.assertEqual(bot.buffered_parts(111), ["а" * 10000])
        self.assertEqual(self.client.mock_calls, [])

    def test_a_smaller_part_still_fits_after_a_refusal(self):
        self.start()
        self.part(111, "а" * 15000, 1)
        self.assertIn("ограничен", self.part(111, "б" * 6000, 2)[0])
        self.assertIn("часть 2", self.part(111, "в" * 100, 3)[0])

    def test_oversized_file_is_refused_by_the_buffer_limit(self):
        self.start()
        out = self.send_doc(111, self.doc(("а" * (LIMIT + 1)).encode()), date=at(5))
        self.assertIn("ограничен", out[0])
        self.assertEqual(bot.buffered_parts(111), [])

    def test_joined_request_sent_by_go_never_exceeds_the_limit(self):
        self.ready("а" * 9999, "б" * 9999)
        self.go()
        sent = self.stream_calls()[0].kwargs["messages"][-1]["content"]
        self.assertTrue(sent.startswith("а" * 9999 + "\n\n" + "б" * 9999))


class GoTests(PromptBase):
    def test_go_calls_the_model_exactly_once_with_the_joined_text(self):
        self.write_facts(111, ["факт 1"])
        self.start()
        self.run_handler(bot.handle_message, 111, "часть C", origin_from(BOT_ID, at(30)), date=at(900))
        self.part(111, "часть A", 10)
        self.part(111, "часть B", 20)
        self.assertEqual(self.client.mock_calls, [])
        out = self.go()
        self.assertEqual(len(self.stream_calls()), 1)
        self.assertEqual(self.client.messages.create.call_count, 0)  # роутер выключен (режим opus)
        kw = self.stream_calls()[0].kwargs
        self.assertEqual(kw["model"], bot.MODELS["opus"])
        last = kw["messages"][-1]
        self.assertEqual(last["role"], "user")
        self.assertTrue(last["content"].startswith("часть A\n\nчасть B\n\nчасть C"))
        self.assertIn("факт 1", last["content"])  # канон добавлен, как у обычного сообщения
        self.assertEqual(kw["system"][0]["cache_control"]["type"], "ephemeral")  # кэш system, как обычно
        self.assertEqual(out[-1], "ответ")
        self.assertEqual(bot.conversations[111][0]["content"], "часть A\n\nчасть B\n\nчасть C")
        self.assertEqual(bot.conversations[111][-1]["content"], "ответ")

    def test_first_reply_is_chars_and_estimate_then_generation(self):
        self.ready("а" * 5000, "б" * 5000)
        out = self.go()
        joined = 5000 + 2 + 5000
        self.assertEqual(len(out), 2)
        self.assertIn(f"{joined} символов", out[0])
        self.assertIn("$", out[0])
        self.assertIn("opus", out[0])
        self.assertNotIn("ааааа", out[0])  # текст запроса в ответе не повторяется
        self.assertEqual(out[1], "ответ")

    def test_go_leaves_the_mode_and_clears_the_buffer(self):
        self.ready("часть A")
        self.go()
        self.assertFalse(bot.prompt_active(111))
        self.assertNotIn(111, bot.forward_buffers)

    def test_second_go_does_not_send_again(self):
        self.ready("часть A")
        self.go()
        out = self.go()
        self.assertIn("не включён", out[0])
        self.assertEqual(len(self.stream_calls()), 1)

    def test_go_without_the_mode(self):
        out = self.go()
        self.assertIn("/prompt", out[0])
        self.assertEqual(self.client.mock_calls, [])

    def test_go_with_an_empty_buffer_does_not_call_the_model_and_keeps_the_mode(self):
        self.start()
        out = self.go()
        self.assertIn("буфер пуст", out[0])
        self.assertEqual(self.client.mock_calls, [])
        self.assertTrue(bot.prompt_active(111))
        self.assertEqual(bot.conversations, {})

    def test_go_in_import_mode_does_not_send_the_import_buffer(self):
        self.command(bot.cmd_import, 111, "/import")
        self.part(111, "часть A", 1)
        out = self.go()
        self.assertIn("/prompt", out[0])
        self.assertEqual(self.client.mock_calls, [])
        self.assertTrue(bot.import_active(111))

    def test_auto_mode_goes_through_the_router_and_labels_the_estimate(self):
        self.ready("часть A")
        with mock.patch.object(bot, "load_mode", return_value="auto"):
            out = self.go()
        self.assertEqual(self.client.messages.create.call_count, 1)  # классификатор
        self.assertEqual(len(self.stream_calls()), 1)
        self.assertIn("роутер", out[0])

    def test_manual_sonnet_is_estimated_by_sonnet_prices(self):
        self.ready("а" * 10000)
        with mock.patch.object(bot, "load_mode", return_value="sonnet"):
            out = self.go()
        self.assertIn("sonnet", out[0])
        self.assertEqual(self.stream_calls()[0].kwargs["model"], bot.MODELS["sonnet"])

    def test_go_after_a_rejected_long_paste_end_to_end(self):
        long_text = "а" * (bot.MAX_INPUT_CHARS + 1)
        rejected = self.part(111, long_text, 1000)
        self.assertIn("/prompt", rejected[0])
        self.assertEqual(self.client.mock_calls, [])
        self.start()
        self.part(111, long_text, 1001)  # окно вставки не мешает: режим сборки раньше paste_check
        self.go()
        self.assertEqual(len(self.stream_calls()), 1)
        self.assertEqual(self.stream_calls()[0].kwargs["messages"][-1]["content"].count("а"), len(long_text))


class GoLimitTests(PromptBase):
    def test_daily_limit_blocks_go_before_any_model_call_or_cost_line(self):
        self.ready("часть A")
        self.spend(111, 3.5)  # лимит в тестах $3.00
        out = self.go()
        self.assertEqual(len(out), 1)  # только отказ: строки со стоимостью нет
        self.assertIn("дневной лимит", out[0])
        self.assertEqual(self.client.mock_calls, [])
        self.assertEqual(bot.conversations, {})

    def test_blocked_go_keeps_the_buffer_and_the_mode(self):
        self.ready("часть A", "часть B")
        self.spend(111, 3.5)
        self.go()
        self.assertTrue(bot.prompt_active(111))
        self.assertEqual(bot.buffered_parts(111), ["часть A", "часть B"])
        with mock.patch.object(bot, "DAILY_LIMIT_USD", 100.0):
            out = self.go()
        self.assertEqual(len(self.stream_calls()), 1)
        self.assertEqual(out[-1], "ответ")

    def test_spend_under_the_limit_passes(self):
        self.ready("часть A")
        self.spend(111, 2.9)
        self.go()
        self.assertEqual(len(self.stream_calls()), 1)

    def test_disabled_limit_passes(self):
        self.ready("часть A")
        self.spend(111, 50)
        with mock.patch.object(bot, "DAILY_LIMIT_USD", 0):
            self.go()
        self.assertEqual(len(self.stream_calls()), 1)

    def test_collecting_parts_is_not_blocked_by_the_limit_and_costs_nothing(self):
        self.spend(111, 3.5)
        self.start()
        self.assertIn("часть 1", self.part(111, "часть A", 1)[0])
        self.assertEqual(self.client.mock_calls, [])


class EstimateTests(unittest.TestCase):
    def test_opus_twenty_thousand_chars(self):
        lo, hi = bot.estimate_prompt_cost(20000, "opus")
        # вход 20000 / 2.8 ≈ 7143 токена по $4/M; выход 5000 и 10000 токенов по $20/M
        self.assertAlmostEqual(lo, 7142.857 * 4 / 1e6 + 5000 * 20 / 1e6, places=4)
        self.assertAlmostEqual(hi, 7142.857 * 4 / 1e6 + 10000 * 20 / 1e6, places=4)
        self.assertEqual((f"{lo:.2f}", f"{hi:.2f}"), ("0.13", "0.23"))

    def test_cheaper_models_cost_less(self):
        opus = bot.estimate_prompt_cost(20000, "opus")
        sonnet = bot.estimate_prompt_cost(20000, "sonnet")
        haiku = bot.estimate_prompt_cost(20000, "haiku")
        self.assertLess(sonnet[1], opus[0])
        self.assertLess(haiku[1], sonnet[0])


class AccessTests(PromptBase):
    def test_outsider_gets_silence_and_no_buffer(self):
        self.assertEqual(self.command(bot.cmd_prompt, OUTSIDER, "/prompt"), [])
        self.assertEqual(self.command(bot.cmd_go, OUTSIDER, "/go"), [])
        self.assertNotIn(OUTSIDER, bot.forward_buffers)
        self.assertEqual(self.client.mock_calls, [])

    def test_outsider_cannot_send_somebody_elses_buffer(self):
        self.ready("часть A", user_id=111)
        self.assertEqual(self.command(bot.cmd_go, OUTSIDER, "/go"), [])
        self.assertEqual(self.client.mock_calls, [])
        self.assertEqual(bot.buffered_parts(111), ["часть A"])

    def test_outsider_text_never_reaches_a_buffer_or_the_model(self):
        self.assertEqual(self.part(OUTSIDER, "а" * 5000, 1), [])
        self.assertEqual(self.client.mock_calls, [])
        self.assertNotIn(OUTSIDER, bot.forward_buffers)

    def test_one_user_go_does_not_send_the_other_users_buffer(self):
        self.ready("часть A", user_id=111)
        self.start(222)  # у 222 буфер пуст
        out = self.go(222)
        self.assertIn("буфер пуст", out[0])
        self.assertEqual(self.client.mock_calls, [])
        self.assertEqual(bot.buffered_parts(111), ["часть A"])


class ExitTests(PromptBase):
    def test_cancel_clears_the_buffer_and_the_mode(self):
        self.ready("часть A", "часть B")
        out = self.command(bot.cmd_cancel, 111, "/cancel")
        self.assertIn("запрос отменён", out[0])
        self.assertIn("частей: 2", out[0])
        self.assertFalse(bot.prompt_active(111))
        self.assertNotIn(111, bot.forward_buffers)
        self.assertEqual(self.client.mock_calls, [])
        self.assertIn("/prompt", self.go()[0])
        self.assertEqual(self.client.mock_calls, [])

    def test_new_clears_the_buffer_and_the_mode(self):
        self.ready("часть A")
        self.command(bot.cmd_new, 111, "/new")
        self.assertFalse(bot.prompt_active(111))
        self.assertNotIn(111, bot.forward_buffers)
        self.go()
        self.assertEqual(self.client.mock_calls, [])

    def test_timeout_turns_the_mode_off_and_go_does_not_send(self):
        self.ready("часть A")
        self.now += bot.FORWARD_BUFFER_TTL + 1
        out = self.go()
        self.assertIn("режим /prompt выключен", out[0])
        self.assertIn("30 минут", out[0])
        self.assertEqual(self.client.mock_calls, [])
        self.assertFalse(bot.prompt_active(111))
        self.assertIn("не включён", self.go()[0])  # уведомление один раз, дальше обычный ответ

    def test_text_after_the_timeout_is_not_sent_to_the_model(self):
        self.ready("часть A")
        self.now += bot.FORWARD_BUFFER_TTL + 1
        out = self.part(111, "а" * 100, 5000)
        self.assertIn("режим /prompt выключен", out[0])
        self.assertEqual(self.client.mock_calls, [])
        self.assertEqual(bot.conversations, {})

    def test_every_part_extends_the_timeout(self):
        self.start()
        for k in range(4):
            self.now += bot.FORWARD_BUFFER_TTL - 60
            self.part(111, f"часть {k}", k)
        self.assertTrue(bot.prompt_active(111))
        self.go()
        self.assertEqual(len(self.stream_calls()), 1)

    def test_new_prompt_after_the_timeout_starts_clean(self):
        self.ready("часть A")
        self.now += bot.FORWARD_BUFFER_TTL + 1
        self.start()
        self.assertEqual(bot.buffered_parts(111), [])
        self.assertTrue(bot.prompt_active(111))


class ModeExclusionTests(PromptBase):
    def test_prompt_is_refused_during_import(self):
        self.command(bot.cmd_import, 111, "/import")
        self.part(111, "часть A", 1)
        out = self.start()
        self.assertIn("/import", out[0])
        self.assertFalse(bot.prompt_active(111))
        self.assertTrue(bot.import_active(111))
        self.assertEqual(bot.buffered_parts(111), ["часть A"])

    def test_import_is_refused_during_prompt(self):
        self.ready("часть A")
        out = self.command(bot.cmd_import, 111, "/import")
        self.assertIn("/go", out[0])
        self.assertTrue(bot.prompt_active(111))
        self.assertFalse(bot.import_active(111))
        self.assertEqual(bot.buffered_parts(111), ["часть A"])

    def test_prompt_does_not_swallow_forwarded_scene_parts(self):
        self.run_handler(bot.handle_message, 111, "часть сцены", origin_from(BOT_ID, at(10)), date=at(500))
        out = self.start()
        self.assertIn("/keep", out[0])
        self.assertFalse(bot.prompt_active(111))
        self.assertEqual(bot.buffered_parts(111), ["часть сцены"])

    def test_done_and_keep_in_prompt_mode_point_to_go(self):
        self.ready("часть A")
        self.assertIn("/go", self.command(bot.cmd_done, 111, "/done")[0])
        self.assertIn("/go", self.command(bot.cmd_keep, 111, "/keep")[0])
        self.assertFalse(self.store.scenes)
        self.assertTrue(bot.prompt_active(111))
        self.assertEqual(self.client.mock_calls, [])

    def test_context_shows_the_mode(self):
        self.ready("часть A")
        self.assertIn("режим /prompt", self.command(bot.cmd_context, 111, "/context")[0])


class FailureAndPrivacyTests(PromptBase):
    def test_failed_generation_restores_the_buffer_and_history(self):
        self.ready("часть A", "часть B")
        self.client.messages.stream.side_effect = RuntimeError("boom")
        out = self.go()
        self.assertIn("повтори /go", out[-1])
        self.assertTrue(bot.prompt_active(111))
        self.assertEqual(bot.buffered_parts(111), ["часть A", "часть B"])
        self.assertEqual(bot.conversations[111], [])  # запрос не остался висеть в истории
        # повтор после починки: ровно ещё один вызов
        self.client.messages.stream.reset_mock()
        final = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ответ")], usage=self.usage, stop_reason="end_turn"
        )
        self.client.messages.stream.side_effect = lambda **kw: FakeStream(final)
        out = self.go()
        self.assertEqual(len(self.stream_calls()), 1)
        self.assertEqual(out[-1], "ответ")
        self.assertFalse(bot.prompt_active(111))

    def test_request_text_is_not_written_to_the_log(self):
        marker = "МАРКЕРЗАПРОСА"
        with self.assertLogs(bot.logger, level="DEBUG") as cm:
            self.start()
            self.part(111, marker + " а" * 50, 1)
            self.go()
            self.run_handler(bot.handle_message, 111, "а" * 5000, date=at(9000))  # отказ по длине
        self.assertNotIn(marker, "\n".join(cm.output))
        self.assertTrue(any("/go" in line and "chars" in line for line in cm.output))

    def test_request_text_is_not_echoed_in_replies(self):
        marker = "МАРКЕРЗАПРОСА"
        self.start()
        replies = self.part(111, marker * 10, 1) + self.go()[:1]
        self.assertNotIn(marker, "\n".join(replies))


if __name__ == "__main__":
    unittest.main()
