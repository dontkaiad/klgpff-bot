"""Вспомогательные вызовы без размышления, диагностика ответа, подсказка про /extract и
команда /extract <номер сцены>. Без сети. Тексты нейтральные.

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import unittest
from unittest import mock

import rag
from support import Base2, bot, extraction_json, words
from test_import_limit import OUTSIDER
from test_scenes import FakeConn

THINKING_OFF = {"type": "disabled"}


class ExtractBase(Base2):
    def seed(self, texts, user=111):
        self.store.scenes[user] = [{"request": f"запрос {i + 1}", "text": t, "summary": None} for i, t in enumerate(texts)]

    def extract(self, arg="1", user=111):
        return self.run_handler(bot.cmd_extract, user, f"/extract {arg}".strip())


# ---------------------------------------------- размышление выключено у вспомогательных вызовов
class AuxCallTests(ExtractBase):
    def test_extraction_and_summary_calls_disable_thinking(self):
        self.command(bot.cmd_import, 111, "/import")
        self.run_handler(bot.handle_message, 111, words(20))
        self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(self.extraction_calls()[0].kwargs["thinking"], THINKING_OFF)
        self.assertEqual(self.summary_calls()[0].kwargs["thinking"], THINKING_OFF)

    def test_default_is_disabled_and_adaptive_restores_the_models_default(self):
        self.assertEqual(bot.AUX_THINKING, "disabled")
        self.command(bot.cmd_import, 111, "/import")
        self.run_handler(bot.handle_message, 111, words(20))
        with mock.patch.object(bot, "AUX_THINKING", "adaptive"):
            self.command(bot.cmd_done, 111, "/done")
        self.assertNotIn("thinking", self.extraction_calls()[0].kwargs)
        self.assertNotIn("thinking", self.summary_calls()[0].kwargs)

    def test_scene_written_by_the_bot_uses_the_same_settings(self):
        self.stream_text = words(300)
        self.say(111, "напиши сцену")
        self.assertEqual(self.extraction_calls()[0].kwargs["thinking"], THINKING_OFF)

    def test_diagnostic_log_has_stop_reason_tokens_and_block_types_without_text(self):
        self.extract_reply = extraction_json(["СЕКРЕТНЫЙ ФАКТ"])
        self.extract_blocks = ("thinking", "text")
        self.stream_text = words(300)
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.say(111, "напиши сцену")
        line = next(l for l in cm.output if "extraction call:" in l)
        self.assertIn("stop=end_turn in=1000 out=200 blocks=thinking,text", line)
        self.assertNotIn("СЕКРЕТН", "\n".join(cm.output))

    def test_summary_call_is_logged_too(self):
        self.command(bot.cmd_import, 111, "/import")
        self.run_handler(bot.handle_message, 111, words(20))
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.command(bot.cmd_done, 111, "/done")
        self.assertTrue(any("summary call: stop=end_turn in=1000 out=200 blocks=text" in l for l in cm.output))

    def test_truncated_answer_with_only_a_thinking_block_is_visible_in_the_log(self):
        self.extract_stop = "max_tokens"
        self.extract_blocks = ("thinking",)  # весь лимит ушёл на размышление: текста нет
        self.stream_text = words(300)
        with self.assertLogs(bot.logger, level="INFO") as cm:
            out = self.say(111, "напиши сцену")
        self.assertTrue(any("stop=max_tokens" in l and "blocks=thinking" in l for l in cm.output))
        self.assertIn("/extract", out[-1])


class CutHintTests(ExtractBase):
    def test_done_hint_points_to_extract_with_the_scene_number_not_to_done(self):
        self.extract_stop = "max_tokens"
        self.extract_reply = "{"
        self.command(bot.cmd_import, 111, "/import")
        self.run_handler(bot.handle_message, 111, words(20))
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(out[-1], "извлечение фактов оборвалось, попробуй /extract 1")
        self.assertNotIn("/done ещё раз", "\n".join(out))

    def test_scene_that_is_not_saved_yet_gets_the_keep_then_extract_hint(self):
        self.extract_stop = "max_tokens"
        self.extract_reply = "{"
        self.stream_text = words(300)
        out = self.say(111, "напиши сцену")
        self.assertIn("/keep", out[-1])
        self.assertIn("/extract <номер>", out[-1])

    def test_explicit_commands_report_unparsable_answers_and_failures_auto_scenes_stay_silent(self):
        self.seed([words(30)])
        self.extract_reply = "{не json"
        self.assertIn("не получилось", self.extract()[-1])
        self.client.messages.create.side_effect = lambda **kw: (_ for _ in ()).throw(RuntimeError("api down"))
        self.assertIn("не получилось", self.extract()[-1])
        self.client.messages.create.side_effect = self._create
        self.extract_reply = "{не json"
        self.stream_text = words(300)
        self.assertEqual(self.say(111, "напиши сцену"), [words(300)])  # автоматический путь молчит


# ------------------------------------------------------------------------ /extract
class ExtractCommandTests(ExtractBase):
    def setUp(self):
        super().setUp()
        self.seed([words(20, 100), words(30, 500)])

    def test_runs_extraction_for_the_numbered_scene_like_after_done(self):
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        out = self.extract("2")
        self.assertEqual(out[0], "извлекаю кандидатов в факты из сцены 2...")
        self.assertIn("1. факт 1\n2. факт 2", out[1])
        self.assertIn("/accept", out[1])
        calls = self.extraction_calls()
        self.assertEqual(len(calls), 1)
        self.assertIn(words(30, 500), calls[0].kwargs["messages"][0]["content"])  # сцена 2, не сцена 1
        self.assertNotIn(words(20, 100), calls[0].kwargs["messages"][0]["content"])
        self.assertEqual(calls[0].kwargs["model"], bot.EXTRACT_MODEL)
        self.assertEqual(calls[0].kwargs["thinking"], THINKING_OFF)

    def test_number_matches_the_scenes_listing(self):
        listing = "\n".join(self.run_handler(bot.cmd_scenes, 111, "/scenes"))
        self.assertIn("2. 15.11.2026 · запрос 2 · 30 сл.", listing)
        self.extract("2")
        self.assertIn(words(30, 500), self.extraction_calls()[0].kwargs["messages"][0]["content"])

    def test_nothing_new_is_said_explicitly(self):
        self.assertEqual(self.extract("1")[-1], "новых фактов в тексте не найдено.")

    def test_repeat_does_not_duplicate_pending_candidates(self):
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        self.extract("1")
        out = self.extract("1")
        self.assertEqual([t for _, t in self.cands.pending_candidates(111)], ["факт 1", "факт 2"])
        self.assertEqual(out[-1], "новых фактов в тексте не найдено.")

    def test_candidates_are_only_candidates_nothing_is_written_to_facts(self):
        self.write_facts(111, ["факт канона 1"])
        self.extract_reply = extraction_json(["факт 1"])
        self.extract("1")
        self.assertEqual(self.read_facts(111), ["факт канона 1"])
        self.index_facts.assert_not_called()

    def test_does_not_save_or_summarize_anything(self):
        self.extract("1")
        self.assertEqual(self.summary_calls(), [])
        self.assertEqual(len(self.store.scenes[111]), 2)
        self.assertEqual(self.client.messages.stream.call_count, 0)

    def test_bad_arguments_cost_nothing(self):
        for arg in ("", "abc", "1.5"):
            self.assertIn("/extract 1", self.extract(arg)[0], arg)
        self.assertEqual(self.client.mock_calls, [])

    def test_missing_scene(self):
        for arg in ("3", "0", "-1"):
            self.assertEqual(self.extract(arg), [f"сцены {arg} нет, список: /scenes"])
        self.assertEqual(self.client.mock_calls, [])

    def test_storage_failure(self):
        self.store.fail = True
        self.assertIn("недоступна", self.extract("1")[0])
        self.assertEqual(self.client.mock_calls, [])

    def test_outsider_gets_silence(self):
        self.assertEqual(self.extract("1", user=OUTSIDER), [])
        self.assertEqual(self.client.mock_calls, [])

    def test_exhausted_daily_limit_says_so_and_does_not_call_the_model(self):
        self.spend(111, 3.5)
        out = self.extract("1")
        self.assertEqual(len(out), 1)
        self.assertIn("дневной лимит", out[0])
        self.assertEqual(self.extraction_calls(), [])

    def test_disabled_extraction(self):
        with mock.patch.object(bot, "EXTRACT_ENABLED", False):
            self.assertIn("EXTRACT_ENABLED", self.extract("1")[0])
        self.assertEqual(self.extraction_calls(), [])

    def test_truncated_answer_points_back_to_extract_with_the_same_number(self):
        self.extract_stop = "max_tokens"
        self.extract_reply = '{"new_facts":["факт'
        self.assertEqual(self.extract("2")[-1], "извлечение фактов оборвалось, попробуй /extract 2")

    def test_partial_json_is_salvaged(self):
        self.extract_stop = "max_tokens"
        self.extract_reply = '{"new_facts":["факт 1","факт 2","обор'
        self.assertIn("1. факт 1\n2. факт 2", self.extract("1")[-1])

    def test_cost_goes_to_usage_and_the_log_has_no_text(self):
        self.extract_reply = extraction_json(["СЕКРЕТНЫЙ ФАКТ"])
        before = bot.spent_today(111)
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.extract("1")
        self.assertGreater(bot.spent_today(111), before)
        line = next(l for l in cm.output if "/extract scene 1" in l)
        self.assertIn("extracted=1", line)
        self.assertRegex(line, r"extract_cost=\$0\.\d{4}")
        self.assertNotIn("СЕКРЕТН", "\n".join(cm.output))


class SceneTextSqlTests(unittest.TestCase):
    def test_uses_the_ordinal_number_and_handles_missing(self):
        log = []
        with mock.patch.object(rag, "_connect", lambda: FakeConn(log, lambda s, p: [("текст",)])):
            self.assertEqual(rag.get_scene_text(42, 3), "текст")
        self.assertEqual(log[0][2], (42, 2))  # OFFSET number-1
        self.assertIn("ORDER BY id", log[0][1])
        with mock.patch.object(rag, "_connect", lambda: FakeConn([], lambda s, p: [])):
            self.assertIsNone(rag.get_scene_text(42, 9))
        self.assertIsNone(rag.get_scene_text(42, 0))
        self.assertIsNone(rag.get_scene_text(42, -2))


if __name__ == "__main__":
    unittest.main()
