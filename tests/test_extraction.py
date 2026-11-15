"""Извлечение кандидатов в факты и /accept. Без сети: Anthropic, Voyage и Postgres
подменены. Тексты нейтральные («факт N», «w<номер>»).

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import asyncio
import json
import unittest
from unittest import mock

import rag
from support import Base2, bot, extract, extraction_json, words
from test_import_limit import OUTSIDER
from test_scenes import FakeConn


# ------------------------------------------------------------ чистые функции
class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract.parse_extraction('{"new_facts": ["факт 1", "факт 2"], "conflicts": []}'),
                         (["факт 1", "факт 2"], []))

    def test_fenced_and_chatty_answers(self):
        body = '{"new_facts": ["факт 1"], "conflicts": ["конфликт 1"]}'
        for raw in (f"```json\n{body}\n```", f"Вот результат:\n{body}\nГотово.", f"```\n{body}\n```"):
            self.assertEqual(extract.parse_extraction(raw), (["факт 1"], ["конфликт 1"]), raw)

    def test_broken_or_wrong_shape_gives_none_not_an_exception(self):
        for raw in ("", None, "не json", '{"new_facts": ["факт 1"', "[1, 2]", '"строка"', '{"other": 1}', "{"):
            self.assertIsNone(extract.parse_extraction(raw), raw)

    def test_at_most_8_facts_and_5_conflicts(self):
        raw = extraction_json([f"факт {i}" for i in range(12)], [f"конфликт {i}" for i in range(9)])
        facts, conflicts = extract.parse_extraction(raw)
        self.assertEqual(len(facts), 8)
        self.assertEqual(facts[0], "факт 0")
        self.assertEqual(len(conflicts), 5)

    def test_non_strings_and_blank_items_are_dropped(self):
        raw = json.dumps({"new_facts": ["факт 1", 5, None, "  ", ["x"], {"a": 1}], "conflicts": None})
        self.assertEqual(extract.parse_extraction(raw), (["факт 1"], []))

    def test_dict_conflicts_are_flattened_and_long_facts_cut(self):
        raw = json.dumps({"new_facts": ["а" * 1000],
                          "conflicts": [{"fact": "факт 3", "quote": "цитата"}]}, ensure_ascii=False)
        facts, conflicts = extract.parse_extraction(raw)
        self.assertEqual(len(facts[0]), extract.MAX_FACT_CHARS)
        self.assertEqual(conflicts, ["факт 3 — цитата"])


class DedupeTests(unittest.TestCase):
    def test_normalized_match_ignores_case_punctuation_and_yo(self):
        known = ["Героиня — рыжая!", "У него есть ёж"]
        out = extract.dedupe_normalized(["героиня рыжая", "у него есть еж.", "новый факт"], known)
        self.assertEqual(out, ["новый факт"])

    def test_pin_mark_is_ignored(self):
        self.assertEqual(extract.dedupe_normalized(["\U0001F4CC факт 1"], ["факт 1"]), [])

    def test_duplicates_inside_the_batch(self):
        self.assertEqual(extract.dedupe_normalized(["факт 1", "Факт 1.", "факт 2"], []), ["факт 1", "факт 2"])

    def test_empty_after_normalization_is_dropped(self):
        self.assertEqual(extract.dedupe_normalized(["...", "факт"], []), ["факт"])


class MessageTests(unittest.TestCase):
    def test_list_conflicts_and_hint(self):
        msg = extract.candidates_message([(4, "факт 1"), (5, "факт 2")], ["конфликт 1"])
        self.assertIn("4. факт 1\n5. факт 2", msg)
        self.assertIn("⚠️ противоречия канону:\n- конфликт 1", msg)
        self.assertIn("/accept 4 5 или /accept all; остальные сгорят через 24 часа", msg)

    def test_conflicts_only_has_no_accept_hint(self):
        msg = extract.candidates_message([], ["конфликт 1"])
        self.assertIn("противоречия", msg)
        self.assertNotIn("/accept", msg)

    def test_prompt_follows_the_spec(self):
        for piece in ('"new_facts"', '"conflicts"', "до 8 коротких фактов", "НЕ дублируют и не противоречат канону",
                      "без откровенных подробностей", "Не включай разовые события сюжета", "до 12 слов"):
            self.assertIn(piece, extract.EXTRACT_SYSTEM)


# ------------------------------------------------- rag: SQL-формы на фейковой БД
class CandidateSqlTests(unittest.TestCase):
    def conn(self, on_sql):
        self.log = []
        return mock.patch.object(rag, "_connect", lambda: FakeConn(self.log, on_sql))

    def test_add_purges_first_and_continues_numbering(self):
        def on_sql(sql, params):
            return [(7,)] if sql.startswith("SELECT coalesce(max(num)") else []

        with self.conn(on_sql):
            out = rag.add_candidates(42, ["факт 1", "факт 2"])
        self.assertEqual(out, [(8, "факт 1"), (9, "факт 2")])
        self.assertTrue(self.log[0][1].startswith("DELETE FROM fact_candidates WHERE created_at <"))
        self.assertIn("24 hours", self.log[0][1])
        rows = next(e[2] for e in self.log if e[0] == "executemany")
        self.assertEqual(rows, [(42, 8, "факт 1"), (42, 9, "факт 2")])

    def test_add_nothing_does_not_touch_the_db(self):
        with self.conn(lambda s, p: []):
            self.assertEqual(rag.add_candidates(42, []), [])
        self.assertEqual(self.log, [])

    def test_claim_takes_only_pending_requested_numbers(self):
        def on_sql(sql, params):
            if sql.startswith("SELECT num, status"):
                return [(1, "pending"), (2, "accepted"), (3, "pending")]
            if sql.startswith("UPDATE fact_candidates SET status = 'accepted'"):
                self.update_params = params
                return [(3, "факт 3"), (1, "факт 1")]
            return []

        with self.conn(on_sql):
            claimed, already, missing = rag.claim_candidates(42, [1, 2, 9])
        self.assertEqual(claimed, [(1, "факт 1"), (3, "факт 3")])
        self.assertEqual((already, missing), ([2], [9]))
        self.assertEqual(self.update_params, (42, [1]))  # в UPDATE только ожидающие из запрошенных
        update_sql = next(e[1] for e in self.log if e[1].startswith("UPDATE"))
        self.assertIn("status = 'pending'", update_sql)
        self.assertIn("RETURNING", update_sql)

    def test_claim_all_and_nothing_pending(self):
        with self.conn(lambda s, p: [(1, "accepted")] if s.startswith("SELECT num, status") else []):
            self.assertEqual(rag.claim_candidates(42, None), ([], [], []))
        self.assertFalse([e for e in self.log if e[1].startswith("UPDATE")])  # нечего обновлять

    def test_release_only_accepted(self):
        with self.conn(lambda s, p: []):
            rag.release_candidates(42, [3])
            rag.release_candidates(42, [])
        sqls = [e[1] for e in self.log]
        self.assertEqual(len(sqls), 1)
        self.assertIn("SET status = 'pending'", sqls[0])
        self.assertIn("status = 'accepted'", sqls[0])


class FilterSimilarTests(unittest.TestCase):
    """Эмбеддинги подменены: близость задаётся векторами, к фактам через ответ SQL."""

    VEC = {"факт 1": [1, 0], "факт 2": [0, 1], "почти факт 2": [0.01, 1], "факт 3": [1, 1],
           "ожидающий": [0, 1]}

    def run_filter(self, cands, pending, fact_sims, threshold=0.92):
        sims = iter(fact_sims)
        embed = mock.Mock(side_effect=lambda texts, kind: [self.VEC[t] for t in texts])
        with mock.patch.object(rag, "_embed", embed), \
                mock.patch.object(rag, "_connect", lambda: FakeConn([], lambda s, p: [(next(sims),)])):
            return rag.filter_similar(42, cands, pending, threshold), embed

    def test_similar_to_an_existing_fact_is_dropped(self):
        out, embed = self.run_filter(["факт 1", "факт 2"], [], [0.95, 0.30])
        self.assertEqual(out, ["факт 2"])
        self.assertEqual(embed.call_args[0][1], "document")

    def test_threshold_is_inclusive_below_kept(self):
        self.assertEqual(self.run_filter(["факт 1"], [], [0.9199])[0], ["факт 1"])
        self.assertEqual(self.run_filter(["факт 1"], [], [0.92])[0], [])

    def test_similar_to_a_pending_candidate_is_dropped(self):
        out, _ = self.run_filter(["факт 1", "почти факт 2"], ["ожидающий"], [0.1, 0.1])
        self.assertEqual(out, ["факт 1"])

    def test_similar_to_an_already_kept_candidate_is_dropped(self):
        out, _ = self.run_filter(["факт 2", "почти факт 2"], [], [0.1, 0.1])
        self.assertEqual(out, ["факт 2"])

    def test_empty_index_counts_as_not_similar(self):
        out, _ = self.run_filter(["факт 1"], [], [None])
        self.assertEqual(out, ["факт 1"])

    def test_no_candidates_no_calls(self):
        with mock.patch.object(rag, "_embed") as embed:
            self.assertEqual(rag.filter_similar(42, [], ["x"], 0.92), [])
        embed.assert_not_called()


# ----------------------------------------------------- извлечение: триггеры и поток
class ExtractionFlowTests(Base2):
    def generate(self, text, user=111):
        self.stream_text = text
        return self.say(user, "напиши сцену")

    def test_scene_of_enough_words_triggers_extraction_after_the_scene(self):
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        out = self.generate(words(300))
        self.assertEqual(len(self.extraction_calls()), 1)
        self.assertEqual(out[0], words(300))  # сначала сцена
        self.assertIn("кандидаты в факты", out[-1])  # потом отдельное сообщение
        self.assertIn("1. факт 1\n2. факт 2", out[-1])

    def test_short_answer_does_not_trigger_extraction(self):
        self.generate(words(299))
        self.assertEqual(self.extraction_calls(), [])

    def test_threshold_follows_the_setting(self):
        with mock.patch.object(bot, "EXTRACT_MIN_WORDS", 10):
            self.generate(words(10))
        self.assertEqual(len(self.extraction_calls()), 1)

    def test_extract_enabled_false(self):
        with mock.patch.object(bot, "EXTRACT_ENABLED", False):
            out = self.generate(words(400))
        self.assertEqual(self.extraction_calls(), [])
        self.assertEqual(out, [words(400)])

    def test_extraction_uses_the_configured_model_and_a_canon_list(self):
        self.write_facts(111, ["факт канона 1", "📌 факт канона 2"])
        self.generate(words(300))
        kw = self.extraction_calls()[0].kwargs
        self.assertEqual(kw["model"], bot.EXTRACT_MODEL)
        self.assertEqual(bot.EXTRACT_MODEL, bot.MODELS["haiku"])
        self.assertEqual(kw["max_tokens"], bot.EXTRACT_MAX_TOKENS)
        self.assertIn("- факт канона 1\n- факт канона 2", kw["messages"][0]["content"])  # без маркера 📌

    def test_nothing_is_written_to_facts_without_accept(self):
        self.write_facts(111, ["факт канона 1"])
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        self.generate(words(300))
        self.assertEqual(self.read_facts(111), ["факт канона 1"])
        self.index_facts.assert_not_called()
        self.assertEqual(self.cands.statuses(111), {1: "pending", 2: "pending"})

    def test_at_most_8_candidates_shown(self):
        self.extract_reply = extraction_json([f"факт {i}" for i in range(12)])
        out = self.generate(words(300))
        self.assertEqual(len(self.cands.rows[111]), 8)
        self.assertNotIn("9. ", out[-1])

    def test_broken_json_is_a_warning_and_silence(self):
        self.extract_reply = "{не json"
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.generate(words(300))
        self.assertEqual(out, [words(300)])  # пользователь получил только сцену
        self.assertTrue(any("не разобран" in l for l in cm.output), cm.output)
        self.assertEqual(self.cands.rows, {})

    def test_model_failure_never_breaks_the_reply(self):
        self.client.messages.create.side_effect = RuntimeError("api down")
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.generate(words(300))
        self.assertEqual(out, [words(300)])
        self.assertTrue(any("extraction failed" in l for l in cm.output))

    def test_storage_failure_is_silent_too(self):
        self.extract_reply = extraction_json(["факт 1"])
        self.cands.fail = True
        out = self.generate(words(300))
        self.assertEqual(out, [words(300)])

    def test_dedupe_against_existing_facts_by_normalized_text(self):
        self.write_facts(111, ["Героиня рыжая."])
        self.extract_reply = extraction_json(["героиня  рыжая", "факт 2"])
        self.generate(words(300))
        self.assertEqual([t for _, t in self.cands.pending_candidates(111)], ["факт 2"])

    def test_dedupe_against_pending_candidates(self):
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        self.generate(words(300))
        self.extract_reply = extraction_json(["Факт 2!", "факт 3"])
        out = self.generate(words(300, 5000))
        self.assertEqual([t for _, t in self.cands.pending_candidates(111)], ["факт 1", "факт 2", "факт 3"])
        self.assertIn("3. факт 3", out[-1])  # нумерация продолжается
        self.assertNotIn("факт 2", out[-1].split("/accept")[0].replace("3. факт 3", ""))

    def test_similarity_dedupe_is_applied(self):
        self.cands.drop_similar = {"факт 1"}
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        self.generate(words(300))
        self.assertEqual([t for _, t in self.cands.pending_candidates(111)], ["факт 2"])

    def test_similarity_failure_falls_back_to_normalized_dedupe(self):
        self.cands.similar_fails = True
        self.extract_reply = extraction_json(["факт 1"])
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.generate(words(300))
        self.assertIn("1. факт 1", out[-1])
        self.assertTrue(any("близости" in l for l in cm.output))

    def test_conflicts_only_message(self):
        self.extract_reply = extraction_json([], ["факт 3 — «цитата»"])
        out = self.generate(words(300))
        self.assertIn("противоречия канону", out[-1])
        self.assertNotIn("/accept", out[-1])

    def test_nothing_found_is_silent_for_scenes(self):
        out = self.generate(words(300))
        self.assertEqual(out, [words(300)])

    def test_only_one_scene_message_when_extraction_is_off_the_path(self):
        out = self.generate(words(50))
        self.assertEqual(len(out), 1)


class ExtractionCostTests(Base2):
    def test_cost_goes_to_usage_cost_command_and_daily_limit(self):
        self.stream_text = words(300)
        before = bot.spent_today(111)
        self.say(111, "напиши сцену")
        haiku = bot.MODELS["haiku"]
        usage = bot.load_usage(111)
        self.assertIn(haiku, usage)  # учтено под моделью извлечения
        self.assertEqual(usage[haiku]["input"], 1000)
        self.assertGreater(bot.spent_today(111), before)
        cost_text = "\n".join(self.run_handler(bot.cmd_cost, 111, "/cost"))
        self.assertIn(bot.model_label(haiku), cost_text)

    def test_done_log_has_extracted_and_cost(self):
        self.stream_text = words(300)
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.say(111, "напиши сцену")
        done = next(l for l in cm.output if "done:" in l)
        self.assertIn("extracted=2", done)
        self.assertRegex(done, r"extract_cost=\$0\.\d{4}")

    def test_done_log_without_extraction(self):
        self.stream_text = words(50)
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.say(111, "привет")
        done = next(l for l in cm.output if "done:" in l)
        self.assertIn("extracted=0 extract_cost=$0.0000", done)

    def test_exhausted_limit_skips_extraction_silently_with_warning(self):
        self.spend(111, 3.5)
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            result = asyncio.run(bot.run_extraction(self.update(111, "x")[0], 111, 111, words(400)))
        self.assertEqual(result, (0, 0.0))
        self.assertEqual(self.extraction_calls(), [])
        self.assertTrue(any("extraction skipped" in l for l in cm.output))

    def test_done_command_is_skipped_at_the_limit_without_a_message(self):
        self.command(bot.cmd_import, 111, "/import")
        self.run_handler(bot.handle_message, 111, words(20))
        self.spend(111, 3.5)
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(len(out), 1)  # только «сохранена», без извлечения
        self.assertEqual(self.extraction_calls(), [])


# ------------------------------------------------------------------- /accept
class AcceptTests(Base2):
    def setUp(self):
        super().setUp()
        self.write_facts(111, ["факт канона 1", "факт канона 2"])
        self.cands.add_candidates(111, ["кандидат 1", "кандидат 2", "кандидат 3"])

    def accept(self, arg, user=111):
        return "\n".join(self.run_handler(bot.cmd_accept, user, f"/accept {arg}".strip()))

    def test_accept_all(self):
        out = self.accept("all")
        self.assertEqual(self.read_facts(111), ["факт канона 1", "факт канона 2", "кандидат 1", "кандидат 2", "кандидат 3"])
        self.assertIn("добавлено фактов: 3 (в /facts № 3-5)", out)
        self.assertIn("3. кандидат 1", out)
        self.assertEqual(set(self.cands.statuses(111).values()), {"accepted"})

    def test_accept_numbers_adds_only_those_in_order(self):
        out = self.accept("3 1")
        self.assertEqual(self.read_facts(111)[2:], ["кандидат 1", "кандидат 3"])
        self.assertIn("добавлено фактов: 2 (в /facts № 3-4)", out)
        self.assertEqual(self.cands.statuses(111), {1: "accepted", 2: "pending", 3: "accepted"})

    def test_single_fact_and_comma_separated(self):
        self.assertIn("в /facts № 3)", self.accept("2"))
        self.accept("1, 3")
        self.assertEqual(self.read_facts(111)[2:], ["кандидат 2", "кандидат 1", "кандидат 3"])

    def test_indexing_and_backup_follow_the_remember_path(self):
        self.accept("1 2")
        self.index_facts.assert_called_once()
        self.assertEqual(self.index_facts.call_args[0][0], 111)
        self.assertEqual(self.index_facts.call_args[0][1][-2:], ["кандидат 1", "кандидат 2"])
        self.assertEqual(json.loads((self.backup_dir / "111.json").read_text(encoding="utf-8"))[-1], "кандидат 2")

    def test_repeat_accept_does_not_duplicate(self):
        self.accept("1 2")
        out = self.accept("1 2")
        self.assertEqual(self.read_facts(111).count("кандидат 1"), 1)
        self.assertEqual(len(self.read_facts(111)), 4)
        self.assertIn("уже приняты раньше: 1, 2", out)
        self.assertEqual(self.index_facts.call_count, 1)  # повтор ничего не переиндексировал
        self.assertIn("добавлено фактов: 1", self.accept("all"))  # all берёт только оставшийся третий
        self.assertEqual(len(self.read_facts(111)), 5)
        self.assertIn("нет ожидающих", self.accept("all"))
        self.assertEqual(len(self.read_facts(111)), 5)

    def test_unknown_numbers_and_mixed_input(self):
        out = self.accept("2 9")
        self.assertIn("нет таких кандидатов: 9", out)
        self.assertEqual(self.read_facts(111)[2:], ["кандидат 2"])

    def test_only_unknown_numbers_change_nothing(self):
        out = self.accept("7 8")
        self.assertIn("нет таких кандидатов: 7, 8", out)
        self.assertEqual(len(self.read_facts(111)), 2)
        self.index_facts.assert_not_called()

    def test_invalid_arguments(self):
        for arg in ("", "abc", "0", "-1", "1 x", "all 1"):
            out = self.accept(arg)
            self.assertIn("/accept 1 3 5 или /accept all", out, arg)
        self.assertEqual(len(self.read_facts(111)), 2)
        self.index_facts.assert_not_called()

    def test_nothing_pending(self):
        self.accept("all")
        self.assertIn("нет ожидающих кандидатов", self.accept("all"))

    def test_failed_write_releases_the_candidates(self):
        with mock.patch.object(bot, "save_facts", side_effect=OSError("disk")):
            out = self.accept("1 2")
        self.assertIn("не удалось записать факты", out)
        self.assertEqual(self.cands.statuses(111), {1: "pending", 2: "pending", 3: "pending"})
        self.accept("1 2")  # повтор работает
        self.assertEqual(len(self.read_facts(111)), 4)

    def test_db_failure_is_reported(self):
        self.cands.fail = True
        self.assertIn("недоступны", self.accept("all"))
        self.assertEqual(len(self.read_facts(111)), 2)

    def test_candidates_are_per_user(self):
        self.assertIn("нет ожидающих", self.accept("all", user=222))
        self.assertEqual(len(self.read_facts(111)), 2)

    def test_outsider_gets_silence(self):
        self.assertEqual(self.run_handler(bot.cmd_accept, OUTSIDER, "/accept all"), [])
        self.assertEqual(self.cands.statuses(111), {1: "pending", 2: "pending", 3: "pending"})

    def test_accept_does_not_call_the_model(self):
        self.accept("all")
        self.assertEqual(self.client.mock_calls, [])

    def test_numbers_stay_stable_when_a_new_batch_arrives(self):
        self.accept("1")
        self.cands.add_candidates(111, ["кандидат 4"])
        self.accept("4")
        self.assertEqual(self.read_facts(111)[2:], ["кандидат 1", "кандидат 4"])


# ------------------------------------------------ обрезанный ответ (stop_reason=max_tokens)
class SalvageTests(unittest.TestCase):
    def test_complete_elements_survive_and_the_cut_one_is_dropped(self):
        raw = '{"new_facts":["факт 1","факт 2","факт 3 обор'
        self.assertEqual(extract.salvage_truncated(raw), (["факт 1", "факт 2"], []))

    def test_both_arrays(self):
        raw = '{"new_facts":["факт 1","факт 2"],"conflicts":["конфликт 1","конфликт 2 обор'
        self.assertEqual(extract.salvage_truncated(raw), (["факт 1", "факт 2"], ["конфликт 1"]))

    def test_object_conflicts_cut_in_the_middle_are_dropped(self):
        raw = '{"new_facts":["факт 1"],"conflicts":[{"fact":"факт 3","quote":"цитата"},{"fact":"факт 4","quo'
        self.assertEqual(extract.salvage_truncated(raw), (["факт 1"], ["факт 3 — цитата"]))

    def test_escaped_quotes_and_fences(self):
        raw = '```json\n{"new_facts":["он сказал \\"да\\"","факт 2","оборв'
        self.assertEqual(extract.salvage_truncated(raw), (['он сказал "да"', "факт 2"], []))

    def test_nothing_complete_gives_none(self):
        for raw in ('{"new_facts":["факт', '{"new_facts":[', "", None, "просто текст", '{"conflicts":[{"fact":"x'):
            self.assertIsNone(extract.salvage_truncated(raw), raw)

    def test_limits_still_apply(self):
        raw = '{"new_facts":[' + ",".join(f'"факт {i}"' for i in range(12)) + ',"обор'
        self.assertEqual(len(extract.salvage_truncated(raw)[0]), 8)

    def test_prompt_asks_for_compact_json_with_limits(self):
        for piece in ("компактный JSON", "до 15 слов", "не больше 8", "conflicts не больше 5"):
            self.assertIn(piece, extract.EXTRACT_SYSTEM)


class TruncationFlowTests(Base2):
    def generate(self, text=None):
        self.stream_text = text or words(300)
        return self.say(111, "напиши сцену")

    def test_default_max_tokens_is_3000_and_is_passed_to_the_model(self):
        self.assertEqual(bot.EXTRACT_MAX_TOKENS, 3000)
        self.generate()
        self.assertEqual(self.extraction_calls()[0].kwargs["max_tokens"], 3000)

    def test_partial_json_is_parsed_with_a_warning_and_without_a_user_error(self):
        self.extract_stop = "max_tokens"
        self.extract_reply = '{"new_facts":["факт 1","факт 2","факт 3 обор'
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.generate()
        self.assertEqual(len(self.extraction_calls()), 1)  # повторного вызова нет
        self.assertIn("1. факт 1\n2. факт 2", out[-1])
        self.assertNotIn("оборвалось", "\n".join(out))
        self.assertNotIn("факт 3", out[-1])
        self.assertTrue(any("обрезан" in l and "взяты полные элементы: 2" in l for l in cm.output), cm.output)

    def test_nothing_to_salvage_tells_the_user_once_and_does_not_retry(self):
        self.extract_stop = "max_tokens"
        self.extract_reply = '{"new_facts":["факт 1 обор'
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.generate()
        self.assertEqual(out[0], words(300))  # сцена дошла
        self.assertEqual(out[1:], ["извлечение фактов оборвалось, попробуй /done ещё раз"])
        self.assertEqual(len(self.extraction_calls()), 1)  # автоповтора нет
        self.assertTrue(any("обрезан" in l and "достать нечего" in l for l in cm.output))
        self.assertEqual(self.cands.rows, {})
        self.assertGreater(bot.spent_today(111), 0)  # потраченное учтено

    def test_done_command_reports_the_truncation_too(self):
        self.command(bot.cmd_import, 111, "/import")
        self.run_handler(bot.handle_message, 111, words(20))
        self.extract_stop = "max_tokens"
        self.extract_reply = "{"
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(out[-1], "извлечение фактов оборвалось, попробуй /done ещё раз")
        self.assertEqual(len(self.extraction_calls()), 1)

    def test_truncated_flag_with_a_complete_json_is_a_normal_answer(self):
        self.extract_stop = "max_tokens"
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        out = self.generate()
        self.assertIn("1. факт 1\n2. факт 2", out[-1])
        self.assertNotIn("оборвалось", "\n".join(out))

    def test_unparsable_answer_that_is_not_truncated_stays_silent(self):
        self.extract_reply = "{не json"
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.generate()
        self.assertEqual(out, [words(300)])
        self.assertTrue(any("не разобран" in l for l in cm.output))

    def test_done_log_counts_zero_extracted_for_a_failed_extraction(self):
        self.extract_stop = "max_tokens"
        self.extract_reply = "{"
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.generate()
        self.assertIn("extracted=0", next(l for l in cm.output if "done:" in l))


if __name__ == "__main__":
    unittest.main()
