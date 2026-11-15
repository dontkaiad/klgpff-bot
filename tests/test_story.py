"""Хроника сюжета: выжимка при сохранении (/done, /keep), блок <story_so_far>, /scenes N,
/context. Без сети: Anthropic, Voyage и Postgres подменены. Тексты нейтральные.

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import datetime
import unittest
from unittest import mock

import rag
from support import Base2, bot, words
from test_import_limit import OUTSIDER
from test_scenes import FakeConn

FAILED_LINE = "выжимка не получилась"


class StoryBase(Base2):
    def seed(self, summaries, user=111):
        """Сцены юзера с выжимками (None: без выжимки), в порядке сохранения."""
        self.store.scenes[user] = [
            {"request": f"запрос {i + 1}", "text": f"текст сцены {i + 1}", "summary": s}
            for i, s in enumerate(summaries)
        ]

    def import_scene(self, text_words=20, user=111):
        self.command(bot.cmd_import, user, "/import")
        self.run_handler(bot.handle_message, user, words(text_words))

    def last_request(self):
        return self.client.messages.stream.call_args.kwargs["messages"][-1]["content"]

    def scenes_on(self):
        self.command(bot.cmd_context, 111, "/context scenes on")


# ------------------------------------------------------------ выжимка при сохранении
class SummaryOnSaveTests(StoryBase):
    def test_done_creates_the_summary_with_the_configured_model(self):
        self.import_scene()
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(out, ["✓ сохранена: сцена 1 (частей: 1, слов: 20)", "новых фактов в тексте не найдено."])
        calls = self.summary_calls()
        self.assertEqual(len(calls), 1)
        kw = calls[0].kwargs
        self.assertEqual(kw["model"], bot.SUMMARY_MODEL)
        self.assertEqual(bot.SUMMARY_MODEL, bot.MODELS["haiku"])
        self.assertEqual(kw["max_tokens"], 800)
        self.assertEqual(kw["messages"][0]["content"], words(20))
        self.assertEqual(self.store.scenes[111][0]["summary"], "выжимка сцены")

    def test_keep_creates_the_summary(self):
        h = bot.get_history(111)
        h += [{"role": "user", "content": "запрос 1"}, {"role": "assistant", "content": "сцена 1"}]
        self.command(bot.cmd_keep, 111, "/keep")
        self.assertEqual(len(self.summary_calls()), 1)
        self.assertEqual(self.store.scenes[111][0]["summary"], "выжимка сцены")

    def test_summary_goes_before_extraction_in_done(self):
        self.import_scene()
        self.command(bot.cmd_done, 111, "/done")
        systems = [c.kwargs.get("system") for c in self.client.messages.create.call_args_list]
        self.assertEqual(systems, [bot.SUMMARY_SYSTEM, bot.extract.EXTRACT_SYSTEM])

    def test_prompt_follows_the_spec(self):
        for piece in ("до 150 слов", "кто участвует", "где и когда", "что произошло", "отношения",
                      "линии остались открытыми", "Нейтральные формулировки", "без откровенных подробностей",
                      "Не выдумывай"):
            self.assertIn(piece, bot.SUMMARY_SYSTEM)

    def test_failed_call_still_saves_the_scene_and_says_so_once(self):
        self.client.messages.create.side_effect = lambda **kw: (_ for _ in ()).throw(RuntimeError("api down")) \
            if kw.get("system") == bot.SUMMARY_SYSTEM else self._create(**kw)
        self.import_scene()
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(self.store.scenes[111][0]["text"], words(20))  # сцена сохранена
        self.assertIsNone(self.store.scenes[111][0].get("summary"))
        self.assertEqual(out.count(FAILED_LINE), 1)
        self.assertTrue(any("summary failed" in l for l in cm.output))
        self.assertEqual(len(self.extraction_calls()), 1)  # /done продолжился

    def test_truncated_answer_is_a_failure_not_a_summary(self):
        self.summary_stop = "max_tokens"
        self.import_scene()
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.command(bot.cmd_done, 111, "/done")
        self.assertIn(FAILED_LINE, out)
        self.assertIsNone(self.store.scenes[111][0].get("summary"))
        self.assertTrue(any("обрезан" in l for l in cm.output))
        self.assertGreater(bot.spent_today(111), 0)  # обрезанный вызов всё равно оплачен

    def test_empty_answer_is_a_failure(self):
        self.summary_reply = "   "
        self.import_scene()
        self.assertIn(FAILED_LINE, self.command(bot.cmd_done, 111, "/done"))

    def test_storage_failure_when_writing_the_summary(self):
        self.import_scene()
        with mock.patch.object(bot.rag, "set_summary", side_effect=RuntimeError("db")):
            out = self.command(bot.cmd_done, 111, "/done")
        self.assertIn(FAILED_LINE, out)
        self.assertIn("✓ сохранена", out[0])

    def test_exhausted_daily_limit_skips_the_summary_silently(self):
        self.import_scene()
        self.spend(111, 3.5)
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(self.summary_calls(), [])
        self.assertNotIn(FAILED_LINE, out)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(self.store.scenes[111]), 1)  # сцена сохранена
        self.assertTrue(any("summary skipped" in l for l in cm.output))

    def test_cost_goes_to_usage_cost_and_the_daily_limit(self):
        self.import_scene()
        before = bot.spent_today(111)
        self.command(bot.cmd_done, 111, "/done")
        usage = bot.load_usage(111)[bot.MODELS["haiku"]]
        self.assertEqual(usage["input"], 2 * 1000)  # выжимка и извлечение по 1000 токенов входа
        self.assertGreater(bot.spent_today(111), before)
        self.assertIn(bot.model_label(bot.MODELS["haiku"]), "\n".join(self.run_handler(bot.cmd_cost, 111, "/cost")))

    def test_summary_cost_alone_is_counted(self):
        with mock.patch.object(bot, "EXTRACT_ENABLED", False):
            self.import_scene()
            self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(bot.load_usage(111)[bot.MODELS["haiku"]]["input"], 1000)

    def test_duplicate_with_a_summary_is_not_summarized_again(self):
        self.import_scene()
        self.command(bot.cmd_done, 111, "/done")
        self.import_scene()
        self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(len(self.summary_calls()), 1)

    def test_duplicate_without_a_summary_gets_one(self):
        self.store.scenes[111] = [{"request": "r", "text": words(20), "summary": None}]
        self.import_scene()
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertIn("уже сохранена: сцена 1", out[0])
        self.assertEqual(self.store.scenes[111][0]["summary"], "выжимка сцены")

    def test_long_answer_is_capped(self):
        self.summary_reply = "а" * 5000
        self.import_scene()
        self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(len(self.store.scenes[111][0]["summary"]), bot.SUMMARY_MAX_CHARS)

    def test_outsider_triggers_nothing(self):
        self.run_handler(bot.cmd_done, OUTSIDER, "/done")
        self.run_handler(bot.cmd_keep, OUTSIDER, "/keep")
        self.assertEqual(self.client.mock_calls, [])


# ------------------------------------------------------------------ блок <story_so_far>
class StoryBlockTests(StoryBase):
    def setUp(self):
        super().setUp()
        self.write_facts(111, ["факт канона 1"])
        self.seed(["выжимка 1", "выжимка 2", "выжимка 3"])

    def test_default_has_no_block_and_does_not_even_read_the_chronicle(self):
        with mock.patch.object(bot.rag, "story_chronicle") as chronicle:
            self.say(111, "привет")
        self.assertNotIn("story_so_far", str(self.client.messages.stream.call_args.kwargs))
        chronicle.assert_not_called()

    def test_after_scenes_on_the_block_is_present_then_off_again_absent(self):
        self.scenes_on()
        self.say(111, "привет")
        self.assertIn("<story_so_far>", self.last_request())
        self.command(bot.cmd_context, 111, "/context scenes off")
        self.say(111, "ещё")
        self.assertNotIn("story_so_far", self.last_request())

    def test_block_stands_after_the_canon_in_the_tail_never_in_system(self):
        self.scenes_on()
        self.say(111, "привет")
        kwargs = self.client.messages.stream.call_args.kwargs
        last = kwargs["messages"][-1]["content"]
        self.assertLess(last.index("</canon>"), last.index("<story_so_far>"))
        self.assertTrue(last.endswith("</story_so_far>"))
        self.assertNotIn("story_so_far", str(kwargs["system"]))
        for m in kwargs["messages"][:-1]:
            self.assertNotIn("story_so_far", str(m))

    def test_prefix_for_the_cache_is_the_same_with_and_without_the_block(self):
        self.say(111, "привет")
        without = self.client.messages.stream.call_args.kwargs["messages"][:-1]
        self.scenes_on()
        self.say(111, "ещё")
        with_story = self.client.messages.stream.call_args.kwargs["messages"][:-1]
        self.assertEqual(with_story[:len(without)], without)

    def test_text_numbering_and_order_follow_the_saving_order(self):
        block = bot.build_story_block([(1, "выжимка 1"), (2, "выжимка 2"), (4, "выжимка 4")])
        self.assertTrue(block.startswith("<story_so_far>\nХроника ранее сохранённых сцен автора по порядку. "
                                         "Это не канон. Если она противоречит <canon>, прав канон. "
                                         "Продолжай сюжет с учётом событий и развития отношений, не повторяй "
                                         "сцены и не пересказывай их."))
        self.assertTrue(block.endswith("1. выжимка 1\n2. выжимка 2\n4. выжимка 4\n</story_so_far>"))

    def test_scenes_without_a_summary_are_skipped_and_numbers_stay(self):
        self.seed(["выжимка 1", None, "выжимка 3"])
        self.scenes_on()
        self.say(111, "привет")
        self.assertIn("1. выжимка 1\n3. выжимка 3", self.last_request())
        self.assertNotIn("2. ", self.last_request().split("<story_so_far>")[1])

    def test_no_scenes_or_no_summaries_means_no_block_and_the_request_is_as_before(self):
        for summaries in ([], [None, None]):
            self.seed(summaries)
            self.scenes_on()
            self.say(111, "привет")
            self.assertNotIn("story_so_far", self.last_request())
            self.assertEqual(self.last_request(), f"привет\n\n{bot.build_canon_block(['факт канона 1'])}")

    def test_scene_count_limit_takes_the_freshest_and_notes_the_rest(self):
        items = [(i, f"выжимка {i}") for i in range(1, 6)]
        with mock.patch.object(bot, "STORY_MAX_SCENES", 2):
            block = bot.build_story_block(items)
        self.assertIn("(ранних сцен не вошло в хронику: 3)", block)
        self.assertTrue(block.endswith("4. выжимка 4\n5. выжимка 5\n</story_so_far>"))
        self.assertNotIn("3. выжимка 3", block)

    def test_char_limit_takes_the_freshest_that_fit_in_order(self):
        items = [(i, "а" * 100) for i in range(1, 6)]
        size = len("1. " + "а" * 100) + 1
        with mock.patch.object(bot, "STORY_MAX_CHARS", 3 * size):
            chosen, skipped = bot.select_chronicle(items)
        self.assertEqual([n for n, _ in chosen], [3, 4, 5])
        self.assertEqual(skipped, 2)
        with mock.patch.object(bot, "STORY_MAX_CHARS", 3 * size - 1):  # на один символ меньше
            self.assertEqual([n for n, _ in bot.select_chronicle(items)[0]], [4, 5])

    def test_nothing_fits_means_no_block(self):
        with mock.patch.object(bot, "STORY_MAX_CHARS", 5):
            self.assertEqual(bot.build_story_block([(1, "выжимка 1")]), "")

    def test_limit_defaults(self):
        self.assertEqual((bot.STORY_MAX_SCENES, bot.STORY_MAX_CHARS), (30, 12000))

    def test_zero_limits_switch_the_chronicle_off(self):
        self.scenes_on()
        with mock.patch.object(bot, "STORY_MAX_SCENES", 0):
            self.say(111, "привет")
        self.assertNotIn("story_so_far", self.last_request())

    def test_thirty_one_scenes_the_oldest_is_dropped_with_a_note(self):
        self.seed([f"выжимка {i}" for i in range(1, 32)])
        self.scenes_on()
        self.say(111, "привет")
        block = self.last_request().split("<story_so_far>")[1]
        self.assertIn("(ранних сцен не вошло в хронику: 1)", block)
        self.assertNotIn("\n1. выжимка 1\n", block)
        self.assertIn("\n2. выжимка 2\n", block)
        self.assertIn("\n31. выжимка 31\n", block)

    def test_chronicle_failure_means_no_block_and_a_warning(self):
        self.scenes_on()
        self.store.fail = True
        with self.assertLogs(bot.logger, level="WARNING") as cm:
            self.say(111, "привет")
        self.assertNotIn("story_so_far", self.last_request())
        self.assertTrue(any("хроника" in l for l in cm.output))
        self.client.messages.stream.assert_called_once()  # бот ответил как обычно

    def test_fragments_default_off_and_order_canon_story_fragments(self):
        self.assertEqual(bot.SCENE_TOP_K, 0)
        search = mock.Mock(return_value=[(2, "фрагмент сцены 2")])
        self.scenes_on()
        with mock.patch.object(bot.rag, "search_scenes", search):
            self.say(111, "привет")
            search.assert_not_called()  # SCENE_TOP_K=0
            with mock.patch.object(bot, "SCENE_TOP_K", 3):
                self.say(111, "ещё")
        last = self.last_request()
        self.assertLess(last.index("</canon>"), last.index("<story_so_far>"))
        self.assertLess(last.index("</story_so_far>"), last.index("<memory_scenes>"))

    def test_done_log_reports_the_story_size(self):
        self.scenes_on()
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.say(111, "привет")
        self.assertIn(" story=3 ", next(l for l in cm.output if "done:" in l))

    def test_summary_texts_never_reach_the_log(self):
        self.seed(["СЕКРЕТНАЯ ВЫЖИМКА"])
        self.scenes_on()
        with self.assertLogs(bot.logger, level="INFO") as cm:
            self.say(111, "привет")
        self.assertNotIn("СЕКРЕТНАЯ", "\n".join(cm.output))


# ------------------------------------------------------------------- /scenes N, /context
class ScenesNCommandTests(StoryBase):
    def setUp(self):
        super().setUp()
        self.seed(["выжимка 1", None])

    def scenes(self, arg="", user=111):
        return "\n".join(self.run_handler(bot.cmd_scenes, user, f"/scenes {arg}".strip()))

    def test_shows_the_summary_without_the_full_text_and_without_the_model(self):
        self.store.scenes[111][0]["text"] = "ПОЛНЫЙ ТЕКСТ СЦЕНЫ"
        out = self.scenes("1")
        self.assertIn("сцена 1 · 15.11.2026 · запрос 1", out)
        self.assertIn("выжимка 1", out)
        self.assertNotIn("ПОЛНЫЙ ТЕКСТ", out)
        self.assertEqual(self.client.mock_calls, [])

    def test_scene_without_a_summary(self):
        self.assertIn("выжимки нет", self.scenes("2"))

    def test_bad_input(self):
        self.assertIn("/scenes 3", self.scenes("abc"))
        self.assertIn("сцены 9 нет", self.scenes("9"))
        self.assertIn("сцены 0 нет", self.scenes("0"))
        self.assertEqual(self.client.mock_calls, [])

    def test_list_marks_scenes_with_and_without_a_summary(self):
        out = self.scenes()
        self.assertIn("1. 15.11.2026 · запрос 1 · 3 сл. · выжимка есть", out)
        self.assertIn("2. 15.11.2026 · запрос 2 · 3 сл. · без выжимки", out)

    def test_db_failure_and_outsider(self):
        self.store.fail = True
        self.assertIn("недоступна", self.scenes("1"))
        self.assertEqual(self.scenes("1", user=OUTSIDER), "")


class ContextChronicleTests(StoryBase):
    def setUp(self):
        super().setUp()
        self.seed(["СЕКРЕТНАЯ ВЫЖИМКА 1", None, "СЕКРЕТНАЯ ВЫЖИМКА 3"])

    def ctx(self):
        return "\n".join(self.command(bot.cmd_context, 111, "/context"))

    def test_counts_size_and_scenes_without_a_summary_without_the_model_and_texts(self):
        out = self.ctx()
        size = len("СЕКРЕТНАЯ ВЫЖИМКА 1") + len("СЕКРЕТНАЯ ВЫЖИМКА 3")
        self.assertIn(f"хроника сюжета: 2 сцен, {size} симв.; без выжимки: 1", out)
        self.assertIn("сохранено сцен: 3", out)
        self.assertNotIn("СЕКРЕТН", out)
        self.assertEqual(self.client.mock_calls, [])

    def test_off_is_marked_and_on_is_not(self):
        self.assertIn("не подмешивается", self.ctx())
        self.scenes_on()
        self.assertNotIn("не подмешивается", self.ctx())

    def test_limited_chronicle_says_how_many_will_go(self):
        self.scenes_on()
        with mock.patch.object(bot, "STORY_MAX_SCENES", 1):
            self.assertIn("в запрос войдут последние 1", self.ctx())

    def test_fragments_line(self):
        self.assertIn("фрагменты сцен: выкл (SCENE_TOP_K=0)", self.ctx())
        with mock.patch.object(bot, "SCENE_TOP_K", 3):
            self.assertIn("фрагменты сцен: до 3", self.ctx())

    def test_unavailable_storage(self):
        self.store.fail = True
        out = self.ctx()
        self.assertIn("хроника сюжета: н/д", out)
        self.assertIn("сохранено сцен: н/д", out)

    def test_scenes_on_reply_mentions_the_chronicle(self):
        out = self.command(bot.cmd_context, 111, "/context scenes on")[0]
        self.assertIn("хроника сюжета", out)


# ------------------------------------------------------- SQL-формы на фейковой БД
class StorySqlTests(unittest.TestCase):
    def conn(self, on_sql):
        self.log = []
        return mock.patch.object(rag, "_connect", lambda: FakeConn(self.log, on_sql))

    def test_schema_adds_the_summary_column_after_the_scenes_table(self):
        sqls = [" ".join(x.split()) for x in rag._SCHEMA]
        i_table = next(i for i, x in enumerate(sqls) if x.startswith("CREATE TABLE IF NOT EXISTS scenes ("))
        i_alter = sqls.index("ALTER TABLE scenes ADD COLUMN IF NOT EXISTS summary text")
        self.assertGreater(i_alter, i_table)

    def test_list_scenes_returns_a_summary_flag(self):
        created = datetime.datetime(2026, 11, 15)
        with self.conn(lambda s, p: [(created, "r1", 10, True), (created, "r2", 20, False)]):
            self.assertEqual(rag.list_scenes(42), [(created, "r1", 10, True), (created, "r2", 20, False)])
        self.assertIn("coalesce(summary, '') <> ''", self.log[0][1])
        self.assertIn("ORDER BY id", self.log[0][1])

    def test_story_chronicle_numbers_by_position_among_all_scenes(self):
        with self.conn(lambda s, p: [(1, "выжимка 1"), (3, "выжимка 3")]):
            self.assertEqual(rag.story_chronicle(42), [(1, "выжимка 1"), (3, "выжимка 3")])
        sql = self.log[0][1]
        self.assertIn("row_number() OVER (ORDER BY id)", sql)
        self.assertIn("ORDER BY n", sql)

    def test_set_and_get_summary_use_the_ordinal_number(self):
        with self.conn(lambda s, p: [(7,)]):
            self.assertTrue(rag.set_summary(42, 3, "выжимка"))
        self.assertEqual(self.log[0][2], ("выжимка", 42, 42, 2))  # OFFSET number-1
        with self.conn(lambda s, p: []):
            self.assertFalse(rag.set_summary(42, 3, "выжимка"))
        created = datetime.datetime(2026, 11, 15)
        with self.conn(lambda s, p: [(created, "r", "", 5)]):
            self.assertEqual(rag.get_scene_summary(42, 1), (created, "r", None, 5))  # пустая выжимка = нет
        with self.conn(lambda s, p: []):
            self.assertIsNone(rag.get_scene_summary(42, 9))
        self.assertIsNone(rag.get_scene_summary(42, 0))


if __name__ == "__main__":
    unittest.main()
