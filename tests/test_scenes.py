"""Память сцен: чанкер, запись и поиск (rag), команды /keep /scenes /unkeep,
инъекция справки в запрос. Без сети: Voyage, Postgres и Anthropic подменены.

Тексты нейтральные («сцена N», «слово N»): настоящих сцен в репозитории нет.
SQL на реальной базе проверяется отдельно, на сервере (см. docs/ARCHITECTURE.md).

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import asyncio
import datetime
import os
import re
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1:a")
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import claude_tg_bot as bot  # noqa: E402
import rag  # noqa: E402

ALLOWED = (111, 222)
OUTSIDER = 999


def paragraph(start, words):
    """Абзац из слов «w<номер>» с уникальными номерами: потерю видно по номерам."""
    return " ".join(f"w{start + i}" for i in range(words))


def make_scene(n_paras, words_per_para):
    return [paragraph(i * 1000, words_per_para) for i in range(n_paras)]


def paras_of(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# --------------------------------------------------------------- чанкер
class ChunkerTests(unittest.TestCase):
    def test_nothing_lost_and_order_kept(self):
        paras = make_scene(40, 60)  # 2400 слов
        chunks = rag.chunk_scene("\n\n".join(paras))
        self.assertGreater(len(chunks), 1)
        rejoined = [p for c in chunks for p in paras_of(c)]
        self.assertEqual(rejoined, paras)

    def test_paragraph_is_never_split(self):
        paras = make_scene(40, 60)
        for chunk in rag.chunk_scene("\n\n".join(paras)):
            for p in paras_of(chunk):
                self.assertIn(p, paras)

    def test_chunk_size_in_range(self):
        for chunk in rag.chunk_scene("\n\n".join(make_scene(40, 60))):
            words = len(chunk.split())
            self.assertGreaterEqual(words, 250)
            self.assertLessEqual(words, rag.CHUNK_MAX_WORDS)

    def test_long_paragraph_stays_whole(self):
        big = paragraph(0, 500)
        chunks = rag.chunk_scene("\n\n".join([paragraph(9000, 50), big, paragraph(9100, 50)]))
        self.assertIn(big, chunks)  # абзац-великан ровно один чанк, хвост к нему не клеится
        self.assertEqual([p for c in chunks for p in paras_of(c)],
                         [paragraph(9000, 50), big, paragraph(9100, 50)])

    def test_short_tail_merges_into_previous_chunk(self):
        paras = make_scene(6, 60) + [paragraph(7000, 20)]  # 360 слов + хвост 20
        chunks = rag.chunk_scene("\n\n".join(paras))
        self.assertEqual([p for c in chunks for p in paras_of(c)], paras)
        self.assertTrue(all(len(c.split()) >= rag.CHUNK_TAIL_WORDS for c in chunks))

    def test_short_scene_is_one_chunk(self):
        self.assertEqual(rag.chunk_scene("слово 1\n\nслово 2"), ["слово 1\n\nслово 2"])

    def test_blank_text_gives_no_chunks(self):
        self.assertEqual(rag.chunk_scene(""), [])
        self.assertEqual(rag.chunk_scene(" \n\n  \n"), [])

    def test_crlf_and_spaced_blank_lines_are_paragraph_breaks(self):
        text = "слово 1\r\n\r\nслово 2\n  \nслово 3"
        self.assertEqual(paras_of(text), ["слово 1", "слово 2", "слово 3"])
        self.assertEqual(
            paras_of(rag.chunk_scene(text)[0]),
            ["слово 1", "слово 2", "слово 3"],
        )


# --------------------------------------------- rag: запись и поиск на фейковой БД
class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class FakeConn:
    """Записывает SQL и выдаёт строки из on_sql(sql, params)."""

    def __init__(self, log, on_sql):
        self.log = log
        self.on_sql = on_sql

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append(("execute", " ".join(sql.split()), params))
        return FakeResult(self.on_sql(" ".join(sql.split()), params))

    @contextmanager
    def cursor(self):
        log = self.log

        class Cur:
            def executemany(self, sql, rows):
                log.append(("executemany", " ".join(sql.split()), list(rows)))

        yield Cur()


class SaveSceneTests(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.scene_exists = False
        self.insert_returns_row = True

    def on_sql(self, sql, params):
        if sql.startswith("SELECT id FROM scenes"):
            return [(7,)] if self.scene_exists else []
        if sql.startswith("INSERT INTO scenes"):
            return [(7,)] if self.insert_returns_row else []
        if sql.startswith("SELECT count(*)"):
            return [(3,)]
        return []

    def embed(self, texts, input_type):
        self.log.append(("embed", input_type, list(texts)))
        return [[0.1] * 3 for _ in texts]

    def run_save(self, text, embed=None):
        with mock.patch.object(rag, "_connect", lambda: FakeConn(self.log, self.on_sql)), \
                mock.patch.object(rag, "_embed", embed or self.embed):
            return rag.save_scene(42, "запрос 1", text)

    def kinds(self):
        return [e[0] for e in self.log]

    def test_every_chunk_is_written_and_nothing_is_lost(self):
        paras = make_scene(40, 60)
        number, created = self.run_save("\n\n".join(paras))
        self.assertEqual((number, created), (3, True))
        rows = next(e[2] for e in self.log if e[0] == "executemany")
        embedded = next(e[2] for e in self.log if e[0] == "embed")
        self.assertEqual(len(rows), len(embedded))
        self.assertEqual([r[2] for r in rows], list(range(len(rows))))  # chunk_index подряд
        self.assertEqual([p for r in rows for p in paras_of(r[3])], paras)
        self.assertTrue(all(r[0] == 7 and r[1] == 42 for r in rows))
        self.assertEqual([e[1] for e in self.log if e[0] == "embed"], ["document"])

    def test_embeddings_come_before_any_write(self):
        self.run_save("\n\n".join(make_scene(10, 60)))
        first_write = next(i for i, e in enumerate(self.log) if e[0] == "execute" and e[1].startswith("INSERT"))
        self.assertLess(self.kinds().index("embed"), first_write)

    def test_embed_failure_writes_nothing(self):
        def boom(texts, input_type):
            raise RuntimeError("voyage down")

        with self.assertRaises(RuntimeError):
            self.run_save("\n\n".join(make_scene(10, 60)), embed=boom)
        self.assertFalse([e for e in self.log if e[0] == "executemany"])
        self.assertFalse([e for e in self.log if e[0] == "execute" and e[1].startswith(("INSERT", "DELETE"))])

    def test_duplicate_is_not_saved_and_not_embedded(self):
        self.scene_exists = True
        number, created = self.run_save("\n\n".join(make_scene(10, 60)))
        self.assertEqual((number, created), (3, False))
        self.assertNotIn("embed", self.kinds())
        self.assertNotIn("executemany", self.kinds())

    def test_race_duplicate_on_insert_writes_no_chunks(self):
        self.insert_returns_row = False
        self.scene_exists = False
        # SELECT id вернёт строку только после «гонки»: после неудачного INSERT
        calls = {"n": 0}
        base = self.on_sql

        def on_sql(sql, params):
            if sql.startswith("SELECT id FROM scenes"):
                calls["n"] += 1
                return [] if calls["n"] == 1 else [(7,)]
            return base(sql, params)

        self.on_sql = on_sql
        number, created = self.run_save("\n\n".join(make_scene(10, 60)))
        self.assertEqual((number, created), (3, False))
        self.assertNotIn("executemany", self.kinds())

    def test_same_text_same_hash_for_one_user(self):
        self.assertEqual(rag.scene_hash("  сцена 1\n"), rag.scene_hash("сцена 1"))
        self.assertNotEqual(rag.scene_hash("сцена 1"), rag.scene_hash("сцена 2"))

    def test_blank_scene_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_save("  \n\n ")


class SearchScenesTests(unittest.TestCase):
    def run_search(self, rows_per_query, queries=("запрос 1", "запрос 2"), top_k=3, has_chunks=True):
        log = []
        it = iter(rows_per_query)

        def on_sql(sql, params):
            if sql.startswith("SELECT 1 FROM scene_chunks"):
                return [(1,)] if has_chunks else []
            return next(it)

        embed = mock.Mock(return_value=[[0.1] * 3 for _ in queries])
        with mock.patch.object(rag, "_connect", lambda: FakeConn(log, on_sql)), \
                mock.patch.object(rag, "_embed", embed):
            out = rag.search_scenes(42, list(queries), top_k=top_k)
        return out, embed, log

    def test_no_chunks_means_empty_without_voyage_call(self):
        out, embed, _ = self.run_search([], has_chunks=False)
        self.assertEqual(out, [])
        embed.assert_not_called()

    def test_top_k_by_score_with_dedup_across_queries(self):
        q1 = [(1, "фрагмент 1", 0.9, 1), (2, "фрагмент 2", 0.5, 2), (3, "фрагмент 3", 0.4, 2)]
        q2 = [(2, "фрагмент 2", 0.8, 2), (4, "фрагмент 4", 0.7, 3), (5, "фрагмент 5", 0.1, 3)]
        out, embed, _ = self.run_search([q1, q2], top_k=3)
        self.assertEqual(out, [(1, "фрагмент 1"), (2, "фрагмент 2"), (3, "фрагмент 4")])
        self.assertEqual(embed.call_args[0][1], "query")

    def test_no_queries_or_zero_top_k(self):
        self.assertEqual(rag.search_scenes(42, [], top_k=3), [])
        self.assertEqual(rag.search_scenes(42, ["запрос 1"], top_k=0), [])


# ------------------------------------------------ команды на фейковом хранилище
class FakeStore:
    """Подмена rag.save_scene/list_scenes/delete_scene с той же семантикой, что у
    SQL: номер = позиция с единицы, дубль по тексту, удаление сдвигает номера."""

    def __init__(self):
        self.scenes = {}  # user_id -> [{"request", "text"}]
        self.fail = False

    def save_scene(self, user_id, request, text):
        if self.fail:
            raise RuntimeError("db down")
        items = self.scenes.setdefault(user_id, [])
        text = text.strip()
        for i, s in enumerate(items):
            if s["text"] == text:
                return i + 1, False
        items.append({"request": request, "text": text})
        return len(items), True

    def list_scenes(self, user_id):
        if self.fail:
            raise RuntimeError("db down")
        return [
            (datetime.datetime(2026, 11, 15, 12, 0), s["request"], len(s["text"].split()), bool(s.get("summary")))
            for s in self.scenes.get(user_id, [])
        ]

    def get_scene_summary(self, user_id, number):
        if self.fail:
            raise RuntimeError("db down")
        items = self.scenes.get(user_id, [])
        if not 1 <= number <= len(items):
            return None
        s = items[number - 1]
        return datetime.datetime(2026, 11, 15, 12, 0), s["request"], s.get("summary"), len(s["text"].split())

    def get_scene_text(self, user_id, number):
        if self.fail:
            raise RuntimeError("db down")
        items = self.scenes.get(user_id, [])
        return items[number - 1]["text"] if 1 <= number <= len(items) else None

    def set_summary(self, user_id, number, summary):
        if self.fail:
            raise RuntimeError("db down")
        items = self.scenes.get(user_id, [])
        if not 1 <= number <= len(items):
            return False
        items[number - 1]["summary"] = summary
        return True

    def story_chronicle(self, user_id):
        if self.fail:
            raise RuntimeError("db down")
        return [(i + 1, s["summary"]) for i, s in enumerate(self.scenes.get(user_id, [])) if s.get("summary")]

    def delete_scene(self, user_id, number):
        if self.fail:
            raise RuntimeError("db down")
        items = self.scenes.get(user_id, [])
        if 1 <= number <= len(items):
            items.pop(number - 1)
            return True
        return False


class CommandBase(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        bot.conversations.clear()
        bot.reply_texts.clear()
        self.client = mock.Mock()
        self.client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="выжимка сцены")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            stop_reason="end_turn",
        )
        for p in (
            mock.patch.object(bot, "ALLOWED_USER_IDS", set(ALLOWED)),
            mock.patch.object(bot, "client", self.client),
            mock.patch.object(bot, "add_usage"),
            mock.patch.object(bot, "over_daily_limit", return_value=False),
            mock.patch.object(bot.rag, "get_scene_summary", side_effect=self.store.get_scene_summary),
            mock.patch.object(bot.rag, "set_summary", side_effect=self.store.set_summary),
            mock.patch.object(bot.rag, "story_chronicle", side_effect=self.store.story_chronicle),
            mock.patch.object(bot.rag, "save_scene", side_effect=self.store.save_scene),
            mock.patch.object(bot.rag, "list_scenes", side_effect=self.store.list_scenes),
            mock.patch.object(bot.rag, "delete_scene", side_effect=self.store.delete_scene),
            mock.patch.object(bot, "alert_error", new=mock.AsyncMock()),
            mock.patch.object(bot, "alert_warn", new=mock.AsyncMock()),
            mock.patch.object(bot, "send_alert", new=mock.AsyncMock()),
        ):
            p.start()
            self.addCleanup(p.stop)

    def call(self, handler, user_id, text, reply_to=None):
        replies = []

        async def reply_text(t):
            replies.append(t)

        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=user_id),
            message=SimpleNamespace(text=text, reply_text=reply_text, reply_to_message=reply_to),
        )
        asyncio.run(handler(update, None))
        return replies

    def talk(self, user_id, pairs):
        """История диалога: [(запрос, ответ), ...]."""
        h = bot.get_history(user_id)
        for q, a in pairs:
            h.append({"role": "user", "content": q})
            h.append({"role": "assistant", "content": a})


def bot_message(mid, text, is_bot=True):
    return SimpleNamespace(message_id=mid, text=text, from_user=SimpleNamespace(is_bot=is_bot))


class KeepTests(CommandBase):
    def test_without_reply_keeps_last_assistant_message_with_its_request(self):
        self.talk(111, [("запрос 1", "сцена 1"), ("запрос 2", "сцена 2")])
        self.assertEqual(self.call(bot.cmd_keep, 111, "/keep"), ["✓ сохранена: сцена 1"])
        self.assertEqual([(s["request"], s["text"]) for s in self.store.scenes[111]], [("запрос 2", "сцена 2")])

    def test_reply_to_bot_message_keeps_that_message(self):
        self.talk(111, [("запрос 1", "сцена 1"), ("запрос 2", "сцена 2")])
        out = self.call(bot.cmd_keep, 111, "/keep", reply_to=bot_message(5, "сцена 1"))
        self.assertEqual(out, ["✓ сохранена: сцена 1"])
        self.assertEqual([(s["request"], s["text"]) for s in self.store.scenes[111]], [("запрос 1", "сцена 1")])

    def test_unknown_reply_text_uses_first_60_chars_as_request(self):
        long_text = "слово " * 40
        self.call(bot.cmd_keep, 111, "/keep", reply_to=bot_message(5, long_text))
        self.assertEqual(self.store.scenes[111][0]["request"], " ".join(long_text.split())[:60])

    def test_reply_to_non_bot_message_saves_nothing(self):
        out = self.call(bot.cmd_keep, 111, "/keep", reply_to=bot_message(5, "сцена 1", is_bot=False))
        self.assertEqual(len(out), 1)
        self.assertFalse(self.store.scenes)

    def test_nothing_to_keep(self):
        out = self.call(bot.cmd_keep, 111, "/keep")
        self.assertIn("нечего сохранять", out[0])
        self.assertFalse(self.store.scenes)

    def test_duplicate_is_reported_not_saved_twice(self):
        self.talk(111, [("запрос 1", "сцена 1")])
        self.call(bot.cmd_keep, 111, "/keep")
        out = self.call(bot.cmd_keep, 111, "/keep")
        self.assertEqual(out, ["уже сохранена: сцена 1"])
        self.assertEqual(len(self.store.scenes[111]), 1)

    def test_same_text_for_two_users_is_not_a_duplicate(self):
        self.talk(111, [("запрос 1", "сцена 1")])
        self.talk(222, [("запрос 1", "сцена 1")])
        self.call(bot.cmd_keep, 111, "/keep")
        self.assertEqual(self.call(bot.cmd_keep, 222, "/keep"), ["✓ сохранена: сцена 1"])

    def test_storage_failure_is_reported_without_crash(self):
        self.talk(111, [("запрос 1", "сцена 1")])
        self.store.fail = True
        out = self.call(bot.cmd_keep, 111, "/keep")
        self.assertIn("не удалось сохранить", out[0])

    def test_scene_command_prefix_is_stripped_from_request(self):
        self.talk(111, [(f"{bot.SCENE_PROMPT_PREFIX}встреча в порту", "сцена 1")])
        self.call(bot.cmd_keep, 111, "/keep")
        self.assertEqual(self.store.scenes[111][0]["request"], "встреча в порту")

    def test_reply_to_any_part_of_long_answer_keeps_whole_scene(self):
        full = "\n\n".join(paragraph(i * 1000, 400) for i in range(6))  # > 4096 символов
        parts = bot.split_by_paragraphs(full)
        self.assertGreater(len(parts), 2)
        bot.remember_reply_parts(111, [10 + i for i in range(len(parts))], full)
        self.talk(111, [("запрос 1", full)])
        self.call(bot.cmd_keep, 111, "/keep", reply_to=bot_message(11, parts[1]))
        self.assertEqual(self.store.scenes[111][0]["text"], full)
        self.assertEqual(self.store.scenes[111][0]["request"], "запрос 1")

    def test_reply_texts_map_is_bounded(self):
        for i in range(bot.REPLY_TEXTS_MAX + 50):
            bot.remember_reply_parts(1, [i], "x")
        self.assertEqual(len(bot.reply_texts), bot.REPLY_TEXTS_MAX)
        self.assertNotIn((1, 0), bot.reply_texts)
        self.assertIn((1, bot.REPLY_TEXTS_MAX + 49), bot.reply_texts)


class NumberingTests(CommandBase):
    def keep_three(self):
        self.talk(111, [("запрос 1", "сцена 1"), ("запрос 2", "сцена 2"), ("запрос 3", "сцена 3")])
        for i in (1, 2, 3):
            self.call(bot.cmd_keep, 111, "/keep", reply_to=bot_message(i, f"сцена {i}"))

    def listing(self, user_id=111):
        out = self.call(bot.cmd_scenes, user_id, "/scenes")
        return "\n".join(out)

    def test_scenes_are_numbered_from_one_in_save_order(self):
        self.keep_three()
        lines = self.listing().splitlines()[2:]
        self.assertEqual([l.split(".")[0] for l in lines], ["1", "2", "3"])
        for i, line in enumerate(lines, 1):
            self.assertIn(f"запрос {i}", line)
            self.assertIn("15.11.2026", line)
            self.assertIn("2 сл. · выжимка есть", line)  # /keep сохраняет и выжимку

    def test_unkeep_removes_the_scene_listed_under_that_number(self):
        self.keep_three()
        line2 = next(l for l in self.listing().splitlines() if l.startswith("2."))
        self.assertIn("запрос 2", line2)
        out = self.call(bot.cmd_unkeep, 111, "/unkeep 2")
        self.assertIn("удалена: сцена 2", out[0])
        self.assertEqual([s["text"] for s in self.store.scenes[111]], ["сцена 1", "сцена 3"])
        after = self.listing().splitlines()[2:]
        self.assertTrue(after[1].startswith("2.") and "запрос 3" in after[1])  # номера сдвинулись

    def test_unkeep_bad_input(self):
        self.keep_three()
        for arg in ("", "abc", "0", "4", "-1"):
            out = self.call(bot.cmd_unkeep, 111, f"/unkeep {arg}".strip())
            self.assertEqual(len(out), 1)
        self.assertEqual(len(self.store.scenes[111]), 3)

    def test_empty_list(self):
        self.assertIn("пусто", self.call(bot.cmd_scenes, 111, "/scenes")[0])

    def test_request_is_cut_to_60_chars_on_one_line(self):
        self.store.scenes[111] = [{"request": "слово\n" * 40, "text": "сцена 1"}]
        line = self.listing().splitlines()[2]
        req = line.split(" · ")[1]
        self.assertLessEqual(len(req), 60)
        self.assertNotIn("\n", req)

    def test_listing_splits_at_line_boundaries_without_loss(self):
        rows = [(datetime.datetime(2026, 11, 15), f"запрос {i}", 100 + i, False) for i in range(1, 201)]
        chunks = bot.scenes_messages(rows, limit=500)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 500 + len("🎬 сцены:\n\n") for c in chunks))
        lines = [l for c in chunks for l in c.splitlines() if re.match(r"^\d+\. ", l)]
        self.assertEqual([int(l.split(".")[0]) for l in lines], list(range(1, 201)))
        self.assertTrue(all(l.endswith(" сл. · без выжимки") for l in lines))  # ни одна строка не порвана

    def test_storage_failure_gives_message_not_crash(self):
        self.store.fail = True
        self.assertIn("недоступна", self.call(bot.cmd_scenes, 111, "/scenes")[0])
        self.assertIn("недоступна", self.call(bot.cmd_unkeep, 111, "/unkeep 1")[0])


class AccessTests(CommandBase):
    def test_outsider_gets_silence_and_touches_no_storage(self):
        self.talk(OUTSIDER, [("запрос 1", "сцена 1")])
        for handler, text in (
            (bot.cmd_keep, "/keep"),
            (bot.cmd_scenes, "/scenes"),
            (bot.cmd_unkeep, "/unkeep 1"),
        ):
            self.assertEqual(self.call(handler, OUTSIDER, text), [], text)
        bot.rag.save_scene.assert_not_called()
        bot.rag.list_scenes.assert_not_called()
        bot.rag.delete_scene.assert_not_called()

    def test_users_do_not_see_each_others_scenes(self):
        self.talk(111, [("запрос 1", "сцена 1")])
        self.call(bot.cmd_keep, 111, "/keep")
        self.assertIn("пусто", self.call(bot.cmd_scenes, 222, "/scenes")[0])
        self.assertEqual(self.call(bot.cmd_unkeep, 222, "/unkeep 1")[0][:8], "сцены 1 ")


# ------------------------------------------------------ инъекция в запрос
HIST = [
    {"role": "user", "content": "запрос 1"},
    {"role": "assistant", "content": "ответ 1"},
    {"role": "user", "content": "запрос 2"},
]
FRAGMENTS = [(2, "фрагмент сцены 2"), (5, "фрагмент сцены 5")]
CANON = bot.build_canon_block(["факт 1", "факт 2"])


class BuildMessagesTests(unittest.TestCase):
    def test_empty_scenes_do_not_change_the_request(self):
        self.assertEqual(
            bot.build_messages(HIST, CANON, bot.build_scenes_block([])),
            bot.build_messages(HIST, CANON),
        )
        self.assertNotIn("memory_scenes", str(bot.build_messages(HIST, CANON)))

    def test_block_text_and_labels(self):
        block = bot.build_scenes_block(FRAGMENTS)
        self.assertTrue(block.startswith("<memory_scenes>\n"))
        self.assertTrue(block.endswith("</memory_scenes>"))
        self.assertIn("Это не канон. Если они противоречат <canon>, прав канон.", block)
        self.assertIn("не пересказывай и не копируй дословно", block)
        self.assertIn("[сцена 2]\nфрагмент сцены 2", block)
        self.assertIn("[сцена 5]\nфрагмент сцены 5", block)

    def test_block_stands_after_canon_in_the_tail_of_last_user_message(self):
        msgs = bot.build_messages(HIST, CANON, bot.build_scenes_block(FRAGMENTS))
        last = msgs[-1]["content"]
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertTrue(last.startswith("запрос 2\n\n<canon>"))
        self.assertLess(last.index("</canon>"), last.index("<memory_scenes>"))
        self.assertTrue(last.endswith("</memory_scenes>"))

    def test_block_without_canon_still_in_the_tail(self):
        msgs = bot.build_messages(HIST, "", bot.build_scenes_block(FRAGMENTS))
        self.assertTrue(msgs[-1]["content"].startswith("запрос 2\n\n<memory_scenes>"))

    def test_history_and_earlier_messages_are_untouched(self):
        before = [dict(m) for m in HIST]
        with_scenes = bot.build_messages(HIST, CANON, bot.build_scenes_block(FRAGMENTS))
        without = bot.build_messages(HIST, CANON)
        self.assertEqual(HIST, before)  # история в памяти чистая
        self.assertEqual(with_scenes[:-1], without[:-1])  # префикс кеша тот же


class FakeStream:
    def __init__(self, final):
        self.final = final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.final


class GenerateTests(unittest.TestCase):
    def setUp(self):
        bot.conversations.clear()
        bot.reply_texts.clear()
        self.history = [dict(m) for m in HIST]
        final = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ответ 2")],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
            stop_reason="end_turn",
        )
        self.client = mock.Mock()
        self.client.messages.stream.side_effect = lambda **kw: FakeStream(final)
        self.search = mock.Mock(return_value=list(FRAGMENTS))
        for p in (
            mock.patch.object(bot, "client", self.client),
            mock.patch.object(bot, "get_relevant_facts", return_value=(["факт 1", "факт 2"], "all")),
            mock.patch.object(bot.rag, "search_scenes", self.search),
            mock.patch.object(bot, "add_usage"),
            mock.patch.object(bot, "load_prompt", return_value="системный промпт"),
            mock.patch.object(bot, "alert_error", new=mock.AsyncMock()),
            mock.patch.object(bot, "alert_warn", new=mock.AsyncMock()),
            mock.patch.object(bot, "send_alert", new=mock.AsyncMock()),
            mock.patch.object(bot, "SCENE_TOP_K", 3),
            mock.patch.object(bot, "scenes_enabled", return_value=True),  # по умолчанию память сцен выключена
            mock.patch.object(bot.rag, "story_chronicle", return_value=[]),
        ):
            p.start()
            self.addCleanup(p.stop)

    def generate(self):
        replies = []

        async def reply_text(t):
            replies.append(t)
            return SimpleNamespace(message_id=100 + len(replies))

        async def send_action(a):
            return None

        update = SimpleNamespace(
            effective_chat=SimpleNamespace(send_action=send_action),
            message=SimpleNamespace(reply_text=reply_text),
        )
        with self.assertLogs(bot.logger, level="INFO") as cm:
            asyncio.run(bot._generate_and_reply(update, 1, 111, self.history, model_override="m"))
        return self.client.messages.stream.call_args.kwargs, cm.output, replies

    def test_block_goes_to_the_tail_after_canon_never_to_system(self):
        kwargs, _, _ = self.generate()
        last = kwargs["messages"][-1]["content"]
        self.assertLess(last.index("<canon>"), last.index("<memory_scenes>"))
        self.assertTrue(last.endswith("</memory_scenes>"))
        self.assertIn("[сцена 2]", last)
        self.assertNotIn("memory_scenes", str(kwargs["system"]))
        for m in kwargs["messages"][:-1]:
            self.assertNotIn("memory_scenes", str(m))

    def test_search_uses_the_same_queries_as_facts_and_top_k(self):
        self.generate()
        args, kw = self.search.call_args
        self.assertEqual(args[0], 111)
        self.assertEqual(args[1], bot.retrieval_queries(HIST))
        self.assertEqual(kw["top_k"], 3)

    def test_done_log_reports_scene_count(self):
        _, log, _ = self.generate()
        done = next(l for l in log if "done:" in l)
        self.assertIn(" scenes=2 ", done + " ")

    def test_no_scenes_request_is_exactly_as_before(self):
        self.search.return_value = []
        kwargs, log, _ = self.generate()
        self.assertEqual(kwargs["messages"], bot.build_messages(HIST, CANON))
        self.assertNotIn("memory_scenes", str(kwargs))
        self.assertIn(" scenes=0 ", next(l for l in log if "done:" in l) + " ")

    def test_search_failure_sends_request_without_block_and_warns(self):
        self.search.side_effect = RuntimeError("voyage down")
        kwargs, log, replies = self.generate()
        self.assertEqual(kwargs["messages"], bot.build_messages(HIST, CANON))
        self.assertNotIn("memory_scenes", str(kwargs))
        self.assertEqual(replies, ["ответ 2"])  # бот ответил как обычно
        self.assertTrue(any("WARNING" in l and "память сцен" in l for l in log), log)
        self.assertIn(" scenes=0 ", next(l for l in log if "done:" in l) + " ")

    def test_scene_text_never_reaches_the_log(self):
        _, log, _ = self.generate()
        self.assertNotIn("фрагмент сцены", "\n".join(log))

    def test_top_k_zero_disables_the_lookup(self):
        with mock.patch.object(bot, "SCENE_TOP_K", 0):
            kwargs, _, _ = self.generate()
        self.search.assert_not_called()
        self.assertNotIn("memory_scenes", str(kwargs))

    def test_reply_parts_are_remembered_for_keep(self):
        full = "\n\n".join(paragraph(i * 1000, 400) for i in range(6))
        final = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=full)],
            usage=SimpleNamespace(
                input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
            stop_reason="end_turn",
        )
        self.client.messages.stream.side_effect = lambda **kw: FakeStream(final)
        _, _, replies = self.generate()
        self.assertGreater(len(replies), 1)
        for i in range(len(replies)):
            self.assertEqual(bot.reply_texts[(1, 101 + i)], full)


if __name__ == "__main__":
    unittest.main()
