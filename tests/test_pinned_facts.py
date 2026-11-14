"""Тесты закреплённых фактов (маркер 📌). Без сети: Voyage и Postgres подменены,
LLM не вызывается.

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import asyncio
import json
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

import claude_tg_bot as bot  # noqa: E402

PIN = "\U0001F4CC"  # 📌

# Индексы 0..9. Закреплены 0, 3 и 7 (3 с variation selector, 7 с пробелами вокруг
# маркера). Тексты нарочно обезличены: настоящих фактов в репозитории нет.
RAW = [
    f"{PIN} факт 0",
    "факт 1",
    "факт 2",
    f"{PIN}\ufe0f факт 3",
    "факт 4",
    "факт 5",
    "факт 6",
    f"  {PIN}  факт 7",
    "факт 8",
    "факт 9",
]
CLEAN = [f"факт {i}" for i in range(10)]
PINNED_IDX = {0, 3, 7}
PINNED_CLEAN = [CLEAN[0], CLEAN[3], CLEAN[7]]


def hist(text):
    return [{"role": "user", "content": text}]


class FakeSearch:
    """Подмена rag.search_facts_multi: «релевантные» факты зависят от запроса,
    exclude и limit соблюдаются так же, как в SQL."""

    def __init__(self, texts=CLEAN):
        self.texts = texts
        self.calls = []

    def __call__(self, user_id, queries, top_k=30, limit=50, exclude=None):
        self.calls.append(
            {"queries": list(queries), "top_k": top_k, "limit": limit, "exclude": set(exclude or ())}
        )
        idx = [i for i in range(len(self.texts)) if i not in (exclude or ())]
        if not idx:
            return []
        shift = len(queries[0]) % len(idx)
        ranked = idx[shift:] + idx[:shift]  # «ранжирование» зависит от запроса
        top = sorted(ranked[:limit])
        return [self.texts[i] for i in top]


class Base(unittest.TestCase):
    def setUp(self):
        self.search = FakeSearch()
        patches = [
            mock.patch.object(bot, "load_facts", return_value=list(RAW)),
            mock.patch.object(bot.rag, "search_facts_multi", side_effect=self.search),
            mock.patch.object(bot, "FACTS_FULL_LIMIT", 0),  # поиск включён всегда
            mock.patch.object(bot, "RAG_TOP_K", 4),
            mock.patch.object(bot, "RAG_MAX_FACTS", 4),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class TestHelpers(unittest.TestCase):
    def test_is_pinned_and_strip(self):
        cases = {
            f"{PIN} факт": (True, "факт"),
            f"{PIN}️ факт": (True, "факт"),
            f"  {PIN}факт": (True, "факт"),
            f"{PIN}": (True, ""),
            "обычный факт": (False, "обычный факт"),
            f"факт {PIN} внутри": (False, f"факт {PIN} внутри"),  # не в начале: не маркер
        }
        for raw, (pinned, clean) in cases.items():
            self.assertEqual(bot.is_pinned(raw), pinned, raw)
            self.assertEqual(bot.strip_pin(raw), clean, raw)


class TestBelowLimit(Base):
    def test_all_facts_returned_without_marker(self):
        with mock.patch.object(bot, "FACTS_FULL_LIMIT", 150):
            facts, source = bot.get_relevant_facts(1, hist("сцена: ссора"))
        self.assertEqual(facts, CLEAN)  # все, в авторском порядке
        self.assertEqual(source, "all (<= 150)")
        self.assertEqual(self.search.calls, [])  # поиск не вызывался
        self.assertFalse(any(PIN in f for f in facts))

    def test_limit_boundary_is_inclusive(self):
        with mock.patch.object(bot, "FACTS_FULL_LIMIT", len(RAW)):
            facts, source = bot.get_relevant_facts(1, hist("x"))
        self.assertEqual(len(facts), len(RAW))
        with mock.patch.object(bot, "FACTS_FULL_LIMIT", len(RAW) - 1):
            facts, source = bot.get_relevant_facts(1, hist("x"))
        self.assertTrue(source.startswith("semantic"))


class TestAboveLimit(Base):
    def test_pinned_present_for_any_query(self):
        for query in ("сцена: ссора", "ещё одну", "а теперь в баре, под дождём", "x"):
            facts, source = bot.get_relevant_facts(1, hist(query))
            self.assertEqual(facts[:3], PINNED_CLEAN, query)  # первыми, в авторском порядке
            self.assertEqual(len(facts), len(set(facts)), f"дубли при {query!r}")
            self.assertFalse(any(PIN in f for f in facts), f"маркер в выдаче при {query!r}")
            self.assertEqual(source, "semantic (1 queries) + 3 pinned")

    def test_pinned_do_not_count_in_rag_limit(self):
        facts, _ = bot.get_relevant_facts(1, hist("сцена: ссора"))
        call = self.search.calls[-1]
        self.assertEqual(call["exclude"], PINNED_IDX)  # закреплённые исключены из поиска
        self.assertEqual(call["limit"], 4)  # RAG_MAX_FACTS без вычета закреплённых
        self.assertEqual(len(facts), 3 + 4)  # 3 закреплённых + ровно RAG_MAX_FACTS

    def test_rag_part_in_authorial_order_after_pinned(self):
        facts, _ = bot.get_relevant_facts(1, hist("сцена: ссора"))
        rag_part = facts[3:]
        positions = [CLEAN.index(f) for f in rag_part]
        self.assertEqual(positions, sorted(positions))

    def test_no_pinned_keeps_plain_source(self):
        plain = [f for f in RAW if not bot.is_pinned(f)]
        with mock.patch.object(bot, "load_facts", return_value=plain):
            facts, source = bot.get_relevant_facts(1, hist("сцена: ссора"))
        self.assertEqual(source, "semantic (1 queries)")  # формат как до закреплённых
        self.assertEqual(self.search.calls[-1]["exclude"], set())

    def test_two_queries_in_source(self):
        history = [
            {"role": "user", "content": "первое"},
            {"role": "assistant", "content": "ответ"},
            {"role": "user", "content": "ещё"},
        ]
        _, source = bot.get_relevant_facts(1, history)
        self.assertEqual(source, "semantic (2 queries) + 3 pinned")

    def test_duplicates_with_stale_index_are_removed(self):
        # Индекс устарел после ручной правки файла: поиск вернул закреплённый факт
        # (с маркером и без) и обычный. Дублей и маркера в выдаче быть не должно.
        stale = [f"{PIN} факт 0", "факт 7", "факт 2"]
        with mock.patch.object(bot.rag, "search_facts_multi", return_value=stale):
            facts, _ = bot.get_relevant_facts(1, hist("x"))
        self.assertEqual(facts, PINNED_CLEAN + ["факт 2"])

    def _with_facts(self, raw):
        """load_facts и «индекс» поиска согласованы между собой."""
        search = FakeSearch([bot.strip_pin(f) for f in raw])
        return (
            mock.patch.object(bot, "load_facts", return_value=raw),
            mock.patch.object(bot.rag, "search_facts_multi", side_effect=search),
        )

    def test_warning_when_more_than_40_pinned(self):
        many = [f"{PIN} закреплённый {i}" for i in range(41)] + ["обычный"]
        p1, p2 = self._with_facts(many)
        with p1, p2, self.assertLogs(bot.logger, level="WARNING") as cm:
            facts, source = bot.get_relevant_facts(1, hist("x"))
        self.assertEqual(len(cm.output), 1, cm.output)  # только предупреждение про 41
        self.assertIn("закреплённых фактов 41", cm.output[0])
        self.assertEqual(source, "semantic (1 queries) + 41 pinned")  # поиск отработал, не fallback
        self.assertEqual(len(facts), 42)  # все 41 закреплённых + 1 из поиска

    def test_no_warning_at_40(self):
        forty = [f"{PIN} закреплённый {i}" for i in range(40)] + ["обычный"]
        p1, p2 = self._with_facts(forty)
        with p1, p2, mock.patch.object(bot.logger, "warning") as warn:
            facts, source = bot.get_relevant_facts(1, hist("x"))
        warn.assert_not_called()
        self.assertEqual(source, "semantic (1 queries) + 40 pinned")
        self.assertEqual(len(facts), 41)

    def test_all_pinned_skips_search(self):
        only_pinned = [f"{PIN} a", f"{PIN} b"]
        with mock.patch.object(bot, "load_facts", return_value=only_pinned):
            facts, source = bot.get_relevant_facts(1, hist("x"))
        self.assertEqual(facts, ["a", "b"])
        self.assertEqual(source, "all pinned")
        self.assertEqual(self.search.calls, [])

    def test_db_error_falls_back_to_all_without_marker(self):
        with mock.patch.object(bot.rag, "search_facts_multi", side_effect=RuntimeError("db down")):
            facts, source = bot.get_relevant_facts(1, hist("x"))
        self.assertEqual(facts, CLEAN)
        self.assertIn("vector DB unavailable", source)  # от этой строки зависит алерт

    def test_empty_index_backfills_and_returns_all(self):
        with mock.patch.object(bot.rag, "search_facts_multi", return_value=[]), mock.patch.object(
            bot, "reindex_user_facts"
        ) as reindex:
            facts, source = bot.get_relevant_facts(1, hist("x"))
        reindex.assert_called_once_with(1)
        self.assertEqual(facts, CLEAN)
        self.assertIn("index empty", source)


class TestCommandsAndIndexing(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "facts").mkdir()
        (self.tmp / "backup").mkdir()
        for p in (
            mock.patch.object(bot, "FACTS_DIR", self.tmp / "facts"),
            mock.patch.object(bot, "FACTS_BACKUP_DIR", self.tmp / "backup"),
            mock.patch.object(bot, "ALLOWED_USER_IDS", set()),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _remember(self, text, user_id=555):
        replies = []

        async def reply_text(t):
            replies.append(t)

        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
            message=SimpleNamespace(text=f"/remember {text}", reply_text=reply_text),
        )
        asyncio.run(bot.cmd_remember(update, None))
        return replies

    def test_remember_keeps_marker_in_file_and_indexes_clean_text(self):
        (self.tmp / "facts" / "555.json").write_text(
            json.dumps(["старый факт"], ensure_ascii=False), encoding="utf-8"
        )
        with mock.patch.object(bot.rag, "index_facts") as index:
            replies = self._remember(f"{PIN} закреплённый факт")
        saved = json.loads((self.tmp / "facts" / "555.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, ["старый факт", f"{PIN} закреплённый факт"])  # маркер сохранён
        backup = json.loads((self.tmp / "backup" / "555.json").read_text(encoding="utf-8"))
        self.assertEqual(backup, saved)
        index.assert_called_once()  # факт проиндексирован
        user_id, texts = index.call_args.args
        self.assertEqual(user_id, 555)
        self.assertEqual(texts, ["старый факт", "закреплённый факт"])  # без маркера
        self.assertTrue(replies and "запомнила (2)" in replies[0])

    def test_plain_remember_unchanged(self):
        with mock.patch.object(bot.rag, "index_facts") as index:
            self._remember("обычный факт")
        saved = json.loads((self.tmp / "facts" / "555.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, ["обычный факт"])
        self.assertEqual(index.call_args.args[1], ["обычный факт"])

    def test_restore_from_backup_indexes_without_marker(self):
        (self.tmp / "backup" / "777.json").write_text(
            json.dumps([f"{PIN} закреплённый", "обычный"], ensure_ascii=False), encoding="utf-8"
        )
        with mock.patch.object(bot.rag, "index_facts") as index:
            bot.restore_facts_from_backups()
        restored = json.loads((self.tmp / "facts" / "777.json").read_text(encoding="utf-8"))
        self.assertEqual(restored, [f"{PIN} закреплённый", "обычный"])  # файл с маркером
        self.assertEqual(index.call_args.args, (777, ["закреплённый", "обычный"]))

    def test_facts_command_shows_marker(self):
        # /facts показывает факты как хранятся, чтобы было видно, какие закреплены.
        (self.tmp / "facts" / "555.json").write_text(
            json.dumps([f"{PIN} закреплённый факт", "обычный"], ensure_ascii=False), encoding="utf-8"
        )
        sent = []

        async def reply_text(t):
            sent.append(t)

        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=555),
            message=SimpleNamespace(text="/facts", reply_text=reply_text),
        )
        asyncio.run(bot.cmd_facts(update, None))
        self.assertIn(f"1. {PIN} закреплённый факт", sent[0])


if __name__ == "__main__":
    unittest.main()
