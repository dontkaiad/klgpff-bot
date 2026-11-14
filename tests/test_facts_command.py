"""/facts: нумерация как у /forget, нарезка на сообщения, доступ. На моках, без сети.

Тексты фактов нейтральные ("факт N"), реальный канон в тесты не попадает.
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

import claude_tg_bot as bot  # noqa: E402

PIN = "\U0001F4CC"
ALLOWED = (111, 222)


class FactsCommandBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "facts").mkdir()
        (self.tmp / "backup").mkdir()
        for p in (
            mock.patch.object(bot, "FACTS_DIR", self.tmp / "facts"),
            mock.patch.object(bot, "FACTS_BACKUP_DIR", self.tmp / "backup"),
            mock.patch.object(bot, "ALLOWED_USER_IDS", set(ALLOWED)),
            mock.patch.object(bot.rag, "index_facts"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def write_facts(self, user_id, facts):
        (self.tmp / "facts" / f"{user_id}.json").write_text(
            json.dumps(facts, ensure_ascii=False), encoding="utf-8"
        )

    def call(self, handler, user_id, text):
        replies = []

        async def reply_text(t):
            replies.append(t)

        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=user_id),
            message=SimpleNamespace(text=text, reply_text=reply_text),
        )
        asyncio.run(handler(update, None))
        return replies


class NumberingTests(FactsCommandBase):
    FACTS = ["факт 1", f"{PIN} закреплённый факт 2", "факт 3", f"{PIN}️ закреплённый факт 4", "факт 5"]

    def test_numbers_match_forget(self):
        # Каждый номер из /facts должен удалять в /forget именно этот факт.
        self.write_facts(111, self.FACTS)
        listing = "\n\n".join(self.call(bot.cmd_facts, 111, "/facts"))
        pairs = re.findall(r"^(\d+)\. (.*)$", listing, re.M)
        self.assertEqual(len(pairs), len(self.FACTS))
        for num, text in pairs:
            self.write_facts(111, self.FACTS)
            out = self.call(bot.cmd_forget, 111, f"/forget {num}")
            self.assertEqual(out, [f"удалён: {text}"], num)

    def test_numbering_is_file_position_from_one(self):
        self.write_facts(111, self.FACTS)
        first = self.call(bot.cmd_facts, 111, "/facts")[0]
        for i, fact in enumerate(self.FACTS, start=1):
            self.assertIn(f"{i}. {fact}", first)

    def test_pinned_shown_as_is_and_nothing_added_without_pins(self):
        self.write_facts(111, ["факт 1", "факт 2"])
        out = "\n".join(self.call(bot.cmd_facts, 111, "/facts"))
        self.assertEqual(out, "🧠 факты:\n\n1. факт 1\n\n2. факт 2")
        self.assertNotIn(PIN, out)
        self.write_facts(111, [f"{PIN} факт 1", "факт 2"])
        out = "\n".join(self.call(bot.cmd_facts, 111, "/facts"))
        self.assertIn(f"1. {PIN} факт 1", out)

    def test_empty_list(self):
        self.assertEqual(self.call(bot.cmd_facts, 111, "/facts"), ["пусто. добавь через /remember <факт>"])


class SplitTests(FactsCommandBase):
    def test_long_list_split_on_fact_boundaries_without_loss(self):
        facts = [f"факт {i} " + "я" * 300 for i in range(60)]  # около 20 000 символов
        self.write_facts(111, facts)
        chunks = self.call(bot.cmd_facts, 111, "/facts")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 4096)
            self.assertLessEqual(len(c), bot.FACTS_MSG_LIMIT + len("🧠 факты:\n\n"))
        whole = "\n\n".join(chunks)
        for i, fact in enumerate(facts, start=1):
            # факт целиком и ровно в одном сообщении
            self.assertEqual(sum(f"{i}. {fact}" in c for c in chunks), 1, i)
            self.assertIn(f"{i}. {fact}", whole)
        nums = [int(n) for n in re.findall(r"^(\d+)\. ", whole, re.M)]
        self.assertEqual(nums, list(range(1, 61)))

    def test_fact_longer_than_limit_is_not_lost(self):
        huge = "я" * 9000
        facts = ["факт 1", huge, "факт 3"]
        self.write_facts(111, facts)
        chunks = self.call(bot.cmd_facts, 111, "/facts")
        for c in chunks:
            self.assertLessEqual(len(c), 4096)
        whole = "".join(chunks)
        self.assertEqual(whole.count("я"), 9000)
        self.assertIn("1. факт 1", whole)
        self.assertIn("3. факт 3", whole)
        self.assertEqual(len(re.findall(r"^3\. ", "\n".join(chunks), re.M)), 1)


class AccessTests(FactsCommandBase):
    def test_both_allowed_ids_get_their_list(self):
        for uid in ALLOWED:
            self.write_facts(uid, [f"факт {uid}"])
            self.assertIn(f"1. факт {uid}", self.call(bot.cmd_facts, uid, "/facts")[0])

    def test_foreign_id_gets_nothing_and_facts_are_not_read(self):
        self.write_facts(999, ["факт чужого"])
        self.write_facts(111, ["факт хозяина"])
        with mock.patch.object(bot, "load_facts") as load:
            out = self.call(bot.cmd_facts, 999, "/facts")
        self.assertEqual(out, [])
        load.assert_not_called()

    def test_user_sees_only_own_facts(self):
        self.write_facts(111, ["факт хозяина"])
        self.write_facts(222, ["факт второго"])
        out = "\n".join(self.call(bot.cmd_facts, 222, "/facts"))
        self.assertIn("факт второго", out)
        self.assertNotIn("факт хозяина", out)


if __name__ == "__main__":
    unittest.main()
