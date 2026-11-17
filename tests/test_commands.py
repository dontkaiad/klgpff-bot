"""Набор команд и лог команд. На моках, без сети и без API.

Запуск: TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import asyncio
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1:a")
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import claude_tg_bot as bot  # noqa: E402

KEPT = {"start", "help", "new", "model", "regenerate", "scene", "addprompt", "cost", "remember", "forget", "facts", "keep", "scenes", "unkeep", "import", "done", "cancel", "prompt", "go", "accept", "context", "extract"}
REMOVED = {"system", "summarize", "setsystem"}


class CommandSetTests(unittest.TestCase):
    def test_removed_handlers_are_gone(self):
        for name in REMOVED:
            self.assertFalse(hasattr(bot, f"cmd_{name}"), name)

    def test_kept_handlers_exist(self):
        for name in KEPT:
            self.assertTrue(callable(getattr(bot, f"cmd_{name}", None)), name)

    def test_help_lists_only_kept_commands(self):
        listed = set(re.findall(r"^/(\w+)", bot.HELP_TEXT.replace("💸 ", "").replace("ℹ️ ", ""), re.M))
        self.assertTrue(listed.isdisjoint(REMOVED), listed & REMOVED)
        self.assertTrue(listed <= KEPT, listed - KEPT)

    def test_menu_lists_only_kept_commands(self):
        menu = {c for c, _ in bot.BOT_COMMANDS}
        self.assertTrue(menu.isdisjoint(REMOVED))
        self.assertTrue(menu <= KEPT)


class HandlerRegistrationTests(unittest.TestCase):
    """CommandHandler("x", cmd_x) в main(): имя команды, имя функции и меню сходятся."""

    def setUp(self):
        src = (Path(__file__).resolve().parent.parent / "claude_tg_bot.py").read_text(encoding="utf-8")
        self.pairs = re.findall(r'CommandHandler\("(\w+)", (\w+)\)', src)

    def test_every_handler_name_exists_and_matches_command(self):
        self.assertTrue(self.pairs)
        for cmd, func in self.pairs:
            self.assertEqual(func, f"cmd_{cmd}")
            self.assertTrue(callable(getattr(bot, func, None)), func)

    def test_registered_commands_are_exactly_the_kept_set(self):
        self.assertEqual({c for c, _ in self.pairs}, KEPT)

    def test_menu_and_help_cover_every_command_except_start(self):
        menu = {c for c, _ in bot.BOT_COMMANDS}
        self.assertEqual(menu, KEPT - {"start"})
        listed = set(re.findall(r"^/(\w+)", bot.HELP_TEXT.replace("💸 ", "").replace("ℹ️ ", ""), re.M))
        self.assertEqual(listed, KEPT - {"start"})


class LogCommandTests(unittest.TestCase):
    def _run(self, text, user_id=42):
        update = SimpleNamespace(
            effective_message=SimpleNamespace(text=text),
            effective_user=SimpleNamespace(id=user_id),
        )
        with self.assertLogs(bot.logger, level="INFO") as cm:
            asyncio.run(bot.log_command(update, None))
        return cm.output

    def test_logs_name_and_user_only(self):
        out = self._run("/scene секретное описание сцены", user_id=42)
        self.assertEqual(len(out), 1)
        self.assertIn("command /scene user=42", out[0])
        self.assertNotIn("секретное", out[0])

    def test_strips_bot_suffix_and_case(self):
        self.assertIn("command /model user=7", self._run("/Model@some_bot auto", user_id=7)[0])

    def test_unknown_command_is_logged(self):
        self.assertIn("command /setsystem user=1", self._run("/setsystem", user_id=1)[0])

    def test_no_text_is_ignored(self):
        update = SimpleNamespace(effective_message=SimpleNamespace(text=None), effective_user=None)
        with mock.patch.object(bot.logger, "info") as info:
            asyncio.run(bot.log_command(update, None))
        info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
