"""Общая подпорка для тестов /import, /context, извлечения и /accept: база с
замоканными Anthropic, Voyage и Postgres. Тестов здесь нет.

Тексты нейтральные («часть N», «факт N»): настоящих сцен и фактов в репозитории нет.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1:a")
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_tg_bot as bot  # noqa: E402
import extract  # noqa: E402
from test_import_limit import Base, at, origin_from  # noqa: E402
from test_scenes import FakeStream  # noqa: E402

__all__ = ["Base2", "FakeCandStore", "at", "bot", "extract", "extraction_json", "origin_from", "words"]


class FakeCandStore:
    """Подмена rag.*_candidates с той же семантикой, что у SQL: номер это счётчик
    юзера (max+1), claim берёт только pending, release возвращает."""

    def __init__(self):
        self.rows = {}  # user_id -> [{"num", "text", "status"}]
        self.fail = False
        self.drop_similar = set()  # тексты, которые filter_similar считает дублями
        self.similar_fails = False

    def _rows(self, uid):
        if self.fail:
            raise RuntimeError("db down")
        return self.rows.setdefault(uid, [])

    def add_candidates(self, uid, texts):
        rows = self._rows(uid)
        start = max([r["num"] for r in rows], default=0)
        out = []
        for i, t in enumerate(texts):
            rows.append({"num": start + i + 1, "text": t, "status": "pending"})
            out.append((start + i + 1, t))
        return out

    def pending_candidates(self, uid):
        return [(r["num"], r["text"]) for r in self._rows(uid) if r["status"] == "pending"]

    def claim_candidates(self, uid, nums=None):
        rows = self._rows(uid)
        pending = {r["num"] for r in rows if r["status"] == "pending"}
        accepted = {r["num"] for r in rows if r["status"] == "accepted"}
        if nums is None:
            target, already, missing = sorted(pending), [], []
        else:
            asked = set(nums)
            target = sorted(asked & pending)
            already = sorted(asked & accepted)
            missing = sorted(asked - pending - accepted)
        claimed = []
        for r in rows:
            if r["num"] in target and r["status"] == "pending":
                r["status"] = "accepted"
                claimed.append((r["num"], r["text"]))
        return sorted(claimed), already, missing

    def release_candidates(self, uid, nums):
        for r in self._rows(uid):
            if r["num"] in nums and r["status"] == "accepted":
                r["status"] = "pending"

    def filter_similar(self, uid, candidates, pending_texts, threshold):
        if self.similar_fails:
            raise RuntimeError("voyage down")
        return [c for c in candidates if c not in self.drop_similar]

    def statuses(self, uid):
        return {r["num"]: r["status"] for r in self.rows.get(uid, [])}


def extraction_json(new_facts, conflicts=()):
    return json.dumps({"new_facts": list(new_facts), "conflicts": list(conflicts)}, ensure_ascii=False)


def words(n, start=0):
    return " ".join(f"w{start + i}" for i in range(n))


class Base2(Base):
    """Base плюс: файловое хранилище в tmp, фейковые сцены и кандидаты, ответы
    клиента по типу вызова (извлечение отличается системным промптом), режим opus."""

    def setUp(self):
        super().setUp()
        self.cands = FakeCandStore()
        self.facts_dir = self.tmp / "facts"
        self.backup_dir = self.tmp / "backup"
        self.models_dir = self.tmp / "models"
        for d in (self.facts_dir, self.backup_dir, self.models_dir):
            d.mkdir()
        self.stream_text = "ответ"
        self.extract_reply = extraction_json([])
        self.extract_stop = "end_turn"  # "max_tokens" имитирует обрезанный ответ
        self.index_facts = mock.Mock()
        for p in (
            mock.patch.object(bot, "FACTS_DIR", self.facts_dir),
            mock.patch.object(bot, "FACTS_BACKUP_DIR", self.backup_dir),
            mock.patch.object(bot, "MODELS_DIR", self.models_dir),
            mock.patch.object(bot, "load_mode", return_value="opus"),
            mock.patch.object(bot, "get_relevant_facts", side_effect=lambda uid, h: (bot.load_facts(uid), "all")),
            mock.patch.object(bot.rag, "index_facts", self.index_facts),
            mock.patch.object(bot.rag, "save_scene", side_effect=self.store.save_scene),
            mock.patch.object(bot.rag, "list_scenes", side_effect=self.store.list_scenes),
            mock.patch.object(bot.rag, "add_candidates", side_effect=self.cands.add_candidates),
            mock.patch.object(bot.rag, "pending_candidates", side_effect=self.cands.pending_candidates),
            mock.patch.object(bot.rag, "claim_candidates", side_effect=self.cands.claim_candidates),
            mock.patch.object(bot.rag, "release_candidates", side_effect=self.cands.release_candidates),
            mock.patch.object(bot.rag, "filter_similar", side_effect=self.cands.filter_similar),
            mock.patch.object(bot, "EXTRACT_ENABLED", True),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.usage = SimpleNamespace(
            input_tokens=1000, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        self.client.messages.stream.side_effect = lambda **kw: FakeStream(SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.stream_text)], usage=self.usage, stop_reason="end_turn"))
        self.client.messages.create.side_effect = self._create

    def _create(self, **kw):
        text = self.extract_reply if kw.get("system") == extract.EXTRACT_SYSTEM else "opus"
        stop = self.extract_stop if kw.get("system") == extract.EXTRACT_SYSTEM else "end_turn"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=self.usage, stop_reason=stop)

    # --- помощники
    def extraction_calls(self):
        return [c for c in self.client.messages.create.call_args_list if c.kwargs.get("system") == extract.EXTRACT_SYSTEM]

    def write_facts(self, user_id, facts):
        (self.facts_dir / f"{user_id}.json").write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")

    def read_facts(self, user_id):
        return bot.load_facts(user_id)

    def doc(self, data=b"", name="fic.txt", size=None, fail=False):
        async def download_as_bytearray():
            if fail:
                raise RuntimeError("tg down")
            return bytearray(data)

        async def get_file():
            return SimpleNamespace(download_as_bytearray=download_as_bytearray)

        return SimpleNamespace(file_name=name, file_size=len(data) if size is None else size, get_file=get_file)

    def send_doc(self, user_id, document, **extra):
        return self.run_handler(bot.handle_document, user_id, None, document=document, **extra)

    def command(self, handler, user_id, text):
        return self.run_handler(handler, user_id, text)
