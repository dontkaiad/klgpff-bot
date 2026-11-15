"""Режим /import, документы, /context и «память сцен выключена по умолчанию».
Без сети: Anthropic, Voyage и Postgres подменены. Тексты нейтральные.

    TELEGRAM_BOT_TOKEN=1:a ANTHROPIC_API_KEY=x python3 -m unittest discover -s tests -v
"""

import unittest
from unittest import mock

from support import Base2, at, bot, extraction_json, origin_from, words
from test_import_limit import BOT_ID, OUTSIDER


class ImportBase(Base2):
    def start_import(self, user_id=111):
        return self.command(bot.cmd_import, user_id, "/import")

    def part(self, user_id, text, seconds=0):
        return self.run_handler(bot.handle_message, user_id, text, date=at(seconds))


class ImportModeTests(ImportBase):
    def test_import_reply(self):
        self.assertEqual(self.start_import(), ["присылай части или .txt файл, в конце /done"])

    def test_parts_go_to_buffer_without_the_model(self):
        self.start_import()
        replies = [self.part(111, words(10, 100 * k), k) for k in range(3)]
        self.assertEqual([r[0] for r in replies], ["часть 1, 10 слов", "часть 2, 10 слов", "часть 3, 10 слов"])
        self.assertEqual(self.client.mock_calls, [])  # ни stream, ни роутер, ни извлечение
        self.assertEqual(bot.conversations, {})
        self.assertTrue(bot.import_active(111))

    def test_long_text_is_accepted_in_import_mode(self):
        self.start_import()
        long_part = "а" * (bot.MAX_INPUT_CHARS + 500)  # вне режима это был бы отказ
        self.assertEqual(self.part(111, long_part, 1), ["часть 1, 1 слов"])
        self.assertEqual(self.client.mock_calls, [])

    def test_forwarded_messages_use_the_original_date(self):
        self.start_import()
        self.run_handler(bot.handle_message, 111, "часть B", origin_from(BOT_ID, at(20)), date=at(500))
        self.run_handler(bot.handle_message, 111, "часть A", origin_from(BOT_ID, at(10)), date=at(600))
        self.part(111, "часть C", 700)
        self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(self.store.scenes[111][0]["text"], "часть A\n\nчасть B\n\nчасть C")

    def test_foreign_forward_is_just_text_in_import_mode(self):
        self.start_import()
        out = self.run_handler(bot.handle_message, 111, "чужое", origin_from(12345, at(1), is_bot=False), date=at(5))
        self.assertEqual(out, ["часть 1, 1 слов"])

    def test_duplicate_part_is_not_added_twice(self):
        self.start_import()
        self.part(111, "часть A", 1)
        self.assertEqual(self.part(111, "часть A", 2), ["эта часть уже принята (всего 1)"])

    def test_buffer_limit_stops_accepting_and_reports(self):
        with mock.patch.object(bot, "IMPORT_BUFFER_MAX_CHARS", 100):
            self.start_import()
            self.part(111, "а" * 60, 1)
            out = self.part(111, "б" * 60, 2)  # 60 + 60 > 100
            out2 = self.part(111, "в" * 10, 3)  # меньше влезло бы, но дальше не принимаем? нет: влезает
        self.assertIn("буфер ограничен 100 символов", out[0])
        self.assertEqual(out2, ["часть 2, 1 слов"])  # маленькая часть ещё помещается
        self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(self.store.scenes[111][0]["text"], "а" * 60 + "\n\n" + "в" * 10)

    def test_commands_still_work_in_import_mode(self):
        self.start_import()
        self.assertIn("🧭", self.command(bot.cmd_context, 111, "/context")[0])

    def test_keep_in_import_mode_points_to_done(self):
        self.start_import()
        self.part(111, "часть A", 1)
        out = self.command(bot.cmd_keep, 111, "/keep")
        self.assertIn("/done", out[0])
        self.assertFalse(self.store.scenes)
        self.assertTrue(bot.import_active(111))


class DoneTests(ImportBase):
    def import_three(self):
        self.start_import()
        for text, sec in (("часть C", 30), ("часть A", 10), ("часть B", 20)):
            self.part(111, text, sec)

    def test_done_joins_by_date_saves_scene_and_leaves_the_mode(self):
        self.import_three()
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(out[0], "✓ сохранена: сцена 1 (частей: 3, слов: 6)")
        self.assertEqual(self.store.scenes[111][0]["text"], "часть A\n\nчасть B\n\nчасть C")
        self.assertFalse(bot.import_active(111))
        self.assertNotIn(111, bot.forward_buffers)

    def test_done_starts_extraction_once_on_the_joined_text(self):
        self.import_three()
        self.extract_reply = extraction_json(["факт 1", "факт 2"])
        out = self.command(bot.cmd_done, 111, "/done")
        calls = self.extraction_calls()
        self.assertEqual(len(calls), 1)
        self.assertIn("часть A\n\nчасть B\n\nчасть C", calls[0].kwargs["messages"][0]["content"])
        self.assertIn("1. факт 1", out[1])
        self.assertIn("/accept", out[1])
        self.client.messages.stream.assert_not_called()  # генерации сцены нет

    def test_done_with_nothing_new_says_so(self):
        self.import_three()
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(out[1], "новых фактов в тексте не найдено.")

    def test_done_duplicate_shows_the_existing_number(self):
        self.import_three()
        self.command(bot.cmd_done, 111, "/done")
        self.start_import()
        for text, sec in (("часть C", 30), ("часть A", 10), ("часть B", 20)):
            self.part(111, text, sec)
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(out[0], "уже сохранена: сцена 1 (частей: 3, слов: 6)")
        self.assertEqual(len(self.store.scenes[111]), 1)

    def test_done_storage_failure_keeps_buffer_and_mode(self):
        self.import_three()
        self.store.fail = True
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertIn("не удалось сохранить", out[0])
        self.assertTrue(bot.import_active(111))
        self.store.fail = False
        self.assertIn("✓ сохранена", self.command(bot.cmd_done, 111, "/done")[0])

    def test_done_without_import_mode(self):
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertIn("не включён", out[0])
        self.assertEqual(self.extraction_calls(), [])

    def test_done_with_empty_buffer(self):
        self.start_import()
        out = self.command(bot.cmd_done, 111, "/done")
        self.assertIn("буфер пуст", out[0])
        self.assertFalse(bot.import_active(111))
        self.assertFalse(self.store.scenes)

    def test_extraction_does_not_start_without_done(self):
        self.start_import()
        self.part(111, words(400), 1)
        self.command(bot.cmd_cancel, 111, "/cancel")
        self.assertEqual(self.extraction_calls(), [])
        self.assertEqual(self.client.mock_calls, [])


class CancelAndExpiryTests(ImportBase):
    def test_cancel_clears_without_saving_or_extracting(self):
        self.start_import()
        self.part(111, "часть A", 1)
        out = self.command(bot.cmd_cancel, 111, "/cancel")
        self.assertIn("импорт отменён", out[0])
        self.assertFalse(bot.import_active(111))
        self.assertNotIn(111, bot.forward_buffers)
        self.assertFalse(self.store.scenes)
        self.assertEqual(self.client.mock_calls, [])

    def test_new_clears_mode_and_buffer(self):
        self.start_import()
        self.part(111, "часть A", 1)
        self.command(bot.cmd_new, 111, "/new")
        self.assertFalse(bot.import_active(111))
        self.assertNotIn(111, bot.forward_buffers)
        self.assertIn("не включён", self.command(bot.cmd_done, 111, "/done")[0])

    def test_after_cancel_messages_go_to_the_model_again(self):
        self.start_import()
        self.command(bot.cmd_cancel, 111, "/cancel")
        self.part(111, "привет", 1)
        self.client.messages.stream.assert_called_once()

    def test_cancel_with_nothing_to_cancel(self):
        self.assertEqual(self.command(bot.cmd_cancel, 111, "/cancel"), ["нечего отменять."])

    def test_timeout_turns_mode_off_and_reports_on_next_message(self):
        self.start_import()
        self.part(111, "часть A", 1)
        self.now += bot.FORWARD_BUFFER_TTL + 1
        out = self.part(111, "часть B", 2)
        self.assertIn("режим /import выключен", out[0])
        self.assertIn("не обработано", out[0])
        self.assertEqual(self.client.mock_calls, [])  # эта часть в модель не ушла
        self.assertNotIn(111, bot.forward_buffers)
        self.part(111, "привет", 3)  # дальше обычный режим
        self.client.messages.stream.assert_called_once()

    def test_timeout_is_reported_once(self):
        self.start_import()
        self.now += bot.FORWARD_BUFFER_TTL + 1
        self.part(111, "x", 1)
        self.assertNotIn("режим /import выключен", self.part(111, "y", 2)[0])

    def test_done_after_timeout_reports_it(self):
        self.start_import()
        self.part(111, "часть A", 1)
        self.now += bot.FORWARD_BUFFER_TTL + 1
        self.assertIn("режим /import выключен", self.command(bot.cmd_done, 111, "/done")[0])
        self.assertFalse(self.store.scenes)

    def test_each_part_extends_the_timeout(self):
        self.start_import()
        self.part(111, "часть A", 1)
        self.now += bot.FORWARD_BUFFER_TTL - 10
        self.part(111, "часть B", 2)
        self.now += bot.FORWARD_BUFFER_TTL - 10
        self.assertIn("✓ сохранена", self.command(bot.cmd_done, 111, "/done")[0])

    def test_new_import_after_timeout_starts_clean(self):
        self.start_import()
        self.part(111, "часть A", 1)
        self.now += bot.FORWARD_BUFFER_TTL + 1
        self.start_import()
        self.assertEqual(self.part(111, "часть B", 2), ["часть 1, 2 слов"])


class AccessTests(ImportBase):
    def test_outsider_gets_silence_everywhere(self):
        doc = self.doc(b"text")
        for handler, text in (
            (bot.cmd_import, "/import"), (bot.cmd_done, "/done"), (bot.cmd_cancel, "/cancel"),
            (bot.cmd_accept, "/accept all"), (bot.cmd_context, "/context"),
            (bot.cmd_context, "/context scenes on"), (bot.handle_message, "текст"),
        ):
            self.assertEqual(self.run_handler(handler, OUTSIDER, text), [], text)
        self.assertEqual(self.send_doc(OUTSIDER, doc), [])
        self.assertEqual(self.client.mock_calls, [])
        self.assertNotIn(OUTSIDER, bot.forward_buffers)
        self.assertFalse(bot.scenes_enabled(OUTSIDER))


class DocumentTests(ImportBase):
    def test_txt_and_md_in_import_mode_go_to_buffer_not_to_the_model(self):
        self.start_import()
        a = self.send_doc(111, self.doc("часть один".encode("utf-8"), "fic.txt"), date=at(1))
        b = self.send_doc(111, self.doc("﻿часть два".encode("utf-8"), "fic.MD"), date=at(2))  # с BOM
        self.assertEqual(a, ["часть 1, 2 слов"])
        self.assertEqual(b, ["часть 2, 2 слов"])
        self.assertEqual(self.client.mock_calls, [])
        self.command(bot.cmd_done, 111, "/done")
        self.assertEqual(self.store.scenes[111][0]["text"], "часть один\n\nчасть два")

    def test_file_over_200kb_is_refused_by_declared_size_without_download(self):
        self.start_import()
        d = self.doc(b"x", size=bot.IMPORT_DOC_MAX_BYTES + 1)
        out = self.send_doc(111, d)
        self.assertIn("файл больше 200 КБ", out[0])
        self.assertNotIn(111, [k for k, v in bot.forward_buffers.items() if v["parts"]])

    def test_file_over_200kb_is_refused_by_real_size(self):
        self.start_import()
        d = self.doc(b"x" * (bot.IMPORT_DOC_MAX_BYTES + 1), size=0)  # размер в метаданных не указан
        self.assertIn("не принят", self.send_doc(111, d)[0])

    def test_file_at_the_limit_is_read(self):
        self.start_import()
        with mock.patch.object(bot, "IMPORT_BUFFER_MAX_CHARS", 10**9):
            out = self.send_doc(111, self.doc(b"x" * bot.IMPORT_DOC_MAX_BYTES), date=at(1))
        self.assertEqual(out, ["часть 1, 1 слов"])

    def test_non_utf8_empty_and_wrong_extension_are_refused(self):
        self.start_import()
        self.assertIn("не в UTF-8", self.send_doc(111, self.doc("текст".encode("cp1251")))[0])
        self.assertEqual(self.send_doc(111, self.doc(b"  \n"))[0], "файл пустой")
        self.assertIn("только .txt и .md", self.send_doc(111, self.doc(b"x", "fic.pdf"))[0])

    def test_download_failure_is_reported(self):
        self.start_import()
        self.assertIn("не смогла прочитать файл", self.send_doc(111, self.doc(fail=True))[0])

    def test_outside_import_txt_is_not_sent_to_the_model(self):
        gets = []
        d = self.doc(b"x")
        real = d.get_file

        async def spy():
            gets.append(1)
            return await real()

        d.get_file = spy
        out = self.send_doc(111, d)
        self.assertIn("не отправлен в модель", out[0])
        self.assertIn("/import", out[0])
        self.assertEqual(gets, [])  # файл даже не скачивали
        self.assertEqual(self.client.mock_calls, [])
        self.assertEqual(bot.conversations, {})

    def test_outside_import_other_files_are_silent(self):
        self.assertEqual(self.send_doc(111, self.doc(b"x", "pic.pdf")), [])

    def test_document_after_timeout_is_reported_not_processed(self):
        self.start_import()
        self.now += bot.FORWARD_BUFFER_TTL + 1
        self.assertIn("режим /import выключен", self.send_doc(111, self.doc(b"x"))[0])


# --------------------------------------------- память сцен: выкл по умолчанию, /context
class ScenesDefaultOffTests(ImportBase):
    FRAGMENTS = [(2, "фрагмент сцены 2")]

    def setUp(self):
        super().setUp()
        self.search = mock.Mock(return_value=list(self.FRAGMENTS))
        p = mock.patch.object(bot.rag, "search_scenes", self.search)
        p.start()
        self.addCleanup(p.stop)

    def last_request(self):
        return self.client.messages.stream.call_args.kwargs["messages"][-1]["content"]

    def test_default_request_has_no_scene_block_and_no_search(self):
        self.part(111, "привет", 1)
        self.assertNotIn("memory_scenes", self.last_request())
        self.search.assert_not_called()  # даже БД не трогаем

    def test_on_then_off(self):
        self.command(bot.cmd_context, 111, "/context scenes on")
        self.part(111, "привет", 1)
        self.assertIn("<memory_scenes>", self.last_request())
        self.assertIn("фрагмент сцены 2", self.last_request())
        self.command(bot.cmd_context, 111, "/context scenes off")
        self.part(111, "ещё", 2)
        self.assertNotIn("memory_scenes", self.last_request())

    def test_switch_is_per_user_and_stored_in_a_file(self):
        self.command(bot.cmd_context, 111, "/context scenes on")
        self.assertTrue(bot.scenes_enabled(111))
        self.assertFalse(bot.scenes_enabled(222))
        self.assertTrue((self.models_dir / "111.scenes").exists())

    def test_block_is_not_mixed_in_while_off_even_with_hits(self):
        self.part(111, "привет", 1)
        self.part(111, "ещё", 2)
        for call in self.client.messages.stream.call_args_list:
            self.assertNotIn("memory_scenes", str(call.kwargs))


class ContextTests(ImportBase):
    def setUp(self):
        super().setUp()
        for p in (
            mock.patch.object(bot, "load_prompt", return_value="СЕКРЕТНЫЙ ПРОМПТ 12345"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def ctx(self, text="/context"):
        return "\n".join(self.command(bot.cmd_context, 111, text))

    def test_shows_sizes_modes_and_limit_without_texts_and_without_the_model(self):
        self.write_facts(111, ["СЕКРЕТНЫЙ ФАКТ А", "📌 СЕКРЕТНЫЙ ФАКТ Б"])
        self.store.scenes[111] = [{"request": "СЕКРЕТНЫЙ ЗАПРОС", "text": "СЕКРЕТНАЯ СЦЕНА"}] * 2
        out = self.ctx()
        self.assertEqual(self.client.mock_calls, [])
        for secret in ("СЕКРЕТН", "12345"):
            self.assertNotIn(secret, out)
        self.assertIn("system-промпт: 22 симв.", out)
        self.assertIn("факты: все (2)", out)
        self.assertIn("память сцен: выкл, сохранено сцен: 2", out)
        self.assertIn(f"режим модели: opus ({bot.MODELS['opus']})", out)
        self.assertIn("сегодня (UTC+3): ≈ $0.00 из $3.00", out)

    def test_scenes_line_follows_the_switch(self):
        self.ctx("/context scenes on")
        self.assertIn(f"память сцен: вкл (до {bot.SCENE_TOP_K} фрагментов", self.ctx())

    def test_facts_line_when_search_is_on(self):
        self.write_facts(111, [f"факт {i}" for i in range(5)] + ["📌 закреплённый"])
        with mock.patch.object(bot, "FACTS_FULL_LIMIT", 3):
            out = self.ctx()
        self.assertIn(f"факты: 1 закреплённых + поиск до {bot.RAG_MAX_FACTS} из 6", out)

    def test_no_facts_and_unavailable_storage(self):
        self.store.fail = True
        out = self.ctx()
        self.assertIn("факты: нет", out)
        self.assertIn("сохранено сцен: н/д", out)

    def test_auto_mode_and_import_state(self):
        with mock.patch.object(bot, "load_mode", return_value="auto"):
            self.start_import_for(111)
            out = self.ctx()
        self.assertIn("режим модели: auto (роутер)", out)
        self.assertIn("режим /import: идёт", out)

    def start_import_for(self, uid):
        self.command(bot.cmd_import, uid, "/import")

    def test_spend_is_shown(self):
        self.spend(111, 1.0)
        self.assertIn("≈ $1.00 из $3.00", self.ctx())

    def test_bad_arguments(self):
        for arg in ("scenes", "scenes maybe", "foo", "scenes on now"):
            out = self.command(bot.cmd_context, 111, f"/context {arg}")
            self.assertIn("использование", out[0], arg)
        self.assertFalse(bot.scenes_enabled(111))


if __name__ == "__main__":
    unittest.main()
