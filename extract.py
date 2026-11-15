"""Извлечение кандидатов в факты из текста сцены: промпт, устойчивый разбор ответа
модели, нормализация и нечёткая дедупликация, текст сообщения для пользователя.

Модуль чистый (без сети и состояния): вызов модели, эмбеддинги и БД остаются в
claude_tg_bot.py и rag.py. В факты ничего не попадает без /accept.
"""

import json
import re

MAX_CANDIDATES = 8  # новых фактов за одно извлечение
MAX_CONFLICTS = 5
MAX_FACT_CHARS = 300  # длиннее: обрезаем (факты короткие)
MAX_CONFLICT_CHARS = 200

EXTRACT_SYSTEM = (
    "Тебе даны факты канона и текст сцены. Верни JSON "
    '{"new_facts": [...], "conflicts": [...]}. '
    "new_facts: до 8 коротких фактов об обстановке, персонажах, привычках, отношениях, "
    "которые явно сказаны в тексте и НЕ дублируют и не противоречат канону. "
    "Формулируй в стиле существующих фактов, нейтрально, без откровенных подробностей. "
    "Не включай разовые события сюжета. "
    "conflicts: места, где текст противоречит канону (факт и короткая цитата до 12 слов). "
    "Ответ: только JSON, без пояснений."
)


def build_extract_input(facts: list, text: str, max_chars: int) -> str:
    """Пользовательское сообщение для модели: канон списком и сцена."""
    canon = "\n".join(f"- {f}" for f in facts) if facts else "(фактов пока нет)"
    return f"<canon>\n{canon}\n</canon>\n\n<scene>\n{text[:max_chars]}\n</scene>"


def _json_object(raw: str):
    """Достаёт JSON-объект из ответа: чистый JSON или (если вокруг пояснение либо
    ```-блок) текст от первой «{» до последней «}». None, если не вышло."""
    raw = (raw or "").strip()
    for candidate in (raw, raw[raw.find("{"): raw.rfind("}") + 1] if "{" in raw else ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _clean(value, limit: int) -> str:
    return " ".join(str(value).split())[:limit].strip()


def parse_extraction(raw: str):
    """(new_facts, conflicts) из ответа модели или None, если разобрать нельзя.
    Лишние и нестроковые элементы отбрасываются, new_facts режется до 8."""
    obj = _json_object(raw)
    if obj is None or not ({"new_facts", "conflicts"} & set(obj)):
        return None
    facts = []
    for item in obj.get("new_facts") or []:
        if isinstance(item, str) and _clean(item, MAX_FACT_CHARS):
            facts.append(_clean(item, MAX_FACT_CHARS))
    conflicts = []
    for item in obj.get("conflicts") or []:
        if isinstance(item, dict):
            item = " — ".join(str(v) for v in item.values() if v)
        if isinstance(item, str) and _clean(item, MAX_CONFLICT_CHARS):
            conflicts.append(_clean(item, MAX_CONFLICT_CHARS))
    return facts[:MAX_CANDIDATES], conflicts[:MAX_CONFLICTS]


def normalize_fact(text: str) -> str:
    """Для сравнения: без регистра, ё=е, без знаков препинания и маркера закрепления."""
    s = text.lower().replace("ё", "е").replace("\U0001F4CC", "")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def dedupe_normalized(candidates: list, known: list) -> list:
    """Убирает кандидатов, совпавших после нормализации с известными текстами
    (факты, ожидающие кандидаты) и друг с другом."""
    seen = {normalize_fact(k) for k in known}
    out = []
    for c in candidates:
        key = normalize_fact(c)
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def candidates_message(numbered: list, conflicts: list) -> str:
    """Сообщение пользователю (без LLM): список кандидатов с номерами для /accept,
    противоречия канону, подсказка. numbered: [(номер, текст)]."""
    parts = []
    if numbered:
        lines = "\n".join(f"{n}. {t}" for n, t in numbered)
        parts.append(f"🧠 кандидаты в факты:\n\n{lines}")
    if conflicts:
        lines = "\n".join(f"- {c}" for c in conflicts)
        parts.append(f"⚠️ противоречия канону:\n{lines}")
    if numbered:
        example = " ".join(str(n) for n, _ in numbered[:3])
        parts.append(
            f"/accept {example} или /accept all; остальные сгорят через 24 часа. "
            "Без /accept в факты ничего не попадёт."
        )
    return "\n\n".join(parts)
