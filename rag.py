"""RAG для фактов и памяти сцен: Voyage AI эмбеддинги + Postgres/pgvector.

Таблица "facts" хранит по одной строке на факт юзера. Таблицы "scenes" и
"scene_chunks" хранят сохранённые вручную сцены и их чанки с эмбеддингами,
"fact_candidates" живущие сутки кандидаты в факты. В "scenes" есть колонка summary:
выжимка сцены для хроники сюжета.
При недоступности Postgres / Voyage / зависимостей функции бросают исключение —
вызывающая сторона (claude_tg_bot) ловит и делает fallback (для фактов: полный
список, для сцен: запрос уходит без справки).

Поиск точный (seq scan по косинусному расстоянию), без ANN-индекса: на сотнях
и тысячах фактов он быстрее и без потерь recall. HNSW имеет смысл от десятков
тысяч строк.

ENV:
- DATABASE_URL (postgresql://роль:пароль@хост:5432/база)
- VOYAGE_API_KEY

Расширение vector создаёт админ БД (CREATE EXTENSION vector); у роли бота прав
на это нет и не нужно.
"""

import hashlib
import logging
import os
import re

logger = logging.getLogger(__name__)

EMBED_MODEL = "voyage-3-lite"  # нативная размерность 512
EMBED_DIM = 512

DATABASE_URL = os.environ.get("DATABASE_URL", "")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")

try:
    import psycopg
    import voyageai

    _DEPS_OK = True
except Exception as e:  # noqa: BLE001 — модуль не должен валить импорт бота
    logger.warning(f"RAG deps unavailable, поиск отключён: {e}")
    _DEPS_OK = False

_voyage = None
_schema_ready = False

_SCHEMA = (
    f"""
CREATE TABLE IF NOT EXISTS facts (
    user_id    bigint  NOT NULL,
    fact_index integer NOT NULL,
    fact_text  text    NOT NULL,
    embedding  vector({EMBED_DIM}) NOT NULL,
    PRIMARY KEY (user_id, fact_index)
)
""",
    # Память сцен. Номер сцены в /scenes это позиция среди сцен юзера по id,
    # отдельной колонки под номер нет (после удаления номера сдвигаются).
    """
CREATE TABLE IF NOT EXISTS scenes (
    id         bigserial   PRIMARY KEY,
    user_id    bigint      NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    request    text        NOT NULL,
    text       text        NOT NULL,
    text_hash  text        NOT NULL,
    UNIQUE (user_id, text_hash)
)
""",
    # Выжимка сцены (хроника сюжета). Миграция для баз, где scenes уже есть.
    "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS summary text",
    f"""
CREATE TABLE IF NOT EXISTS scene_chunks (
    id          bigserial PRIMARY KEY,
    scene_id    bigint    NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    user_id     bigint    NOT NULL,
    chunk_index integer   NOT NULL,
    chunk_text  text      NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL
)
""",
    "CREATE INDEX IF NOT EXISTS scene_chunks_user_idx ON scene_chunks (user_id)",
    "CREATE INDEX IF NOT EXISTS scene_chunks_scene_idx ON scene_chunks (scene_id)",
    # Кандидаты в факты: живут 24 часа, в facts попадают только по /accept.
    # num это номер для /accept: счётчик на юзера (max(num)+1), не позиция среди
    # живых строк, поэтому номера в старом сообщении не съезжают, когда сгорает
    # более ранняя партия.
    """
CREATE TABLE IF NOT EXISTS fact_candidates (
    id         bigserial   PRIMARY KEY,
    user_id    bigint      NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    num        integer     NOT NULL,
    text       text        NOT NULL,
    status     text        NOT NULL DEFAULT 'pending'
)
""",
    "CREATE INDEX IF NOT EXISTS fact_candidates_user_idx ON fact_candidates (user_id, num)",
)


def _connect():
    """Новое соединение (нагрузка мизерная, пул не нужен). Бросает RuntimeError,
    если RAG не сконфигурирован. Схему создаёт один раз за процесс."""
    global _schema_ready
    if not _DEPS_OK:
        raise RuntimeError("voyageai / psycopg не установлены")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL не задан")
    conn = psycopg.connect(DATABASE_URL, connect_timeout=5)
    if not _schema_ready:
        for stmt in _SCHEMA:
            conn.execute(stmt)
        conn.commit()
        _schema_ready = True
        logger.info("RAG: таблицы facts, scenes, scene_chunks, fact_candidates готовы")
    return conn


def _voyage_client():
    global _voyage
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY не задан")
    if _voyage is None:
        _voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _voyage


def _embed(texts: list, input_type: str) -> list:
    result = _voyage_client().embed(texts, model=EMBED_MODEL, input_type=input_type)
    return result.embeddings


def _vec(embedding) -> str:
    """Вектор в текстовом виде pgvector: '[0.1,0.2,...]' (кастуем ::vector в SQL,
    отдельный адаптер не нужен)."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def index_facts(user_id: int, facts: list) -> None:
    """Эмбеддит все факты юзера и (пере)заливает их в Postgres.

    Эмбеддинги считаются ДО транзакции, а замена идёт атомарно: если Voyage
    упал, старый индекс остаётся нетронутым (раньше между delete и upsert
    индекс был пуст). Убранные факты не остаются в индексе.
    """
    vectors = _embed(list(facts), "document") if facts else []
    with _connect() as conn:
        conn.execute("DELETE FROM facts WHERE user_id = %s", (user_id,))
        if facts:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO facts (user_id, fact_index, fact_text, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    [
                        (user_id, i, fact, _vec(vec))
                        for i, (fact, vec) in enumerate(zip(facts, vectors))
                    ],
                )
    if facts:
        logger.info(f"RAG: проиндексировано {len(facts)} фактов юзера {user_id}")
    else:
        logger.info(f"RAG: факты юзера {user_id} очищены из индекса")


def search_facts_multi(
    user_id: int, queries: list, top_k: int = 30, limit: int = 50, exclude=None
) -> list:
    """Поиск сразу по нескольким запросам (один батч эмбеддингов). Берёт top_k на
    каждый запрос, склеивает без дублей, оставляет `limit` самых близких и
    возвращает их в авторском порядке (по fact_index) — так канон читается как
    написан, а не вперемешку.

    exclude: fact_index, которые не искать (закреплённые факты бот добавляет в
    выдачу сам, они не должны занимать места в top_k и limit)."""
    vecs = _embed(list(queries), "query")
    excl = sorted(set(exclude)) if exclude else []
    best = {}  # fact_index -> (score, text)
    with _connect() as conn:
        for vec in vecs:
            qv = _vec(vec)
            rows = conn.execute(
                "SELECT fact_index, fact_text, 1 - (embedding <=> %s::vector) AS score "
                "FROM facts WHERE user_id = %s AND NOT (fact_index = ANY(%s::int[])) "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (qv, user_id, excl, qv, top_k),
            ).fetchall()
            for idx, text, score in rows:
                prev = best.get(idx)
                if prev is None or score > prev[0]:
                    best[idx] = (score, text)
    top = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
    return [text for _, (_, text) in sorted(top, key=lambda kv: kv[0])]


def search_facts(user_id: int, query: str, top_k: int = 10) -> list:
    """Релевантные query факты юзера в авторском порядке (один запрос)."""
    return search_facts_multi(user_id, [query], top_k=top_k, limit=top_k)


# --- Память сцен ---
# Чанки режутся по абзацам, абзац не рвётся: чанк это целое число абзацев.
CHUNK_MAX_WORDS = 350  # новый абзац не добавляем, если чанк перевалит за это
CHUNK_TAIL_WORDS = 100  # хвост короче этого приклеивается к предыдущему чанку
EMBED_BATCH = 64  # текстов за один вызов Voyage (его лимит 128)
REQUEST_MAX_CHARS = 500  # сколько символов request храним


def _words(text: str) -> int:
    return len(text.split())


def chunk_scene(text: str, max_words: int = CHUNK_MAX_WORDS) -> list:
    """Режет сцену на чанки примерно по 250-350 слов по границам абзацев.

    Абзацы копятся, пока чанк не перевалил бы за max_words; абзац длиннее лимита
    остаётся одним чанком (не режем посреди абзаца). Короткий хвост склеивается с
    предыдущим чанком. Склейка чанков через пустую строку возвращает абзацы
    сцены в исходном порядке, ничего не теряется."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    cur = []
    cur_words = 0
    for para in paras:
        w = _words(para)
        if cur and cur_words + w > max_words:
            chunks.append(cur)
            cur, cur_words = [], 0
        cur.append(para)
        cur_words += w
    if cur:
        prev_words = sum(_words(x) for x in chunks[-1]) if chunks else 0
        # хвост клеим, только если чанк не раздуется за max_words + CHUNK_TAIL_WORDS
        # (после абзаца-великана клеить нельзя: чанк и так уже выше лимита)
        if chunks and cur_words < CHUNK_TAIL_WORDS and prev_words + cur_words <= max_words + CHUNK_TAIL_WORDS:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return ["\n\n".join(c) for c in chunks]


def scene_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _embed_batched(texts: list, input_type: str) -> list:
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(_embed(texts[i : i + EMBED_BATCH], input_type))
    return out


_NUMBER_SQL = "SELECT count(*) FROM scenes WHERE user_id = %s AND id <= %s"


def save_scene(user_id: int, request: str, text: str):
    """Сохраняет сцену с чанками. Возвращает (номер, создана_ли): для дубля по
    (user_id, text_hash) ничего не пишет и возвращает номер уже сохранённой.

    Порядок: проверка дубля (чтение), эмбеддинги, потом одна транзакция с записью
    сцены и всех чанков. Сбой Voyage или БД до коммита ничего не оставляет."""
    text = text.strip()
    h = scene_hash(text)
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM scenes WHERE user_id = %s AND text_hash = %s", (user_id, h)
        ).fetchone()
        if row is not None:
            return conn.execute(_NUMBER_SQL, (user_id, row[0])).fetchone()[0], False
    chunks = chunk_scene(text)
    if not chunks:
        raise ValueError("пустая сцена")
    vectors = _embed_batched(chunks, "document")
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Voyage вернул {len(vectors)} векторов на {len(chunks)} чанков")
    with _connect() as conn:
        row = conn.execute(
            "INSERT INTO scenes (user_id, request, text, text_hash) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, text_hash) DO NOTHING "
            "RETURNING id",
            (user_id, request[:REQUEST_MAX_CHARS], text, h),
        ).fetchone()
        if row is None:  # параллельный /keep успел раньше
            row = conn.execute(
                "SELECT id FROM scenes WHERE user_id = %s AND text_hash = %s", (user_id, h)
            ).fetchone()
            return conn.execute(_NUMBER_SQL, (user_id, row[0])).fetchone()[0], False
        scene_id = row[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO scene_chunks (scene_id, user_id, chunk_index, chunk_text, embedding) "
                "VALUES (%s, %s, %s, %s, %s::vector)",
                [
                    (scene_id, user_id, i, chunk, _vec(vec))
                    for i, (chunk, vec) in enumerate(zip(chunks, vectors))
                ],
            )
        number = conn.execute(_NUMBER_SQL, (user_id, scene_id)).fetchone()[0]
    logger.info(f"RAG: сцена {number} юзера {user_id} сохранена ({len(chunks)} чанков)")
    return number, True


def list_scenes(user_id: int) -> list:
    """Сцены юзера по порядку сохранения: [(created_at, request, слов, есть_выжимка)].
    Позиция в списке с единицы это номер сцены для /scenes, /unkeep и меток в промпте."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT created_at, request, "
            "cardinality(regexp_split_to_array(btrim(text), '\\s+')), "
            "coalesce(summary, '') <> '' "
            "FROM scenes WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
    return [(created, request, words, bool(has)) for created, request, words, has in rows]


def get_scene_summary(user_id: int, number: int):
    """(created_at, request, выжимка или None, слов) сцены с порядковым номером number
    (с единицы) или None, если такой сцены нет."""
    if number < 1:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT created_at, request, summary, "
            "cardinality(regexp_split_to_array(btrim(text), '\\s+')) "
            "FROM scenes WHERE user_id = %s ORDER BY id OFFSET %s LIMIT 1",
            (user_id, number - 1),
        ).fetchone()
    if row is None:
        return None
    created, request, summary, words = row
    return created, request, (summary or None), words


def set_summary(user_id: int, number: int, summary: str) -> bool:
    """Записывает выжимку сцене с порядковым номером number. False, если сцены нет."""
    with _connect() as conn:
        row = conn.execute(
            "UPDATE scenes SET summary = %s WHERE user_id = %s AND id = ("
            "SELECT id FROM scenes WHERE user_id = %s ORDER BY id OFFSET %s LIMIT 1"
            ") RETURNING id",
            (summary, user_id, user_id, number - 1),
        ).fetchone()
    return row is not None


def story_chronicle(user_id: int) -> list:
    """Хроника сюжета: [(номер сцены, выжимка)] по порядку сохранения, только сцены с
    выжимкой. Номер это порядковый номер среди всех сцен юзера (как в /scenes)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT n, summary FROM ("
            "SELECT row_number() OVER (ORDER BY id) AS n, summary FROM scenes WHERE user_id = %s"
            ") t WHERE coalesce(summary, '') <> '' ORDER BY n",
            (user_id,),
        ).fetchall()
    return [(int(n), summary) for n, summary in rows]


def delete_scene(user_id: int, number: int) -> bool:
    """Удаляет сцену с порядковым номером number (с единицы) вместе с чанками
    (ON DELETE CASCADE). Одно выражение: номер и удаление не расходятся, даже
    если между /scenes и /unkeep что-то сохранилось. False, если номера нет."""
    if number < 1:
        return False
    with _connect() as conn:
        row = conn.execute(
            "DELETE FROM scenes WHERE user_id = %s AND id = ("
            "SELECT id FROM scenes WHERE user_id = %s ORDER BY id OFFSET %s LIMIT 1"
            ") RETURNING id",
            (user_id, user_id, number - 1),
        ).fetchone()
    return row is not None


def search_scenes(user_id: int, queries: list, top_k: int = 3) -> list:
    """top_k ближайших по косинусу чанков сцен юзера: [(номер_сцены, текст)] от
    самого близкого. Запросы эмбеддятся одним батчем, на каждый берётся top_k,
    дубли склеиваются по максимуму близости. Нет сцен или запросов: [] без
    вызова Voyage."""
    if not queries or top_k < 1:
        return []
    with _connect() as conn:
        if conn.execute(
            "SELECT 1 FROM scene_chunks WHERE user_id = %s LIMIT 1", (user_id,)
        ).fetchone() is None:
            return []
        vecs = _embed(list(queries), "query")
        best = {}  # chunk id -> (score, номер сцены, текст)
        for vec in vecs:
            qv = _vec(vec)
            rows = conn.execute(
                "SELECT c.id, c.chunk_text, 1 - (c.embedding <=> %s::vector) AS score, "
                "(SELECT count(*) FROM scenes s WHERE s.user_id = c.user_id AND s.id <= c.scene_id) "
                "FROM scene_chunks c WHERE c.user_id = %s "
                "ORDER BY c.embedding <=> %s::vector LIMIT %s",
                (qv, user_id, qv, top_k),
            ).fetchall()
            for cid, chunk, score, number in rows:
                prev = best.get(cid)
                if prev is None or score > prev[0]:
                    best[cid] = (score, number, chunk)
    top = sorted(best.values(), key=lambda v: v[0], reverse=True)[:top_k]
    return [(number, chunk) for _, number, chunk in top]


# --- Кандидаты в факты ---
# Срок жизни 24 часа: просроченное удаляется при любом обращении к таблице.
_PURGE_SQL = "DELETE FROM fact_candidates WHERE created_at < now() - interval '24 hours'"


def add_candidates(user_id: int, texts: list) -> list:
    """Сохраняет кандидатов (status pending). Возвращает [(номер, текст)]. Номера
    продолжают нумерацию живых кандидатов юзера."""
    if not texts:
        return []
    with _connect() as conn:
        conn.execute(_PURGE_SQL)
        start = conn.execute(
            "SELECT coalesce(max(num), 0) FROM fact_candidates WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
        rows = [(user_id, start + i + 1, t) for i, t in enumerate(texts)]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO fact_candidates (user_id, num, text) VALUES (%s, %s, %s)", rows
            )
    return [(num, text) for _, num, text in rows]


def pending_candidates(user_id: int) -> list:
    """Ожидающие подтверждения кандидаты юзера: [(номер, текст)] по номеру."""
    with _connect() as conn:
        conn.execute(_PURGE_SQL)
        rows = conn.execute(
            "SELECT num, text FROM fact_candidates WHERE user_id = %s AND status = 'pending' "
            "ORDER BY num",
            (user_id,),
        ).fetchall()
    return [(num, text) for num, text in rows]


def claim_candidates(user_id: int, nums=None):
    """Помечает выбранных кандидатов принятыми (nums=None: все ожидающие).
    Возвращает (взято [(номер, текст)], уже_принятые [номера], нет_таких [номера]).
    Один UPDATE ... RETURNING: повторный вызов с теми же номерами ничего не берёт.
    Если запись фактов после этого не удалась, вызывающий возвращает строки через
    release_candidates."""
    with _connect() as conn:
        conn.execute(_PURGE_SQL)
        live = conn.execute(
            "SELECT num, status FROM fact_candidates WHERE user_id = %s", (user_id,)
        ).fetchall()
        pending = {n for n, st in live if st == "pending"}
        accepted = {n for n, st in live if st == "accepted"}
        if nums is None:
            target = sorted(pending)
            already, missing = [], []
        else:
            asked = set(nums)
            target = sorted(asked & pending)
            already = sorted(asked & accepted)
            missing = sorted(asked - pending - accepted)
        claimed = []
        if target:
            claimed = conn.execute(
                "UPDATE fact_candidates SET status = 'accepted' "
                "WHERE user_id = %s AND status = 'pending' AND num = ANY(%s::int[]) "
                "RETURNING num, text",
                (user_id, target),
            ).fetchall()
    return sorted((num, text) for num, text in claimed), already, missing


def release_candidates(user_id: int, nums: list) -> None:
    """Откат claim_candidates, если факты записать не удалось."""
    if not nums:
        return
    with _connect() as conn:
        conn.execute(
            "UPDATE fact_candidates SET status = 'pending' "
            "WHERE user_id = %s AND status = 'accepted' AND num = ANY(%s::int[])",
            (user_id, list(nums)),
        )


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def filter_similar(user_id: int, candidates: list, pending_texts: list, threshold: float) -> list:
    """Оставляет кандидатов, у которых косинусная близость < threshold к каждому
    существующему факту (по готовым эмбеддингам в таблице facts), к каждому
    ожидающему кандидату и к уже оставленным в этой партии. Эмбеддинги новых
    текстов считает Voyage одним батчем (input_type document, как у фактов)."""
    if not candidates:
        return []
    vecs = _embed_batched(list(candidates) + list(pending_texts), "document")
    cvecs, pvecs = vecs[: len(candidates)], vecs[len(candidates):]
    kept, kept_vecs = [], []
    with _connect() as conn:
        for text, vec in zip(candidates, cvecs):
            row = conn.execute(
                "SELECT 1 - min(embedding <=> %s::vector) FROM facts WHERE user_id = %s",
                (_vec(vec), user_id),
            ).fetchone()
            sim_facts = float(row[0]) if row and row[0] is not None else 0.0
            if sim_facts >= threshold:
                continue
            if any(cosine(vec, other) >= threshold for other in pvecs + kept_vecs):
                continue
            kept.append(text)
            kept_vecs.append(vec)
    return kept
