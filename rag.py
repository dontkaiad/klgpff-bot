"""RAG для фактов и памяти сцен: Voyage AI эмбеддинги + Postgres/pgvector.

Таблица "facts" хранит по одной строке на факт юзера. Таблицы "scenes" и
"scene_chunks" хранят сохранённые вручную сцены и их чанки с эмбеддингами.
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
        logger.info("RAG: таблицы facts, scenes, scene_chunks готовы")
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
    """Сцены юзера по порядку сохранения: [(created_at, request, слов)]. Позиция в
    списке с единицы это номер сцены для /scenes, /unkeep и меток «сцена N»."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT created_at, request, "
            "cardinality(regexp_split_to_array(btrim(text), '\\s+')) "
            "FROM scenes WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
    return [(created, request, words) for created, request, words in rows]


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
