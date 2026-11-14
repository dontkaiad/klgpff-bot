"""RAG для фактов: Voyage AI эмбеддинги + Postgres/pgvector.

Таблица "facts" хранит по одной строке на факт юзера. При недоступности
Postgres / Voyage / зависимостей функции бросают исключение — вызывающая
сторона (claude_tg_bot) ловит и делает fallback на полный список фактов.

Поиск точный (seq scan по косинусному расстоянию), без ANN-индекса: на сотнях
и тысячах фактов он быстрее и без потерь recall. HNSW имеет смысл от десятков
тысяч строк.

ENV:
- DATABASE_URL (postgresql://роль:пароль@хост:5432/база)
- VOYAGE_API_KEY

Расширение vector создаёт админ БД (CREATE EXTENSION vector); у роли бота прав
на это нет и не нужно.
"""

import logging
import os

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

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS facts (
    user_id    bigint  NOT NULL,
    fact_index integer NOT NULL,
    fact_text  text    NOT NULL,
    embedding  vector({EMBED_DIM}) NOT NULL,
    PRIMARY KEY (user_id, fact_index)
)
"""


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
        conn.execute(_SCHEMA)
        conn.commit()
        _schema_ready = True
        logger.info("RAG: таблица facts готова")
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
    user_id: int, queries: list, top_k: int = 30, limit: int = 50
) -> list:
    """Поиск сразу по нескольким запросам (один батч эмбеддингов). Берёт top_k на
    каждый запрос, склеивает без дублей, оставляет `limit` самых близких и
    возвращает их в авторском порядке (по fact_index) — так канон читается как
    написан, а не вперемешку."""
    vecs = _embed(list(queries), "query")
    best = {}  # fact_index -> (score, text)
    with _connect() as conn:
        for vec in vecs:
            qv = _vec(vec)
            rows = conn.execute(
                "SELECT fact_index, fact_text, 1 - (embedding <=> %s::vector) AS score "
                "FROM facts WHERE user_id = %s "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (qv, user_id, qv, top_k),
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
