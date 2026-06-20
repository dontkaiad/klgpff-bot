"""RAG для фактов: Voyage AI эмбеддинги + Qdrant векторный поиск.

Коллекция "facts" хранит по одному вектору на факт юзера. При недоступности
Qdrant / Voyage / зависимостей функции бросают исключение — вызывающая
сторона (claude_tg_bot) ловит и делает fallback на полный список фактов.

ENV:
- QDRANT_HOST (дефолт "qdrant"), QDRANT_PORT (дефолт 6333)
- VOYAGE_API_KEY
"""

import logging
import os
import uuid

logger = logging.getLogger(__name__)

COLLECTION = "facts"
EMBED_MODEL = "voyage-3-lite"  # нативная размерность 512
EMBED_DIM = 512

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")

try:
    import voyageai
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    _DEPS_OK = True
except Exception as e:  # noqa: BLE001 — модуль не должен валить импорт бота
    logger.warning(f"RAG deps unavailable, поиск отключён: {e}")
    _DEPS_OK = False

_qdrant = None
_voyage = None


def _ensure_clients():
    """Ленивая инициализация. Бросает RuntimeError если RAG не сконфигурирован."""
    global _qdrant, _voyage
    if not _DEPS_OK:
        raise RuntimeError("voyageai / qdrant-client не установлены")
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY не задан")
    if _voyage is None:
        _voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
    if _qdrant is None:
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        _ensure_collection(qc)
        _qdrant = qc
    return _qdrant, _voyage


def _ensure_collection(qc):
    names = {c.name for c in qc.get_collections().collections}
    if COLLECTION not in names:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info(f"создана Qdrant-коллекция {COLLECTION!r} (dim={EMBED_DIM})")


def _user_filter(user_id: int):
    return Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    )


def _point_id(user_id: int, idx: int) -> str:
    # детерминированный id — переиндексация перезаписывает те же точки
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"klgpff:{user_id}:{idx}"))


def _embed(texts: list, input_type: str) -> list:
    _, vo = _ensure_clients()
    result = vo.embed(texts, model=EMBED_MODEL, input_type=input_type)
    return result.embeddings


def index_facts(user_id: int, facts: list) -> None:
    """Эмбеддит все факты юзера и (пере)заливает их в Qdrant.

    Сначала удаляет прежние точки юзера, чтобы убранные факты не оставались
    в индексе.
    """
    qc, _ = _ensure_clients()
    qc.delete(collection_name=COLLECTION, points_selector=_user_filter(user_id))
    if not facts:
        logger.info(f"RAG: факты юзера {user_id} очищены из индекса")
        return
    vectors = _embed(list(facts), "document")
    points = [
        PointStruct(
            id=_point_id(user_id, i),
            vector=vec,
            payload={"user_id": user_id, "fact_text": fact, "fact_index": i},
        )
        for i, (fact, vec) in enumerate(zip(facts, vectors))
    ]
    qc.upsert(collection_name=COLLECTION, points=points)
    logger.info(f"RAG: проиндексировано {len(points)} фактов юзера {user_id}")


def search_facts(user_id: int, query: str, top_k: int = 10) -> list:
    """Возвращает релевантные query факты юзера (по убыванию схожести)."""
    qc, _ = _ensure_clients()
    qvec = _embed([query], "query")[0]
    response = qc.query_points(
        collection_name=COLLECTION,
        query=qvec,
        query_filter=_user_filter(user_id),
        limit=top_k,
    )
    return [p.payload["fact_text"] for p in response.points if p.payload]
