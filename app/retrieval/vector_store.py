"""Chroma vector store wrapper.

Supports either a local on-disk Chroma (PersistentClient) or Chroma Cloud
via `chromadb.CloudClient` depending on environment configuration in
`app.core.config.settings`.

When `settings.chroma_api_key` is set the code attempts to connect to the
CloudClient using tenant/database; otherwise it falls back to a local
PersistentClient using `settings.chroma_db_path`.
"""
import logging
import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings

logger = logging.getLogger(__name__)

_client = None
_collection = None


def get_collection():
    """Return a Chroma collection; prefer CloudClient when configured."""
    global _client, _collection
    if _collection is not None:
        return _collection

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )

    # Prefer Chroma Cloud when API key is present
    if settings.chroma_api_key:
        try:
            # CloudClient may accept host in newer SDKs; we keep the call
            # conservative and pass the required auth/tenant/database fields.
            _client = chromadb.CloudClient(
                api_key=settings.chroma_api_key,
                tenant=settings.chroma_tenant,
                database=settings.chroma_database,
            )
            _collection = _client.get_or_create_collection(
                name="drug_label_chunks",
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Connected to Chroma Cloud collection 'drug_label_chunks'")
            return _collection
        except Exception as exc:
            logger.warning("Chroma Cloud connection failed, falling back to local: %s", exc)

    # Fallback to local persistent client
    _client = chromadb.PersistentClient(path=settings.chroma_db_path)
    _collection = _client.get_or_create_collection(
        name="drug_label_chunks",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Using local Chroma collection at %s", settings.chroma_db_path)
    return _collection


_known_drugs = None



def get_known_drugs(user_id: int | None = None) -> list[str]:
    try:
        collection = get_collection()
        where = {"user_id": user_id} if user_id is not None else None
        res = collection.get(where=where, include=["metadatas"])
        if res and res.get("metadatas"):
            drugs = list(set(m["drug_name"] for m in res["metadatas"] if m and "drug_name" in m))
            return sorted(drugs)
    except Exception:
        pass
    return []


def normalize_drug_name(name: str | list | tuple | None, user_id: int | None = None) -> str | None:
    if not name:
        return None
    if isinstance(name, (list, tuple)):
        # If a list of drugs is passed, take the first valid non-empty drug name
        valid_items = [str(x).strip() for x in name if x and str(x).strip()]
        if not valid_items:
            return None
        name = valid_items[0]
    elif not isinstance(name, str):
        name = str(name)

    name_lower = name.lower().strip()
    if not name_lower:
        return None

    knowns = get_known_drugs(user_id=user_id)
    # Prefer exact match with uppercase canonical drug name
    for known in knowns:
        if known.isupper() and known.lower().strip() == name_lower:
            return known
    for known in knowns:
        if known.lower().strip() == name_lower:
            return known
    return name.strip().upper()


def list_indexed_drugs(user_id: int | None = None) -> list[dict]:
    """Return a list of indexed drugs with chunk count and stats for a specific user."""
    try:
        collection = get_collection()
        where = {"user_id": user_id} if user_id is not None else None
        res = collection.get(where=where, include=["metadatas"])
        if not res or not res.get("metadatas"):
            return []

        drug_counts: dict[str, int] = {}
        for m in res["metadatas"]:
            if m and "drug_name" in m:
                # If user_id is provided, double check it matches
                if user_id is not None and m.get("user_id") != user_id:
                    continue
                d = m["drug_name"].strip().upper()
                drug_counts[d] = drug_counts.get(d, 0) + 1

        return [
            {"drug_name": d, "chunk_count": count}
            for d, count in sorted(drug_counts.items(), key=lambda x: x[0])
        ]
    except Exception as exc:
        logger.warning("Failed to list indexed drugs: %s", exc)
        return []


def delete_drug_documents(drug_name: str, user_id: int | None = None, document_id: str | None = None) -> int:
    """Delete all chunks for a specified drug/document belonging to a specific user."""
    global _known_drugs
    _known_drugs = None
    collection = get_collection()

    where = {}
    if user_id is not None:
        where["user_id"] = user_id

    all_res = collection.get(where=where if where else None, include=["metadatas"])
    all_ids = all_res.get("ids") or []
    all_metas = all_res.get("metadatas") or []

    target_clean = drug_name.strip().lower()
    ids_to_delete = []
    for i, m in enumerate(all_metas):
        if not m:
            continue
        # Verify ownership match
        if user_id is not None and m.get("user_id") != user_id:
            continue
        if document_id and m.get("document_id") != document_id:
            continue
        if m.get("drug_name", "").strip().lower() == target_clean:
            ids_to_delete.append(all_ids[i])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for drug '%s' (user_id=%s)", len(ids_to_delete), drug_name, user_id)

    # Rebuild BM25 keyword index
    try:
        from app.retrieval.keyword_index import ensure_bm25_index
        ensure_bm25_index(force_rebuild=True)
    except Exception as exc:
        logger.warning("Failed to rebuild BM25 index after deleting '%s': %s", drug_name, exc)

    return len(ids_to_delete)


def upsert_chunks(chunks_with_ids: list[dict]):
    """chunks_with_ids: [{id, text, metadata}, ...]
    metadata must include: drug_name, section, page_number, document_id, is_table, is_boxed_warning, and optional user_id
    """
    global _known_drugs
    _known_drugs = None  # Invalidate cache
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks_with_ids],
        documents=[c["text"] for c in chunks_with_ids],
        metadatas=[c["metadata"] for c in chunks_with_ids],
    )


def vector_search(query: str, top_k: int, drug_name: str | list[str] | tuple | None = None, user_id: int | None = None) -> list[dict]:
    collection = get_collection()

    target_drugs = []
    if isinstance(drug_name, (list, tuple)):
        for d in drug_name:
            nd = normalize_drug_name(d, user_id=user_id)
            if nd and nd not in target_drugs:
                target_drugs.append(nd)
    elif drug_name:
        nd = normalize_drug_name(drug_name, user_id=user_id)
        if nd:
            target_drugs.append(nd)

    where_conditions = []
    if len(target_drugs) == 1:
        where_conditions.append({"drug_name": target_drugs[0]})
    elif len(target_drugs) > 1:
        where_conditions.append({"drug_name": {"$in": target_drugs}})

    if user_id is not None:
        where_conditions.append({"user_id": user_id})

    if len(where_conditions) == 1:
        where = where_conditions[0]
    elif len(where_conditions) > 1:
        where = {"$and": where_conditions}
    else:
        where = None

    try:
        results = collection.query(query_texts=[query], n_results=top_k, where=where)
    except Exception as exc:
        logger.warning("Chroma vector query error: %s", exc)
        return []

    out = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
    return out


