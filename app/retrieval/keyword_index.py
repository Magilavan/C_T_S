"""Keyword-side of hybrid retrieval. Vector search alone misses exact terms
like drug names, mg values, and section numbers — BM25 catches those."""
from rank_bm25 import BM25Okapi
import re

_index = None
_corpus_meta = []  # parallel list of {id, text, metadata}


_STOP_WORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "is", "are", "was", "were", "what", "its", "for", "with", "on", "at", "by", "from", "it"}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9%.]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def build_index(all_chunks: list[dict] | None = None):
    """all_chunks: [{id, text, metadata}, ...] — builds BM25 index."""
    global _index, _corpus_meta
    if all_chunks is None:
        ensure_bm25_index(force_rebuild=True)
        return
    _corpus_meta = all_chunks
    tokenized = [_tokenize(c["text"]) for c in all_chunks]
    _index = BM25Okapi(tokenized)


def ensure_bm25_index(force_rebuild: bool = False):
    """Restores/rebuilds BM25 index from stored Chroma database chunks."""
    global _index, _corpus_meta
    if _index is not None and not force_rebuild:
        return

    try:
        from app.retrieval.vector_store import get_collection
        collection = get_collection()
        res = collection.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        if ids and docs:
            all_chunks = []
            for i in range(len(ids)):
                all_chunks.append({
                    "id": ids[i],
                    "text": docs[i] if (i < len(docs) and docs[i]) else "",
                    "metadata": metas[i] if (i < len(metas) and metas[i]) else {},
                })
            build_index(all_chunks)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to auto-build BM25 index from Chroma: %s", e)


def keyword_search(query: str, top_k: int, drug_name: str | list[str] | tuple | None = None, user_id: int | None = None) -> list[dict]:
    global _index, _corpus_meta
    if _index is None:
        ensure_bm25_index()

    if _index is None or not _corpus_meta:
        return []

    from app.retrieval.vector_store import normalize_drug_name
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

    scores = _index.get_scores(_tokenize(query))
    ranked = sorted(zip(_corpus_meta, scores), key=lambda x: x[1], reverse=True)
    out = []
    for chunk, score in ranked:
        # Enforce user isolation: if user_id is provided, only return chunks owned by this user
        if user_id is not None:
            chunk_user_id = chunk["metadata"].get("user_id")
            if chunk_user_id is not None and chunk_user_id != user_id:
                continue

        if target_drugs and chunk["metadata"].get("drug_name") not in target_drugs:
            continue
        if score <= 0:
            continue
        out.append({**chunk, "bm25_score": score})
        if len(out) >= top_k:
            break
    return out



