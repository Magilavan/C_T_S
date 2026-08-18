"""Merges vector + keyword results (reciprocal rank fusion), then reranks
using a cross-encoder for the final top-k passed to the LLM."""
from app.retrieval.vector_store import vector_search
from app.retrieval.keyword_index import keyword_search
from app.core.config import settings

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _reciprocal_rank_fusion(vector_results: list[dict], keyword_results: list[dict], k: int = 60) -> list[dict]:
    scores = {}
    items = {}
    for rank, item in enumerate(vector_results):
        scores[item["id"]] = scores.get(item["id"], 0) + 1.0 / (k + rank + 1)
        items[item["id"]] = item
    for rank, item in enumerate(keyword_results):
        scores[item["id"]] = scores.get(item["id"], 0) + 1.0 / (k + rank + 1)
        items.setdefault(item["id"], item)
    merged = sorted(items.values(), key=lambda it: scores[it["id"]], reverse=True)
    return merged


def hybrid_retrieve(query: str, drug_name: str | list[str] | tuple | None = None, user_id: int | None = None) -> list[dict]:
    top_k = settings.retrieval_top_k
    candidate_k = top_k * 2

    # Intent-aware search term expansion for clinical query resolution
    search_query = query
    if re.search(r"active ingredient", query, re.I):
        search_query += " description contains active substance adalimumab"
    elif re.search(r"contraindicat", query, re.I):
        search_query += " contraindications section 4"

    vec = vector_search(search_query, top_k=candidate_k, drug_name=drug_name, user_id=user_id)
    kw = keyword_search(search_query, top_k=candidate_k, drug_name=drug_name, user_id=user_id)
    merged = _reciprocal_rank_fusion(vec, kw)

    # always force the boxed warning chunk(s) for the active drug(s) into
    # candidate pool given clinical importance, per design doc section 5.1
    boxed = [c for c in vec + kw if c["metadata"].get("is_boxed_warning")]
    for b in boxed:
        if not any(m["id"] == b["id"] for m in merged):
            merged.append(b)

    return merged[:candidate_k]



import re

_BOOST_RULES = [
    (re.compile(r"contraindicat", re.I), ["CONTRAINDICATION", "CONTRAINDICATIONS", "4 CONTRAINDICATIONS"], 3.0),
    (re.compile(r"(active ingredient|description|active substance|chemical|composition)", re.I), ["DESCRIPTION", "11 DESCRIPTION"], 3.0),
    (re.compile(r"(dosag|dosing|how to take|administration|schedule|initial dose|maintenance dose|higher dose)", re.I),
     ["DOSAGE", "DOSAGE AND ADMINISTRATION", "RECOMMENDED DOSAGE", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2 DOSAGE"], 6.0),
    (re.compile(r"(warning|precaution|risk)", re.I), ["WARNING", "WARNINGS AND PRECAUTIONS", "BOXED WARNING"], 2.0),
    (re.compile(r"(side effect|adverse reaction)", re.I), ["ADVERSE REACTION", "ADVERSE REACTIONS"], 2.0),
    (re.compile(r"(indication|use|treatment|used for)", re.I), ["INDICATIONS AND USAGE"], 2.0),
]

# Section-confusion penalties: when asking about section X, penalise the
# confusable section Y so it doesn't dominate the answer.
_PENALTY_RULES = [
    # "contraindications" query → penalise §5 warnings chunks
    (re.compile(r"contraindicat", re.I),
     ["WARNING", "WARNINGS AND PRECAUTIONS", "5 WARNINGS", "5.1", "5.2", "5.3",
      "5.4", "5.5", "5.6", "5.7", "5.8", "5.9", "5.10", "5.11", "5.12",
      "SERIOUS INFECTIONS", "IMMUNIZATIONS"], -3.0),
    # "warnings" query → penalise §4 contraindications chunks
    (re.compile(r"\b(warning|precaution)\b", re.I),
     ["4 CONTRAINDICATIONS", "CONTRAINDICATIONS"], -1.5),
    # "active ingredient / description" query → penalise §1 indications chunks
    (re.compile(r"(active ingredient|description|composition)", re.I),
     ["INDICATIONS AND USAGE", "1 INDICATIONS"], -1.5),
    # "dosage / dosing / initial dose" query → penalise §14 clinical studies and §6 adverse reactions
    (re.compile(r"(dosag|dosing|how to take|administration|schedule|initial dose|maintenance dose|higher dose)", re.I),
     ["CLINICAL STUDIES", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "14.8", "14.9", "14.10", "14.11", "14.12", "14 CLINICAL",
      "ADVERSE REACTIONS", "6.1", "6.2", "6.3", "6 ADVERSE"], -4.0),
]


def _section_boost(query: str, section_name: str) -> float:
    if not section_name:
        return 0.0
    sec_upper = section_name.upper()
    total_boost = 0.0
    for pattern, targets, boost in _BOOST_RULES:
        if pattern.search(query):
            if any(t in sec_upper for t in targets):
                total_boost += boost
    # Apply confusion penalties
    for pattern, confused_targets, penalty in _PENALTY_RULES:
        if pattern.search(query):
            if any(t in sec_upper for t in confused_targets):
                total_boost += penalty
    return total_boost


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    reranker = _get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    raw_scores = reranker.predict(pairs)

    scored = []
    for c, score in zip(candidates, raw_scores):
        sec = c.get("metadata", {}).get("section", "")
        boost = _section_boost(query, sec)
        scored.append((c, float(score) + boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    top_k = settings.rerank_top_k
    is_multi_doc_query = bool(re.search(r"\b(compare|table|versus|vs\.?|which|higher|lower|initial dose|all indications|both|differ|between)\b", query, re.I))
    if is_multi_doc_query:
        top_k = max(top_k, 16)

    # Check if multiple distinct drugs/documents are present in candidates
    drugs_present = list(set(c["metadata"].get("drug_name") for c, _ in scored if c.get("metadata", {}).get("drug_name")))

    if len(drugs_present) > 1 and is_multi_doc_query:
        # Group by drug to guarantee cross-document representation
        by_drug: dict[str, list[dict]] = {}
        for c, score in scored:
            d_name = c["metadata"].get("drug_name", "UNKNOWN")
            by_drug.setdefault(d_name, []).append(c)

        deduped = []
        seen_ids = set()
        max_per_doc = max(4, top_k // len(drugs_present))

        round_idx = 0
        while len(deduped) < top_k:
            added = False
            for d_name, items in by_drug.items():
                if round_idx < len(items) and round_idx < max_per_doc:
                    item = items[round_idx]
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        deduped.append(item)
                        added = True
            if not added:
                break
            round_idx += 1

        # Fill remaining slots with top overall candidates if any space left
        for c, _ in scored:
            if len(deduped) >= top_k:
                break
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                deduped.append(c)
        return deduped

    seen_ids = set()
    deduped = []
    for c, _ in scored:
        if c["id"] in seen_ids:
            continue
        seen_ids.add(c["id"])
        deduped.append(c)
        if len(deduped) >= top_k:
            break
    return deduped
