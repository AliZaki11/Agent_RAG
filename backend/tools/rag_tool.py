"""
RAG Tool — Hybrid Dense + BM25 Retrieval + Cross-Encoder Reranking
"""
from __future__ import annotations

import logging
from typing import Optional

from pinecone import Pinecone
from backend.config import PINECONE_API_KEY, PINECONE_INDEX, RERANKER_MODEL
from backend.utils import embed

logger = logging.getLogger(__name__)

# ── Lazy singletons ─────────────────────────────────────────────
_pc:        Optional[Pinecone]    = None
_index                             = None
_reranker                          = None

# ── BM25 state (built at ingest time) ──────────────────────────
_bm25_corpus: list[str] = []
_bm25                    = None


def _get_index():
    global _pc, _index
    if _index is None:
        _pc    = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX)
        logger.info(f"Pinecone index '{PINECONE_INDEX}' connected.")
    return _index


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def build_bm25(texts: list[str]) -> None:
    """Call this after ingesting new documents."""
    global _bm25_corpus, _bm25
    try:
        from rank_bm25 import BM25Okapi
        _bm25_corpus = texts
        _bm25        = BM25Okapi([t.split() for t in texts])
        logger.info(f"BM25 index built on {len(texts)} documents.")
    except ImportError:
        logger.warning("rank-bm25 not installed. BM25 disabled.")


# ── Search functions ─────────────────────────────────────────────

def dense_search(query: str, top_k: int = 10) -> list[dict]:
    q_emb = embed([query])[0]
    res   = _get_index().query(vector=q_emb, top_k=top_k, include_metadata=True)
    return [
        {"id": m["id"], "text": m["metadata"].get("text", ""), "score": m["score"]}
        for m in res["matches"]
    ]


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    if _bm25 is None:
        return []
    scores  = _bm25.get_scores(query.split())
    ranked  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"id": str(i), "text": _bm25_corpus[i], "score": float(scores[i])}
        for i in ranked
    ]


def hybrid_retrieve(query: str, alpha: float = 0.7, top_k: int = 8) -> list[str]:
    """
    Reciprocal Rank Fusion of dense + BM25,
    followed by Cross-Encoder reranking. Returns top-5 texts.
    """
    dense  = dense_search(query, top_k=top_k)
    sparse = bm25_search(query,  top_k=top_k)

    # ── RRF scoring ──
    scores: dict[str, float] = {}
    for i, d in enumerate(dense):
        scores[d["id"]] = scores.get(d["id"], 0) + alpha * (1 / (i + 1))
    for i, s in enumerate(sparse):
        scores[s["id"]] = scores.get(s["id"], 0) + (1 - alpha) * (1 / (i + 1))

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]   # type: ignore

    id2text    = {x["id"]: x["text"] for x in dense + sparse}
    candidates = [id2text[rid] for rid in ranked_ids if rid in id2text]

    if not candidates:
        return []

    # ── Cross-Encoder rerank ──
    reranker  = _get_reranker()
    pairs     = [(query, c) for c in candidates]
    rr_scores = reranker.predict(pairs)

    ranked = [
        text for _, text in
        sorted(zip(rr_scores, candidates), reverse=True)
    ]
    return ranked[:5]
