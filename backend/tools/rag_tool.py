"""
RAG Tool — Hybrid Dense + BM25 + CrossEncoder reranking
Fixes: auto-create Pinecone index, persistent BM25, prefixed IDs, cached embed,
       grounding heuristic, preload reranker, dynamic top_k.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from backend.config import (PINECONE_API_KEY, PINECONE_INDEX,
                             RERANKER_MODEL, EMBEDDING_DIM)
from backend.utils import cached_embed

logger = logging.getLogger(__name__)
BM25_PATH = Path("bm25_corpus.json")

_pc:       Optional[Pinecone] = None
_index                         = None
_reranker                      = None
_bm25_corpus: list[str]        = []
_bm25                          = None


# ── Pinecone (auto-create) ───────────────────────────────────────

def _ensure_index(pc: Pinecone, name: str) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if name not in existing:
        logger.info(f"Creating Pinecone index '{name}' ({EMBEDDING_DIM}d cosine)…")
        pc.create_index(
            name      = name,
            dimension = EMBEDDING_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info(f"Index '{name}' created.")

def _get_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _ensure_index(_pc, PINECONE_INDEX)
        _index = _pc.Index(PINECONE_INDEX)
        logger.info(f"Pinecone '{PINECONE_INDEX}' connected.")
    return _index


# ── Reranker ─────────────────────────────────────────────────────

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def preload_reranker() -> None:
    """Call at startup to avoid first-request latency spike."""
    _get_reranker()


# ── BM25 persistence ─────────────────────────────────────────────

def _save_corpus(texts: list[str]) -> None:
    try:
        BM25_PATH.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error(f"BM25 save failed: {e}")

def _load_corpus() -> list[str]:
    if not BM25_PATH.exists():
        return []
    try:
        return json.loads(BM25_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"BM25 load failed: {e}")
        return []

def _build_bm25_index(texts: list[str]) -> None:
    global _bm25_corpus, _bm25
    if not texts:
        return
    try:
        from rank_bm25 import BM25Okapi
        _bm25_corpus = texts
        _bm25        = BM25Okapi([t.split() for t in texts])
        logger.info(f"BM25 built on {len(texts)} docs.")
    except ImportError:
        logger.warning("rank-bm25 not installed — BM25 disabled.")

def build_bm25(texts: list[str]) -> None:
    _build_bm25_index(texts)
    _save_corpus(texts)

def load_bm25_from_disk() -> None:
    texts = _load_corpus()
    if texts:
        _build_bm25_index(texts)


# ── Grounding heuristic ──────────────────────────────────────────

def grounding_score(answer: str, context: str) -> float:
    if not answer or not context:
        return 0.0
    a = set(answer.lower().split())
    c = set(context.lower().split())
    return round(len(a & c) / max(len(a), 1), 3)


# ── Search ───────────────────────────────────────────────────────

def dense_search(query: str, top_k: int = 10, namespace: str = "") -> list[dict]:
    q_emb  = list(cached_embed(query))         # cached_embed returns tuple → convert
    kwargs: dict = dict(vector=q_emb, top_k=top_k, include_metadata=True)
    if namespace:
        kwargs["namespace"] = namespace
    res = _get_index().query(**kwargs)
    return [
        {"id": f"dense_{m['id']}",
         "text": m["metadata"].get("text", ""),
         "score": m["score"]}
        for m in res["matches"]
    ]

def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(query.split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"id": f"sparse_{i}", "text": _bm25_corpus[i], "score": float(scores[i])}
            for i in ranked]


# ── Hybrid retrieve ──────────────────────────────────────────────

def hybrid_retrieve(
    query:     str,
    alpha:     float = 0.7,
    top_k:     int   = 0,
    namespace: str   = "",
) -> tuple[list[str], float]:
    """
    Returns (top_chunks: list[str], grounding_score: float).
    """
    if top_k == 0:
        top_k = 20 if len(query.split()) > 20 else 10

    dense  = dense_search(query, top_k=top_k, namespace=namespace)
    sparse = bm25_search(query,  top_k=top_k)

    rrf: dict[str, float] = {}
    for i, d in enumerate(dense):
        rrf[d["id"]] = rrf.get(d["id"], 0) + alpha * (1 / (i + 1))
    for i, s in enumerate(sparse):
        rrf[s["id"]] = rrf.get(s["id"], 0) + (1 - alpha) * (1 / (i + 1))

    ranked_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]   # type: ignore
    id2text    = {x["id"]: x["text"] for x in dense + sparse}
    candidates = [id2text[rid] for rid in ranked_ids if rid in id2text]

    if not candidates:
        return [], 0.0

    reranker  = _get_reranker()
    rr_scores = reranker.predict([(query, c) for c in candidates])
    ranked    = [t for _, t in sorted(zip(rr_scores, candidates), reverse=True)]
    top       = ranked[:5]

    g_score = grounding_score(query, " ".join(top))
    return top, g_score
