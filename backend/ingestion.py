from __future__ import annotations
import io, uuid, logging
from pathlib import Path
from typing import Union

from pinecone import Pinecone, ServerlessSpec
from backend.config import PINECONE_API_KEY, PINECONE_INDEX, EMBEDDING_DIM
from backend.utils import embed
from backend.tools.rag_tool import build_bm25

logger     = logging.getLogger(__name__)
BATCH_SIZE = 100
_all_ingested_texts: list[str] = []

_pc_inst   = None
_idx_inst  = None

def _ensure_index(pc: Pinecone, name: str) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if name not in existing:
        logger.info(f"Creating Pinecone index '{name}'…")
        pc.create_index(
            name=name, dimension=EMBEDDING_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

def _get_index():
    global _pc_inst, _idx_inst
    if _idx_inst is None:
        _pc_inst = Pinecone(api_key=PINECONE_API_KEY)
        _ensure_index(_pc_inst, PINECONE_INDEX)
        _idx_inst = _pc_inst.Index(PINECONE_INDEX)
    return _idx_inst


# ── Text extraction ──────────────────────────────────────────────

def extract_text_from_bytes(filename: str, content: bytes) -> str:
    """Extract plain text from file bytes. Supports txt, md, pdf."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages  = [p.extract_text() or "" for p in reader.pages]
            text   = "\n\n".join(p.strip() for p in pages if p.strip())
            if text.strip():
                return text
            logger.warning("PDF text extraction returned empty — falling back to raw decode")
        except Exception as exc:
            logger.warning(f"PDF extraction failed ({exc}), falling back to raw decode")
    return content.decode("utf-8", errors="replace")


def extract_text_from_path(path: Union[str, Path]) -> str:
    path = Path(path)
    return extract_text_from_bytes(path.name, path.read_bytes())


# ── Chunking ─────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Sliding-window word chunker with overlap."""
    words = text.split()
    step  = max(1, size - overlap)
    return [
        " ".join(words[i: i + size])
        for i in range(0, len(words), step)
        if " ".join(words[i: i + size]).strip()
    ]


# ── Upsert ───────────────────────────────────────────────────────

def ingest(chunks: list[str], namespace: str = "") -> int:
    if not chunks:
        return 0
    global _all_ingested_texts
    index = _get_index()
    total = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch      = chunks[start: start + BATCH_SIZE]
        embeddings = embed(batch)
        vectors    = [(str(uuid.uuid4()), emb, {"text": chunk})
                      for emb, chunk in zip(embeddings, batch)]
        kwargs: dict = {"vectors": vectors}
        if namespace:
            kwargs["namespace"] = namespace
        index.upsert(**kwargs)
        total += len(vectors)
        _all_ingested_texts.extend(batch)
        logger.info(f"Upserted batch {start // BATCH_SIZE + 1}: {len(vectors)} vectors")

    build_bm25(_all_ingested_texts)
    logger.info(f" Ingest complete: {total} vectors, BM25 rebuilt.")
    return total


def ingest_file(path: Union[str, Path], **kwargs) -> int:
    text   = extract_text_from_path(path)
    chunks = chunk_text(text)
    logger.info(f"Ingesting '{path}' → {len(chunks)} chunks")
    return ingest(chunks, **kwargs)
