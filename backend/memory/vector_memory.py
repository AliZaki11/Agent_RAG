import uuid, logging
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from backend.config import PINECONE_API_KEY, MEMORY_INDEX, EMBEDDING_DIM
from backend.utils import embed

logger     = logging.getLogger(__name__)
_pc_inst   = None
_idx_inst  = None

def _ensure_index(pc: Pinecone, name: str) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if name not in existing:
        logger.info(f"Creating memory index '{name}'…")
        pc.create_index(
            name=name, dimension=EMBEDDING_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

def _get_index():
    global _pc_inst, _idx_inst
    if _idx_inst is None:
        _pc_inst = Pinecone(api_key=PINECONE_API_KEY)
        _ensure_index(_pc_inst, MEMORY_INDEX)
        _idx_inst = _pc_inst.Index(MEMORY_INDEX)
    return _idx_inst

def store_memory(text: str, meta: Optional[dict] = None) -> str:
    mem_id = str(uuid.uuid4())
    emb    = embed([text])[0]
    meta   = {**(meta or {}), "text": text}
    _get_index().upsert([(mem_id, emb, meta)])
    return mem_id

def retrieve_memory(query: str, top_k: int = 3) -> list[str]:
    q_emb = embed([query])[0]
    res   = _get_index().query(vector=q_emb, top_k=top_k, include_metadata=True)
    return [m["metadata"]["text"] for m in res["matches"]]

def delete_memory(mem_id: str) -> None:
    _get_index().delete(ids=[mem_id])
