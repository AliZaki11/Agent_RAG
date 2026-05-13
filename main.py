"""
FastAPI Application — Agentic RAG API
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.orchestrator import run, short_mem
from backend.ingestion     import ingest, chunk_text

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Agentic RAG API starting…")
    yield
    logger.info("🛑 Agentic RAG API shutting down.")


app = FastAPI(
    title       = "Agentic RAG API",
    description = "Multi-agent Retrieval-Augmented Generation pipeline",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Schemas ──────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:  str
    namespace: Optional[str] = ""

class QueryResponse(BaseModel):
    answer:     str
    trace:      list[dict]
    grounded:   bool
    iterations: int

class IngestRequest(BaseModel):
    texts:     list[str]
    namespace: Optional[str] = ""


# ── Routes ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "memory_entries": len(short_mem)}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question cannot be empty")

    answer, trace = run(req.question)
    grounded = any(t.get("grounded") for t in trace)

    return QueryResponse(
        answer     = answer,
        trace      = trace,
        grounded   = grounded,
        iterations = len(trace),
    )


@app.post("/ingest")
async def ingest_texts(req: IngestRequest, background_tasks: BackgroundTasks):
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts list cannot be empty")

    background_tasks.add_task(ingest, req.texts, req.namespace)
    return {"status": "ingestion queued", "chunks": len(req.texts)}


@app.post("/ingest/file")
async def ingest_file_upload(
    file: UploadFile = File(...),
    namespace: str   = "",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    content = await file.read()
    text    = content.decode("utf-8", errors="replace")
    chunks  = chunk_text(text)

    background_tasks.add_task(ingest, chunks, namespace)
    return {
        "status":    "ingestion queued",
        "filename":  file.filename,
        "chunks":    len(chunks),
    }


@app.delete("/memory")
async def clear_memory():
    short_mem.clear()
    return {"status": "short-term memory cleared"}
