import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

def _require(key: str) -> str:
    v = os.getenv(key)
    if not v:
        sys.exit(f"[config]  Missing env var: {key}  →  Add it to your .env file.")
    return v

OPENROUTER_API_KEY = _require("OPENROUTER_API_KEY")
PINECONE_API_KEY   = _require("PINECONE_API_KEY")
TAVILY_API_KEY     = _require("TAVILY_API_KEY")

PINECONE_INDEX  = os.getenv("PINECONE_INDEX",  "agentic-rag")
MEMORY_INDEX    = os.getenv("MEMORY_INDEX",    "memory-index")
REDIS_URL       = os.getenv("REDIS_URL",       "redis://localhost:6379/0")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536           # matches text-embedding-3-small

# deepseek-chat (V3) = fast & capable for chat — deepseek-r1 is a slow reasoning model
LLM_MODEL       = "openrouter/deepseek/deepseek-chat"

RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
