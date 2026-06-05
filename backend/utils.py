import time, logging, requests
from functools import lru_cache
from backend.config import OPENROUTER_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

def embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    if not texts:
        return []
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://agentic-rag.app",
        "X-Title":       "Agentic RAG",
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers,
                              json={"model": EMBEDDING_MODEL, "input": texts},
                              timeout=30)
            r.raise_for_status()
            return [x["embedding"] for x in r.json()["data"]]
        except requests.HTTPError as e:
            if r.status_code in (401, 403):
                raise RuntimeError("Invalid OPENROUTER_API_KEY") from e
            logger.warning(f"Embed HTTP {r.status_code} attempt {attempt+1}")
        except Exception as e:
            logger.warning(f"Embed attempt {attempt+1} failed: {e}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Embedding failed after {retries} attempts")

@lru_cache(maxsize=2000)
def cached_embed(text: str) -> tuple:
    """Single-text embed with LRU cache. Returns tuple (hashable for lru_cache)."""
    return tuple(embed([text])[0])
