import os, logging
from backend.config import TAVILY_API_KEY

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
from langchain_community.tools.tavily_search import TavilySearchResults

logger  = logging.getLogger(__name__)
_search = TavilySearchResults(k=5)

def web_search(query: str) -> str:
    try:
        results = _search.run(query)
        if isinstance(results, list):
            return "\n\n".join(
                f"[{i+1}] {r.get('url','')}\n{r.get('content','').strip()}"
                for i, r in enumerate(results)
            )
        return str(results)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"[web_search] Failed: {e}"
