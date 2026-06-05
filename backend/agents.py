from crewai import Agent, LLM
from crewai.tools import tool
from backend.config import OPENROUTER_API_KEY, LLM_MODEL
from backend.tools.rag_tool import hybrid_retrieve
from backend.tools.web_tool import web_search as _web_search

# LLM via OpenRouter — LiteLLM prefix "openrouter/" is required by CrewAI
llm = LLM(
    model    = LLM_MODEL,   # e.g. "openrouter/deepseek/deepseek-chat"
    api_key  = OPENROUTER_API_KEY,
    base_url = "https://openrouter.ai/api/v1",
    temperature = 0,
)

@tool("hybrid_rag_retrieve")
def rag_tool(query: str) -> str:
    """
    Retrieve the most relevant document chunks from the vector knowledge base
    using hybrid dense + sparse retrieval and cross-encoder reranking.
    Use when the question relates to uploaded documents or domain knowledge.
    """
    chunks, _ = hybrid_retrieve(query)
    return "\n---\n".join(chunks) if chunks else "No relevant documents found."

@tool("web_search")
def web_tool(query: str) -> str:
    """
    Search the web for current or general information.
    Use when the question requires up-to-date data not in the knowledge base.
    """
    return _web_search(query)

def router_agent() -> Agent:
    return Agent(
        role      = "Query Router",
        goal      = "Classify the user's question into the correct information source.",
        backstory = (
            "You are a routing specialist. You analyse questions and decide "
            "whether to answer from the local knowledge base (rag), "
            "the internet (web), or recent conversation history (memory)."
        ),
        llm=llm, verbose=False,
    )

def retriever_agent() -> Agent:
    return Agent(
        role      = "Information Retriever",
        goal      = "Retrieve the most relevant information and formulate a precise answer.",
        backstory = (
            "You are a skilled researcher. You use the appropriate tool to retrieve "
            "information and synthesise a clear, grounded answer."
        ),
        tools=[rag_tool, web_tool],
        llm=llm, verbose=True,
    )

def critic_agent() -> Agent:
    return Agent(
        role      = "Answer Critic",
        goal      = "Verify the answer is grounded in retrieved evidence.",
        backstory = (
            "You are a rigorous fact-checker. You examine the retriever's answer "
            "and verify every claim against the provided context."
        ),
        llm=llm, verbose=False,
    )
