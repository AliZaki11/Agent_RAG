"""
Question Generation — builds study questions, MCQs, and flashcards.
"""
from __future__ import annotations
import json, logging, re
from typing import Literal

import requests
from backend.config       import OPENROUTER_API_KEY, LLM_MODEL
from backend.tools.rag_tool import hybrid_retrieve

logger = logging.getLogger(__name__)
QuestionType = Literal["factual", "mcq", "flashcard", "analytical"]

# ── LLM call ─────────────────────────────────────────────────────

def _llm(prompt: str) -> str:
    url     = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://agentic-rag.app",
        "X-Title":       "Agentic RAG",
    }
    # Strip LiteLLM prefix — REST API uses plain model names
    model = LLM_MODEL.removeprefix("openrouter/")
    resp  = requests.post(
        url, headers=headers,
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.7},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ── Prompts ───────────────────────────────────────────────────────

_PROMPTS: dict[QuestionType, str] = {

"factual": """You are an expert educator. Based ONLY on the context below, generate {count} factual questions with short, direct answers.

Context:
{context}

Return ONLY a valid JSON array (no markdown, no code fences):
[
  {{"question": "...", "answer": "..."}},
  ...
]""",

"mcq": """You are an expert educator. Based ONLY on the context below, generate {count} multiple-choice questions. Each must have exactly 4 options (A,B,C,D) and one correct answer.

Context:
{context}

Return ONLY a valid JSON array (no markdown, no code fences):
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A",
    "explanation": "..."
  }},
  ...
]""",

"flashcard": """You are an expert educator. Based ONLY on the context below, generate {count} study flashcards. Each has a concise front (concept/term) and detailed back (definition/explanation).

Context:
{context}

Return ONLY a valid JSON array (no markdown, no code fences):
[
  {{"front": "...", "back": "..."}},
  ...
]""",

"analytical": """You are an expert educator. Based ONLY on the context below, generate {count} deep analytical questions requiring critical thinking. Include a model answer.

Context:
{context}

Return ONLY a valid JSON array (no markdown, no code fences):
[
  {{"question": "...", "model_answer": "..."}},
  ...
]""",
}

# ── JSON extractor ────────────────────────────────────────────────

def _extract_json(raw: str) -> list:
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    m   = re.search(r'\[.*\]', raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array found:\n{raw[:400]}")
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(m.group()))
        except Exception:
            raise ValueError(f"Could not parse JSON:\n{raw[:400]}")

# ── Public API ────────────────────────────────────────────────────

def generate_questions(
    topic:         str          = "",
    question_type: QuestionType = "factual",
    count:         int          = 5,
) -> list[dict]:
    count = max(1, min(count, 20))
    query = topic.strip() if topic.strip() else "main topics key concepts summary"

    # hybrid_retrieve returns (chunks, grounding_score) — unpack correctly
    try:
        chunks, _ = hybrid_retrieve(query, top_k=min(count * 2, 20))
    except Exception as exc:
        logger.warning(f"Retrieval failed: {exc}")
        chunks = []

    if not chunks:
        raise ValueError(
            "No documents found in the knowledge base. "
            "Please upload at least one document before generating questions."
        )

    context = "\n\n---\n\n".join(
        c.get("text", "") if isinstance(c, dict) else str(c)
        for c in chunks[:10]
    )[:6000]

    prompt    = _PROMPTS[question_type].format(count=count, context=context)
    logger.info(f"Generating {count} {question_type} questions ({len(context)} chars context)")
    raw       = _llm(prompt)
    questions = _extract_json(raw)
    logger.info(f"Generated {len(questions)} questions.")
    return questions
