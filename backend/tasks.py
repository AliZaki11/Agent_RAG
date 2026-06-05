from crewai import Task

def router_task(agent) -> Task:
    return Task(
        description=(
            "Classify the following question into exactly one route.\n\n"
            "Question: {question}\n\n"
            "Conversation memory:\n{memory}\n\n"
            "Rules:\n"
            "- 'rag'    → question is about uploaded documents, internal knowledge base, domain content.\n"
            "- 'web'    → question needs current events, live data, news, or anything time-sensitive.\n"
            "- 'memory' → question explicitly references the current conversation "
            "             (says 'you said', 'earlier', 'before', 'previously').\n\n"
            "Think step by step, then respond with ONLY valid JSON (no markdown, no extra text):\n"
            '{{"route": "rag" | "web" | "memory", "reason": "<one sentence>"}}'
        ),
        expected_output='JSON with keys "route" and "reason".',
        agent=agent,
    )

def retriever_task(agent, router_task_ref) -> Task:
    return Task(
        description=(
            "Retrieve information and answer the question based on the route decision.\n\n"
            "Question: {question}\n\n"
            "Conversation memory (use if route=memory):\n{memory}\n\n"
            "Instructions:\n"
            "- route=rag    → call hybrid_rag_retrieve with the question.\n"
            "- route=web    → call web_search with a focused search query.\n"
            "- route=memory → answer directly from the conversation memory above.\n"
            "- Always capture the raw retrieved text as context.\n"
            "- LANGUAGE: Detect the language of the Question. "
            "  If Arabic → answer in Arabic. If English → answer in English.\n"
            "- FORMATTING (mandatory):\n"
            "  * Do NOT use markdown bold (**text**) or headers (# ## ###).\n"
            "  * Use plain numbered lists (1. 2. 3.) or dashes (- ) for lists.\n"
            "  * Separate paragraphs with a blank line.\n\n"
            "Respond with ONLY valid JSON (no markdown, no extra text):\n"
            '{{"context": "<raw retrieved text>", "answer": "<your plain-text answer>"}}'
        ),
        expected_output='JSON with keys "context" and "answer".',
        context=[router_task_ref],
        agent=agent,
    )

def critic_task(agent, retriever_task_ref) -> Task:
    return Task(
        description=(
            "Review the retriever's answer and verify it is grounded in the context.\n\n"
            "Question: {question}\n\n"
            "Instructions:\n"
            "- Check every factual claim against the provided context.\n"
            "- If all claims are supported → grounded: true.\n"
            "- If any claim is unsupported → grounded: false.\n"
            "- Improve or correct the final_answer if needed.\n"
            "- LANGUAGE: Respond in the SAME language as the question.\n"
            "- FORMATTING for final_answer (mandatory):\n"
            "  * Do NOT use markdown bold (**text**) or headers (# ## ###).\n"
            "  * Use plain numbered lists or dashes for lists.\n"
            "  * Separate paragraphs with a blank line.\n\n"
            "Respond with ONLY valid JSON (no markdown, no extra text):\n"
            '{{"grounded": true, "final_answer": "<plain-text answer>", "confidence": 0.0}}'
        ),
        expected_output='JSON with keys "grounded", "final_answer", "confidence".',
        context=[retriever_task_ref],
        agent=agent,
    )
