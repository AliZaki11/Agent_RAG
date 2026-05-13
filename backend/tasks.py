"""
CrewAI Tasks — Router · Retriever · Critic
"""
from crewai import Task


def router_task(agent) -> Task:
    return Task(
        description=(
            "Analyse the following question and decide the best source:\n\n"
            "Question: {question}\n\n"
            "Rules:\n"
            "- Return 'rag'    if the question is about documents, files, or domain knowledge.\n"
            "- Return 'web'    if the question needs current/live information.\n"
            "- Return 'memory' if the question references something from this conversation.\n\n"
            "Respond with ONLY valid JSON, no extra text:\n"
            '{"route": "rag" | "web" | "memory"}'
        ),
        expected_output='JSON object with key "route".',
        agent=agent,
    )


def retriever_task(agent, router_task_ref) -> Task:
    return Task(
        description=(
            "Using the route selected by the Router, retrieve information "
            "and answer the question.\n\n"
            "Question: {question}\n\n"
            "Instructions:\n"
            "- If route is 'rag':    use hybrid_rag_retrieve tool.\n"
            "- If route is 'web':    use web_search tool.\n"
            "- If route is 'memory': use the conversation context provided.\n"
            "- Always include the source context in your output.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"context": "...", "answer": "..."}'
        ),
        expected_output='JSON object with keys "context" and "answer".',
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
            "- If all claims are supported → set grounded: true.\n"
            "- If any claim is unsupported or hallucinated → set grounded: false.\n"
            "- Improve or correct the final_answer if needed.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"grounded": true|false, "final_answer": "...", "confidence": 0.0-1.0}'
        ),
        expected_output='JSON with keys "grounded", "final_answer", "confidence".',
        context=[retriever_task_ref],
        agent=agent,
    )
