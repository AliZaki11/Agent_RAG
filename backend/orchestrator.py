from __future__ import annotations
import json, re, logging, time
from typing import Any

from crewai import Crew, Process
from backend.agents import router_agent, retriever_agent, critic_agent
from backend.tasks  import router_task, retriever_task, critic_task
from backend.memory.short_term    import ShortMemory
from backend.memory.vector_memory import store_memory
from backend.tools.rag_tool       import grounding_score, load_bm25_from_disk, preload_reranker

logger    = logging.getLogger(__name__)
MAX_ITERS = 3
short_mem = ShortMemory(maxlen=20)


def _parse(raw: str) -> dict[str, Any]:
    """Extract last JSON object from raw string. Falls back to json_repair."""
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        logger.warning(f"No JSON in output: {raw[:200]}")
        return {}
    candidate = match.group()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json
        return json.loads(repair_json(candidate))
    except Exception:
        logger.error(f"JSON parse failed: {raw[:200]}")
        return {}


def _extract_route(text: str) -> str:
    m = re.search(r'"route"\s*:\s*"(rag|web|memory)"', text, re.I)
    return m.group(1) if m else "rag"


def _build_crew():
    r_a   = router_agent()
    ret_a = retriever_agent()
    c_a   = critic_agent()
    rt    = router_task(r_a)
    ret_t = retriever_task(ret_a, rt)
    ct    = critic_task(c_a, ret_t)
    return Crew(
        agents=[r_a, ret_a, c_a],
        tasks=[rt, ret_t, ct],
        process=Process.sequential,
        verbose=True,
    )


def run(question: str) -> tuple[str, list[dict]]:
    t0 = time.perf_counter()

    # Exact-match cache hit
    cached = short_mem.find(question)
    if cached:
        return cached, [{"grounded": True, "route": "memory", "source": "cache"}]

    crew  = _build_crew()
    trace = []
    base_q = question

    for iteration in range(MAX_ITERS):
        q           = base_q if iteration == 0 else _escalate(base_q, iteration, trace)
        mem_context = short_mem.format_for_prompt(n=5)
        logger.info(f"Iter {iteration+1}/{MAX_ITERS} — q: {q[:80]}")

        try:
            raw     = crew.kickoff(inputs={"question": q, "memory": mem_context})
            raw_str = str(raw)
            parsed  = _parse(raw_str)
            parsed["route"]  = _extract_route(raw_str)
            parsed["reason"] = parsed.get("reason", "")
        except Exception as e:
            logger.error(f"Crew error: {e}")
            parsed = {"route": "rag", "reason": str(e)}

        parsed["iteration"] = iteration + 1
        trace.append(parsed)

        if parsed.get("grounded"):
            answer  = parsed.get("final_answer", "")
            conf    = parsed.get("confidence", 1.0)
            context = parsed.get("context", answer)
            g_score = grounding_score(answer, context)
            parsed["grounding_heuristic"] = g_score
            logger.info(f" Grounded (conf={conf}, heuristic={g_score}) in {time.perf_counter()-t0:.1f}s")
            short_mem.add(question, answer)
            try:
                store_memory(f"Q: {question}\nA: {answer}")
            except Exception:
                pass
            return answer, trace

    elapsed = time.perf_counter() - t0
    logger.warning(f" No grounded answer after {MAX_ITERS} iterations ({elapsed:.1f}s)")
    return (
        "I was unable to find a fully verified answer. "
        "Please rephrase your question or upload more relevant documents.",
        trace,
    )


def _escalate(question: str, iteration: int, trace: list[dict]) -> str:
    last   = trace[-1] if trace else {}
    ctx    = last.get("context", "")
    conf   = last.get("confidence", 0)
    if iteration == 1:
        if len(ctx.split()) < 30:
            return question + " — search more broadly and retrieve additional documents."
        return question + " — cross-verify using web search."
    return question + " — use web search only, ignore the local knowledge base."
