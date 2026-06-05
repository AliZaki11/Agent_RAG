from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

@dataclass
class MemoryEntry:
    question: str
    answer:   str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ShortMemory:
    def __init__(self, maxlen: int = 20):
        self._buf: deque[MemoryEntry] = deque(maxlen=maxlen)

    def add(self, question: str, answer: str) -> None:
        self._buf.append(MemoryEntry(question=question, answer=answer))

    def recent(self, n: int = 5) -> list[MemoryEntry]:
        return list(self._buf)[-n:]

    def format_for_prompt(self, n: int = 5) -> str:
        entries = self.recent(n)
        if not entries:
            return "(no prior conversation)"
        lines = ["[Conversation history]"]
        for e in entries:
            lines += [f"Q: {e.question}", f"A: {e.answer}"]
        return "\n".join(lines)

    def find(self, query: str) -> Optional[str]:
        for e in reversed(self._buf):
            if e.question.strip().lower() == query.strip().lower():
                return e.answer
        return None

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
