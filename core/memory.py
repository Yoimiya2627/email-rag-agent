"""
Sliding-window conversation memory for multi-turn chat.
Keeps the last max_turns (user, assistant) pairs in context.
"""
import threading
from dataclasses import dataclass, field
from typing import List


@dataclass
class Turn:
    role: str   # "user" or "assistant"
    content: str


class ConversationMemory:
    """Thread-safe sliding window. FastAPI runs requests on a thread pool, and
    the same session_id can be touched concurrently (e.g. background streaming
    persisting `assistant` while a new `user` request reads history). Without
    a lock the underlying list can be torn during slice-rebind."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: List[Turn] = []
        self._lock = threading.Lock()

    def add(self, role: str, content: str) -> None:
        with self._lock:
            self._turns.append(Turn(role=role, content=content))
            max_messages = self.max_turns * 2
            if len(self._turns) > max_messages:
                self._turns = self._turns[-max_messages:]

    def to_messages(self) -> List[dict]:
        with self._lock:
            return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self) -> None:
        with self._lock:
            self._turns = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)
