"""
Sliding-window conversation memory for multi-turn chat.
Keeps the last max_turns (user, assistant) pairs in context.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Turn:
    role: str   # "user" or "assistant"
    content: str


class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    def add(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        # Keep only the last max_turns pairs (2 messages per turn)
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def to_messages(self) -> List[dict]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self) -> None:
        self._turns = []

    def __len__(self) -> int:
        return len(self._turns)
