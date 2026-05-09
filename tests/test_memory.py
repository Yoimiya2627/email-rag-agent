"""Tests for core/memory.py and api/main.py session-store helper."""
import threading

import pytest

from core.memory import ConversationMemory


def test_default_window_holds_five_turns():
    mem = ConversationMemory()
    assert mem.max_turns == 5


def test_sliding_window_drops_oldest_pairs():
    mem = ConversationMemory(max_turns=5)
    for i in range(12):
        role = "user" if i % 2 == 0 else "assistant"
        mem.add(role, f"msg-{i}")

    msgs = mem.to_messages()
    assert len(msgs) == 10
    # Oldest two messages (msg-0, msg-1) are dropped; first kept message is msg-2.
    assert msgs[0]["content"] == "msg-2"
    assert msgs[-1]["content"] == "msg-11"


def test_to_messages_preserves_role_and_order():
    mem = ConversationMemory()
    mem.add("user", "hello")
    mem.add("assistant", "hi there")
    msgs = mem.to_messages()
    assert msgs == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_clear_resets_state():
    mem = ConversationMemory()
    mem.add("user", "x")
    mem.add("assistant", "y")
    mem.clear()
    assert len(mem) == 0
    assert mem.to_messages() == []


def test_get_session_isolates_session_ids_and_returns_same_instance():
    from api.main import _get_session

    a = _get_session("session-a")
    b = _get_session("session-b")
    a_again = _get_session("session-a")

    assert a is a_again, "Same session_id must return the same memory instance"
    assert a is not b, "Different session_ids must yield different memory instances"


def test_concurrent_add_does_not_lose_messages():
    mem = ConversationMemory(max_turns=10000)  # window large enough so nothing trims
    threads = []
    per_thread = 50

    def writer(tid: int):
        for i in range(per_thread):
            mem.add("user", f"t{tid}-m{i}")

    for t in range(8):
        threads.append(threading.Thread(target=writer, args=(t,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(mem) == 8 * per_thread


def test_max_turns_one_keeps_only_latest_pair():
    mem = ConversationMemory(max_turns=1)
    mem.add("user", "u1")
    mem.add("assistant", "a1")
    mem.add("user", "u2")
    mem.add("assistant", "a2")

    msgs = mem.to_messages()
    assert len(msgs) == 2  # max_messages = max_turns * 2 = 2
    assert msgs[0]["content"] == "u2"
    assert msgs[1]["content"] == "a2"
