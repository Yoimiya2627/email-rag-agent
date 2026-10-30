"""Summarize JSONL agent traces into operational metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.tracing import load_events, summarize_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize agent trace JSONL events")
    parser.add_argument("--input", default=None, help="Trace JSONL path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = summarize_events(load_events(args.input))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("Agent trace summary")
    print("-" * 60)
    print(f"runs               : {summary['runs']}")
    print(f"tool calls         : {summary['tool_calls']}")
    print(f"tool errors        : {summary['tool_errors']}")
    print(f"approval required  : {summary['approval_required']}")
    print(f"avg tool latency ms: {summary['avg_tool_latency_ms']}")


if __name__ == "__main__":
    main()
