"""Verify /chat/stream emits tokens incrementally (real SSE), not in one burst."""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings as cfg

URL = f"{cfg.API_URL}/chat/stream"
payload = json.dumps({"query": "Q3预算评审会议的发件人是谁？", "session_id": "stream-test"}).encode("utf-8")

req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
t0 = time.time()
arrivals = []
with urllib.request.urlopen(req, timeout=120) as resp:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if not line.startswith("data: "):
            continue
        body = line[len("data: "):]
        dt = time.time() - t0
        arrivals.append((dt, body[:60]))

print(f"received {len(arrivals)} SSE events in {time.time()-t0:.2f}s")
for i, (dt, body) in enumerate(arrivals[:5]):
    print(f"  [{i}] t={dt:.2f}s  {body}")
print("  ...")
for i, (dt, body) in enumerate(arrivals[-3:]):
    print(f"  [-{3-i}] t={dt:.2f}s  {body}")

# Real streaming criterion: first token should arrive well before last token.
if len(arrivals) >= 2:
    spread = arrivals[-1][0] - arrivals[0][0]
    print(f"\nfirst→last spread: {spread:.2f}s")
    if spread > 0.5:
        print("✓ tokens arrived incrementally (real streaming)")
    else:
        print("✗ tokens arrived in a burst (fake streaming)")
