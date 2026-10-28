"""Astra review 2026-10-28: is Qwen 9B's 24K cut a lack of reply room or a runaway reasoning? Replays the exact r2 H5
request (input 1 723 tokens, reply cut at 24 576) with the same seed and a big window, changing ONLY num_predict:
24 576 (as in r2) and 55 296 (approved budget). If the bigger cap ends in a tool call, the room was the problem; if it
is cut again or ends without an action, more budget only buys more thinking.

  python probe_qwen9b_reply_cap.py   -> ../runs/probe-qwen9b-reply-cap-20261028.json
"""
import json
import time
import urllib.request
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
base = json.loads((RUNS / "qwen9b-cut-request-20261028.json").read_text(encoding="utf-8"))
rows = []
for cap in (24576, 55296):
    req = dict(base, stream=False, keep_alive="10m", options=dict(base["options"], num_ctx=110592, num_predict=cap))
    t = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(req).encode(),
                                                                 {"Content-Type": "application/json"}), timeout=3600).read())
    m = d.get("message") or {}
    row = dict(num_predict=cap, seconds=round(time.time() - t), done_reason=d.get("done_reason"), input=d.get("prompt_eval_count"),
               reply=d.get("eval_count"), thinking_chars=len(m.get("thinking") or ""), content_chars=len(m.get("content") or ""),
               tool_call=[c["function"]["name"] for c in m.get("tool_calls") or []],
               thinking_tail=(m.get("thinking") or "")[-400:])
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "thinking_tail"}), flush=True)
(RUNS / "probe-qwen9b-reply-cap-20261028.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
