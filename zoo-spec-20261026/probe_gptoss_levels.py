"""gpt-oss latency/format probe v2 (Astra 2026-10-28, GPT-OSS-LATENCY.md): streaming, so a long reasoning is told
apart from no progress; reasoning low/medium/high x 2 contents x 2 seeds = 12 requests, 90 s cap each; byte-exact
check of the written file, not just HTTP 200; after a cap, a 1-token request measures whether the old generation
still occupies Ollama. Same tools and sampling as the harness (run_light.tool_specs, high_matrix gpt-oss options).

  python probe_gptoss_levels.py   -> ../runs/probe-gptoss-levels-20261028.json (rows written as they finish)
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_light  # noqa: E402
import high_matrix as hm  # noqa: E402

URL = "http://127.0.0.1:11434/api/chat"
CAP = 90
OUT = Path(__file__).resolve().parent.parent / "runs" / "probe-gptoss-levels-20261028.json"
OPTS = next(o for k, tag, cfg, think, o in hm.MATRIX if k == "gpt-oss-20b" and cfg == "high")
CASES = {
    "triple_quotes": '"""Candidate tests for the engine."""\nfrom engine import Engine\n\n\ndef test_submit():\n    """One id per key."""\n    assert Engine().submit("a", "k")\n',
    "escapes": 'TEXT = "line one\\nline two\\t\'quoted\' \\"double\\""\nPATH = r"C:\\temp\\x"\nprint(TEXT, PATH)\n',
}


def stream(req, cap):
    """Returns timing and the assembled message; stops reading at the cap (closing the connection)."""
    t0 = time.time()
    first = first_content = None
    thinking, content, calls, last = [], [], [], {}
    try:
        resp = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(dict(req, stream=True)).encode(),
                                                             {"Content-Type": "application/json"}), timeout=cap)
        for line in resp:
            if time.time() - t0 > cap:
                resp.close()
                return dict(status="cap", seconds=round(time.time() - t0, 1), first_token_s=first,
                            thinking_chars=sum(map(len, thinking)), content_chars=sum(map(len, content)))
            chunk = json.loads(line)
            m = chunk.get("message") or {}
            if first is None and (m.get("thinking") or m.get("content") or m.get("tool_calls")):
                first = round(time.time() - t0, 1)
            if first_content is None and (m.get("content") or m.get("tool_calls")):
                first_content = round(time.time() - t0, 1)
            thinking.append(m.get("thinking") or "")
            content.append(m.get("content") or "")
            calls += m.get("tool_calls") or []
            if chunk.get("done"):
                last = chunk
    except urllib.error.HTTPError as e:
        return dict(status=e.code, seconds=round(time.time() - t0, 1), error=e.read().decode("utf-8", "replace")[:240])
    except (TimeoutError, OSError) as e:
        return dict(status="cap", seconds=round(time.time() - t0, 1), first_token_s=first, error=str(e)[:80],
                    thinking_chars=sum(map(len, thinking)))
    ev, evd = last.get("eval_count") or 0, (last.get("eval_duration") or 0) / 1e9
    return dict(status=200, seconds=round(time.time() - t0, 1), first_token_s=first, first_action_s=first_content,
                load_s=round((last.get("load_duration") or 0) / 1e9, 1),
                prompt_eval_s=round((last.get("prompt_eval_duration") or 0) / 1e9, 1), eval_tokens=ev,
                tok_per_s=round(ev / evd, 1) if evd else None, thinking_chars=sum(map(len, thinking)),
                calls=calls, done_reason=last.get("done_reason"))


def busy_after():
    """Latency of a 1-token request: large -> the capped generation was still running."""
    t0 = time.time()
    req = dict(model="gpt-oss:20b", messages=[{"role": "user", "content": "Say ok."}], think="low", stream=False,
               options=dict(OPTS, num_predict=1, seed=1), keep_alive="10m")
    try:
        urllib.request.urlopen(urllib.request.Request(URL, json.dumps(req).encode(), {"Content-Type": "application/json"}),
                               timeout=600).read()
    except Exception as e:
        return f"error {e}"[:80]
    return round(time.time() - t0, 1)


rows = []
print("warm-up", busy_after(), flush=True)
for level in ("low", "medium", "high"):
    for name, text in CASES.items():
        for seed in (11, 22):
            msgs = [{"role": "system", "content": "You are a coding assistant. Use the tools. Exactly one tool call."},
                    {"role": "user", "content": "Call write_file with path candidate_tests.py and EXACTLY this content "
                                                "(byte for byte, between the markers):\n<<<BEGIN\n" + text + "END>>>"}]
            req = dict(model="gpt-oss:20b", messages=msgs, tools=run_light.tool_specs("high"), think=level,
                       options=dict(OPTS, seed=seed), keep_alive="10m")
            r = stream(req, CAP)
            calls = r.pop("calls", [])
            got = (calls[0]["function"].get("arguments") or {}).get("content") if calls else None
            r.update(level=level, case=name, seed=seed, tool_call=bool(calls), exact=got == text,
                     near=got is not None and got.strip() == text.strip())
            if r["status"] != 200:
                r["next_request_latency_s"] = busy_after()
            rows.append(r)
            print(json.dumps(r), flush=True)
            OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
for level in ("low", "medium", "high"):
    rs = [r for r in rows if r["level"] == level]
    print(f"{level:7s} ok {sum(r['status'] == 200 for r in rs)}/4  exact {sum(r['exact'] for r in rs)}/4  "
          f"near {sum(r['near'] for r in rs)}/4  cap {sum(r['status'] == 'cap' for r in rs)}  "
          f"500 {sum(r['status'] == 500 for r in rs)}  median s "
          f"{sorted(r['seconds'] for r in rs)[len(rs) // 2]}")
