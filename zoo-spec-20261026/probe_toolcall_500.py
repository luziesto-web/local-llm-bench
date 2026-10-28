"""Short probe (Astra review 2026-10-28): which file contents make Ollama answer 500 "error parsing tool call" for
gpt-oss, and does a retry help (same seed / next seed). Native tools exactly as in the harness (run_light.tool_specs).

  python probe_toolcall_500.py   -> ../runs/probe-toolcall-500-20261028.json
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

OPTS = next(o for k, tag, cfg, think, o in hm.MATRIX if k == "gpt-oss-20b" and cfg == "high")
CASES = {
    "short": 'print("ok")\n',
    "triple_quotes": '"""Candidate tests for the engine."""\nfrom engine import Engine\n\n\ndef test_submit():\n    """One id per key."""\n    assert Engine().submit("a", "k")\n',
    "escapes": 'TEXT = "line one\\nline two\\t\'quoted\' \\"double\\""\nPATH = r"C:\\temp\\x"\nprint(TEXT, PATH)\n',
    "unicode": '# Проверка: зарплата ≥ 50 000 € — ok ✅\nNAME = "Кирилл 🚀"\nprint(NAME)\n',
    "realistic": '"""Candidate tests."""\nimport json\nfrom engine import Engine\n\nCASES = {"a": [1, 2], "b": {"x": "y"}}\n\nif __name__ == "__main__":\n    e = Engine()\n    print(f"ok {json.dumps(CASES)}")\n    assert e.submit("a", "same") == e.submit("a", "same"), "dup"\n    print("""done\n""")\n',
}


def ask(content, seed):
    msgs = [{"role": "system", "content": "You are a coding assistant. Use the tools. Exactly one tool call."},
            {"role": "user", "content": "Call write_file with path candidate_tests.py and EXACTLY this content "
                                        "(byte for byte, between the markers):\n<<<BEGIN\n" + content + "END>>>"}]
    req = dict(model="gpt-oss:20b", messages=msgs, tools=run_light.tool_specs("high"), stream=False, think="high",
               options=dict(OPTS, seed=seed), keep_alive="10m")
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(req).encode(),
                                                          {"Content-Type": "application/json"}), timeout=300)
        d = json.loads(r.read())
        calls = (d.get("message") or {}).get("tool_calls") or []
        got = ((calls[0]["function"].get("arguments") or {}).get("content") if calls else None)
        return dict(status=200, seconds=round(time.time() - t, 1), tool_call=bool(calls), exact=got == content,
                    near=(got or "").strip() == content.strip())
    except urllib.error.HTTPError as e:
        return dict(status=e.code, seconds=round(time.time() - t, 1), error=e.read().decode("utf-8", "replace")[:240])
    except (TimeoutError, OSError) as e:   # a stuck generation is a result too (first run 28.10 hung > 10 min)
        return dict(status="timeout", seconds=round(time.time() - t, 1), error=str(e)[:120])


rows = []
for name, content in CASES.items():
    for seed in (11, 22, 33):
        r = ask(content, seed)
        r.update(case=name, seed=seed)
        if r["status"] == 500:
            r["retry_same_seed"] = ask(content, seed)["status"]
            r["retry_next_seed"] = ask(content, seed + 1)["status"]
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        (Path(__file__).resolve().parent.parent / "runs" / "probe-toolcall-500-20261028.partial.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
out = Path(__file__).resolve().parent.parent / "runs" / "probe-toolcall-500-20261028.json"
out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
for name in CASES:
    rs = [r for r in rows if r["case"] == name]
    print(f"{name:14s} 500: {sum(r['status'] == 500 for r in rs)}/3  timeout: {sum(r['status'] == 'timeout' for r in rs)}/3  "
          f"exact: {sum(bool(r.get('exact')) for r in rs)}/3")
