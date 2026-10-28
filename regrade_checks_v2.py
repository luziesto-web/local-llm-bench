"""Re-grade saved solutions with checks v2 (Astra review 2026-10-28: H2/H5 barrier races and grouped H5, H8 pipeline.py
guard, H9 shuffled input, M3 return values). No model calls: each saved workspace is copied to a temp folder and the
new acceptance runs on the copy. Original episode.json files are not touched; results go to
runs/regrade-checks-v2-20261028.json with the old and the new verdict.

  python regrade_checks_v2.py
"""
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import run_light

TASKS = {"H2", "H5", "H8", "H9", "M3"}
RUNS = Path(__file__).resolve().parent / "runs"
rows = []
for ep in sorted(RUNS.glob("*-run/*/episode.json")):
    task = ep.parent.name
    if task not in TASKS or not (ep.parent / "workspace").is_dir() or "_invalid" in ep.parts:
        continue
    old = json.loads(ep.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "workspace"
        shutil.copytree(ep.parent / "workspace", work)
        try:
            new = run_light.worker(work, task, "acceptance", timeout=60)
        except Exception as exc:
            new = dict(success=None, output="", error=f"{type(exc).__name__}: {exc}")
    groups = [l for l in new.get("output", "").splitlines() if l.startswith(("PASS ", "FAIL "))]
    rows.append(dict(episode=str(ep.parent.relative_to(RUNS)), task=task, old=old.get("code_pass"),
                     new=new["success"], groups=groups, error=new.get("error", "")[-300:]))
out = RUNS / "regrade-checks-v2-20261028.json"
out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(rows), "saved solutions re-graded ->", out)
for task in sorted(TASKS):
    c = Counter((r["old"], r["new"]) for r in rows if r["task"] == task)
    print(task, "old->new:", dict(c))
for r in rows:
    if r["old"] and not r["new"]:
        print("LOST", r["episode"], "|", r["error"].strip().splitlines()[-1][:160] if r["error"].strip() else r["groups"])
groups = Counter(g for r in rows if r["task"] == "H5" for g in r["groups"])
print("H5 groups:", dict(groups))
