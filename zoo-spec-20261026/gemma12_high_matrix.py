"""Gemma 4 12B — High config matrix (owner 2026-10-31: "крутануть её в разных конфигах на L тестах").

Copy of gemma26_high_matrix.py for the dense 12B (no use_mmap: 7.2 GB, 73% on GPU in the passport).
The budget comes from calibration-approved-gemma.json["gemma4-12b"] and the script refuses to start unless its
status is "approved" (owner decision; proposed 16K/8K on 2026-10-31).

Original 26B notes:

Official Google sampling for the family (T 1.0, top_p 0.95, top_k 64); High adjustments: the owner-approved High
budget (calibration-approved-gemma.json), native tools, 12-minute High limit, and the same "check yourself" system hint the other models had.
Configs follow GEMMA4_SELECTION_PLAN.md: th-official, nt-official, th-official-syshint, nt-official-syshint.

Screening (default): 4 configs x H0, H3, H8, seed 11, task by task -> runs/gemma4-12b-high-matrix-20261031
Round 2 (--round2 cfg[,cfg]): finalists x H1, H2, H4, H6, H7, H9, seed 11 -> runs/gemma4-12b-high-round2-20261031,
the same tasks and seed as zoo-high-round2-20261029, so Gemma compares with the other models directly.

  python gemma12_high_matrix.py --dry-run
  python gemma12_high_matrix.py                 (refuses to start while another model is loaded)
  python gemma12_high_matrix.py --report
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import zoo_night as z
import high_matrix  # noqa: F401  (imports light_matrix: the -syshint system-prompt patch)

KEY, TAG = "gemma4-12b", "gemma4:12b-it-qat"
GOOGLE = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
APPROVED = json.loads((Path(__file__).resolve().parent / "calibration-approved-gemma.json").read_text(encoding="utf-8"))[KEY]
BUDGET = {"num_ctx": APPROVED["num_ctx"], "num_predict": APPROVED["num_predict"]}
CONFIGS = {"th-official": True, "nt-official": False, "th-official-syshint": True, "nt-official-syshint": False}
SCREEN = ("H0", "H3", "H8")
ROUND2 = ("H1", "H2", "H4", "H6", "H7", "H9")
SEED = 11
NAME = "gemma4-12b-high-matrix-20261031"
ACTIVE, TASKS = list(CONFIGS), SCREEN


def build_plan(inv, only=None, profile="all", levels=("high",)):
    if TAG not in inv:
        raise SystemExit(f"missing tag {TAG}")
    runs = []
    for task in TASKS:                       # task by task: every config on H0, then the next task
        for cfg in ACTIVE:
            runs.append(dict(phase=1, level="high", model_key=KEY, tag=TAG, original=TAG, config_id=cfg,
                             think=CONFIGS[cfg], options=dict(GOOGLE, **BUDGET, seed=SEED), seed=SEED, task_id=task,
                             contract="explicit", structured=False, deviation=False, hint=False, baked=False,
                             tool_mode="native", purpose="Gemma 4 12B High config matrix",
                             est_seconds=z.runner.LEVELS["high"]["seconds"]))
    return dict(originals={KEY: TAG}, missing=[], baked=[], runs=runs)


def report():
    f = z.RUNS / NAME / "results.jsonl"
    score = collections.defaultdict(lambda: [0, 0.0, 0])
    for line in f.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("outcome", "episode") != "episode":
            continue
        r = d.get("row") or {}
        s = score[d["config_id"]]
        s[0] += bool(r.get("automated_workflow_pass"))
        s[1] += float(r.get("wall_seconds") or 0) / 60
        s[2] += 1
    ranked = sorted(score, key=lambda c: (-score[c][0], score[c][1]))
    print(json.dumps({c: dict(points=score[c][0], played=score[c][2], minutes=round(score[c][1], 1)) for c in ranked},
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run of the same name")
    ap.add_argument("--round2", help="comma-separated finalist configs for the hard tasks")
    a = ap.parse_args()
    assert z.runner.LEVELS["high"]["seconds"] == 720
    if a.round2:
        ACTIVE = [c for c in a.round2.split(",") if c]
        assert ACTIVE and set(ACTIVE) <= set(CONFIGS), ACTIVE
        TASKS, NAME = ROUND2, "gemma4-12b-high-round2-20261031"
    if a.report:
        report()
        sys.exit(0)
    z.build_plan = build_plan
    if a.dry_run:
        for r in build_plan({TAG: {}})["runs"]:
            print(r["config_id"], r["task_id"], r["think"], r["options"])
        sys.exit(0)
    if APPROVED.get("status") != "approved":
        raise SystemExit(f"Budget for {KEY} is {APPROVED.get('status')!r}, not approved: the owner must approve it first.")
    if (z.RUNS / NAME).exists() and not a.resume:
        raise SystemExit(f"Refusing to overwrite existing run: {NAME}")
    if z.runner.api("ps").get("models"):
        raise SystemExit("Ollama has a model loaded (another run?); wait for it to finish.")
    try:
        sys.exit(z.run(argparse.Namespace(name=NAME, hours=6, new_deadline=True, models=None, profile="all",
                                          levels=["high"])))
    finally:
        z._api("generate", dict(model=TAG, keep_alive=0), 120)
