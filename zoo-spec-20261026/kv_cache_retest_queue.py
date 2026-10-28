#!/usr/bin/env python3
"""Run the q8 KV-cache A/B retest, then the authorized High mmap follow-up.

Queue: XS baked tags across SuperLight; S baked profiles on L0/L2/L4/L5;
M K2-Q5 winner and Granite-8B low-olldef on M1/M2/M3/M4; then GLM and
Nemotron High H1-H5 with vendor-mmap. All work is serial and resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent
sys.path.insert(0, str(ROUND2))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROUND2 / "superlight"))

import superlight as sl  # noqa: E402
import zoo_night as zoo  # noqa: E402


RUNS = ROUND2 / "runs"
QUEUE_NAME = "q8-cache-zoo-then-high-mmap-20261027"
QUEUE_ROOT = RUNS / QUEUE_NAME
SUPERLIGHT_NAME = "superlight-q8-cache-retest-20261027"
ZOO_NAME = "zoo-night-q8-cache-retest-20261027"
HIGH_NAME = "zoo-high-mmap-q8-followup-20261027"
XS_MODELS = ["gemma4:e4b-T-XS1", "qwen3.5:4b-T-XS2", "granite4.2:3b-T-XS3"]
XS_TO_BASE = {
    "gemma4:e4b-T-XS1": "gemma4:e4b",
    "qwen3.5:4b-T-XS2": "qwen3.5:4b",
    "granite4.2:3b-T-XS3": "granite4.2:3b",
}
LIGHT_TASKS = {"L0", "L2", "L4", "L5"}
MEDIUM_TASKS = {"M1", "M2", "M3", "M4"}
KEEP_CONFIGS = {
    ("granite4.2-3b", "th-ibm"): LIGHT_TASKS,
    ("gemma4-e4b", "bakedtag-gemma4-e4b-T-S2"): LIGHT_TASKS,
    ("k2-horizon-7b-q5", "baked-k2-horizon-7b-q5-t"): MEDIUM_TASKS,
    ("granite4.2-8b", "low-olldef"): MEDIUM_TASKS,
}
ORIGINAL_BUILD_PLAN = zoo.build_plan


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def update_queue(**fields) -> None:
    state = {}
    path = QUEUE_ROOT / "status.json"
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    state.update(fields, updated_at=utc_now(), pid=os.getpid())
    write_json(path, state)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def run_superlight() -> None:
    out = RUNS / SUPERLIGHT_NAME
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["model"], row["config"], row["task"], row["i"]))

    data = sl.dataset()
    total = sum(len(cases) for cases in data.values()) * len(XS_MODELS) * len(sl.CONFIGS)
    count = len(done)
    baseline_path = RUNS / "superlight-20261027" / "results.jsonl"
    version = zoo.runner.api("version").get("version")
    write_json(out / "campaign.json", {
        "name": SUPERLIGHT_NAME,
        "cache_requested": "q8_0",
        "flash_attention_requested": True,
        "ollama_version": version,
        "models": XS_MODELS,
        "configs": list(sl.CONFIGS),
        "tasks": list(data),
        "seed": sl.SEED,
        "request_count_expected": total,
        "started_at": utc_now(),
    })

    update_queue(phase="superlight", finished=False, done=count, total=total,
                 current={"model": None, "task": None, "config": None})
    # Task-major order: every XS model/config finishes one task before the next.
    for task, cases in data.items():
        for model in XS_MODELS:
            for cfg, use_schema in sl.CONFIGS.items():
                for i, case in enumerate(cases):
                    key = (model, cfg, task, i)
                    if key in done:
                        count += 1
                        continue
                    update_queue(phase="superlight", finished=False, done=count, total=total,
                                 current={"model": model, "task": task, "config": cfg, "case": i})
                    text, schema = sl.prompt(task, case)
                    try:
                        data_out, seconds = sl.call(model, text, schema if use_schema else None)
                        content = data_out.get("message", {}).get("content", "")
                        answer = sl.parse(content)
                        ok, kind = sl.check(task, case, answer)
                        eval_count = data_out.get("eval_count")
                        eval_duration = data_out.get("eval_duration")
                        row = dict(model=model, config=cfg, task=task, i=i, ok=ok, kind=kind,
                                   seconds=round(seconds, 3), eval_count=eval_count,
                                   eval_duration=eval_duration,
                                   gen_tok_s=round(eval_count / (eval_duration / 1e9), 2)
                                   if eval_count and eval_duration else None,
                                   answer=content[:300], cache_requested="q8_0")
                    except Exception as exc:
                        row = dict(model=model, config=cfg, task=task, i=i, ok=False, kind="infra",
                                   seconds=None, eval_count=None, eval_duration=None,
                                   gen_tok_s=None, answer=f"{type(exc).__name__}: {exc}"[:300],
                                   cache_requested="q8_0")
                    append_jsonl(result_path, row)
                    done.add(key)
                    count += 1
                    if count % 10 == 0:
                        update_queue(phase="superlight", finished=False, done=count, total=total,
                                     current={"model": model, "task": task, "config": cfg, "case": i})

    # Compare per case with the previous seed-locked baseline. Recheck the first
    # changed case in each cell once; a single flip is a signal, not a verdict.
    old = {}
    if baseline_path.exists():
        for line in baseline_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            old[(row["model"], row["config"], row["task"], row["i"])] = row
    new = {}
    for line in result_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        new[(row["model"], row["config"], row["task"], row["i"])] = row
    rechecks_path = out / "rechecks.jsonl"
    comparison = []
    for baked, base in XS_TO_BASE.items():
        for cfg in sl.CONFIGS:
            for task in data:
                cell_keys = [(base, cfg, task, i) for i in range(len(data[task]))]
                old_cell = [old[k] for k in cell_keys if k in old]
                new_cell = [new.get((baked, cfg, task, i)) for i in range(len(data[task]))]
                new_cell = [r for r in new_cell if r is not None]
                flips = [i for i in range(len(data[task]))
                         if (base, cfg, task, i) in old and (baked, cfg, task, i) in new
                         and bool(old[(base, cfg, task, i)]["ok"])
                         != bool(new[(baked, cfg, task, i)]["ok"])]
                recheck = None
                if flips:
                    i = flips[0]
                    prompt_text, schema = sl.prompt(task, data[task][i])
                    try:
                        d, sec = sl.call(baked, prompt_text, schema if sl.CONFIGS[cfg] else None)
                        ans = sl.parse(d.get("message", {}).get("content", ""))
                        ok, kind = sl.check(task, data[task][i], ans)
                        recheck = dict(model=baked, config=cfg, task=task, i=i, ok=ok, kind=kind,
                                       seconds=round(sec, 3), answer=d.get("message", {}).get("content", "")[:300])
                    except Exception as exc:
                        recheck = dict(model=baked, config=cfg, task=task, i=i, ok=False, kind="infra",
                                       seconds=None, answer=f"{type(exc).__name__}: {exc}"[:300])
                    append_jsonl(rechecks_path, recheck)
                comparison.append({
                    "baked_model": baked, "baseline_model": base, "config": cfg, "task": task,
                    "baseline_n": len(old_cell), "baseline_pass": sum(bool(r["ok"]) for r in old_cell),
                    "q8_n": len(new_cell), "q8_pass": sum(bool(r["ok"]) for r in new_cell),
                    "case_level_flips": len(flips), "one_flip_recheck": recheck,
                })
    write_json(out / "comparison.json", {"compared_at": utc_now(), "cells": comparison})
    update_queue(phase="superlight", finished=True, done=total, total=total,
                 current=None, result=str(result_path))


def selected_zoo_plan(inv, only=None, profile="core", levels=("light", "medium")):
    full = ORIGINAL_BUILD_PLAN(inv, list({key for key, _ in KEEP_CONFIGS}), "core", ["light", "medium"])
    selected = [r for r in full["runs"] if (r["model_key"], r["config_id"]) in KEEP_CONFIGS
                and r["task_id"] in KEEP_CONFIGS[(r["model_key"], r["config_id"])] and r["seed"] == 11]
    task_order = ["L0", "L2", "L4", "L5", "M1", "M2", "M3", "M4"]
    model_order = {"granite4.2-3b": 0, "gemma4-e4b": 1,
                   "k2-horizon-7b-q5": 0, "granite4.2-8b": 1}
    selected.sort(key=lambda r: (task_order.index(r["task_id"]), model_order[r["model_key"]]))
    if len(selected) != 16:
        raise RuntimeError(f"Expected 16 Light/Medium baked-profile episodes; found {len(selected)}. "
                           f"Present: {[(r['model_key'], r['config_id'], r['task_id']) for r in selected]}")
    if any(r["model_key"] == "k2-horizon-7b-q5" and r["options"].get("num_ctx") != 24576 for r in selected):
        raise RuntimeError("K2 Q5 winning profile is expected at 24K context")
    return dict(full, runs=selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-superlight", action="store_true")
    parser.add_argument("--skip-agents", action="store_true")
    parser.add_argument("--skip-high", action="store_true")
    args = parser.parse_args()
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    (QUEUE_ROOT / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    update_queue(phase="starting", finished=False, pid=os.getpid(), started_at=utc_now(),
                 cache_requested="q8_0", flash_attention_requested=True)

    if not args.skip_superlight:
        run_superlight()

    if not args.skip_agents:
        zoo.spec.SEEDS[("screen", 1)] = [11]
        zoo.build_plan = selected_zoo_plan
        update_queue(phase="light-medium", finished=False, done=0, total=16, current=None)
        zoo.run(argparse.Namespace(name=ZOO_NAME, hours=12.0, profile="core",
                                   models=list({key for key, _ in KEEP_CONFIGS}),
                                   levels=["light", "medium"], new_deadline=True))
        inner_status = RUNS / ZOO_NAME / "status.json"
        if not inner_status.exists() or not json.loads(inner_status.read_text(encoding="utf-8")).get("finished"):
            raise RuntimeError("Light/Medium q8 retest did not complete successfully; High remains queued")

    if not args.skip_high:
        update_queue(phase="high-mmap", finished=False, done=0, total=10, current=None)
        zoo.build_plan = ORIGINAL_BUILD_PLAN
        import high_mmap_rescue
        sys.argv = [str(HERE / "high_mmap_rescue.py"), "--name", HIGH_NAME, "--hours", "12"]
        high_mmap_rescue.main()
        high_status = RUNS / HIGH_NAME / "status.json"
        if not high_status.exists() or not json.loads(high_status.read_text(encoding="utf-8")).get("finished"):
            raise RuntimeError("High mmap follow-up did not complete")

    update_queue(phase="complete", finished=True, done=1, total=1, current=None,
                 finished_at=utc_now())


if __name__ == "__main__":
    main()
