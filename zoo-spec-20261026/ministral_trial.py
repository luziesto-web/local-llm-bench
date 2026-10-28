"""Ministral 3 8B trial: does it deserve a place in the zoo? (user request 2026-10-27)

Stage 1 "best params": Ministral 3 Instruct with the vendor-range configs from zoo_test_spec
          (t005, t015, t005-json, t015-json) on Light L0, L2, L4, L5. The winner goes to stage 2.
          Ministral 3 Reasoning has one vendor config (rsn-p8192) and skips stage 1.
Stage 2 "competition material": the exact task sets the other models already ran
          (Light v5 + Light10: L0, L2-L11; Medium: M1-M5), seed 42 like those contests,
          same round2 tools, limits and acceptance. Then a comparison table and a verdict
          against the kill criterion fixed BEFORE the run (see CRITERION).

Never deletes or creates models/tags. Refuses to start while another contest is running.

  python ministral_trial.py plan
  python ministral_trial.py run [--name ministral-trial-YYYYMMDD] [--variants instruct reasoning] [--force]
  python ministral_trial.py report --name ministral-trial-YYYYMMDD
"""
import argparse
import contextlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import zoo_night as zn          # also patches run_light (structured format, implicit contract, hint)
import light10_cases            # noqa: F401  registers L6..L11 in the shared CASES map
from zoo_night import spec, runner, RUNS, CASES

INSTRUCT, REASONING = "ministral3-8b-instruct", "ministral3-8b-reasoning"
STAGE1_CONFIGS = ["t005", "t015", "t005-json", "t015-json"]
STAGE1_TASKS = ["L0", "L2", "L4", "L5"]
LIGHT_TASKS = ["L0", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"]
MEDIUM_TASKS = ["M1", "M2", "M3", "M4", "M5"]
SEED = 42
# Fixed before the run. Light10 = the ten tasks Qwen 4B (6/10) and Granite 4.2 8B (5/10) ran;
# Medium = Granite 4.2 8B 3/5, Qwen 9B 4/5, K2 4/5. One seed: a screen, not a ranking.
LIGHT10 = ["L0", "L2", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"]
CRITERION = dict(light10_workflow_min=5, medium_workflow_min=3)
# Earlier runs of the other models on the same material (episode.json folders in round2/runs).
REFERENCES = {
    "qwen3.5:4b-t (light10)": "light10-qwen4b-20261027-qwen4b",
    "granite4.2:8b low (light10)": "light10-granite42-8b-20261027-granite42_8b",
    "qwen3.5:4b-t (light v5)": "light-contest-v5-clean-seed42-qwen4b",
    "gemma4:e4b-t (light v5)": "light-contest-v5-clean-seed42-gemma",
    "ministral-3:8b-t T0.3 (light v5)": "light-contest-v5-clean-seed42-ministral",
    "qwen3.5:9b-t (medium v4)": "medium-contest-v4-json-seed42-qwen9b",
    "k2-horizon:7b-q5-t (medium v4)": "medium-contest-v4-json-seed42-k2q5",
    "qwen3.5:4b-t (medium v4)": "medium-contest-v4-json-seed42-qwen4b",
    "gemma4:e4b-t (medium v4)": "medium-contest-v4-json-seed42-gemma",
    "granite4.2:8b low (medium)": "granite42-return-medium-20261027-granite42_8b",
    "ministral-3:8b-t T0.3 (medium v4)": "medium-contest-v4-json-seed42-ministral",
}


def busy():
    """Another contest process alive, or a non-Ministral model loaded in Ollama."""
    reasons = []
    ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
                         "'run_light|run_medium|run_high|zoo_night.py run|run_8b|run_modes|light10' -and "
                         "$_.CommandLine -notmatch 'Get-CimInstance' } | ForEach-Object { $_.CommandLine }"],
                        capture_output=True, text=True)
    reasons += [f"process: {line.strip()[:150]}" for line in ps.stdout.splitlines() if line.strip()]
    loaded = [m["name"] for m in runner.api("ps").get("models", [])]
    reasons += [f"model loaded in Ollama: {m}" for m in loaded if "inistral" not in m]
    return reasons


def config(key, cfg_id):
    return next(c for c in spec.MODELS[key].configs if c.id == cfg_id)


def run_list(inv, key, cfgs, tasks, stage):
    tag = zn.TAG_ALIASES.get(spec.MODELS[key].tag, spec.MODELS[key].tag)
    if tag not in inv:
        raise SystemExit(f"{key}: tag {tag} is not installed")
    out = []
    for cfg in cfgs:
        for task in tasks:
            out.append(dict(stage=stage, model_key=key, tag=tag, config_id=cfg.id, think=cfg.think,
                            options=dict(cfg.options, seed=SEED), contract=cfg.contract, structured=cfg.structured,
                            hint=False, task_id=task, level="light" if task.startswith("L") else "medium",
                            purpose=cfg.purpose))
    return out


def execute(name, r, inv, log):
    label = "run"
    output = RUNS / f"{name}-{zn.slug(r['model_key'])}-{zn.slug(r['config_id'])}-s{SEED}"
    root = RUNS / f"{output.name}-{label}" / r["task_id"]
    if (root / "episode.json").exists():                       # resume / reuse (stage 2 reuses stage-1 tasks)
        row = json.loads((root / "episode.json").read_text(encoding="utf-8"))
        seen = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        if not any(d.get("stage") == r["stage"] and d.get("task_id") == r["task_id"]
                   and d.get("config_id") == r["config_id"] for d in seen):
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(r, row=row, reused=True), ensure_ascii=False, default=str) + "\n")
        return row
    if root.exists():                                           # interrupted: keep, never overwrite
        root.rename(root.with_name(f"{root.name}-interrupted-{datetime.now():%Y%m%d%H%M%S}"))
    output.mkdir(exist_ok=True)
    configuration = dict(interface="ollama-custom-json-tools-v2", level=r["level"],
                         config_id=f"{r['model_key']}/{r['config_id']}", options=r["options"],
                         models={label: dict(tag=r["tag"], digest=inv[r["tag"]]["digest"],
                                             **zn.think_fields(r, inv[r["tag"]]["capabilities"]))},
                         limits=dict(calls=runner.LEVELS[r["level"]]["calls"], seconds=runner.LEVELS[r["level"]]["seconds"]),
                         trial=name, stage=r["stage"], contract=r["contract"], structured=r["structured"],
                         purpose=r["purpose"])
    zn.write_json(output / "configuration.json", configuration)
    zn.CURRENT.update(structured=r["structured"], implicit=r["contract"] == "implicit", hint=False)
    start = zn.machine_snapshot()
    try:
        row = runner.episode(output, label, r["task_id"], configuration)
    except Exception as exc:                                    # infrastructure, not a model score
        row = dict(case_id=r["task_id"], code_pass=False, automated_workflow_pass=False,
                   infra_error=f"runner_error {type(exc).__name__}: {exc}", wall_seconds=0)
    telemetry = zn.episode_telemetry(root, r["options"], start)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(r, row=row, telemetry=telemetry), ensure_ascii=False, default=str) + "\n")
    return row


def score(rows):
    wf = sum(bool(x.get("automated_workflow_pass")) for x in rows)
    code = sum(bool(x.get("code_pass")) for x in rows)
    return (wf, code, -sum(x.get("wall_seconds") or 0 for x in rows))


def reference_rows(folder, tasks):
    rows = {}
    for t in tasks:
        p = RUNS / folder / t / "episode.json"
        if p.exists():
            rows[t] = json.loads(p.read_text(encoding="utf-8"))
    return rows


def cell(e):
    if e is None:
        return "·"
    s = ("C" if e.get("code_pass") else "-") + ("W" if e.get("automated_workflow_pass") else "")
    return s + ("!" if e.get("violations") else "") + ("I" if e.get("infra_error") else "")


def report(name):
    night = RUNS / name
    log = night / "episodes.jsonl"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    plan = json.loads((night / "plan.json").read_text(encoding="utf-8")) if (night / "plan.json").exists() else {}
    lines = [f"# Ministral trial {name}", "",
             "C = code accepted, W = workflow pass (own tests seen passing + honest final), ! = violation, "
             "I = infrastructure error, · = not run. One seed: screen only, review_required.", ""]
    s1 = [r for r in rows if r["stage"] == 1]
    if s1:
        lines += ["## Stage 1 (Instruct, vendor-range configs, Light L0/L2/L4/L5)", "",
                  "| config | " + " | ".join(STAGE1_TASKS) + " | W | C | time |", "|---" * (len(STAGE1_TASKS) + 4) + "|"]
        for cfg in STAGE1_CONFIGS:
            rr = {r["task_id"]: r["row"] for r in s1 if r["config_id"] == cfg}
            if rr:
                w, c, t = score(list(rr.values()))
                lines.append(f"| {cfg} | " + " | ".join(cell(rr.get(t_)) for t_ in STAGE1_TASKS) + f" | {w} | {c} | {-t:.0f}s |")
        lines += ["", f"Winner: **{plan.get('winner', '?')}**", ""]
    s2 = [r for r in rows if r["stage"] == 2]
    # Stage 2 of the Instruct winner reuses its stage-1 episodes (same folder): count them too.
    winner = plan.get("winner")
    have = {(r["model_key"], r["task_id"]) for r in s2}
    s2 += [dict(r, stage=2, reused=True) for r in rows if r["stage"] == 1 and r["config_id"] == winner
           and (r["model_key"], r["task_id"]) not in have]
    for title, tasks in (("Light (v5 + Light10 tasks)", LIGHT_TASKS), ("Medium", MEDIUM_TASKS)):
        lines += [f"## Stage 2 — {title}", "", "| run | " + " | ".join(tasks) + " | W | C |", "|---" * (len(tasks) + 3) + "|"]
        for key in (INSTRUCT, REASONING):
            rr = {r["task_id"]: r["row"] for r in s2 if r["model_key"] == key and r["task_id"] in tasks}
            if rr:
                w, c, _ = score(list(rr.values()))
                lines.append(f"| **{key} {next(iter(r['config_id'] for r in s2 if r['model_key'] == key))}** | "
                             + " | ".join(cell(rr.get(t)) for t in tasks) + f" | {w} | {c} |")
        for label, folder in REFERENCES.items():
            rr = reference_rows(folder, tasks)
            if rr:
                w, c, _ = score(list(rr.values()))
                lines.append(f"| {label} | " + " | ".join(cell(rr.get(t)) for t in tasks) + f" | {w} | {c} |")
        lines.append("")
    lines += ["## Verdict (criterion fixed before the run)", "",
              f"KEEP if Light10 workflow >= {CRITERION['light10_workflow_min']}/10 OR Medium workflow >= "
              f"{CRITERION['medium_workflow_min']}/5, else DROP. Deleting the model is the user's decision.", ""]
    for key in (INSTRUCT, REASONING):
        rr = [r["row"] for r in s2 if r["model_key"] == key]
        if not rr:
            continue
        l10 = sum(bool(r["row"].get("automated_workflow_pass")) for r in s2
                  if r["model_key"] == key and r["task_id"] in LIGHT10)
        med = sum(bool(r["row"].get("automated_workflow_pass")) for r in s2
                  if r["model_key"] == key and r["task_id"] in MEDIUM_TASKS)
        done_l = sum(1 for r in s2 if r["model_key"] == key and r["task_id"] in LIGHT10)
        done_m = sum(1 for r in s2 if r["model_key"] == key and r["task_id"] in MEDIUM_TASKS)
        keep = l10 >= CRITERION["light10_workflow_min"] or med >= CRITERION["medium_workflow_min"]
        complete = done_l == len(LIGHT10) and done_m == len(MEDIUM_TASKS)
        lines.append(f"- {key}: Light10 {l10}/{done_l}, Medium {med}/{done_m} -> "
                     f"**{'KEEP' if keep else 'DROP'}**{'' if complete else ' (INCOMPLETE run)'}")
    text = "\n".join(lines) + "\n"
    (night / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    rp = sub.add_parser("run")
    rp.add_argument("--name", default=f"ministral-trial-{datetime.now():%Y%m%d}")
    rp.add_argument("--variants", nargs="+", choices=["instruct", "reasoning"], default=["instruct", "reasoning"])
    rp.add_argument("--force", action="store_true", help="start even if another contest looks active")
    rep = sub.add_parser("report")
    rep.add_argument("--name", required=True)
    args = p.parse_args()

    if args.command == "report":
        return report(args.name)
    inv = zn.inventory()
    s1 = run_list(inv, INSTRUCT, [config(INSTRUCT, c) for c in STAGE1_CONFIGS], STAGE1_TASKS, 1)
    s2r = run_list(inv, REASONING, [config(REASONING, "rsn-p8192")], LIGHT_TASKS + MEDIUM_TASKS, 2)
    if args.command == "plan":
        est = {"light": 40, "medium": 85}
        print(f"Stage 1: {len(s1)} episodes (~{len(s1) * 40 / 60:.0f} min): {', '.join(STAGE1_CONFIGS)} x {', '.join(STAGE1_TASKS)}")
        print(f"Stage 2 Instruct (winner): {len(LIGHT_TASKS) + len(MEDIUM_TASKS)} episodes "
              f"(~{(len(LIGHT_TASKS) * est['light'] + len(MEDIUM_TASKS) * est['medium']) / 60:.0f} min)")
        print(f"Stage 2 Reasoning rsn-p8192: {len(s2r)} episodes (~{(len(LIGHT_TASKS) * 200 + len(MEDIUM_TASKS) * 250) / 60:.0f} min, "
              "no measurements, guess)")
        print("Busy check now:", busy() or "free")
        return
    reasons = busy()
    if reasons and not args.force:
        raise SystemExit("Another contest/model is active; not starting:\n  " + "\n  ".join(reasons))
    night = RUNS / args.name
    night.mkdir(parents=True, exist_ok=True)
    log = night / "episodes.jsonl"
    plan = dict(started=datetime.now(timezone.utc).isoformat(), variants=args.variants, criterion=CRITERION,
                instruct_tag=s1[0]["tag"], reasoning_tag=s2r[0]["tag"])
    zn.write_json(night / "plan.json", plan)

    def go(runs):
        for i, r in enumerate(runs, 1):
            zn.write_json(night / "status.json", dict(updated_at=datetime.now(timezone.utc), stage=r["stage"],
                                                      model=r["model_key"], config=r["config_id"], task=r["task_id"],
                                                      index=i, total=len(runs)))
            execute(args.name, r, inv, log)

    if "instruct" in args.variants:
        go(s1)
        done = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        by_cfg = {c: [d["row"] for d in done if d["stage"] == 1 and d["config_id"] == c] for c in STAGE1_CONFIGS}
        winner = max(STAGE1_CONFIGS, key=lambda c: score(by_cfg[c]))
        plan["winner"] = winner
        zn.write_json(night / "plan.json", plan)
        with contextlib.suppress(Exception):
            runner.api("generate", dict(model=s1[0]["tag"], keep_alive=0), 120)
        go(run_list(inv, INSTRUCT, [config(INSTRUCT, winner)], LIGHT_TASKS + MEDIUM_TASKS, 2))
    if "reasoning" in args.variants:
        with contextlib.suppress(Exception):     # free memory from Instruct before loading Reasoning
            runner.api("generate", dict(model=s1[0]["tag"], keep_alive=0), 120)
        go(s2r)
    zn.write_json(night / "status.json", dict(updated_at=datetime.now(timezone.utc), finished=True))
    report(args.name)
    print("MINISTRAL_TRIAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
