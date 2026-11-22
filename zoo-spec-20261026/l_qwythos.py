"""Qwythos 9B Q4 against its L classmates (owner 2026-11-22: "квитос прогнать с одноклассниками М и L").

M is already done (runs/medium-qwythos9b-q4km-thcoding-c16p8-s22/-s33, 5/10 vs Qwen3.5 9B 7/10, same config).
L = the same hard tasks H1 H2 H4 H6 H7 H9, 12-minute limit, seeds 11 (L round 2) and 22 (L final).
Qwythos is a Qwen3.5 9B fine-tune, so it replays Qwen3.5 9B's L episodes exactly: the six qwen3.5-9b/th-coding
entries of the saved plan runs/l-high-round2-20261102/plan-*.json (options, budget 110592/55296, think, contract),
with only tag/original/model_key (and seed for s22) replaced. The saved plan is used because the original
qwen3.5:9b tag is no longer in `ollama list`, so high_round2.build_plan cannot rebuild it.

  python l_qwythos.py --dry-run --seed 11
  python l_qwythos.py --start --seed 11 [--hours 2]
  python l_qwythos.py --report
"""
import argparse
import collections
import copy
import json
import sys

import zoo_night as z

KEY, TAG = "qwythos-9b-q4", "qwythos:9b-q4km-text"
SOURCE = z.RUNS / "l-high-round2-20261102"


def name(seed):
    return f"l-qwythos9b-q4-s{seed}-20261122"


def make_plan(seed):
    def build_plan(inv, *args, **kw):
        if TAG not in inv:
            raise SystemExit(f"missing tag {TAG}")
        saved = json.loads(sorted(SOURCE.glob("plan-*.json"))[0].read_text(encoding="utf-8"))
        runs = []
        for r in saved["runs"]:
            if r["model_key"] == "qwen3.5-9b" and r["config_id"] == "th-coding":
                r = copy.deepcopy(r)
                r.update(model_key=KEY, tag=TAG, original=TAG, seed=seed,
                         purpose=f"Qwythos vs L classmates, Qwen3.5 9B th-coding config, seed {seed}")
                r["options"]["seed"] = seed
                runs.append(r)
        assert [r["task_id"] for r in runs] == ["H1", "H2", "H4", "H6", "H7", "H9"], runs
        return dict(originals={KEY: TAG}, missing=[], baked=[], runs=runs)
    return build_plan


def report():
    for seed in (11, 22):
        f = z.RUNS / name(seed) / "results.jsonl"
        if not f.exists():
            print(seed, "not run")
            continue
        won, minutes, played = [], 0.0, 0
        for line in f.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d.get("outcome", "episode") != "episode":
                continue
            r = d.get("row") or {}
            played += 1
            minutes += float(r.get("wall_seconds") or 0) / 60
            if r.get("automated_workflow_pass"):
                won.append(d["task_id"])
        print(seed, f"{len(won)}/{played}", f"{minutes:.1f} min", " ".join(won))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", action="store_true", help="required to actually run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, choices=[11, 22], default=11)
    ap.add_argument("--hours", type=float, default=2.0)
    a = ap.parse_args()
    assert z.runner.LEVELS["high"]["seconds"] == 720
    if a.report:
        report()
        sys.exit(0)
    z.build_plan = make_plan(a.seed)
    if a.dry_run:
        for r in z.build_plan(z.inventory())["runs"]:
            print(r["model_key"], r["tag"], r["config_id"], r["task_id"], r["think"], r["options"])
        sys.exit(0)
    if not a.start:
        raise SystemExit("Not started: pass --start.")
    if (z.RUNS / name(a.seed)).exists() and not a.resume:
        raise SystemExit(f"Refusing to overwrite existing run: {name(a.seed)}")
    if z.runner.api("ps").get("models"):
        raise SystemExit("Ollama has a model loaded (another run?); wait for it to finish.")
    try:
        sys.exit(z.run(argparse.Namespace(name=name(a.seed), hours=a.hours, new_deadline=not a.resume, models=None,
                                          profile="all", levels=["high"])))
    finally:
        z._api("generate", dict(model=TAG, keep_alive=0), 120)
