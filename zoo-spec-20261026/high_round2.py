"""High round 2 (owner 2026-10-29): one unambiguous High config per model.

Round 1 picked winners on H0/H3/H8, which almost every config solves, so gpt-oss, K2 and Nemotron stayed tied.
Round 2 plays the tied configs on the hard High tasks that are closest to the website contest
(H1 settings through three modules, H2 lease fencing, H4 statuses and user decisions, H6 storage migration,
H7 rate limiter, H9 cursor pagination). Standard High limit (12 min), approved High budgets, seed 11, native tools
with the 2026-10-29 harness fix (a plain-text final after a passing run_tests counts as final).
Ranking: W (solved and verified) first, then less time on all episodes. A tie -> rerun that model on seed 22.

  python high_round2.py            -> runs/zoo-high-round2-20261029
  python high_round2.py --dry-run  -> the plan
  python high_round2.py --report   -> winner per model
"""
import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import zoo_night as z

_build = z.build_plan
import high_matrix as hm  # noqa: E402  (configs, approved budgets; also the -syshint system-prompt patch)

NAME = "zoo-high-round2-20261029"
FINALISTS = {"gpt-oss-20b": ["low", "medium", "high-syshint"],
             "k2-horizon-7b-q5": ["q5-template", "q5-template-syshint"],
             "nemotron-3.5-lightning-30b": ["vendor-th", "vendor-th-syshint"]}
# second batch (owner 2026-10-28): Qwen3-Coder tie, queued after the first run
CODER = {"qwen3-coder-30b": ["vendor", "vendor-syshint"]}
TASKS = [("high", t) for t in ("H1", "H2", "H4", "H6", "H7", "H9")]
SEED = 11
# standard High limit: run_light LEVELS high seconds=720 (owner 2026-10-28)


def budget(model_key, level):
    b = hm.APPROVED[model_key]
    if level != "high" and "medium" in b:          # K2: its calibrated Medium budget also serves Light
        return b["medium"]["num_ctx"], b["medium"]["num_predict"]
    return b["num_ctx"], b["num_predict"]


def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
    templates = {}
    for level in ("high",):
        # a template run of that level (its model fields are overwritten below); qwen3.5-9b has no Light plan
        base = _build(inv, ["gemma4-e4b" if level == "light" else "qwen3.5-9b"], "all", (level,))
        templates[level] = next(r for r in base["runs"] if r["level"] == level)
    runs = []
    cfgs = {(k, c): (tag, think, opts) for k, tag, c, think, opts in hm.MATRIX}
    for model_key, configs in FINALISTS.items():           # model by model
        for level, task in TASKS:                          # task by task inside a model
            for cfg in configs:
                tag, think, opts = cfgs[(model_key, cfg)]
                if tag not in inv:
                    raise SystemExit(f"missing tag {tag}")
                ctx, predict = budget(model_key, level)
                r = copy.deepcopy(templates[level])
                r.update(model_key=model_key, tag=tag, original=tag, config_id=cfg, think=think,
                         options=dict(opts, num_ctx=ctx, num_predict=predict, seed=SEED), seed=SEED, task_id=task,
                         level=level, phase=1, baked=False, deviation=False, structured=False, hint=False,
                         tool_mode="native", purpose="High round 2: one config per model",
                         est_seconds=min(z.est_seconds(level, model_key, think), z.runner.LEVELS[level]["seconds"]))
                runs.append(r)
    base = _build(inv, ["qwen3.5-9b"], "all", ("high",))
    base["runs"] = runs
    return base


def report():
    f = Path(__file__).resolve().parent.parent / "runs" / NAME / "results.jsonl"
    score = collections.defaultdict(lambda: [0, 0.0, 0])
    for line in f.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        r = d.get("row") or {}
        if d.get("outcome", "episode") != "episode":
            continue
        ok = r.get("automated_workflow_pass") if d["task_id"].startswith("H") else r.get("code_pass")
        s = score[(d["model_key"], d["config_id"])]
        s[0] += bool(ok)
        s[1] += float(r.get("wall_seconds") or 0) / 60
        s[2] += 1
    out = {}
    for model_key, configs in FINALISTS.items():
        ranked = sorted(configs, key=lambda c: (-score[(model_key, c)][0], score[(model_key, c)][1]))
        out[model_key] = {c: dict(points=score[(model_key, c)][0], played=score[(model_key, c)][2],
                                  minutes=round(score[(model_key, c)][1], 1)) for c in ranked}
        out[model_key]["winner"] = ranked[0] if score[(model_key, ranked[0])][2] else None
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tiebreak", help="model key tied on seed 11: replay its configs on seed 22 -> <NAME>-s22-<model>")
    ap.add_argument("--coder", action="store_true", help="Qwen3-Coder batch -> runs/zoo-high-round2-coder-20261029 (only after the owner reviews it)")
    a = ap.parse_args()
    if a.coder:
        FINALISTS.clear()
        FINALISTS.update(CODER)
        NAME = "zoo-high-round2-coder-20261029"
    if a.tiebreak:
        FINALISTS = {a.tiebreak: FINALISTS[a.tiebreak]}
        SEED = 22
        NAME = f"zoo-high-round2-20261029-s22-{a.tiebreak.split('-')[0]}"
    if a.report:
        report()
        sys.exit(0)
    z.build_plan = build_plan
    if a.dry_run:
        plan = build_plan(z.inventory())
        for r in plan["runs"]:
            print(r["model_key"][:10], r["config_id"], r["level"], r["task_id"], r["options"]["num_ctx"],
                  r["options"]["num_predict"], z.runner.LEVELS[r["level"]]["seconds"])
        print(len(plan["runs"]), "episodes")
        sys.exit(0)
    sys.exit(z.run(argparse.Namespace(name=NAME, hours=10, new_deadline=True, models=None, profile="all",
                                      levels=["high"])))
