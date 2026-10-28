"""One-hour knockout (owner 2026-10-28): the High matrix left ties for gpt-oss, K2 and Nemotron; their High-matrix
results are set aside for config choice and the two tied configs of each play a short, wide test instead:
Light L11, L8, L1, L3 (the hardest Light tasks), Medium M5, M1, High H2 — model by model, task by task.
Shorter caps (Light 3 min, Medium 6 min, High 8 min) and a 70-minute deadline keep it to about an hour.
Ranking: Light/Medium by correct code, High by W (solved and verified); ties -> less time.

  python knockout_1h.py            -> runs/zoo-knockout-1h-20261028
  python knockout_1h.py --report   -> winner per model
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

NAME = "zoo-knockout-1h-20261028"
FINALISTS = {"k2-horizon-7b-q5": ["q5-template", "q5-template-syshint"],
             "gpt-oss-20b": ["low", "medium"],
             "nemotron-3.5-lightning-30b": ["vendor-th", "vendor-th-syshint"]}
# second batch (owner 2026-10-28): Qwen3-Coder tie, queued after the first run
CODER = {"qwen3-coder-30b": ["vendor", "vendor-syshint"]}
TASKS = [("light", "L11"), ("light", "L8"), ("light", "L1"), ("light", "L3"),
         ("medium", "M5"), ("medium", "M1"), ("high", "H2")]
SEED = 11
z.runner.LEVELS["light"]["seconds"] = 180
z.runner.LEVELS["medium"]["seconds"] = 360
z.runner.LEVELS["high"]["seconds"] = 480


def budget(model_key, level):
    b = hm.APPROVED[model_key]
    if level != "high" and "medium" in b:          # K2: its calibrated Medium budget also serves Light
        return b["medium"]["num_ctx"], b["medium"]["num_predict"]
    return b["num_ctx"], b["num_predict"]


def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
    templates = {}
    for level in ("light", "medium", "high"):
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
                         tool_mode="native", purpose="1h knockout of tied High configs",
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
    ap.add_argument("--coder", action="store_true", help="Qwen3-Coder batch -> runs/zoo-knockout-1h-coder-20261028")
    a = ap.parse_args()
    if a.coder:
        FINALISTS.clear()
        FINALISTS.update(CODER)
        NAME = "zoo-knockout-1h-coder-20261028"
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
    sys.exit(z.run(argparse.Namespace(name=NAME, hours=70 / 60, new_deadline=True, models=None, profile="all",
                                      levels=["light", "medium", "high"])))
