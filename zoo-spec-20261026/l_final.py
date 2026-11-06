"""L final (owner 2026-11-06: "запускаем тесты ... после определения победителя надо начинать этап мастермайнда").

Five configs tied at 4/6 after High round 2 (seed 11) replay the same hard tasks H1 H2 H4 H6 H7 H9 on seed 22,
same 12-minute limit, same budgets and options as in their round 2. Qwen3.6 last (both test-time crashes were on it).
Ranking: W over seed 11 + seed 22 (12 episodes), then fewer minutes in total.

  python l_final.py --dry-run
  python l_final.py --start [--hours 6.5] [--resume]
  python l_final.py --report
"""
import argparse
import collections
import json
import sys

import zoo_night as z
import high_round2 as hr2
import gemma12_high_matrix as g12
import new_models_high_screen as nm

NAME = "l-final-s22-20261106"
SEED = 22
# (model_key, config_id, seed-11 run)
FINAL = [("gpt-oss-20b", "low", "zoo-high-round2-20261029"),
         ("k2-horizon-7b-q5", "q5-template", "zoo-high-round2-20261029"),
         ("gemma4-12b", "b16-nt-official", "gemma4-12b-high-round2-20261031-rest"),
         ("flint-40b", "vendor-syshint", "new-models-high-round2-20261102"),
         ("qwen3.6-35b", "nt-instruct", "new-models-high-round2-20261102")]


def build_plan(inv, only=None, profile="all", levels=("high",)):
    hr2.SEED = SEED
    hr2.FINALISTS.clear()
    hr2.FINALISTS.update({"gpt-oss-20b": ["low"], "k2-horizon-7b-q5": ["q5-template"]})
    plan = hr2.build_plan(inv)
    g12.SEED, g12.ACTIVE, g12.TASKS = SEED, ["b16-nt-official"], g12.ROUND2
    gp = g12.build_plan(inv)
    nm.SEED, nm.TASKS = SEED, nm.ROUND2
    nm.ONLY = {"flint-40b/vendor-syshint", "qwen3.6-35b/nt-instruct"}
    nm.ACTIVE = ["flint-40b", "qwen3.6-35b"]
    np_ = nm.build_plan(inv)
    plan["runs"] = plan["runs"] + gp["runs"] + np_["runs"]
    plan.setdefault("originals", {}).update(gp["originals"], **np_["originals"])
    for r in plan["runs"]:
        assert r["seed"] == SEED and r["options"]["seed"] == SEED, r
        r["purpose"] = "L final seed 22"
    return plan


def scores(run):
    s = collections.defaultdict(lambda: [0, 0.0, 0, []])
    f = z.RUNS / run / "results.jsonl"
    if not f.exists():
        return s
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("outcome", "episode") != "episode":
            continue
        r = d.get("row") or {}
        x = s[(d["model_key"], d["config_id"])]
        ok = bool(r.get("automated_workflow_pass"))
        x[0] += ok
        x[1] += float(r.get("wall_seconds") or 0) / 60
        x[2] += 1
        if ok:
            x[3].append(d["task_id"])
    return s


def report():
    fin = scores(NAME)
    rows = []
    for key, cfg, run11 in FINAL:
        a, b = scores(run11)[(key, cfg)], fin[(key, cfg)]
        rows.append(dict(model=key, config=cfg, s11=f"{a[0]}/{a[2]}", s22=f"{b[0]}/{b[2]}", total=a[0] + b[0],
                         minutes=round(a[1] + b[1], 1), solved22=" ".join(sorted(b[3]))))
    rows.sort(key=lambda r: (-r["total"], r["minutes"]))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hours", type=float, default=6.5)
    a = ap.parse_args()
    assert z.runner.LEVELS["high"]["seconds"] == 720
    if a.report:
        report()
        sys.exit(0)
    z.build_plan = build_plan
    if a.dry_run:
        for r in build_plan(z.inventory())["runs"]:
            print(r["model_key"], r["config_id"], r["task_id"], r["think"], r["options"])
        sys.exit(0)
    if not a.start:
        raise SystemExit("Not started: pass --start.")
    if (z.RUNS / NAME).exists() and not a.resume:
        raise SystemExit(f"Refusing to overwrite existing run: {NAME}")
    if z.runner.api("ps").get("models"):
        raise SystemExit("Ollama has a model loaded (another run?); wait for it to finish.")
    try:
        sys.exit(z.run(argparse.Namespace(name=NAME, hours=a.hours, new_deadline=not a.resume, models=None,
                                          profile="all", levels=["high"])))
    finally:
        for tag in ("gpt-oss:20b", hr2.hm.K2Q5 if hasattr(hr2.hm, "K2Q5") else "", g12.TAG, "flint:40b", "qwen3.6:35b"):
            if tag:
                z._api("generate", dict(model=tag, keep_alive=0), 120)
