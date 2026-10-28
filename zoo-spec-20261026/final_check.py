"""Final High calibration check (owner 2026-10-28): the owner-approved budgets pass when every model played all
its episodes (a 12-minute time limit counts as played), no reply was cut and the context peak stayed <= 80%.
A large unused budget is NOT a failure here (the owner raised some budgets on purpose).
Exit 0 = all pass (the High matrix may start), 1 = something to decide.

  python final_check.py runs/zoo-high-calibrate-r4-20261028 --expect 3
"""
import argparse
import json
import sys

from calib_report import RUNS, HERE, episode_stats, slug

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--expect", type=int, required=True)
a = ap.parse_args()
night = HERE.parent / a.run
per = {}
for line in (night / "results.jsonl").read_text(encoding="utf-8").splitlines():
    d = json.loads(line)
    row, o = d.get("row") or {}, d.get("options") or {}
    m = per.setdefault(d["model_key"], dict(ctx=o.get("num_ctx"), cap=o.get("num_predict"), eps=0, infra=[], cut=0,
                                            peak=0, reply=0, solved=0, time_limit=0))
    if d.get("outcome", "episode") != "episode" or row.get("infra_error"):
        m["infra"].append(f"{d['task_id']}: {str(row.get('infra_error'))[:160]}")
        continue
    s = episode_stats(RUNS / f"{night.name}-{slug(d['model_key'])}-{slug(d['config_id'])}-s{d['seed']}-run" / d["task_id"])
    m.update(eps=m["eps"] + 1, cut=m["cut"] + s["cut"], peak=max(m["peak"], s["peak"]), reply=max(m["reply"], s["max_reply"]),
             solved=m["solved"] + bool(row.get("code_pass")), time_limit=m["time_limit"] + bool(row.get("time_limit")))
ok_all = True
for k, m in per.items():
    fill = m["peak"] / m["ctx"]
    ok = m["eps"] == a.expect and not m["cut"] and fill <= 0.8
    ok_all &= ok
    print(f"{'PASS' if ok else 'FAIL'} {k:28s} {m['ctx'] // 1024}K/{m['cap'] // 1024}K played {m['eps']}/{a.expect} "
          f"cut {m['cut']} peak {m['peak']} ({fill:.0%}) max reply {m['reply']} solved {m['solved']} "
          f"time-limit {m['time_limit']} {('infra: ' + ' | '.join(m['infra'])) if m['infra'] else ''}")
sys.exit(0 if ok_all else 1)
